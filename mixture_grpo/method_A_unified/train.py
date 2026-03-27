"""Train with Method A: Unified Mixture GRPO.

Student rollouts are generated online. Teacher rollouts are loaded from a
pre-computed JSONL file. Both are pooled into a single group.

Usage:
    python method_A_unified/train.py \
        --teacher_rollout_path rollouts.jsonl \
        --target_model Qwen/Qwen2.5-0.5B-Instruct \
        --output_dir ./outputs
"""

import argparse
import sys
import os
from datetime import datetime

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from trl import GRPOConfig
from datasets import load_dataset

# Add parent dir to path for shared modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import (DEFAULT_TARGET_MODEL, DEFAULT_LORA_CONFIG,
                     GSM8K_SYSTEM_PROMPT, GSM8K_DATASET,
                     MATH_SYSTEM_PROMPT, MATH_DATASET)
from data import load_teacher_rollouts
from method_A_unified.trainer import UnifiedMixtureGRPOTrainer


def parse_args():
    p = argparse.ArgumentParser(description="Unified Mixture GRPO (Method A)")
    # Model
    p.add_argument("--target_model", type=str, default=DEFAULT_TARGET_MODEL)
    # Data
    p.add_argument("--teacher_rollout_path", type=str, required=True)
    p.add_argument("--dataset_type", type=str, default="gsm8k",
                    choices=["gsm8k", "math"], help="Dataset type.")
    p.add_argument("--dataset", type=str, default=None,
                    help="Override dataset path. Defaults based on dataset_type.")
    p.add_argument("--split", type=str, default="train")
    # Output
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    # Training hyperparams
    p.add_argument("--learning_rate", type=float, default=3e-6)
    p.add_argument("--beta", type=float, default=0.01)
    p.add_argument("--num_generations", type=int, default=5, help="Student completions per prompt.")
    p.add_argument("--num_teacher_per_prompt", type=int, default=1, help="Teacher completions per prompt.")
    p.add_argument("--per_device_train_batch_size", type=int, default=5)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--num_train_epochs", type=int, default=15)
    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--max_completion_length", type=int, default=1024)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--logging_steps", type=int, default=1)
    # LoRA
    p.add_argument("--lora_r", type=int, default=DEFAULT_LORA_CONFIG["r"])
    p.add_argument("--lora_alpha", type=int, default=DEFAULT_LORA_CONFIG["lora_alpha"])
    p.add_argument("--no_lora", action="store_true")
    # Reference sync
    p.add_argument("--ref_sync_steps", type=int, default=0)
    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--report_to", type=str, default="wandb")
    p.add_argument("--resume_from_checkpoint", type=str, default=None,
                    help="Resume from checkpoint dir or 'latest'.")
    return p.parse_args()


def main():
    args = parse_args()
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    # ---- 1. Resolve dataset defaults ------------------------------------
    is_math = args.dataset_type == "math"
    if args.dataset is None:
        args.dataset = MATH_DATASET if is_math else GSM8K_DATASET
    system_prompt = MATH_SYSTEM_PROMPT if is_math else GSM8K_SYSTEM_PROMPT
    question_field = "problem" if is_math else "question"

    # ---- 2. Load teacher rollouts --------------------------------------
    model_config = AutoConfig.from_pretrained(args.target_model)
    print(f"Loading teacher rollouts from {args.teacher_rollout_path} ...")
    teacher_data = load_teacher_rollouts(args.teacher_rollout_path,
                                         vocab_size=model_config.vocab_size,
                                         dataset_type=args.dataset_type)
    print(f"  {len(teacher_data)} problems with teacher rollouts")

    total_runs = sum(len(v["runs"]) for v in teacher_data.values())
    correct_runs = sum(sum(1 for r in v["runs"] if r["reward"] > 0) for v in teacher_data.values())
    print(f"  Teacher accuracy: {correct_runs}/{total_runs} ({100*correct_runs/total_runs:.1f}%)")

    # ---- 3. Build training dataset (prompts only) ----------------------
    if is_math:
        data = load_dataset(args.dataset)[args.split]
    else:
        data = load_dataset(args.dataset, "main", split=args.split)
    print(f"  Dataset ({args.dataset_type}): {len(data)} problems ({args.split} split)")

    prompts, answers, qids = [], [], []
    for i, item in enumerate(data):
        prompts.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item[question_field]},
        ])
        answers.append(item["answer"])
        qids.append(i)

    from datasets import Dataset
    dataset = Dataset.from_dict({
        "prompt": prompts,
        "answer": answers,
        "question_id": qids,
    })

    # ---- 3. Load target model ------------------------------------------
    print(f"Loading target model: {args.target_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    tokenizer.pad_token = tokenizer.eos_token

    # ---- 4. LoRA config ------------------------------------------------
    peft_config = None
    if not args.no_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=DEFAULT_LORA_CONFIG["target_modules"],
            task_type=DEFAULT_LORA_CONFIG["task_type"],
            lora_dropout=DEFAULT_LORA_CONFIG["lora_dropout"],
        )

    # ---- 5. Training config --------------------------------------------
    if args.output_dir is None:
        args.output_dir = f"./outputs/mixture_A_{time_str}"
    if args.run_name is None:
        args.run_name = f"mixture-A-{args.target_model.split('/')[-1]}-{time_str}"

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        learning_rate=args.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        beta=args.beta,
        weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        max_grad_norm=args.max_grad_norm,
        report_to=args.report_to,
        log_on_each_node=False,
        seed=args.seed,
    )

    # ---- 6. Create trainer & train -------------------------------------
    def _dummy_reward(prompts, completions, **kwargs):
        return [0.0] * len(completions)

    trainer = UnifiedMixtureGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[_dummy_reward],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        teacher_data=teacher_data,
        num_teacher_per_prompt=args.num_teacher_per_prompt,
        ref_sync_steps=args.ref_sync_steps,
        dataset_type=args.dataset_type,
    )

    print(f"Method A (Unified Mixture GRPO)")
    print(f"  Student generations: {args.num_generations}, Teacher per prompt: {args.num_teacher_per_prompt}")
    print(f"  Beta: {trainer.beta}, ref_sync_steps: {args.ref_sync_steps}")
    print("Starting training...")

    resume = args.resume_from_checkpoint
    if resume == "latest":
        resume = True
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.output_dir)
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
