"""Online GRPO training using TRL's native GRPOTrainer.

The student model generates its own completions via vLLM during training.
No teacher model or pre-computed rollouts needed. Supports any dataset
registered in shared/datasets_registry.py.

Usage:
    accelerate launch --config_file configs/accelerate_ddp_4gpu.yaml \
        train.py --dataset_type math --output_dir ./outputs
    accelerate launch --config_file configs/accelerate_ddp_4gpu.yaml \
        train.py --dataset_type svamp --output_dir ./outputs
"""

import argparse
import os
import sys
from datetime import datetime

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# Add shared/ to path for the dataset registry
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
from datasets_registry import get_dataset_config, load_eval_data, list_datasets

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def make_reward_func(cfg):
    """Create a TRL-compatible reward function from a dataset config.

    TRL's GRPOTrainer calls reward_func(completions, answer, **kwargs)
    where completions is a list of strings and answer is a list of gold answers.
    """
    def reward_func(completions, answer, **kwargs):
        rewards = []
        for completion, gold in zip(completions, answer):
            if isinstance(completion, list):
                gen_text = completion[-1]["content"] if completion else ""
            else:
                gen_text = str(completion)
            rewards.append(cfg.reward_func(gen_text, gold))
        return rewards
    return reward_func


def prepare_dataset(cfg, split="train"):
    """Load dataset and format for GRPOTrainer.

    GRPOTrainer expects a dataset with 'prompt' (chat messages) and 'answer' columns.
    Uses the dataset registry to handle field name differences across datasets.
    """
    from datasets import Dataset

    manual_split = "train" if cfg.name == "asdiv" else None
    problems = load_eval_data(cfg, split=cfg.split_train, manual_split=manual_split)

    prompts = []
    answers = []
    for prob in problems:
        prompts.append([
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": prob["problem"]},
        ])
        answers.append(prob["answer"])

    return Dataset.from_dict({"prompt": prompts, "answer": answers})


def parse_args():
    p = argparse.ArgumentParser(description="Online GRPO with LoRA.")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--dataset_type", type=str, default="gsm8k",
                    choices=list_datasets(),
                    help="Dataset to train on (from shared/datasets_registry.py).")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--learning_rate", type=float, default=3e-6)
    p.add_argument("--beta", type=float, default=0.001)
    p.add_argument("--num_generations", type=int, default=5)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--num_train_epochs", type=int, default=15)
    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--max_completion_length", type=int, default=1024)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--no_lora", action="store_true")
    p.add_argument("--no_vllm", action="store_true")
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--report_to", type=str, default="wandb")
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Resolve dataset config from registry
    cfg = get_dataset_config(args.dataset_type)
    reward_func = make_reward_func(cfg)

    # Use dataset defaults for max tokens if current args are too small
    args.max_completion_length = max(args.max_completion_length, cfg.max_tokens)

    # Load dataset
    print(f"Loading dataset: {cfg.name} ({cfg.hf_path})")
    train_dataset = prepare_dataset(cfg, split="train")
    print(f"  {len(train_dataset)} training examples")

    # Load model
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
    peft_config = None
    if not args.no_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
        )
        print(f"  LoRA: r={args.lora_r}, alpha={args.lora_alpha}, target=all-linear")

    # Training config
    if args.output_dir is None:
        args.output_dir = f"./outputs/online_grpo_{cfg.name}_{time_str}"
    if args.run_name is None:
        args.run_name = f"online-grpo-{cfg.name}-{args.model.split('/')[-1]}-{time_str}"

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        learning_rate=args.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        beta=args.beta,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        use_vllm=not args.no_vllm,
        **({"vllm_mode": "colocate",
            "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization}
           if not args.no_vllm else {}),
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to=args.report_to,
        log_on_each_node=False,
        bf16=args.bf16,
        seed=args.seed,
    )

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )
    print(f"Dataset: {cfg.name}, Beta: {trainer.beta}")
    print(f"Generation: {'vLLM colocate' if not args.no_vllm else 'native PyTorch'}")

    if args.report_to == "wandb" and trainer.accelerator.is_main_process:
        import wandb
        if wandb.run is not None:
            wandb.config.update({
                "model": args.model,
                "dataset": cfg.name,
                "dataset_hf_path": cfg.hf_path,
                "num_train_examples": len(train_dataset),
                "beta": args.beta,
                "num_generations": args.num_generations,
                "lora_r": args.lora_r if not args.no_lora else None,
                "lora_alpha": args.lora_alpha if not args.no_lora else None,
            })

    print("Starting online GRPO training...")
    resume = args.resume_from_checkpoint
    if resume == "latest":
        resume = True
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.output_dir)
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
