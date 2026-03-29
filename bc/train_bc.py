"""Offline Behavioral Cloning: train student on teacher completions via cross-entropy.

Usage:
    accelerate launch train_bc.py \
        --rollout_path /scratch/mrli/rollouts/math_teacher/rollouts_full.jsonl \
        --target_model /scratch/mrli/models/Qwen2.5-0.5B-Instruct \
        --output_dir /scratch/mrli/checkpoints/bc_math
"""

import argparse
import json
import os
import sys
from datetime import datetime

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# Import local bc/data.py first (before sys.path modification)
from data import BCCollator, BCDataset

# Add offline_grpo to path for configs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "offline_grpo"))
from configs import DEFAULT_LORA_CONFIG, DEFAULT_TARGET_MODEL


class MetricsFileCallback(TrainerCallback):
    """Write all logged metrics to a JSONL file."""

    def __init__(self, output_path: str):
        self._path = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(self._path, "w") as f:
            f.write("")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        record = {"step": state.global_step, "epoch": state.epoch}
        record.update(logs)
        with open(self._path, "a") as f:
            f.write(json.dumps(record) + "\n")


def parse_args():
    p = argparse.ArgumentParser(description="Offline BC training with LoRA.")
    p.add_argument("--target_model", type=str, default=DEFAULT_TARGET_MODEL)
    p.add_argument("--rollout_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--run_name", type=str, default=None)
    # Data
    p.add_argument("--max_length", type=int, default=2304,
                    help="Max sequence length (prompt + completion).")
    p.add_argument("--filter_correct_only", action="store_true",
                    help="Only train on correct completions (reward > 0).")
    # Training hyperparams (aligned with offline GRPO controlled experiment)
    p.add_argument("--learning_rate", type=float, default=3e-6)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--logging_steps", type=int, default=5)
    # LoRA
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--no_lora", action="store_true")
    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--report_to", type=str, default="wandb")
    return p.parse_args()


def main():
    args = parse_args()
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.run_name is None:
        args.run_name = f"bc-{args.target_model.split('/')[-1]}-{time_str}"

    # ---- 1. Load model & tokenizer -------------------------------------
    print(f"Loading model: {args.target_model}")
    model_config = AutoConfig.from_pretrained(args.target_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    tokenizer.pad_token = tokenizer.eos_token

    # ---- 2. LoRA -------------------------------------------------------
    if not args.no_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=DEFAULT_LORA_CONFIG["target_modules"],
            task_type=DEFAULT_LORA_CONFIG["task_type"],
            lora_dropout=DEFAULT_LORA_CONFIG["lora_dropout"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # ---- 3. Load dataset -----------------------------------------------
    print(f"Loading rollouts from {args.rollout_path} ...")
    dataset = BCDataset(
        rollout_path=args.rollout_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        vocab_size=model_config.vocab_size,
        filter_correct_only=args.filter_correct_only,
    )

    # Sanity check: print one sample
    sample = dataset[0]
    prompt_len = (sample["labels"] == -100).sum().item()
    comp_len = (sample["labels"] != -100).sum().item()
    print(f"  Sample 0: prompt_len={prompt_len}, comp_len={comp_len}, total={len(sample['input_ids'])}")

    # ---- 4. Training config --------------------------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,
        learning_rate=args.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_grad_norm=args.max_grad_norm,
        bf16=args.bf16,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        log_on_each_node=False,
        seed=args.seed,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    # ---- 5. Trainer & train --------------------------------------------
    collator = BCCollator(pad_token_id=tokenizer.pad_token_id)
    metrics_path = os.path.join(args.output_dir, "training_metrics.jsonl")
    metrics_callback = MetricsFileCallback(metrics_path)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[metrics_callback],
    )

    # Log config to wandb
    if args.report_to == "wandb" and trainer.accelerator.is_main_process:
        import wandb
        if wandb.run is not None:
            wandb.config.update({
                "target_model": args.target_model,
                "rollout_path": args.rollout_path,
                "num_samples": len(dataset),
                "max_length": args.max_length,
                "filter_correct_only": args.filter_correct_only,
                "lora_r": args.lora_r if not args.no_lora else None,
                "lora_alpha": args.lora_alpha if not args.no_lora else None,
                "lora_enabled": not args.no_lora,
                "method": "bc",
            })

    print("Starting BC training...")
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
