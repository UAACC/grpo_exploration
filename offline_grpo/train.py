"""Step 2: Train a target policy with LoRA using offline GRPO.

Usage:
    python train.py \
        --rollout_path rollouts.jsonl \
        --target_model Qwen/Qwen2.5-1.5B-Instruct \
        --output_dir ./outputs
"""

import argparse
import json
import os
from datetime import datetime

# Disable accelerate's automatic fp32 cast of model outputs BEFORE any
# accelerate / trl / transformers import that might capture the function
# reference. With mixed_precision=bf16 set in the yaml, the model itself
# runs in bf16, but accelerate wraps `forward` with `convert_outputs_to_fp32`,
# which does `tensor.float()` on the logits. At long context that's
# `[batch * num_gen, seq, vocab] * 4 bytes` for *each* of student + reference,
# i.e. ~37 GiB per step at 8K seq × 152K vocab × 4 generations -> OOM on L40s.
# Making the cast a no-op leaves logits in bf16 throughout; the trainer's
# loss computation tolerates bf16 inputs.
import accelerate.utils.operations as _accel_ops
_accel_ops.convert_to_fp32 = lambda tensor: tensor

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig

from configs import DEFAULT_TARGET_MODEL, DEFAULT_LORA_CONFIG
from data import load_rollouts, compute_rewards_and_advantages, build_training_dataset, build_offline_lookup
from trainer import OfflineGRPOTrainer


class MetricsFileCallback(TrainerCallback):
    """Write all logged metrics to a JSONL file at each logging step."""

    def __init__(self, output_path: str):
        self._path = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(self._path, "w") as f:
            f.write("")  # create/truncate

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        record = {"step": state.global_step, "epoch": state.epoch}
        record.update(logs)
        with open(self._path, "a") as f:
            f.write(json.dumps(record) + "\n")


def parse_args():
    p = argparse.ArgumentParser(description="Offline GRPO training with LoRA.")
    # Model
    p.add_argument("--target_model", type=str, default=DEFAULT_TARGET_MODEL)
    p.add_argument("--behavior_model", type=str, default=None, help="For logging only.")
    # Data
    p.add_argument("--rollout_path", type=str, required=True)
    # Output
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    # Training hyperparams
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1, help="KL penalty coefficient.")
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--max_prompt_length", type=int, default=256)
    p.add_argument("--max_completion_length", type=int, default=786)
    p.add_argument("--max_grad_norm", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--logging_steps", type=int, default=1)
    # LoRA
    p.add_argument("--lora_r", type=int, default=DEFAULT_LORA_CONFIG["r"])
    p.add_argument("--lora_alpha", type=int, default=DEFAULT_LORA_CONFIG["lora_alpha"])
    p.add_argument("--no_lora", action="store_true", help="Disable LoRA (full fine-tune).")
    # Reference sync
    p.add_argument("--ref_sync_steps", type=int, default=0,
                    help="Sync reference LoRA adapter every N steps. 0=never (use original base model).")
    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--report_to", type=str, default="wandb")
    p.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from checkpoint dir or 'latest'.")
    return p.parse_args()


def main():
    args = parse_args()
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    # ---- 1. Load student tokenizer (needed by the rollout-safety check) -
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    tokenizer.pad_token = tokenizer.eos_token

    # ---- 2. Load & process offline rollouts ----------------------------
    # `vocab_size` truncates teacher IDs that exceed the student's embedding
    # rows. Passing `student_tokenizer` additionally enforces that the
    # remaining IDs round-trip cleanly under the student tokenizer (the
    # Path-A safety check) — we refuse to feed teacher IDs into the student
    # forward pass when the tokenizers disagree at any in-range ID.
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(args.target_model)
    print(f"Loading rollouts from {args.rollout_path} ...")
    records = load_rollouts(
        args.rollout_path,
        vocab_size=model_config.vocab_size,
        student_tokenizer=tokenizer,
    )
    print(f"  {len(records)} completions loaded")

    records = compute_rewards_and_advantages(records)
    correct = sum(1 for r in records if r["reward"] > 0)
    print(f"  Rewards: {correct}/{len(records)} correct ({100*correct/len(records):.1f}%)")

    dataset = build_training_dataset(records)
    offline_data = build_offline_lookup(records)
    print(f"  Dataset: {len(dataset)} rows, {len(offline_data)} offline entries")

    # ---- 2. Load target model ------------------------------------------
    print(f"Loading target model: {args.target_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None,
    )
    model.config.use_cache = False

    # ---- 3. LoRA config ------------------------------------------------
    peft_config = None
    if not args.no_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=DEFAULT_LORA_CONFIG["target_modules"],
            task_type=DEFAULT_LORA_CONFIG["task_type"],
            lora_dropout=DEFAULT_LORA_CONFIG["lora_dropout"],
        )
        print(f"  LoRA: r={args.lora_r}, alpha={args.lora_alpha}")

    # ---- 4. Training config --------------------------------------------
    if args.output_dir is None:
        args.output_dir = f"./outputs/offline_grpo_{time_str}"
    if args.run_name is None:
        args.run_name = f"offline-grpo-{args.target_model.split('/')[-1]}-{time_str}"

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        learning_rate=args.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        beta=args.beta,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        max_grad_norm=args.max_grad_norm,
        report_to=args.report_to,
        log_on_each_node=False,
        seed=args.seed,
    )

    # ---- 5. Create trainer & train -------------------------------------
    # Dummy reward function (never called — advantages are pre-computed)
    def _dummy_reward(prompts, completions, **kwargs):
        return [0.0] * len(completions)

    metrics_path = os.path.join(args.output_dir, "training_metrics.jsonl")
    metrics_callback = MetricsFileCallback(metrics_path)

    trainer = OfflineGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[_dummy_reward],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        offline_data=offline_data,
        ref_sync_steps=args.ref_sync_steps,
        callbacks=[metrics_callback],
    )
    print(f"Beta: {trainer.beta}, LoRA: {peft_config is not None}, ref_sync_steps: {args.ref_sync_steps}")

    # Log config to wandb
    if args.report_to == "wandb" and trainer.accelerator.is_main_process:
        import wandb
        if wandb.run is not None:
            wandb.config.update({
                "target_model": args.target_model,
                "behavior_model": args.behavior_model or "unknown",
                "rollout_path": args.rollout_path,
                "num_completions": len(records),
                "num_problems": len(records) // args.num_generations,
                "reward_accuracy": correct / len(records),
                "beta": args.beta,
                "lora_r": args.lora_r if not args.no_lora else None,
                "lora_alpha": args.lora_alpha if not args.no_lora else None,
                "lora_enabled": not args.no_lora,
                "ref_sync_steps": args.ref_sync_steps,
            })

    print("Starting training...")

    resume = args.resume_from_checkpoint
    if resume == "latest":
        resume = True  # Trainer will find the last checkpoint in output_dir
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.output_dir)
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
