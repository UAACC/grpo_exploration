"""Train with RWR-style offline GRPO.

Uses pre-collected teacher rollouts with signed correctness rewards:
+1 for correct answers and -1 for wrong answers.

Usage:
    accelerate launch --config_file configs/accelerate_ddp_4gpu.yaml \
        train.py --rollout_path /path/to/rollouts.jsonl \
                 --target_model /path/to/student \
                 --dg_temperature 1.0
"""

import argparse
import json
import os
import sys
from datetime import datetime

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig

# Import local trainer BEFORE adding offline_grpo to sys.path
# (both have trainer.py — local must resolve first)
from trainer import RWROfflineTrainer

# Add offline_grpo for shared data loading utilities
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_this_dir)
sys.path.insert(0, os.path.join(_project_root, "offline_grpo"))

from data import load_rollouts, compute_rewards_and_advantages, build_training_dataset, build_offline_lookup


def assign_signed_correctness_rewards(records: list[dict]) -> list[dict]:
    """Use signed correctness rewards: +1 for correct, -1 for wrong."""
    for rec in records:
        rec["reward"] = 1.0 if rec.get("reward", 0.0) > 0.0 else -1.0
    return records


class MetricsFileCallback(TrainerCallback):
    """Write all logged metrics to a JSONL file at each logging step."""

    def __init__(self, output_path: str):
        self._path = output_path
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
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
    p = argparse.ArgumentParser(description="Reward Weighted Regression training.")
    # Model
    p.add_argument("--target_model", type=str, required=True,
                    help="Path to student model (local or HF hub).")
    p.add_argument("--behavior_model", type=str, default=None,
                    help="Name of behavior model (for logging only).")
    # Data
    p.add_argument("--rollout_path", type=str, required=True,
                    help="Path to teacher rollouts JSONL.")
    # Output
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    # DG-specific hyperparams
    p.add_argument("--dg_temperature", type=float, default=1.0,
                    help="eta in sigmoid(delight/eta). Higher = softer gate. Deprecated.")
    p.add_argument("--dg_gating", type=str, default="completion",
                    choices=["completion", "token"],
                    help="Gating granularity: per-completion or per-token. Deprecated.")
    # Training hyperparams
    p.add_argument("--learning_rate", type=float, default=3e-6)
    p.add_argument("--beta", type=float, default=0.001,
                    help="KL penalty coefficient.")
    p.add_argument("--num_generations", type=int, default=4,
                    help="Completions per prompt in the rollout data.")
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--max_prompt_length", type=int, default=256)
    p.add_argument("--max_completion_length", type=int, default=2048)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--logging_steps", type=int, default=1)
    # LoRA
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_target_modules", type=str, default="all-linear",
                    help="Comma-separated list or 'all-linear'.")
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--no_lora", action="store_true",
                    help="Disable LoRA (full fine-tune).")
    # Reference sync
    p.add_argument("--ref_sync_steps", type=int, default=0,
                    help="Sync reference LoRA every N steps. 0=never.")
    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--report_to", type=str, default="wandb")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    # ---- 1. Load & process offline rollouts ----------------------------
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(args.target_model)
    print(f"Loading rollouts from {args.rollout_path} ...")
    records = load_rollouts(args.rollout_path, vocab_size=model_config.vocab_size)
    print(f"  {len(records)} completions loaded")

    records = compute_rewards_and_advantages(records)
    records = assign_signed_correctness_rewards(records)
    correct = sum(1 for r in records if r["reward"] > 0)
    print(f"  Signed rewards: {correct}/{len(records)} correct ({100*correct/len(records):.1f}%)")

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

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    tokenizer.pad_token = tokenizer.eos_token

    # ---- 3. LoRA config ------------------------------------------------
    peft_config = None
    if not args.no_lora:
        target_modules = args.lora_target_modules
        if target_modules != "all-linear":
            target_modules = [m.strip() for m in target_modules.split(",")]
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
            lora_dropout=args.lora_dropout,
        )
        print(f"  LoRA: r={args.lora_r}, alpha={args.lora_alpha}, targets={target_modules}")

    # ---- 4. Training config --------------------------------------------
    if args.output_dir is None:
        args.output_dir = f"./outputs/RWR_offline_{time_str}"
    if args.run_name is None:
        model_name = args.target_model.rstrip("/").split("/")[-1]
        args.run_name = f"RWR-offline-{model_name}-{time_str}"

    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

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
    def _dummy_reward(prompts, completions, **kwargs):
        return [0.0] * len(completions)

    metrics_path = os.path.join(args.output_dir, "training_metrics.jsonl")
    metrics_callback = MetricsFileCallback(metrics_path)

    trainer = RWROfflineTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[_dummy_reward],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        offline_data=offline_data,
        dg_temperature=args.dg_temperature,
        dg_gating=args.dg_gating,
        ref_sync_steps=args.ref_sync_steps,
        callbacks=[metrics_callback],
    )

    print(f"=== RWR Offline GRPO ===")
    print(f"  temperature (eta), deprecated: {args.dg_temperature}")
    print(f"  DG gating constant 1")
    print(f"  Beta (KL): {trainer.beta}")
    print(f"  LoRA: {peft_config is not None}")
    print(f"  ref_sync_steps: {args.ref_sync_steps}")

    if args.report_to == "wandb" and trainer.accelerator.is_main_process:
        import wandb
        if wandb.run is not None:
            wandb.config.update({
                "method": "rwr-offline-grpo",
                "reward_regime": "signed_correctness_minus1_plus1",
                "target_model": args.target_model,
                "behavior_model": args.behavior_model or "unknown",
                "rollout_path": args.rollout_path,
                "num_completions": len(records),
                "num_problems": len(records) // args.num_generations,
                "reward_accuracy": correct / len(records),
                "dg_temperature": args.dg_temperature,
                "dg_gating": args.dg_gating,
                "beta": args.beta,
                "lora_r": args.lora_r if not args.no_lora else None,
                "lora_alpha": args.lora_alpha if not args.no_lora else None,
            })

    print("Starting training...")
    resume = args.resume_from_checkpoint
    if resume == "latest":
        resume = True
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.output_dir)
    print(f"Model saved to: {args.output_dir}")

    print(f"\n=== Metrics saved to: {metrics_path} ===")


if __name__ == "__main__":
    main()
