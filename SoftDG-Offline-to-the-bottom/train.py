"""SoftDG-Offline training: gate-threshold skipping on offline rollouts.

New vs DG-offline/train.py:
  --reward_coding  : "zero_two" (wrong=0, correct=2) or "signed" (wrong=-1, correct=+1)
  --training_signal: "advantage" (group-normalized) or "raw_reward" (raw value)
  --softdg_gate_threshold: skip completions where gate < threshold
  --target_effective_completions: stop when this many non-zero-signal completions seen
  --num_train_epochs defaults to 20 (early stop via counter; loops dataset as needed)

Usage:
    accelerate launch --config_file configs/accelerate_ddp_4gpu.yaml \\
        train.py \\
        --rollout_path /scratch/shuai14/rollouts/math_teacher/rollouts_math_Qwen2.5-0.5B-Instruct_0.6_pick4.jsonl \\
        --target_model /scratch/shuai14/models/Qwen2.5-0.5B \\
        --reward_coding zero_two \\
        --training_signal advantage \\
        --softdg_gate_threshold 0.2 \\
        --target_effective_completions 48000
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig

# Make DG-offline loader importable.
# _HERE is already sys.path[0] (Python adds the script dir automatically).
# Insert _DG_OFFLINE at position 1 so the local trainer.py (at _HERE) always
# wins over DG-offline/trainer.py when Python resolves "from trainer import ...".
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.abspath(os.path.join(_HERE, ".."))
_DG_OFFLINE = os.path.join(_PROJECT, "DG-offline")
if _DG_OFFLINE not in sys.path:
    sys.path.insert(1, _DG_OFFLINE)

from teacher_agnostic_loader import (
    load_rollouts_text,
    compute_rewards_and_advantages,
    build_training_dataset,
    build_offline_lookup,
)
from trainer import (
    SoftDGOfflineTrainer,
    EffectiveCompletionCounter,
    EarlyStopOnTargetCallback,
)


# ---------------------------------------------------------------------------
# Reward coding helpers
# ---------------------------------------------------------------------------

def remap_rewards_signed(records: list[dict]) -> list[dict]:
    """Remap rewards to signed coding: correct=+1, wrong=-1."""
    for rec in records:
        rec["reward"] = 1.0 if rec.get("reward", 0.0) > 0.0 else -1.0
    return records


def recompute_advantages(records: list[dict], eps: float = 1e-4) -> list[dict]:
    """Recompute group-normalized advantages from current reward values."""
    groups: dict = defaultdict(list)
    for rec in records:
        groups[rec["question_id"]].append(rec)
    for group in groups.values():
        rewards = [r["reward"] for r in group]
        mean_r = sum(rewards) / len(rewards)
        std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5
        for rec in group:
            rec["advantage"] = (rec["reward"] - mean_r) / (std_r + eps)
    return records


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def infer_num_generations(records: list[dict]) -> int:
    counts = Counter(rec["question_id"] for rec in records)
    values = set(counts.values())
    if len(values) != 1:
        sample = sorted(counts.items())[:5]
        raise ValueError(f"Non-uniform completions per question: {sample} ...")
    return values.pop()


class MetricsFileCallback(TrainerCallback):
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


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SoftDG offline training with gate-threshold skipping.")
    # Model
    p.add_argument("--target_model", type=str, required=True)
    p.add_argument("--behavior_model", type=str, default=None)
    # Data
    p.add_argument("--rollout_path", type=str, nargs="+", required=True)
    # Output
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    # SoftDG-specific
    p.add_argument("--reward_coding", type=str, default="zero_two",
                   choices=["zero_two", "signed"],
                   help="Reward coding: 'zero_two' (wrong=0, correct=2) or 'signed' (wrong=-1, correct=+1).")
    p.add_argument("--training_signal", type=str, default="advantage",
                   choices=["advantage", "raw_reward"],
                   help="Training signal fed to the gate and loss.")
    p.add_argument("--loss_type", type=str, default="grpo",
                   choices=["grpo", "dr_grpo"],
                   help="GRPO loss formulation: 'grpo' (sequence-normalized) or 'dr_grpo' (constant-length normalization).")
    p.add_argument("--dg_gating", type=str, default="completion",
                   choices=["completion", "token"],
                   help="DG gating mode: 'completion' (mean surprisal) or 'token' (per-token gate averaged).")
    p.add_argument("--softdg_gate_threshold", type=float, default=0.2,
                   help="Gate threshold: completions with gate < threshold are skipped.")
    p.add_argument("--target_effective_completions", type=int, default=48000,
                   help="Stop when this many completions with nonzero effective signal are seen.")
    p.add_argument("--dg_temperature", type=float, default=1.0,
                   help="eta in sigmoid(delight/eta).")
    # Training hyperparams
    p.add_argument("--learning_rate", type=float, default=3e-6)
    p.add_argument("--beta", type=float, default=0.001)
    p.add_argument("--num_generations", type=int, default=None)
    p.add_argument("--per_device_train_batch_size", type=int, default=3)
    p.add_argument("--gradient_accumulation_steps", type=int, default=3)
    p.add_argument("--num_train_epochs", type=int, default=20,
                   help="Max epochs. Early stop triggers before this if target is reached.")
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--max_prompt_length", type=int, default=256)
    p.add_argument("--max_completion_length", type=int, default=2048)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--logging_steps", type=int, default=5)
    # LoRA
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_target_modules", type=str, default="all-linear")
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--no_lora", action="store_true")
    # Misc
    p.add_argument("--ref_sync_steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--report_to", type=str, default="wandb")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    # ---- 1. Tokenizer ---------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    tokenizer.pad_token = tokenizer.eos_token

    # ---- 2. Load rollouts -----------------------------------------------
    rollout_paths = ", ".join(args.rollout_path)
    print(f"Loading rollouts from {rollout_paths} ...")
    records = load_rollouts_text(args.rollout_path, tokenizer)
    print(f"  {len(records)} completions loaded")

    if args.num_generations is None:
        args.num_generations = infer_num_generations(records)
        print(f"  Inferred num_generations={args.num_generations}")

    # ---- 3. Compute rewards (default: zero_two, 0/2) --------------------
    records = compute_rewards_and_advantages(records)

    # Apply reward coding BEFORE (re)computing advantages.
    if args.reward_coding == "signed":
        records = remap_rewards_signed(records)
        records = recompute_advantages(records)  # recompute from -1/+1 rewards
        print(f"  Reward coding: signed (-1/+1); advantages recomputed")
    else:
        print(f"  Reward coding: zero_two (0/2); advantages from loader")

    correct = sum(1 for r in records if r["reward"] > 0)
    print(f"  Rewards: {correct}/{len(records)} correct ({100*correct/len(records):.1f}%)")
    print(f"  Training signal: {args.training_signal}")

    dataset = build_training_dataset(records)
    offline_data = build_offline_lookup(records)
    print(f"  Dataset: {len(dataset)} rows, {len(offline_data)} offline entries")

    # ---- 4. Load model --------------------------------------------------
    print(f"Loading target model: {args.target_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None,
    )
    model.config.use_cache = False

    # ---- 5. LoRA --------------------------------------------------------
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

    # ---- 6. Training config ---------------------------------------------
    if args.output_dir is None:
        model_name = args.target_model.rstrip("/").split("/")[-1]
        args.output_dir = (
            f"./outputs/softdg_{args.reward_coding}_{args.training_signal}"
            f"_{args.loss_type}_{args.dg_gating}"
            f"_thr{args.softdg_gate_threshold}_{time_str}"
        )
    if args.run_name is None:
        model_name = args.target_model.rstrip("/").split("/")[-1]
        args.run_name = (
            f"softdg-{model_name}-{args.reward_coding}-{args.training_signal}"
            f"-{args.loss_type}-{args.dg_gating}"
            f"-thr{args.softdg_gate_threshold}-eta{args.dg_temperature}-{time_str}"
        )

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
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        max_grad_norm=args.max_grad_norm,
        report_to=args.report_to,
        log_on_each_node=False,
        seed=args.seed,
        loss_type=args.loss_type,
    )

    # ---- 7. Counter and callbacks ---------------------------------------
    counter = EffectiveCompletionCounter(target=args.target_effective_completions)
    early_stop_cb = EarlyStopOnTargetCallback(counter)
    metrics_path = os.path.join(args.output_dir, "training_metrics.jsonl")
    metrics_cb = MetricsFileCallback(metrics_path)

    # ---- 8. Trainer -----------------------------------------------------
    def _dummy_reward(prompts, completions, **kwargs):
        return [0.0] * len(completions)

    trainer = SoftDGOfflineTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[_dummy_reward],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        offline_data=offline_data,
        training_signal=args.training_signal,
        dg_temperature=args.dg_temperature,
        dg_gating=args.dg_gating,
        softdg_gate_threshold=args.softdg_gate_threshold,
        counter=counter,
        ref_sync_steps=args.ref_sync_steps,
        callbacks=[early_stop_cb, metrics_cb],
    )

    print(f"=== SoftDG Offline GRPO ===")
    print(f"  reward_coding:               {args.reward_coding}")
    print(f"  training_signal:             {args.training_signal}")
    print(f"  loss_type:                   {args.loss_type}")
    print(f"  dg_gating:                   {args.dg_gating}")
    print(f"  softdg_gate_threshold:       {args.softdg_gate_threshold}")
    print(f"  target_effective_completions:{args.target_effective_completions}")
    print(f"  dg_temperature (eta):        {args.dg_temperature}")
    print(f"  beta (KL):                   {trainer.beta}")
    print(f"  LoRA:                        {peft_config is not None}")
    print(f"  num_train_epochs (max):      {args.num_train_epochs}")

    if args.report_to == "wandb" and trainer.accelerator.is_main_process:
        import wandb
        if wandb.run is not None:
            wandb.config.update({
                "method": "softdg-offline-grpo",
                "reward_coding": args.reward_coding,
                "training_signal": args.training_signal,
                "loss_type": args.loss_type,
                "dg_gating": args.dg_gating,
                "softdg_gate_threshold": args.softdg_gate_threshold,
                "target_effective_completions": args.target_effective_completions,
                "dg_temperature": args.dg_temperature,
                "target_model": args.target_model,
                "behavior_model": args.behavior_model or "unknown",
                "rollout_path": args.rollout_path,
                "num_completions": len(records),
                "num_problems": len(records) // args.num_generations,
                "reward_accuracy": correct / len(records),
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

    # ---- 9. Final counter summary ---------------------------------------
    if trainer.accelerator.is_main_process:
        c = counter
        keep_rate = c.effective / max(c.scanned, 1)
        summary = {
            "effective_completions": c.effective,
            "target_effective_completions": c.target,
            "scanned": c.scanned,
            "keep_rate": keep_rate,
            "skipped_low_gate": c.skipped_low_gate,
            "skipped_zero_signal": c.skipped_zero_signal,
            "reward_coding": args.reward_coding,
            "training_signal": args.training_signal,
            "softdg_gate_threshold": args.softdg_gate_threshold,
            "dg_temperature": args.dg_temperature,
        }
        summary_path = os.path.join(args.output_dir, "training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n=== Training Summary ===")
        print(f"  Effective completions: {c.effective} / {c.target}")
        print(f"  Total scanned:         {c.scanned}")
        print(f"  Keep rate:             {keep_rate:.3f}")
        print(f"  Skipped (low gate):    {c.skipped_low_gate}")
        print(f"  Skipped (zero signal): {c.skipped_zero_signal}")
        print(f"  Summary saved to:      {summary_path}")
        print(f"  Metrics saved to:      {metrics_path}")


if __name__ == "__main__":
    main()
