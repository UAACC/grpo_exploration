"""SoftDG-Token-Mask-Offline training: per-token PG masking on offline rollouts.

Differences vs SoftDG-Offline-to-the-bottom/train.py:
  - Uses SoftDGTokenMaskOfflineTrainer (per-token gating, not per-completion).
  - Token budget = sum(min(completion_length, max_completion_length)) over rollout.
  - Early stop on --target_effective_tokens (tokens, not completions).
  - No --dg_gating flag (always token-level).

Usage:
    accelerate launch --config_file configs/accelerate_ddp_4gpu.yaml \\
        train.py \\
        --rollout_path /scratch/shuai14/rollouts/math_teacher/rollouts_math_Qwen2.5-0.5B-Instruct_0.6_pick4.jsonl \\
        --target_model /scratch/shuai14/models/Qwen2.5-0.5B \\
        --reward_coding signed \\
        --training_signal raw_reward \\
        --loss_type dr_grpo \\
        --softdg_gate_threshold 0.5
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
    SoftDGTokenMaskOfflineTrainer,
    EffectiveTokenCounter,
    EarlyStopOnTokenBudgetCallback,
)


# ---------------------------------------------------------------------------
# Reward coding helpers
# ---------------------------------------------------------------------------

def remap_rewards_signed(records: list[dict]) -> list[dict]:
    for rec in records:
        rec["reward"] = 1.0 if rec.get("reward", 0.0) > 0.0 else -1.0
    return records


def recompute_advantages(records: list[dict], eps: float = 1e-4) -> list[dict]:
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


def compute_token_budget(records: list[dict], max_completion_length: int) -> int:
    return sum(
        min(len(rec["completion_ids"]), max_completion_length) for rec in records
    )


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
    p = argparse.ArgumentParser(
        description="SoftDG Token-Mask offline training: per-token PG masking."
    )
    # Model
    p.add_argument("--target_model", type=str, required=True)
    p.add_argument("--behavior_model", type=str, default=None)
    # Data
    p.add_argument("--rollout_path", type=str, nargs="+", required=True)
    # Output
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    # SoftDG token-mask specific
    p.add_argument("--reward_coding", type=str, default="signed",
                   choices=["zero_two", "signed"],
                   help="Reward coding: 'zero_two' (0/2) or 'signed' (-1/+1).")
    p.add_argument("--training_signal", type=str, default="raw_reward",
                   choices=["advantage", "raw_reward"],
                   help="Signal fed to the per-token gate and PG loss.")
    p.add_argument("--loss_type", type=str, default="dr_grpo",
                   choices=["grpo", "dr_grpo"],
                   help="GRPO loss formulation.")
    p.add_argument("--softdg_gate_threshold", type=float, default=0.5,
                   help="Tokens with per-token gate < threshold excluded from PG.")
    p.add_argument("--target_effective_tokens", type=int, default=None,
                   help="Stop when this many effective tokens seen. "
                        "Defaults to sum of capped completion lengths (token budget).")
    p.add_argument("--dg_temperature", type=float, default=1.0,
                   help="eta in sigmoid(delight/eta).")
    p.add_argument("--pg_weighting", action="store_true", default=True,
                   help="Scale selected-token PG by DG(token_t): "
                        "PG_signal_t = training_signal_i * DG(token_t). Default: True.")
    p.add_argument("--no_pg_weighting", dest="pg_weighting", action="store_false",
                   help="Disable DG-weighting: use binary token-mask PG (original behaviour).")
    # Training hyperparams
    p.add_argument("--learning_rate", type=float, default=3e-6)
    p.add_argument("--beta", type=float, default=0.001)
    p.add_argument("--num_generations", type=int, default=None)
    p.add_argument("--per_device_train_batch_size", type=int, default=3)
    p.add_argument("--gradient_accumulation_steps", type=int, default=3)
    p.add_argument("--num_train_epochs", type=int, default=20)
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

    # ---- 3. Compute rewards and remap -----------------------------------
    records = compute_rewards_and_advantages(records)

    if args.reward_coding == "signed":
        records = remap_rewards_signed(records)
        records = recompute_advantages(records)
        print(f"  Reward coding: signed (-1/+1); advantages recomputed")
    else:
        print(f"  Reward coding: zero_two (0/2); advantages from loader")

    correct = sum(1 for r in records if r["reward"] > 0)
    print(f"  Rewards: {correct}/{len(records)} correct ({100*correct/len(records):.1f}%)")
    print(f"  Training signal: {args.training_signal}")

    # ---- 4. Token budget ------------------------------------------------
    token_budget = compute_token_budget(records, args.max_completion_length)
    if args.target_effective_tokens is None:
        args.target_effective_tokens = token_budget
    print(f"  Token budget (sum of capped completion lengths): {token_budget}")
    print(f"  Target effective tokens: {args.target_effective_tokens}")

    dataset = build_training_dataset(records)
    offline_data = build_offline_lookup(records)
    print(f"  Dataset: {len(dataset)} rows, {len(offline_data)} offline entries")

    # ---- 5. Load model --------------------------------------------------
    print(f"Loading target model: {args.target_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None,
    )
    model.config.use_cache = False

    # ---- 6. LoRA --------------------------------------------------------
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

    # ---- 7. Training config ---------------------------------------------
    wgate_suffix = "_wgate" if args.pg_weighting else ""
    if args.output_dir is None:
        args.output_dir = (
            f"./outputs/softdg_tm_{args.reward_coding}_{args.training_signal}"
            f"_{args.loss_type}_thr{args.softdg_gate_threshold}{wgate_suffix}_{time_str}"
        )
    if args.run_name is None:
        model_name = args.target_model.rstrip("/").split("/")[-1]
        args.run_name = (
            f"softdg-tm-{model_name}-{args.reward_coding}-{args.training_signal}"
            f"-{args.loss_type}-thr{args.softdg_gate_threshold}"
            f"-eta{args.dg_temperature}{wgate_suffix}-{time_str}"
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

    # ---- 8. Counter and callbacks ---------------------------------------
    counter = EffectiveTokenCounter(target=args.target_effective_tokens)
    early_stop_cb = EarlyStopOnTokenBudgetCallback(counter)
    metrics_path = os.path.join(args.output_dir, "training_metrics.jsonl")
    metrics_cb = MetricsFileCallback(metrics_path)

    # ---- 9. Trainer -----------------------------------------------------
    def _dummy_reward(prompts, completions, **kwargs):
        return [0.0] * len(completions)

    trainer = SoftDGTokenMaskOfflineTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[_dummy_reward],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        offline_data=offline_data,
        training_signal=args.training_signal,
        dg_temperature=args.dg_temperature,
        softdg_gate_threshold=args.softdg_gate_threshold,
        pg_weighting=args.pg_weighting,
        counter=counter,
        ref_sync_steps=args.ref_sync_steps,
        callbacks=[early_stop_cb, metrics_cb],
    )

    print(f"=== SoftDG Token-Mask Offline ===")
    print(f"  reward_coding:           {args.reward_coding}")
    print(f"  training_signal:         {args.training_signal}")
    print(f"  loss_type:               {args.loss_type}")
    print(f"  softdg_gate_threshold:   {args.softdg_gate_threshold}")
    print(f"  dg_temperature (eta):    {args.dg_temperature}")
    print(f"  pg_weighting:            {args.pg_weighting}")
    print(f"  token_budget:            {token_budget}")
    print(f"  target_effective_tokens: {args.target_effective_tokens}")
    print(f"  beta (KL):               {trainer.beta}")
    print(f"  LoRA:                    {peft_config is not None}")
    print(f"  num_train_epochs (max):  {args.num_train_epochs}")

    if args.report_to == "wandb" and trainer.accelerator.is_main_process:
        import wandb
        if wandb.run is not None:
            wandb.config.update({
                "method": "softdg-token-mask-offline",
                "reward_coding": args.reward_coding,
                "training_signal": args.training_signal,
                "loss_type": args.loss_type,
                "softdg_gate_threshold": args.softdg_gate_threshold,
                "dg_temperature": args.dg_temperature,
                "pg_weighting": args.pg_weighting,
                "token_budget": token_budget,
                "target_effective_tokens": args.target_effective_tokens,
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

    # ---- 10. Final summary ----------------------------------------------
    if trainer.accelerator.is_main_process:
        c = counter
        keep_rate = c.effective / max(c.scanned, 1)
        budget_reached = c.effective >= c.target

        summary = {
            "method": "softdg-token-mask-offline",
            "reward_coding": args.reward_coding,
            "training_signal": args.training_signal,
            "loss_type": args.loss_type,
            "softdg_gate_threshold": args.softdg_gate_threshold,
            "dg_temperature": args.dg_temperature,
            "pg_weighting": args.pg_weighting,
            "token_budget": token_budget,
            "target_effective_tokens": c.target,
            "effective_tokens": c.effective,
            "scanned_tokens": c.scanned,
            "kept_pg_tokens": c.kept_pg,
            "token_keep_rate": keep_rate,
            "budget_reached": budget_reached,
        }

        if not budget_reached:
            msg = (
                f"WARNING: token budget NOT reached. "
                f"effective={c.effective} < target={c.target}. "
                f"Consider increasing num_train_epochs."
            )
            summary["warning"] = msg
            print(f"\n{msg}")

        summary_path = os.path.join(args.output_dir, "training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n=== Training Summary ===")
        print(f"  Effective tokens:  {c.effective} / {c.target}")
        print(f"  Total scanned:     {c.scanned}")
        print(f"  Token keep rate:   {keep_rate:.4f}")
        print(f"  Kept PG tokens:    {c.kept_pg}")
        print(f"  Budget reached:    {budget_reached}")
        print(f"  Summary saved to:  {summary_path}")
        print(f"  Metrics saved to:  {metrics_path}")


if __name__ == "__main__":
    main()
