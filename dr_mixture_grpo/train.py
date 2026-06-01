"""Train a target student with Dr.Mixture-GRPO on offline teacher rollouts.

Loss = -A * log pi_student where A_i = r_teacher_i - r_mean_student(qid).
The per-qid r_mean_student baseline is recomputed LIVE inside the trainer
at every step by sampling K_s completions from the current policy and
scoring with Math_Verifier. See ``trainer.py:_compute_live_student_baseline``.

Usage::

    accelerate launch dr_mixture_grpo/train.py \\
        --rollout_path /scratch/mrli/rollouts/gsm8k_teacher/rollouts_gsm8k.jsonl \\
        --target_model /scratch/mrli/models/Qwen2.5-0.5B-Instruct \\
        --dataset gsm8k \\
        --output_dir /scratch/mrli/checkpoints/dr_mixture_grpo_gsm8k
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

# Same accelerate fp32 cast no-op as offline_grpo/train.py and awbc/train.py
# (avoids the [batch * num_gen, seq, vocab] fp32 cast on logits OOM).
import accelerate.utils.operations as _accel_ops
_accel_ops.convert_to_fp32 = lambda tensor: tensor

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "offline_grpo"))
from configs import DEFAULT_TARGET_MODEL, DEFAULT_LORA_CONFIG  # noqa: E402

sys.path.insert(0, os.path.join(_PROJECT_ROOT, "shared"))
from datasets_registry import get_dataset_config, list_datasets  # noqa: E402

# Ensure dr_mixture_grpo/ is sys.path[0] so `from trainer import ...` resolves
# here rather than to offline_grpo/trainer.py (data.py's import side-effects
# re-insert offline_grpo at position 0).
sys.path.insert(0, _HERE)
from trainer import DrMixtureGRPOTrainer  # noqa: E402  (import BEFORE data.py)

# data.py's import re-prepends offline_grpo/ to sys.path; that's fine because
# we've already imported trainer.
sys.path.insert(0, _HERE)
from data import (  # noqa: E402
    build_offline_lookup,
    build_training_dataset,
    compute_rewards,
    load_rollouts_text,
)


class MetricsFileCallback(TrainerCallback):
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
    p = argparse.ArgumentParser(description="Dr.Mixture-GRPO training.")
    # Model
    p.add_argument("--target_model", type=str, default=DEFAULT_TARGET_MODEL)
    p.add_argument("--behavior_model", type=str, default=None,
                    help="For logging only (which teacher produced the rollouts).")
    # Data
    p.add_argument("--rollout_path", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True, choices=list_datasets())
    # Live student baseline (computed in-trainer every step).
    p.add_argument("--K_s", type=int, default=5,
                    help="Student samples per unique in-batch qid per step.")
    p.add_argument("--baseline_temperature", type=float, default=0.7,
                    help="Sampling temperature for live student baseline.")
    p.add_argument("--baseline_top_p", type=float, default=1.0)
    p.add_argument("--baseline_max_new_tokens", type=int, default=None,
                    help="Defaults to cfg.max_tokens.")
    p.add_argument("--no_vllm_baseline", action="store_true",
                    help="Use HF generate instead of TRL colocated vLLM for live baseline sampling.")
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.30,
                    help="Fraction of each GPU reserved for colocated vLLM baseline generation.")
    # Output
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    # Training hyperparams
    p.add_argument("--learning_rate", type=float, default=3e-6)
    p.add_argument("--beta", type=float, default=0.001)
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=-1,
                    help=">0 caps training at this many optimizer steps "
                    "(overrides num_train_epochs). Use for smoke tests.")
    p.add_argument("--max_completion_length", type=int, default=1024)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--loss_type", type=str, default="dr_grpo",
                    choices=["grpo", "bnpo", "dr_grpo", "dapo"],
                    help="TRL GRPO loss normalization. Default uses Dr.GRPO.")
    # LoRA
    p.add_argument("--lora_r", type=int, default=DEFAULT_LORA_CONFIG["r"])
    p.add_argument("--lora_alpha", type=int, default=DEFAULT_LORA_CONFIG["lora_alpha"])
    p.add_argument("--no_lora", action="store_true")
    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--report_to", type=str, default="wandb")
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    cfg = get_dataset_config(args.dataset)
    if args.baseline_max_new_tokens is None:
        args.baseline_max_new_tokens = cfg.max_tokens

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    tokenizer.pad_token = tokenizer.eos_token

    # ---- Load teacher rollouts (Path A: re-tokenize under student tokenizer) ----
    print(f"Loading teacher rollouts: {args.rollout_path}")
    records = load_rollouts_text(args.rollout_path, student_tokenizer=tokenizer)
    print(f"  {len(records)} completions loaded")

    # Per-rollout teacher reward; the live student baseline is computed in-trainer.
    records = compute_rewards(records)
    correct = sum(1 for r in records if r["reward"] > 0)
    print(f"  Teacher rewards: {correct}/{len(records)} correct "
          f"({100*correct/len(records):.1f}%)")

    dataset = build_training_dataset(records)
    offline_data = build_offline_lookup(records)
    print(f"  Dataset: {len(dataset)} rows, {len(offline_data)} offline entries")

    # ---- Target model ----
    print(f"Loading target model: {args.target_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None,
    )
    model.config.use_cache = False

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

    if args.output_dir is None:
        args.output_dir = f"./outputs/dr_mixture_grpo_{args.dataset}_{time_str}"
    if args.run_name is None:
        args.run_name = f"dr-mixture-{args.dataset}-{time_str}"

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
        loss_type=args.loss_type,
        use_vllm=not args.no_vllm_baseline,
        **({"vllm_mode": "colocate",
            "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization}
           if not args.no_vllm_baseline else {}),
        temperature=args.baseline_temperature,
        top_p=args.baseline_top_p,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        max_grad_norm=args.max_grad_norm,
        report_to=args.report_to,
        log_on_each_node=False,
        seed=args.seed,
    )

    def _dummy_reward(prompts, completions, **kwargs):
        return [0.0] * len(completions)

    metrics_path = os.path.join(args.output_dir, "training_metrics.jsonl")
    metrics_callback = MetricsFileCallback(metrics_path)

    # Canonical Math_Verifier (project-root package) + gold cleanup (offline_grpo/configs.py).
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from Math_Verifier import is_equiv_multi as _is_equiv_multi  # noqa: E402
    from configs import extract_gsm8k_answer as _extract_gsm8k_answer  # noqa: E402

    trainer = DrMixtureGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[_dummy_reward],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        offline_data=offline_data,
        dataset_name=cfg.name,
        is_equiv_multi=_is_equiv_multi,
        extract_numeric=_extract_gsm8k_answer,
        K_s=args.K_s,
        baseline_temperature=args.baseline_temperature,
        baseline_top_p=args.baseline_top_p,
        baseline_max_new_tokens=args.baseline_max_new_tokens,
        ref_sync_steps=0,
        callbacks=[metrics_callback],
    )

    print(f"Dr.Mixture-GRPO trainer initialized. "
          f"K_s={args.K_s} live samples/step, "
          f"baseline_T={args.baseline_temperature}, "
          f"loss_type={args.loss_type}, "
          f"baseline_gen={'vLLM colocate' if not args.no_vllm_baseline else 'HF generate'}, "
          f"vllm_mem={args.vllm_gpu_memory_utilization if not args.no_vllm_baseline else 'n/a'}, "
          f"LoRA={peft_config is not None}")

    if args.report_to == "wandb" and trainer.accelerator.is_main_process:
        import wandb
        if wandb.run is not None:
            wandb.config.update({
                "method": "dr_mixture_grpo",
                "target_model": args.target_model,
                "rollout_path": args.rollout_path,
                "dataset": args.dataset,
                "K_s": args.K_s,
                "baseline_temperature": args.baseline_temperature,
                "loss_type": args.loss_type,
                "baseline_generation": "vllm_colocate" if not args.no_vllm_baseline else "hf_generate",
                "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization if not args.no_vllm_baseline else None,
                "beta": args.beta,
                "lora_r": args.lora_r if not args.no_lora else None,
                "lora_alpha": args.lora_alpha if not args.no_lora else None,
            })

    print("Starting Dr.Mixture-GRPO training ...")
    resume = args.resume_from_checkpoint
    if resume == "latest":
        resume = True
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.output_dir)
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
