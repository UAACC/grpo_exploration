"""Train with DG-Mixture GRPO.

Online student rollouts (standard GRPO) + offline teacher rollouts (DG-gated).
The teacher loss uses DG-offline's sigmoid gate on advantage * surprisal,
computed entirely from the learner's current policy. No behavior logprobs
required for the teacher loss.

Usage:
    python dg_mixture/train.py \
        --teacher_rollout_path rollouts_full.jsonl \
        --target_model Qwen/Qwen2.5-0.5B-Instruct \
        --output_dir ./outputs \
        --dg_temperature 0.5 \
        --dg_offline_weight 0.3
"""

import argparse
import json
import sys
import os
from datetime import datetime

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TrainerCallback
from trl import GRPOConfig
from datasets import load_dataset


class MetricsFileCallback(TrainerCallback):
    """Write all logged metrics to a JSONL file at each logging step."""

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


# Import local trainer FIRST (before adding offline_grpo to sys.path, which
# also has a trainer.py that would shadow ours).
from dg_mixture.trainer import DGMixtureGRPOTrainer

# Add parent dir + shared for modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "shared"))
from configs import DEFAULT_TARGET_MODEL, DEFAULT_LORA_CONFIG
from datasets_registry import get_dataset_config, load_eval_data

# Add offline_grpo for shared rollout loading + advantage computation
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "offline_grpo"))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "offline_grpo_data", os.path.join(_PROJECT_ROOT, "offline_grpo", "data.py")
)
_offline_data_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_offline_data_mod)
load_rollouts = _offline_data_mod.load_rollouts
compute_rewards_and_advantages = _offline_data_mod.compute_rewards_and_advantages


def load_teacher_rollouts_with_dg_advantages(rollout_path: str, vocab_size: int) -> dict:
    """Load teacher rollouts and compute teacher-only group advantages.

    Reuses offline_grpo/data.py so the advantage normalization matches
    DG-offline's calibration of the eta gate.

    Returns: {qid: {"runs": [{"completion_ids", "reward", "advantage", "response"}, ...]}}
    """
    records = load_rollouts(rollout_path, vocab_size=vocab_size)
    records = compute_rewards_and_advantages(records)

    # Reshape into per-question dict
    teacher_data = {}
    for rec in records:
        qid = rec["question_id"]
        if qid not in teacher_data:
            teacher_data[qid] = {"runs": []}
        teacher_data[qid]["runs"].append({
            "completion_ids": rec["completion_ids"],
            "reward": rec["reward"],
            "advantage": rec["advantage"],
            "response": rec["response"],
        })
    return teacher_data


def parse_args():
    p = argparse.ArgumentParser(description="DG-Mixture GRPO")
    # Model
    p.add_argument("--target_model", type=str, default=DEFAULT_TARGET_MODEL)
    # Data
    p.add_argument("--teacher_rollout_path", type=str, required=True)
    p.add_argument("--dataset_type", type=str, default="math",
                    choices=["gsm8k", "math", "svamp", "asdiv"])
    p.add_argument("--dataset", type=str, default=None,
                    help="Override dataset path. Defaults based on dataset_type.")
    p.add_argument("--split", type=str, default="train")
    # Output
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    # DG-specific
    p.add_argument("--dg_temperature", type=float, default=0.5,
                    help="eta in sigma(delight/eta). Lower = sharper gate.")
    p.add_argument("--dg_gating", type=str, default="completion",
                    choices=["completion", "token"])
    p.add_argument("--dg_offline_weight", type=float, default=0.3,
                    help="lambda: weight for the DG-gated teacher loss.")
    # Training hyperparams
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.01)
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument("--num_teacher_per_prompt", type=int, default=4)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--max_completion_length", type=int, default=786)
    p.add_argument("--max_grad_norm", type=float, default=0.1)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--logging_steps", type=int, default=10)
    # LoRA (default to r=32/alpha=32, matching DG-offline)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--no_lora", action="store_true")
    # Reference sync
    p.add_argument("--ref_sync_steps", type=int, default=0)
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

    # ---- 1. Resolve dataset from registry --------------------------------
    cfg = get_dataset_config(args.dataset_type)
    manual_split = "train" if cfg.name == "asdiv" else None

    # ---- 2. Load teacher rollouts with TEACHER-ONLY group advantages ----
    model_config = AutoConfig.from_pretrained(args.target_model)
    print(f"Loading teacher rollouts from {args.teacher_rollout_path} ...")
    teacher_data = load_teacher_rollouts_with_dg_advantages(
        args.teacher_rollout_path, vocab_size=model_config.vocab_size,
    )
    print(f"  {len(teacher_data)} problems with teacher rollouts")

    total_runs = sum(len(v["runs"]) for v in teacher_data.values())
    correct_runs = sum(sum(1 for r in v["runs"] if r["reward"] > 0)
                       for v in teacher_data.values())
    print(f"  Teacher accuracy: {correct_runs}/{total_runs} ({100*correct_runs/total_runs:.1f}%)")

    # ---- 3. Build training dataset (prompts only, via registry) ---------
    problems = load_eval_data(cfg, split=cfg.split_train, manual_split=manual_split)
    print(f"  Dataset ({cfg.name}): {len(problems)} problems")

    prompts, answers, qids = [], [], []
    for i, prob in enumerate(problems):
        prompts.append([
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": prob["problem"]},
        ])
        answers.append(prob["answer"])
        qids.append(i)

    from datasets import Dataset
    dataset = Dataset.from_dict({
        "prompt": prompts,
        "answer": answers,
        "question_id": qids,
    })

    # ---- 4. Load target model -------------------------------------------
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

    # ---- 5. LoRA config -------------------------------------------------
    peft_config = None
    if not args.no_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=DEFAULT_LORA_CONFIG["target_modules"],
            task_type=DEFAULT_LORA_CONFIG["task_type"],
            lora_dropout=DEFAULT_LORA_CONFIG["lora_dropout"],
        )

    # ---- 6. Training config ---------------------------------------------
    if args.output_dir is None:
        args.output_dir = f"./outputs/dg_mixture_{time_str}"
    if args.run_name is None:
        model_name = args.target_model.rstrip("/").split("/")[-1]
        args.run_name = (f"dg-mixture-{model_name}-eta{args.dg_temperature}"
                         f"-lam{args.dg_offline_weight}-{time_str}")

    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

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
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        max_grad_norm=args.max_grad_norm,
        report_to=args.report_to,
        log_on_each_node=False,
        seed=args.seed,
    )

    # ---- 7. Create trainer & train --------------------------------------
    def _dummy_reward(prompts, completions, **kwargs):
        return [0.0] * len(completions)

    metrics_path = os.path.join(args.output_dir, "training_metrics.jsonl")
    metrics_callback = MetricsFileCallback(metrics_path)

    trainer = DGMixtureGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[_dummy_reward],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        teacher_data=teacher_data,
        dg_offline_weight=args.dg_offline_weight,
        dg_temperature=args.dg_temperature,
        dg_gating=args.dg_gating,
        num_teacher_per_prompt=args.num_teacher_per_prompt,
        ref_sync_steps=args.ref_sync_steps,
        dataset_type=args.dataset_type,
        reward_func=cfg.reward_func,
        callbacks=[metrics_callback],
    )

    print(f"=== DG-Mixture GRPO ===")
    print(f"  Student generations: {args.num_generations}")
    print(f"  Teacher per prompt: {args.num_teacher_per_prompt}")
    print(f"  DG eta: {args.dg_temperature}")
    print(f"  DG gating: {args.dg_gating}")
    print(f"  DG offline weight (lambda): {args.dg_offline_weight}")
    print(f"  Beta (KL on student): {trainer.beta}")
    print(f"  ref_sync_steps: {args.ref_sync_steps}")

    if args.report_to == "wandb" and trainer.accelerator.is_main_process:
        import wandb
        if wandb.run is not None:
            wandb.config.update({
                "method": "dg-mixture-grpo",
                "target_model": args.target_model,
                "rollout_path": args.teacher_rollout_path,
                "num_problems": len(teacher_data),
                "teacher_accuracy": correct_runs / total_runs,
                "dg_temperature": args.dg_temperature,
                "dg_gating": args.dg_gating,
                "dg_offline_weight": args.dg_offline_weight,
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
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
