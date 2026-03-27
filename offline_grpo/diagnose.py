"""Diagnostic script to verify offline GRPO internals.

Run on a compute node (needs GPU):
    srun --account=aip-szepesva --time=0:30:00 --gpus-per-node=1 --cpus-per-task=6 --mem=48G \
        bash -c 'module load python/3.11 cuda/12.6 && \
        source /project/aip-szepesva/mrli/backup_dongheng/.venv/bin/activate && \
        cd /project/aip-szepesva/mrli/backup_dongheng/offline_grpo && \
        python diagnose.py --rollout_path rollouts_test.jsonl \
            --target_model /scratch/mrli/models/Qwen2.5-0.5B-Instruct'
"""

import argparse
import json
import numpy as np
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs import SYSTEM_PROMPT, extract_boxed_answer
from data import load_rollouts, compute_rewards_and_advantages


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollout_path", type=str, required=True)
    p.add_argument("--target_model", type=str, required=True)
    p.add_argument("--max_samples", type=int, default=8, help="Number of samples to inspect in detail")
    args = p.parse_args()

    # ── 1. Check rollout data ────────────────────────────────────────
    print("=" * 60)
    print("1. ROLLOUT DATA ANALYSIS")
    print("=" * 60)

    records = load_rollouts(args.rollout_path)
    print(f"Total completions: {len(records)}")

    # Check logprob ranges
    all_logprobs = []
    for rec in records:
        lps = [lp for lp in rec["behavior_logprobs"] if lp is not None]
        all_logprobs.extend(lps)
    all_logprobs = np.array(all_logprobs)
    print(f"Teacher logprobs: mean={all_logprobs.mean():.4f}, std={all_logprobs.std():.4f}, "
          f"min={all_logprobs.min():.4f}, max={all_logprobs.max():.4f}")

    # Check completion lengths
    lengths = [len(rec["completion_ids"]) for rec in records]
    print(f"Completion lengths: mean={np.mean(lengths):.0f}, min={min(lengths)}, max={max(lengths)}")

    # ── 2. Check rewards and advantages ──────────────────────────────
    print("\n" + "=" * 60)
    print("2. REWARDS AND ADVANTAGES")
    print("=" * 60)

    records = compute_rewards_and_advantages(records)
    rewards = [r["reward"] for r in records]
    advantages = [r["advantage"] for r in records]

    print(f"Rewards:    mean={np.mean(rewards):.3f}, std={np.std(rewards):.3f}")
    print(f"  Correct:  {sum(1 for r in rewards if r > 0)}/{len(rewards)} ({100*sum(1 for r in rewards if r > 0)/len(rewards):.1f}%)")
    print(f"Advantages: mean={np.mean(advantages):.4f}, std={np.std(advantages):.4f}, "
          f"min={min(advantages):.4f}, max={max(advantages):.4f}")

    # Check how many groups have all-same rewards (advantage ≈ 0)
    groups = defaultdict(list)
    for rec in records:
        groups[rec["question_id"]].append(rec["reward"])
    all_same = sum(1 for g in groups.values() if len(set(g)) == 1)
    print(f"Groups with all-same reward: {all_same}/{len(groups)} ({100*all_same/len(groups):.1f}%)")
    print("  → These groups have advantage ≈ 0 and contribute NO gradient signal!")

    # ── 3. Check student logprobs on teacher completions ─────────────
    print("\n" + "=" * 60)
    print("3. STUDENT vs TEACHER LOGPROBS (first few samples)")
    print("=" * 60)

    print(f"Loading student model: {args.target_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    tokenizer.pad_token = tokenizer.eos_token

    for i, rec in enumerate(records[:args.max_samples]):
        # Build prompt
        chat = [
            {"role": "system", "content": rec["system_prompt"]},
            {"role": "user", "content": rec["problem"]},
        ]
        prompt_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        comp_ids = torch.tensor([rec["completion_ids"]])

        # Concat and forward
        input_ids = torch.cat([prompt_ids, comp_ids], dim=1).to(model.device)
        with torch.no_grad():
            logits = model(input_ids).logits

        # Compute student logprobs for completion tokens
        # logits[:, :-1] predicts token at position +1
        comp_start = prompt_ids.shape[1]
        comp_logits = logits[:, comp_start - 1:-1, :]  # shifted: predict each completion token
        log_probs = torch.log_softmax(comp_logits, dim=-1)
        student_logprobs = log_probs.gather(-1, comp_ids.to(model.device).unsqueeze(-1)).squeeze(-1)[0]
        student_logprobs = student_logprobs.cpu().float().numpy()

        teacher_logprobs = np.array(rec["behavior_logprobs"][:len(student_logprobs)])
        teacher_logprobs = np.where(teacher_logprobs == None, 0.0, teacher_logprobs).astype(float)

        # Compute IS ratios
        log_ratio = student_logprobs - teacher_logprobs
        ratio_per_token = np.exp(log_ratio)

        print(f"\n--- Sample {i} (qid={rec['question_id']}, rid={rec['run_id']}, "
              f"reward={rec['reward']}, advantage={rec['advantage']:.4f}) ---")
        print(f"  Completion length: {len(rec['completion_ids'])} tokens")
        print(f"  Teacher logprobs: mean={teacher_logprobs.mean():.4f}, std={teacher_logprobs.std():.4f}")
        print(f"  Student logprobs: mean={student_logprobs.mean():.4f}, std={student_logprobs.std():.4f}")
        print(f"  Log ratio (student - teacher): mean={log_ratio.mean():.4f}, std={log_ratio.std():.4f}")
        print(f"  IS ratio per token: mean={ratio_per_token.mean():.4f}, std={ratio_per_token.std():.4f}")
        print(f"  IS ratio (sequence-level, product): {np.exp(log_ratio.sum()):.6e}")
        print(f"  → If sequence ratio is ~0 or ~inf, IS correction is broken at sequence level")

    # ── 4. Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("4. DIAGNOSIS SUMMARY")
    print("=" * 60)
    if all_same / len(groups) > 0.5:
        print("[WARNING] >50% of groups have all-same rewards → most advantages are ~0")
        print("  → The model gets almost no gradient signal from these groups.")
        print("  → Fix: use more generations per problem, or harder problems.")
    else:
        print("[OK] Reward diversity looks reasonable.")

    print("\nCheck the student vs teacher logprobs above:")
    print("  - If student logprobs are much lower → student finds teacher's text very unlikely")
    print("  - If sequence-level IS ratio → 0 or inf → importance sampling is degenerate")
    print("  - If per-token IS ratios are close to 1.0 → good, models roughly agree")


if __name__ == "__main__":
    main()