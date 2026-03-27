"""Diagnose Method B offline loss effectiveness.

Hypothesis 1: When student's G rollouts are all-correct or all-wrong,
              std_r=0, teacher advantage collapses to 0.0.
Hypothesis 2: IS ratio (π_student / π_teacher) is far from 1.0,
              clipped on almost every token → near-zero gradient.

This script:
  (a) Loads teacher rollouts + student model
  (b) Simulates student rollout reward distribution at different accuracy levels
  (c) Computes actual IS ratios on a sample of teacher completions
  (d) Reports clip fractions and effective gradient magnitude

Usage (1x L40s):
  python diagnose_method_B.py \
      --model_path /scratch/mrli/merged/mixture_B_math_merged \
      --teacher_rollout_path /scratch/mrli/rollouts/math_teacher/rollouts_full.jsonl \
      --dataset_type math \
      --num_samples 200
"""

import argparse
import json
import sys
import os
import random

import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig
from vllm import LLM, SamplingParams

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import (
    GSM8K_SYSTEM_PROMPT, GSM8K_DATASET,
    MATH_SYSTEM_PROMPT, MATH_DATASET,
    DEFAULT_TARGET_MODEL,
    extract_gsm8k_answer, extract_boxed_answer,
)
from data import load_teacher_rollouts


def parse_args():
    p = argparse.ArgumentParser(description="Diagnose Method B offline loss.")
    p.add_argument("--model_path", type=str, required=True,
                    help="Student model (merged or base).")
    p.add_argument("--teacher_rollout_path", type=str, required=True,
                    help="Path to teacher rollout JSONL.")
    p.add_argument("--dataset_type", type=str, default="math",
                    choices=["gsm8k", "math"])
    p.add_argument("--num_samples", type=int, default=200,
                    help="Number of problems to diagnose.")
    p.add_argument("--num_generations", type=int, default=4,
                    help="Student rollouts per prompt (G).")
    p.add_argument("--temperature", type=float, default=0.7,
                    help="Student generation temperature (match training).")
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--max_model_len", type=int, default=2048)
    p.add_argument("--clip_eps", type=float, default=0.2,
                    help="PPO clip epsilon (match Method B).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_math = args.dataset_type == "math"
    system_prompt = MATH_SYSTEM_PROMPT if is_math else GSM8K_SYSTEM_PROMPT

    # ── 1. Load teacher rollouts ──────────────────────────────────────
    print("Loading teacher rollouts...")
    model_config = AutoConfig.from_pretrained(args.model_path)
    teacher_data = load_teacher_rollouts(
        args.teacher_rollout_path,
        vocab_size=model_config.vocab_size,
        dataset_type=args.dataset_type,
    )
    print(f"  {len(teacher_data)} problems loaded")

    # Sample a subset
    all_qids = list(teacher_data.keys())
    sample_qids = random.sample(all_qids, min(args.num_samples, len(all_qids)))

    # ── 2. Load student model via vLLM ────────────────────────────────
    print(f"Loading student model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    llm = LLM(
        model=args.model_path,
        dtype="auto",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
    )

    # ── 3. Generate student rollouts ──────────────────────────────────
    print(f"\nGenerating {args.num_generations} student rollouts per prompt "
          f"for {len(sample_qids)} problems (temp={args.temperature})...")

    # Build prompts
    prompts = []
    prompt_qids = []
    for qid in sample_qids:
        td = teacher_data[qid]
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": td["problem"]},
        ]
        formatted = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        for _ in range(args.num_generations):
            prompts.append(formatted)
            prompt_qids.append(qid)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    outputs = llm.generate(prompts, sampling_params)

    # ── 4. Compute student rewards ────────────────────────────────────
    print("Computing student rewards...")

    # Group by qid
    student_rewards_by_qid = {}
    for i, output in enumerate(outputs):
        qid = prompt_qids[i]
        gen_text = output.outputs[0].text

        if is_math:
            from data import compute_math_correctness
            reward = compute_math_correctness(extract_boxed_answer(gen_text),
                                               teacher_data[qid]["ground_truth"])
        else:
            from data import compute_gsm8k_correctness
            reward = compute_gsm8k_correctness(extract_gsm8k_answer(gen_text),
                                                teacher_data[qid]["ground_truth"])

        if qid not in student_rewards_by_qid:
            student_rewards_by_qid[qid] = []
        student_rewards_by_qid[qid].append(reward)

    # ══════════════════════════════════════════════════════════════════
    # HYPOTHESIS 1: Advantage collapse when student rewards are uniform
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("HYPOTHESIS 1: Teacher advantage collapse (std_r = 0)")
    print("=" * 70)

    n_all_wrong = 0
    n_all_correct = 0
    n_mixed = 0
    n_total = len(sample_qids)

    # Also compute what teacher advantages would be
    teacher_adv_when_mixed = []
    teacher_adv_when_uniform = []

    eps = 1e-4
    for qid in sample_qids:
        rewards = student_rewards_by_qid[qid]
        mean_r = sum(rewards) / len(rewards)
        std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5

        # Check uniformity
        all_zero = all(r == 0.0 for r in rewards)
        all_positive = all(r > 0.0 for r in rewards)
        is_uniform = all_zero or all_positive

        if all_zero:
            n_all_wrong += 1
        elif all_positive:
            n_all_correct += 1
        else:
            n_mixed += 1

        # Compute teacher advantages (Method B formula)
        td = teacher_data[qid]
        for run in td["runs"][:args.num_generations]:
            tea_reward = run["reward"]
            if std_r > eps:
                tea_adv = (tea_reward - mean_r) / (std_r + eps)
                teacher_adv_when_mixed.append(tea_adv)
            else:
                # This is the collapse case
                teacher_adv_when_uniform.append(0.0)

    print(f"\nStudent rollout reward distribution ({args.num_generations} rollouts/prompt):")
    print(f"  All wrong  (reward=0 for all G): {n_all_wrong}/{n_total} ({100*n_all_wrong/n_total:.1f}%)")
    print(f"  All correct (reward>0 for all G): {n_all_correct}/{n_total} ({100*n_all_correct/n_total:.1f}%)")
    print(f"  Mixed (some correct, some wrong):  {n_mixed}/{n_total} ({100*n_mixed/n_total:.1f}%)")

    print(f"\nTeacher advantage statistics:")
    print(f"  Prompts with uniform student rewards → teacher_adv = 0.0: "
          f"{len(teacher_adv_when_uniform)} completions (WASTED)")
    print(f"  Prompts with mixed student rewards → teacher_adv computed: "
          f"{len(teacher_adv_when_mixed)} completions (ACTIVE)")

    if teacher_adv_when_mixed:
        mixed_arr = np.array(teacher_adv_when_mixed)
        print(f"    Mean teacher advantage (when active): {mixed_arr.mean():.4f}")
        print(f"    Std teacher advantage (when active):  {mixed_arr.std():.4f}")
        print(f"    Fraction with teacher_adv = 0:        "
              f"{sum(1 for a in teacher_adv_when_mixed if abs(a) < eps) / len(teacher_adv_when_mixed):.2%}")

    pct_wasted = len(teacher_adv_when_uniform) / (len(teacher_adv_when_uniform) + len(teacher_adv_when_mixed) + 1e-10)
    print(f"\n  >>> {pct_wasted:.1%} of teacher completions produce ZERO gradient (advantage=0)")

    # ══════════════════════════════════════════════════════════════════
    # HYPOTHESIS 2: IS ratio clipping
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("HYPOTHESIS 2: IS ratio clipping (π_student / π_teacher)")
    print("=" * 70)

    # We need student logprobs on teacher completions.
    # Use vLLM's prompt_logprobs to compute this efficiently:
    # Feed [prompt + teacher_completion] as a prompt and get logprobs.

    print(f"\nComputing student logprobs on teacher completions...")

    # Prepare: for each sampled qid, take the first teacher completion
    teacher_prompts = []
    teacher_behavior_logprobs = []
    teacher_completion_lengths = []
    diag_qids = []

    for qid in sample_qids[:100]:  # Limit to 100 for speed
        td = teacher_data[qid]
        if not td["runs"]:
            continue
        run = td["runs"][0]
        cids = run["completion_ids"]
        blps = run["behavior_logprobs"]

        if len(cids) == 0:
            continue

        # Build the full sequence: prompt + teacher completion
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": td["problem"]},
        ]
        formatted = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)

        # Decode teacher completion
        teacher_text = tokenizer.decode(cids, skip_special_tokens=False)
        full_text = formatted + teacher_text

        teacher_prompts.append(full_text)
        teacher_behavior_logprobs.append(blps)
        teacher_completion_lengths.append(len(cids))
        diag_qids.append(qid)

    # Use vLLM with prompt_logprobs to get student's logprobs on the full sequence
    # We generate 0 new tokens but request logprobs for the prompt tokens
    scoring_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,  # Must generate at least 1
        prompt_logprobs=1,  # Get logprobs for prompt tokens
    )
    score_outputs = llm.generate(teacher_prompts, scoring_params)

    # Extract per-token log-ratios
    all_log_ratios = []
    all_ratios = []
    clip_fractions = []
    effective_gradient_fractions = []

    for idx, (output, blps, comp_len) in enumerate(
        zip(score_outputs, teacher_behavior_logprobs, teacher_completion_lengths)
    ):
        # prompt_logprobs is a list of dicts, one per prompt token
        prompt_logprobs = output.prompt_logprobs
        if prompt_logprobs is None:
            continue

        # The completion tokens start after the prompt
        # prompt_logprobs[i] gives logprob for token i given tokens[:i]
        total_prompt_len = len(prompt_logprobs)
        comp_start = total_prompt_len - comp_len

        if comp_start < 0 or comp_start >= total_prompt_len:
            continue

        token_log_ratios = []
        for t in range(comp_len):
            pos = comp_start + t
            if pos >= len(prompt_logprobs) or prompt_logprobs[pos] is None:
                break
            if t >= len(blps):
                break

            # Get student logprob for this token
            # prompt_logprobs[pos] is a dict {token_id: Logprob}
            lp_dict = prompt_logprobs[pos]
            if not lp_dict:
                continue

            # Get the logprob of the actual token (first entry)
            student_lp = list(lp_dict.values())[0].logprob

            teacher_lp = blps[t]
            if teacher_lp is None:
                continue

            log_ratio = student_lp - teacher_lp
            token_log_ratios.append(log_ratio)

        if not token_log_ratios:
            continue

        lr_arr = np.array(token_log_ratios)
        ratio_arr = np.exp(lr_arr)

        all_log_ratios.extend(token_log_ratios)
        all_ratios.extend(ratio_arr.tolist())

        # Clip fraction for this completion
        clipped = np.sum((ratio_arr < 1.0 - args.clip_eps) | (ratio_arr > 1.0 + args.clip_eps))
        clip_frac = clipped / len(ratio_arr)
        clip_fractions.append(clip_frac)

        # Effective gradient: fraction of tokens where gradient passes through
        effective_gradient_fractions.append(1.0 - clip_frac)

    if all_log_ratios:
        lr_arr = np.array(all_log_ratios)
        ratio_arr = np.array(all_ratios)

        print(f"\nIS ratio statistics (π_student / π_teacher) across {len(all_log_ratios)} tokens:")
        print(f"  log(ratio) mean:   {lr_arr.mean():.4f}")
        print(f"  log(ratio) std:    {lr_arr.std():.4f}")
        print(f"  log(ratio) median: {np.median(lr_arr):.4f}")
        print(f"  log(ratio) [5%, 95%]: [{np.percentile(lr_arr, 5):.4f}, {np.percentile(lr_arr, 95):.4f}]")
        print()
        print(f"  ratio mean:   {ratio_arr.mean():.4f}")
        print(f"  ratio median: {np.median(ratio_arr):.4f}")
        print(f"  ratio [5%, 95%]: [{np.percentile(ratio_arr, 5):.4f}, {np.percentile(ratio_arr, 95):.4f}]")
        print()

        # Clip analysis
        in_range = np.sum((ratio_arr >= 1.0 - args.clip_eps) & (ratio_arr <= 1.0 + args.clip_eps))
        below = np.sum(ratio_arr < 1.0 - args.clip_eps)
        above = np.sum(ratio_arr > 1.0 + args.clip_eps)
        total = len(ratio_arr)

        print(f"  Clip analysis (eps={args.clip_eps}):")
        print(f"    ratio < {1-args.clip_eps}: {below}/{total} ({100*below/total:.1f}%) ← clipped low")
        print(f"    ratio in [{1-args.clip_eps}, {1+args.clip_eps}]: {in_range}/{total} ({100*in_range/total:.1f}%) ← gradient passes")
        print(f"    ratio > {1+args.clip_eps}: {above}/{total} ({100*above/total:.1f}%) ← clipped high")
        print()

        cf_arr = np.array(clip_fractions)
        print(f"  Per-completion clip fraction:")
        print(f"    Mean: {cf_arr.mean():.2%}")
        print(f"    Median: {np.median(cf_arr):.2%}")
        print(f"    [25%, 75%]: [{np.percentile(cf_arr, 25):.2%}, {np.percentile(cf_arr, 75):.2%}]")
        print(f"    Completions with >90% tokens clipped: "
              f"{np.sum(cf_arr > 0.9)}/{len(cf_arr)} ({100*np.sum(cf_arr > 0.9)/len(cf_arr):.1f}%)")
        print(f"    Completions with >50% tokens clipped: "
              f"{np.sum(cf_arr > 0.5)}/{len(cf_arr)} ({100*np.sum(cf_arr > 0.5)/len(cf_arr):.1f}%)")

    # ══════════════════════════════════════════════════════════════════
    # COMBINED IMPACT
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("COMBINED IMPACT")
    print("=" * 70)

    h1_wasted = pct_wasted
    if clip_fractions:
        h2_clipped = np.mean(clip_fractions)
    else:
        h2_clipped = 0.0

    # Effective signal = (1 - H1_wasted) * (1 - H2_clipped)
    effective = (1 - h1_wasted) * (1 - h2_clipped)
    print(f"\n  H1: {h1_wasted:.1%} of teacher completions have advantage=0 (wasted)")
    print(f"  H2: {h2_clipped:.1%} of remaining tokens are clipped (weak gradient)")
    print(f"  Combined: only {effective:.1%} of teacher gradient signal is effective")
    print(f"  >>> The offline loss contributes ~{effective:.1%} of its potential to training")


if __name__ == "__main__":
    main()
