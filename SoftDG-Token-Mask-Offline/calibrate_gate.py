#!/usr/bin/env python3
"""Gate calibration for SoftDG-Token-Mask-Offline.

Computes the global per-token gate distribution over the rollout dataset
(no correctness split) and selects candidate thresholds for each of the
three training variants.

Calibration rule (from EXPERIMENT_PLAN_2026-06-19.md):
  - Compute GLOBAL token-gate percentiles over all non-padding completion tokens.
  - Candidate thresholds: p10, p20, p30, p40, p50, p60, p70, p80, p90 + 0.50.
  - Deduplicate within 0.005.
  - Discard thresholds with token keep rate outside [0.10, 0.90].
  - For each candidate threshold, report: keep rate, drop rate,
    kept tokens per epoch, effective nonzero-signal tokens per epoch,
    projected epochs to reach token budget.

Three configs calibrated:
  A: signed + raw_reward  (dr_grpo — loss_type for reference only)
  B: zero_two + advantage (grpo)
  C: signed + advantage   (dr_grpo)

Output files (in --output_dir):
  token_gate_calibration.jsonl          -- per-completion gate summaries
  token_gate_calibration.csv            -- per-config global stats
  token_gate_calibration_thresholds.json -- candidate thresholds per config

Usage:
    python calibrate_gate.py \\
        --rollout_path /scratch/shuai14/rollouts/math_teacher/rollouts_math_Qwen2.5-0.5B-Instruct_0.6_pick4.jsonl \\
        --model_path /scratch/shuai14/models/Qwen2.5-0.5B \\
        --output_dir outputs/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.abspath(os.path.join(_HERE, ".."))
_DG_OFFLINE = os.path.join(_PROJECT, "DG-offline")
if _DG_OFFLINE not in sys.path:
    sys.path.insert(0, _DG_OFFLINE)

from teacher_agnostic_loader import (
    load_rollouts_text,
    compute_rewards_and_advantages,
)


# ---------------------------------------------------------------------------
# Calibration configs — Variants A, B, C from the experiment plan
# ---------------------------------------------------------------------------

CONFIGS = [
    {
        "name": "A",
        "variant": "A",
        "reward_coding": "signed",
        "signal_type": "raw_reward",
        "loss_type": "dr_grpo",
        "description": "signed + raw_reward (dr_grpo)",
    },
    {
        "name": "B",
        "variant": "B",
        "reward_coding": "zero_two",
        "signal_type": "advantage",
        "loss_type": "grpo",
        "description": "zero_two + advantage (grpo)",
    },
    {
        "name": "C",
        "variant": "C",
        "reward_coding": "signed",
        "signal_type": "advantage",
        "loss_type": "dr_grpo",
        "description": "signed + advantage (dr_grpo)",
    },
]


# ---------------------------------------------------------------------------
# Advantage computation per reward coding
# ---------------------------------------------------------------------------

def compute_signal_lookup(records: list[dict], reward_coding: str, signal_type: str,
                           eps: float = 1e-4) -> dict:
    """Return {(question_id, run_id): signal_value} for the given config."""
    # Map raw rewards per coding
    if reward_coding == "signed":
        raw = {(r["question_id"], r["run_id"]): (1.0 if r["reward"] > 0.0 else -1.0)
               for r in records}
    else:  # zero_two
        raw = {(r["question_id"], r["run_id"]): r["reward"] for r in records}

    if signal_type == "raw_reward":
        return raw

    # Recompute group-normalized advantages from mapped rewards
    groups: dict = defaultdict(list)
    for rec in records:
        groups[rec["question_id"]].append(rec)

    advantages = {}
    for group in groups.values():
        group_rewards = [raw[(r["question_id"], r["run_id"])] for r in group]
        mean_r = sum(group_rewards) / len(group_rewards)
        std_r = (sum((rv - mean_r) ** 2 for rv in group_rewards) / len(group_rewards)) ** 0.5
        for rec, rv in zip(group, group_rewards):
            advantages[(rec["question_id"], rec["run_id"])] = (rv - mean_r) / (std_r + eps)
    return advantages


# ---------------------------------------------------------------------------
# Surprisal computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_per_token_surprisal(model, tokenizer, rec: dict,
                                 max_comp_len: int, device) -> list[float]:
    """Return per-token surprisal for one completion under the student model."""
    comp_ids = rec["completion_ids"][:max_comp_len]
    if not comp_ids:
        return []

    messages = [
        {"role": "system", "content": rec["system_prompt"]},
        {"role": "user",   "content": rec["problem"]},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids

    L_p = len(prompt_ids)
    L_c = len(comp_ids)
    full_ids = torch.tensor(prompt_ids + comp_ids, dtype=torch.long).unsqueeze(0).to(device)

    logits = model(full_ids).logits[0]              # (L, V)
    logits_comp = logits[L_p - 1: L_p - 1 + L_c].float()  # (L_c, V)
    log_probs = F.log_softmax(logits_comp, dim=-1)
    comp_tensor = torch.tensor(comp_ids, dtype=torch.long, device=device)
    per_token_logps = log_probs.gather(1, comp_tensor.unsqueeze(1)).squeeze(1)  # (L_c,)

    return (-per_token_logps).cpu().tolist()


# ---------------------------------------------------------------------------
# Gate computation
# ---------------------------------------------------------------------------

def compute_per_token_gates(signal: float, surprisal_tokens: list[float],
                              eta: float) -> list[float]:
    """Return per-token gate values for one completion."""
    gates = []
    for s_t in surprisal_tokens:
        d_t = signal * s_t
        if d_t >= 0:
            g_t = 1.0 / (1.0 + math.exp(-d_t / eta))
        else:
            e = math.exp(d_t / eta)
            g_t = e / (1.0 + e)
        gates.append(g_t)
    return gates


# ---------------------------------------------------------------------------
# Percentile computation
# ---------------------------------------------------------------------------

def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    sorted_v = sorted(values)
    n = len(sorted_v)
    idx = (p / 100.0) * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    frac = idx - lo
    return sorted_v[lo] * (1 - frac) + sorted_v[hi] * frac


def token_keep_rate(gates: list[float], threshold: float) -> float:
    if not gates:
        return 0.0
    return sum(1 for g in gates if g >= threshold) / len(gates)


# ---------------------------------------------------------------------------
# Candidate threshold selection
# ---------------------------------------------------------------------------

def select_candidate_thresholds(
    gates_all: list[float],
    min_keep: float = 0.10,
    max_keep: float = 0.90,
    dedup_tol: float = 0.005,
) -> list[float]:
    """Build candidate threshold set per EXPERIMENT_PLAN rules.

    Rules:
      1. Compute global percentiles: p10, p20, ..., p90.
      2. Always include 0.50.
      3. Sort and deduplicate within dedup_tol.
      4. Discard thresholds with token keep rate outside [min_keep, max_keep].
    """
    pcts = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    candidates = [percentile(gates_all, p) for p in pcts]
    candidates.append(0.50)
    candidates = [c for c in candidates if not math.isnan(c)]

    candidates.sort()
    deduped: list[float] = []
    for c in candidates:
        if not deduped or abs(c - deduped[-1]) > dedup_tol:
            deduped.append(round(c, 4))

    filtered = [
        t for t in deduped
        if min_keep <= token_keep_rate(gates_all, t) <= max_keep
    ]
    return filtered


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Gate calibration for SoftDG-Token-Mask-Offline."
    )
    p.add_argument("--rollout_path", type=str, required=True)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--max_completion_length", type=int, default=2048)
    p.add_argument("--dg_temperature", type=float, default=1.0,
                   help="eta in sigmoid(delight/eta).")
    p.add_argument("--max_completions", type=int, default=None,
                   help="Cap completions processed (for quick smoke tests).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 1. Load model ---------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None,
    ).to(device)
    model.eval()

    # ---- 2. Load rollouts -----------------------------------------------
    print(f"Loading rollouts: {args.rollout_path}")
    records = load_rollouts_text([args.rollout_path], tokenizer)
    records = compute_rewards_and_advantages(records)
    n_correct = sum(1 for r in records if r["reward"] > 0.0)
    n_total = len(records)
    print(f"  {n_total} completions, {n_correct} correct ({100*n_correct/n_total:.1f}%)")

    if args.max_completions is not None:
        records = records[: args.max_completions]
        print(f"  Capped to {len(records)} completions (--max_completions)")

    # ---- 3. Token budget ------------------------------------------------
    token_budget = sum(
        min(len(rec["completion_ids"]), args.max_completion_length)
        for rec in records
    )
    print(f"  Token budget: {token_budget}")

    # ---- 4. Build signal lookups per config ----------------------------
    signal_lookups = {}
    for cfg in CONFIGS:
        signal_lookups[cfg["name"]] = compute_signal_lookup(
            records, cfg["reward_coding"], cfg["signal_type"]
        )

    # ---- 5. Per-completion calibration loop ----------------------------
    print(f"Calibrating {len(records)} completions (eta={args.dg_temperature}) ...")

    # Accumulate per-config global token gate values
    config_all_gates: dict[str, list[float]] = {cfg["name"]: [] for cfg in CONFIGS}
    # Also track per-token signal > 0 for effective token counting
    config_effective_gates: dict[str, list[float]] = {cfg["name"]: [] for cfg in CONFIGS}

    per_completion_records = []

    for i, rec in enumerate(records):
        if i % 2000 == 0:
            print(f"  [{i}/{len(records)}]")

        comp_ids = rec["completion_ids"][: args.max_completion_length]
        if not comp_ids:
            continue

        surprisal_tokens = compute_per_token_surprisal(
            model, tokenizer, rec, args.max_completion_length, device
        )
        if not surprisal_tokens:
            continue

        key = (rec["question_id"], rec["run_id"])
        is_correct = rec["reward"] > 0.0

        comp_rec = {
            "question_id": rec["question_id"],
            "run_id": rec["run_id"],
            "is_correct": is_correct,
            "completion_length": len(comp_ids),
            "mean_surprisal": round(sum(surprisal_tokens) / len(surprisal_tokens), 6),
        }

        for cfg in CONFIGS:
            signal = signal_lookups[cfg["name"]].get(key, 0.0)
            gates = compute_per_token_gates(signal, surprisal_tokens, args.dg_temperature)

            config_all_gates[cfg["name"]].extend(gates)
            if abs(signal) > 1e-12:
                config_effective_gates[cfg["name"]].extend(gates)

            if gates:
                comp_rec[f"gate_{cfg['name']}_mean"] = round(sum(gates) / len(gates), 6)
                comp_rec[f"gate_{cfg['name']}_min"]  = round(min(gates), 6)
                comp_rec[f"gate_{cfg['name']}_max"]  = round(max(gates), 6)
                comp_rec[f"signal_{cfg['name']}"]    = round(signal, 6)

        per_completion_records.append(comp_rec)

    # ---- 6. Write per-completion JSONL ----------------------------------
    jsonl_path = os.path.join(args.output_dir, "token_gate_calibration.jsonl")
    with open(jsonl_path, "w") as f:
        for r in per_completion_records:
            f.write(json.dumps(r) + "\n")
    print(f"Per-completion records: {jsonl_path}")

    # ---- 7. Global stats and candidate thresholds per config ------------
    PCTS = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
    summary_rows = []
    threshold_sets: dict[str, dict] = {}

    for cfg in CONFIGS:
        cname = cfg["name"]
        gates_all = config_all_gates[cname]
        gates_eff = config_effective_gates[cname]

        n_tokens_total = len(gates_all)
        n_tokens_effective_signal = len(gates_eff)

        row: dict = {
            "config": cname,
            "description": cfg["description"],
            "reward_coding": cfg["reward_coding"],
            "signal_type": cfg["signal_type"],
            "loss_type": cfg["loss_type"],
            "n_tokens_total": n_tokens_total,
            "n_tokens_effective_signal": n_tokens_effective_signal,
            "token_budget": token_budget,
            "eta": args.dg_temperature,
        }

        for p in PCTS:
            row[f"gate_p{p:02d}"] = round(percentile(gates_all, p), 4)

        # Candidate thresholds
        candidates = select_candidate_thresholds(gates_all)

        threshold_details = []
        for t in candidates:
            kr = token_keep_rate(gates_all, t)
            dr = 1.0 - kr
            kept_per_epoch = int(round(token_budget * kr))

            # Effective tokens per epoch = tokens passing gate AND with nonzero signal
            # For a per-token estimate: use per-token signal information
            if n_tokens_effective_signal > 0:
                kr_eff = token_keep_rate(gates_eff, t)
                effective_per_epoch = int(round(
                    (n_tokens_effective_signal / max(n_tokens_total, 1)) * token_budget * kr_eff
                ))
            else:
                effective_per_epoch = 0

            proj_epochs = (
                math.ceil(token_budget / max(effective_per_epoch, 1))
                if effective_per_epoch > 0 else float("inf")
            )

            threshold_details.append({
                "threshold": t,
                "token_keep_rate": round(kr, 4),
                "token_drop_rate": round(dr, 4),
                "kept_tokens_per_epoch": kept_per_epoch,
                "effective_tokens_per_epoch": effective_per_epoch,
                "projected_epochs_to_budget": proj_epochs if proj_epochs != float("inf") else "inf",
            })

            # Add to summary row
            tag = f"thr{t:.4f}".replace(".", "p")
            row[f"{tag}_keep_rate"] = round(kr, 4)
            row[f"{tag}_kept_per_epoch"] = kept_per_epoch
            row[f"{tag}_effective_per_epoch"] = effective_per_epoch
            row[f"{tag}_proj_epochs"] = proj_epochs if proj_epochs != float("inf") else 99

        summary_rows.append(row)

        threshold_sets[cname] = {
            "config": cname,
            "description": cfg["description"],
            "reward_coding": cfg["reward_coding"],
            "signal_type": cfg["signal_type"],
            "loss_type": cfg["loss_type"],
            "n_tokens_total": n_tokens_total,
            "token_budget": token_budget,
            "eta": args.dg_temperature,
            "global_percentiles": {f"p{p:02d}": round(percentile(gates_all, p), 4) for p in PCTS},
            "candidate_thresholds": candidates,
            "threshold_details": threshold_details,
        }

    # ---- 8. Write CSV ---------------------------------------------------
    csv_path = os.path.join(args.output_dir, "token_gate_calibration.csv")
    if summary_rows:
        # Collect all fieldnames (rows may have different threshold columns)
        all_fields: list[str] = []
        seen: set = set()
        for row in summary_rows:
            for k in row:
                if k not in seen:
                    all_fields.append(k)
                    seen.add(k)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"Summary CSV: {csv_path}")

    # ---- 9. Write thresholds JSON ----------------------------------------
    thr_path = os.path.join(args.output_dir, "token_gate_calibration_thresholds.json")
    with open(thr_path, "w") as f:
        json.dump(threshold_sets, f, indent=2)
    print(f"Threshold JSON: {thr_path}")

    # ---- 10. Print summary -----------------------------------------------
    print("\n" + "=" * 70)
    print("TOKEN GATE CALIBRATION SUMMARY")
    print(f"  eta={args.dg_temperature}  token_budget={token_budget}")
    print("=" * 70)

    any_valid = False
    for cfg in CONFIGS:
        cname = cfg["name"]
        ts = threshold_sets[cname]
        pcts_info = ts["global_percentiles"]
        details = ts["threshold_details"]

        print(f"\n  Variant {cname}: {cfg['description']}")
        print(f"  n_tokens: {ts['n_tokens_total']}  token_budget: {token_budget}")
        print(f"  gate percentiles: "
              f"p10={pcts_info.get('p10','?')}  p50={pcts_info.get('p50','?')}  "
              f"p90={pcts_info.get('p90','?')}")
        print(f"  Candidate thresholds: {ts['candidate_thresholds']}")

        if details:
            any_valid = True
            print(f"  {'Threshold':>10}  {'KeepRate':>9}  {'KeptTok/ep':>11}  "
                  f"{'EffTok/ep':>10}  {'ProjEp':>7}")
            for d in details:
                print(f"  {d['threshold']:>10.4f}  {d['token_keep_rate']:>9.4f}  "
                      f"{d['kept_tokens_per_epoch']:>11d}  "
                      f"{d['effective_tokens_per_epoch']:>10d}  "
                      f"{d['projected_epochs_to_budget']:>7}")
        else:
            print(f"  WARNING: no candidate thresholds in keep-rate range [0.10, 0.90].")

    print("\n" + "=" * 70)
    if any_valid:
        print("Decision G0: PROCEED — at least one config has candidate thresholds.")
    else:
        print("Decision G0: HOLD — no config has valid thresholds. Investigate.")
    print("=" * 70)

    # ---- 11. Sanity-check command hints ----------------------------------
    print("\nSanity-check command hints:")
    for cfg in CONFIGS:
        cname = cfg["name"]
        details = threshold_sets[cname].get("threshold_details", [])
        if details:
            best_thr = details[0]["threshold"]
            print(f"\n  Variant {cname} ({cfg['description']}):")
            print(f"    REWARD_CODING={cfg['reward_coding']} "
                  f"TRAINING_SIGNAL={cfg['signal_type']} \\")
            print(f"    LOSS_TYPE={cfg['loss_type']} "
                  f"SOFTDG_GATE_THRESHOLD={best_thr} \\")
            print(f"    TARGET_EFFECTIVE_TOKENS=32 \\")
            print(f"    sbatch SoftDG-Token-Mask-Offline/run_math.sh sanity")


if __name__ == "__main__":
    main()
