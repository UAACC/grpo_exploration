#!/usr/bin/env python3
"""Gate calibration for SoftDG token-level gating.

Loads bad-teacher MATH rollouts, computes per-token surprisal under the student
model (no training), and measures the gate distribution for 4 signal configs.
Used to select softdg_gate_threshold before launching full training runs.

Output files (written to --output_dir, default outputs/):
    gate_calibration_v2.jsonl          -- per-completion records
    gate_calibration_v2.csv            -- summary statistics per config
    gate_calibration_v2_thresholds.json -- candidate threshold sets per config

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
# Config definitions — 4 calibration configs as per EXPERIMENT_PLAN
# ---------------------------------------------------------------------------

CONFIGS = [
    {"name": "signed_raw_reward_token",       "reward_coding": "signed",    "signal_type": "raw_reward", "gating": "token"},
    {"name": "zero_two_raw_reward_token",      "reward_coding": "zero_two",  "signal_type": "raw_reward", "gating": "token"},
    {"name": "zero_two_advantage_token",       "reward_coding": "zero_two",  "signal_type": "advantage",  "gating": "token"},
    {"name": "signed_raw_reward_completion",   "reward_coding": "signed",    "signal_type": "raw_reward", "gating": "completion"},
]


# ---------------------------------------------------------------------------
# Data quality flags
# ---------------------------------------------------------------------------

def flag_no_boxed(response: str) -> bool:
    return "\\boxed{" not in response


def flag_truncated(comp_ids: list, max_len: int, margin: int = 10) -> bool:
    return len(comp_ids) >= max_len - margin


def flag_very_short(comp_ids: list, threshold: int = 20) -> bool:
    return len(comp_ids) < threshold


def flag_repeated_lines(response: str, threshold: float = 0.3) -> bool:
    lines = [l.strip() for l in response.split("\n") if l.strip()]
    if len(lines) < 3:
        return False
    n_unique = len(set(lines))
    return (1.0 - n_unique / len(lines)) > threshold


def flag_repeated_tail(response: str, min_repeat_chars: int = 80) -> bool:
    """Detect if the last chunk of text appears verbatim earlier in the response."""
    if len(response) < min_repeat_chars * 2:
        return False
    tail = response[-min_repeat_chars:]
    return tail in response[:-min_repeat_chars]


# ---------------------------------------------------------------------------
# Advantage re-computation per reward coding
# ---------------------------------------------------------------------------

def compute_advantages_by_coding(records: list[dict], reward_coding: str, eps: float = 1e-4) -> dict:
    """Return {(question_id, run_id): advantage} under the given reward coding."""
    if reward_coding == "signed":
        mapped = {(r["question_id"], r["run_id"]): (1.0 if r["reward"] > 0.0 else -1.0) for r in records}
    else:  # zero_two
        mapped = {(r["question_id"], r["run_id"]): r["reward"] for r in records}

    groups: dict = defaultdict(list)
    for rec in records:
        groups[rec["question_id"]].append(rec)

    advantages = {}
    for group in groups.values():
        rewards = [mapped[(r["question_id"], r["run_id"])] for r in group]
        mean_r = sum(rewards) / len(rewards)
        std_r = (sum((rv - mean_r) ** 2 for rv in rewards) / len(rewards)) ** 0.5
        for rec, rv in zip(group, rewards):
            advantages[(rec["question_id"], rec["run_id"])] = (rv - mean_r) / (std_r + eps)

    return advantages


def get_signal(rec: dict, cfg: dict, adv_signed: dict, adv_zero_two: dict) -> float:
    key = (rec["question_id"], rec["run_id"])
    if cfg["reward_coding"] == "signed":
        raw = 1.0 if rec["reward"] > 0.0 else -1.0
        adv = adv_signed.get(key, 0.0)
    else:
        raw = rec["reward"]
        adv = adv_zero_two.get(key, 0.0)
    return raw if cfg["signal_type"] == "raw_reward" else adv


# ---------------------------------------------------------------------------
# Gate computation (pure Python for flexibility)
# ---------------------------------------------------------------------------

def compute_gate(signal: float, surprisal_tokens: list[float], gating: str, eta: float) -> float:
    if not surprisal_tokens:
        return 0.0
    if gating == "completion":
        mean_s = sum(surprisal_tokens) / len(surprisal_tokens)
        d = signal * mean_s
    else:  # "token"
        per_token_gates = []
        for s_t in surprisal_tokens:
            d_t = signal * s_t
            # Numerically stable sigmoid
            if d_t >= 0:
                g_t = 1.0 / (1.0 + math.exp(-d_t / eta))
            else:
                e = math.exp(d_t / eta)
                g_t = e / (1.0 + e)
            per_token_gates.append(g_t)
        return sum(per_token_gates) / len(per_token_gates)

    # completion-level sigmoid
    if d >= 0:
        return 1.0 / (1.0 + math.exp(-d / eta))
    else:
        e = math.exp(d / eta)
        return e / (1.0 + e)


# ---------------------------------------------------------------------------
# Surprisal computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_surprisal(model, tokenizer, rec: dict, max_comp_len: int, device) -> list[float]:
    """Return per-token surprisal list for one completion under the current model."""
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

    # logits[L_p-1 : L_p-1+L_c] predict completion tokens (shift by 1)
    logits = model(full_ids).logits[0]              # (L, V)
    logits_comp = logits[L_p - 1: L_p - 1 + L_c].float()  # (L_c, V)
    log_probs = F.log_softmax(logits_comp, dim=-1)
    comp_tensor = torch.tensor(comp_ids, dtype=torch.long, device=device)
    per_token_logps = log_probs.gather(1, comp_tensor.unsqueeze(1)).squeeze(1)  # (L_c,)

    return (-per_token_logps).cpu().tolist()


# ---------------------------------------------------------------------------
# Aggregate statistics helpers
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


def keep_rate(gates: list[float], threshold: float) -> float:
    if not gates:
        return 0.0
    return sum(1 for g in gates if g >= threshold) / len(gates)


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def select_candidate_thresholds(
    gates_wrong:   list[float],
    gates_correct: list[float],
    gates_all:     list[float],
    min_keep: float = 0.10,
    max_keep: float = 0.80,
    dedup_tol: float = 0.005,
) -> list[float]:
    """Build a candidate threshold set per the EXPERIMENT_PLAN rules."""
    candidates = []

    # From wrong/low-quality distribution: p50, p60, p70, p80, p90
    for p in (50, 60, 70, 80, 90):
        candidates.append(percentile(gates_wrong, p))

    # From correct/effective distribution: p10, p20, p30
    for p in (10, 20, 30):
        candidates.append(percentile(gates_correct, p))

    # Neutral sigmoid boundary
    candidates.append(0.50)

    # Remove NaN
    candidates = [c for c in candidates if not math.isnan(c)]

    # Sort and deduplicate within tol
    candidates.sort()
    deduped = []
    for c in candidates:
        if not deduped or abs(c - deduped[-1]) > dedup_tol:
            deduped.append(round(c, 4))

    # Filter by keep-rate range
    filtered = [t for t in deduped if min_keep <= keep_rate(gates_all, t) <= max_keep]

    return filtered


# ---------------------------------------------------------------------------
# Decision gate check
# ---------------------------------------------------------------------------

def check_decision_gate(
    threshold: float,
    gates_all:     list[float],
    gates_correct: list[float],
    gates_wrong:   list[float],
    gates_flagged_wrong: list[float],
) -> dict:
    """Check if a threshold satisfies the decision criteria from the plan."""
    n_below = sum(1 for g in gates_all if g < threshold)
    kr_correct = keep_rate(gates_correct, threshold)
    kr_wrong   = keep_rate(gates_wrong,   threshold)
    frac_flagged_wrong_below = (
        sum(1 for g in gates_flagged_wrong if g < threshold) / len(gates_flagged_wrong)
        if gates_flagged_wrong else float("nan")
    )
    passes = (
        n_below > 0
        and kr_correct > kr_wrong
        and (math.isnan(frac_flagged_wrong_below) or frac_flagged_wrong_below >= 0.5)
    )
    return {
        "threshold": threshold,
        "passes_decision_gate": passes,
        "low_gate_skips": n_below,
        "correct_keep_rate": round(kr_correct, 4),
        "wrong_keep_rate": round(kr_wrong, 4),
        "frac_flagged_wrong_below": round(frac_flagged_wrong_below, 4)
            if not math.isnan(frac_flagged_wrong_below) else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Gate calibration for SoftDG token-level gating.")
    p.add_argument("--rollout_path", type=str, required=True)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--max_completion_length", type=int, default=2048)
    p.add_argument("--dg_temperature", type=float, default=1.0,
                   help="eta in sigmoid(delight/eta).")
    p.add_argument("--max_completions", type=int, default=None,
                   help="Cap number of completions processed (for quick smoke tests).")
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

    # ---- 2. Load rollouts ------------------------------------------------
    print(f"Loading rollouts: {args.rollout_path}")
    records = load_rollouts_text([args.rollout_path], tokenizer)
    records = compute_rewards_and_advantages(records)
    n_correct = sum(1 for r in records if r["reward"] > 0.0)
    print(f"  {len(records)} completions, {n_correct} correct ({100*n_correct/len(records):.1f}%)")

    if args.max_completions is not None:
        records = records[: args.max_completions]
        print(f"  Capped to {len(records)} completions (--max_completions)")

    # ---- 3. Pre-compute advantages per reward coding ---------------------
    adv_signed   = compute_advantages_by_coding(records, "signed")
    adv_zero_two = compute_advantages_by_coding(records, "zero_two")

    # ---- 4. Mark duplicates per question ---------------------------------
    q_responses: dict = defaultdict(set)
    duplicate_keys: set = set()
    for rec in records:
        key = (rec["question_id"], rec["run_id"])
        resp = rec["response"]
        if resp in q_responses[rec["question_id"]]:
            duplicate_keys.add(key)
        q_responses[rec["question_id"]].add(resp)

    # ---- 5. Per-completion calibration loop ------------------------------
    print(f"Calibrating {len(records)} completions (eta={args.dg_temperature}) ...")
    output_records = []

    for i, rec in enumerate(records):
        if i % 2000 == 0:
            print(f"  [{i}/{len(records)}]")

        comp_ids = rec["completion_ids"][: args.max_completion_length]
        is_correct = rec["reward"] > 0.0

        # Data quality flags
        flags = {
            "no_boxed":       flag_no_boxed(rec["response"]),
            "truncated":      flag_truncated(comp_ids, args.max_completion_length),
            "very_short":     flag_very_short(comp_ids),
            "repeated_lines": flag_repeated_lines(rec["response"]),
            "repeated_tail":  flag_repeated_tail(rec["response"]),
            "duplicate":      (rec["question_id"], rec["run_id"]) in duplicate_keys,
        }
        is_low_quality = any(flags.values())

        # Per-token surprisal
        surprisal_tokens = compute_surprisal(model, tokenizer, rec, args.max_completion_length, device)
        mean_surprisal = sum(surprisal_tokens) / len(surprisal_tokens) if surprisal_tokens else 0.0

        # Gates for each config
        config_gates: dict[str, float] = {}
        config_signals: dict[str, float] = {}
        for cfg in CONFIGS:
            signal = get_signal(rec, cfg, adv_signed, adv_zero_two)
            gate = compute_gate(signal, surprisal_tokens, cfg["gating"], args.dg_temperature)
            config_gates[cfg["name"]]   = gate
            config_signals[cfg["name"]] = signal

        out_rec = {
            "question_id":      rec["question_id"],
            "run_id":           rec["run_id"],
            "reward":           rec["reward"],
            "is_correct":       is_correct,
            "completion_length": len(comp_ids),
            "mean_surprisal":   round(mean_surprisal, 6),
            "is_low_quality":   is_low_quality,
        }
        out_rec.update({f"flag_{k}": v for k, v in flags.items()})
        out_rec.update({f"gate_{k}":   round(v, 6) for k, v in config_gates.items()})
        out_rec.update({f"signal_{k}": round(v, 6) for k, v in config_signals.items()})
        output_records.append(out_rec)

    # ---- 6. Write per-completion JSONL ----------------------------------
    jsonl_path = os.path.join(args.output_dir, "gate_calibration_v2.jsonl")
    with open(jsonl_path, "w") as f:
        for r in output_records:
            f.write(json.dumps(r) + "\n")
    print(f"Per-completion records: {jsonl_path}")

    # ---- 7. Aggregate stats per config ----------------------------------
    PCTS = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    summary_rows = []
    threshold_sets: dict[str, dict] = {}

    for cfg in CONFIGS:
        cname = cfg["name"]
        gates_all     = [r[f"gate_{cname}"] for r in output_records]
        gates_correct = [r[f"gate_{cname}"] for r in output_records if r["is_correct"]]
        gates_wrong   = [r[f"gate_{cname}"] for r in output_records if not r["is_correct"]]
        gates_low_q   = [r[f"gate_{cname}"] for r in output_records if r["is_low_quality"]]
        # Flagged wrong: wrong AND any quality flag
        gates_flagged_wrong = [
            r[f"gate_{cname}"] for r in output_records
            if not r["is_correct"] and r["is_low_quality"]
        ]

        row: dict = {"config": cname}
        row["n_total"]   = len(gates_all)
        row["n_correct"] = len(gates_correct)
        row["n_wrong"]   = len(gates_wrong)
        row["n_low_quality"] = len(gates_low_q)

        for pct in PCTS:
            row[f"gate_p{pct:02d}"]         = round(percentile(gates_all,     pct), 4)
            row[f"gate_correct_p{pct:02d}"] = round(percentile(gates_correct, pct), 4)
            row[f"gate_wrong_p{pct:02d}"]   = round(percentile(gates_wrong,   pct), 4)

        # Keep-rate curve over representative thresholds
        sweep_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for t in sweep_thresholds:
            row[f"kr_all_thr{t:.1f}"]     = round(keep_rate(gates_all,     t), 4)
            row[f"kr_correct_thr{t:.1f}"] = round(keep_rate(gates_correct, t), 4)
            row[f"kr_wrong_thr{t:.1f}"]   = round(keep_rate(gates_wrong,   t), 4)

        summary_rows.append(row)

        # Candidate thresholds
        candidates = select_candidate_thresholds(
            gates_wrong, gates_correct, gates_all
        )
        decision_checks = [
            check_decision_gate(t, gates_all, gates_correct, gates_wrong, gates_flagged_wrong)
            for t in candidates
        ]
        valid_thresholds = [c for c in decision_checks if c["passes_decision_gate"]]

        threshold_sets[cname] = {
            "candidate_thresholds": candidates,
            "valid_thresholds": valid_thresholds,
            "any_valid": len(valid_thresholds) > 0,
            "decision_checks": decision_checks,
            "gate_p10_all":     round(percentile(gates_all, 10), 4),
            "gate_p50_all":     round(percentile(gates_all, 50), 4),
            "gate_p90_all":     round(percentile(gates_all, 90), 4),
            "gate_p50_correct": round(percentile(gates_correct, 50), 4),
            "gate_p50_wrong":   round(percentile(gates_wrong,   50), 4),
        }

    # ---- 8. Write CSV summary -------------------------------------------
    csv_path = os.path.join(args.output_dir, "gate_calibration_v2.csv")
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"Summary CSV: {csv_path}")

    # ---- 9. Write thresholds JSON ----------------------------------------
    thr_path = os.path.join(args.output_dir, "gate_calibration_v2_thresholds.json")
    with open(thr_path, "w") as f:
        json.dump(threshold_sets, f, indent=2)
    print(f"Threshold sets: {thr_path}")

    # ---- 10. Print decision summary --------------------------------------
    print("\n" + "=" * 70)
    print("GATE CALIBRATION SUMMARY")
    print("=" * 70)
    proceed = False
    for cfg in CONFIGS:
        cname = cfg["name"]
        ts = threshold_sets[cname]
        status = "PROCEED" if ts["any_valid"] else "WARNING: no valid threshold"
        print(f"\n  Config: {cname}")
        print(f"  Status: {status}")
        print(f"  Candidates: {ts['candidate_thresholds']}")
        print(f"  gate_p50(all/correct/wrong): "
              f"{ts['gate_p50_all']} / {ts['gate_p50_correct']} / {ts['gate_p50_wrong']}")
        if ts["valid_thresholds"]:
            proceed = True
            for v in ts["valid_thresholds"]:
                print(f"    ✓ thr={v['threshold']:.4f}  "
                      f"kr_correct={v['correct_keep_rate']:.3f}  "
                      f"kr_wrong={v['wrong_keep_rate']:.3f}  "
                      f"low_gate_skips={v['low_gate_skips']}")
        else:
            print(f"    ✗ No threshold passed all decision criteria.")

    print("\n" + "=" * 70)
    if proceed:
        print("Decision: PROCEED with training (at least one config has valid thresholds).")
    else:
        print("Decision: HOLD — calibration did not find valid thresholds. Investigate.")
    print("=" * 70)

    # ---- 11. Write sanity-check command hint ----------------------------
    # Find the first valid threshold for the primary config
    primary = "signed_raw_reward_token"
    primary_ts = threshold_sets.get(primary, {})
    primary_valid = primary_ts.get("valid_thresholds", [])
    if primary_valid:
        first_thr = primary_valid[0]["threshold"]
        print(f"\nSanity-check command (primary config, first valid threshold):")
        print(f"  REWARD_CODING=signed TRAINING_SIGNAL=raw_reward \\")
        print(f"  LOSS_TYPE=dr_grpo DG_GATING=token \\")
        print(f"  SOFTDG_GATE_THRESHOLD={first_thr} TARGET_EFFECTIVE_COMPLETIONS=32 \\")
        print(f"  sbatch SoftDG-Offline-to-the-bottom/run_math.sh train")


if __name__ == "__main__":
    main()
