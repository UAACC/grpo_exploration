#!/usr/bin/env python3
"""Token probability calibration for SoftDG-Token-Mask-Offline.

Recomputes every completion-token probability in a rollout JSONL under the
student model and writes:
  - a full compressed per-token probability dump (jsonl.gz)
  - a per-completion summary file (jsonl)
  - a global percentile summary (json + csv)

Usage:
    python calibrate_token_probabilities.py \\
        --rollout_path /scratch/shuai14/rollouts/math_teacher/rollouts_math_Qwen2.5-0.5B-Instruct_0.6_pick4.jsonl \\
        --model_path /scratch/shuai14/models/Qwen2.5-0.5B
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.abspath(os.path.join(_HERE, ".."))
_DG_OFFLINE = os.path.join(_PROJECT, "DG-offline")
if _DG_OFFLINE not in sys.path:
    sys.path.insert(0, _DG_OFFLINE)

from teacher_agnostic_loader import load_rollouts_text


# ---------------------------------------------------------------------------
# Percentile computation
# ---------------------------------------------------------------------------

def _pct_tag(p: float) -> str:
    """Convert percentile float to a safe key string: 99.9 -> 'p99p9'."""
    return f"p{p}".replace(".", "p")


def compute_percentiles(values: np.ndarray, percentiles: list[float]) -> dict[str, float]:
    if values.size == 0:
        return {str(p): float("nan") for p in percentiles}
    pct_values = np.percentile(values, percentiles).tolist()
    return {str(p): v for p, v in zip(percentiles, pct_values)}


def compute_run_percentiles(values: list[float], percentiles: list[float]) -> dict[str, float]:
    if not values:
        return {_pct_tag(p): None for p in percentiles}
    arr = np.array(values, dtype=np.float32)
    pct_values = np.percentile(arr, percentiles).tolist()
    return {_pct_tag(p): v for p, v in zip(percentiles, pct_values)}


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def build_batches(records: list[dict], max_comp_len: int,
                  batch_size: int, max_tokens_per_batch: int,
                  prompt_overhead: int = 512) -> list[list[dict]]:
    """Split records into batches respecting batch_size and max_tokens_per_batch.

    Token budget uses prompt_overhead + comp_len as the sequence-length estimate
    to account for attention/logit memory that scales with the full sequence.
    """
    batches: list[list[dict]] = []
    current_batch: list[dict] = []
    current_tokens = 0

    for rec in records:
        comp_len = min(len(rec["completion_ids"]), max_comp_len)
        seq_len = prompt_overhead + comp_len

        if current_batch and (
            len(current_batch) >= batch_size
            or current_tokens + seq_len > max_tokens_per_batch
        ):
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(rec)
        current_tokens += seq_len

    if current_batch:
        batches.append(current_batch)

    return batches


# ---------------------------------------------------------------------------
# Batch forward pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_batch_token_logprobs(
    model, tokenizer, batch_records: list[dict],
    max_comp_len: int, device,
) -> list[list[float]]:
    """Return per-token log-probabilities for each record in the batch.

    Returns an empty list for records with zero completion tokens.
    """
    prompt_ids_list: list[list[int]] = []
    comp_ids_list: list[list[int]] = []

    for rec in batch_records:
        messages = [
            {"role": "system", "content": rec["system_prompt"]},
            {"role": "user", "content": rec["problem"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        p_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        c_ids = rec["completion_ids"][:max_comp_len]
        prompt_ids_list.append(p_ids)
        comp_ids_list.append(c_ids)

    full_seqs = [p + c for p, c in zip(prompt_ids_list, comp_ids_list)]
    max_len = max(len(s) for s in full_seqs)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    input_ids = torch.full((len(full_seqs), max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(full_seqs), max_len), dtype=torch.long, device=device)

    for i, seq in enumerate(full_seqs):
        t = torch.tensor(seq, dtype=torch.long)
        input_ids[i, :len(seq)] = t
        attention_mask[i, :len(seq)] = 1

    # Keep bfloat16 for the forward pass; slice to completion positions before
    # converting to float32 to avoid materializing a full (B, L, V) fp32 tensor.
    logits = model(input_ids=input_ids, attention_mask=attention_mask,
                   use_cache=False).logits  # (B, L, V) bfloat16

    results: list[list[float]] = []
    for i, (p_ids, c_ids) in enumerate(zip(prompt_ids_list, comp_ids_list)):
        if not c_ids:
            results.append([])
            continue
        p_len = len(p_ids)
        c_len = len(c_ids)
        # logits[p_len-1] predicts first completion token at position p_len
        comp_logits = logits[i, p_len - 1: p_len - 1 + c_len].float()  # (c_len, V) fp32
        comp_log_probs = F.log_softmax(comp_logits, dim=-1)             # (c_len, V)
        c_ids_t = torch.tensor(c_ids, dtype=torch.long, device=device)
        per_tok_logps = comp_log_probs.gather(1, c_ids_t.unsqueeze(1)).squeeze(1)  # (c_len,)
        results.append(per_tok_logps.cpu().tolist())

    del logits  # free full bfloat16 tensor before next batch
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Token probability calibration for SoftDG-Token-Mask-Offline."
    )
    p.add_argument(
        "--rollout_path", type=str,
        default="/scratch/shuai14/rollouts/math_teacher/"
                "rollouts_math_Qwen2.5-0.5B-Instruct_0.6_pick4.jsonl",
    )
    p.add_argument("--model_path", type=str,
                   default="/scratch/shuai14/models/Qwen2.5-0.5B")
    p.add_argument("--teacher_name", type=str, default="Qwen2.5-0.5B-Instruct")
    p.add_argument("--student_name", type=str, default="Qwen2.5-0.5B")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Default: outputs/token_probability_calibration/"
                        "teacher-TEACHER__student-STUDENT/ relative to this script.")
    p.add_argument("--max_completion_length", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=8,
                   help="Max completions per GPU batch.")
    p.add_argument("--max_tokens_per_batch", type=int, default=12000,
                   help="Max estimated tokens per batch (batch_size × avg_seq_len guard).")
    p.add_argument("--max_completions", type=int, default=None,
                   help="Cap completions processed (for smoke tests).")
    p.add_argument("--percentiles", type=float, nargs="+",
                   default=[0.1, 1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0,
                            60.0, 70.0, 80.0, 90.0, 95.0, 99.0, 99.9],
                   help="Percentile levels to report.")
    p.add_argument("--include_token_text", action="store_true",
                   help="Include decoded token string in per-token dump.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _safe_name(s: str) -> str:
    """Sanitize a model name for use in filenames (removes path separators etc.)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def main():
    args = parse_args()

    pair_tag = f"teacher-{_safe_name(args.teacher_name)}__student-{_safe_name(args.student_name)}"
    if args.output_dir is None:
        args.output_dir = os.path.join(
            _HERE, "outputs", "token_probability_calibration", pair_tag
        )
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 1. Load model ---------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=None,
        ).to(device)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map=None,
        ).to(device)
    model.eval()

    # ---- 2. Load rollouts -----------------------------------------------
    print(f"Loading rollouts: {args.rollout_path}")
    records = load_rollouts_text([args.rollout_path], tokenizer)
    print(f"  {len(records)} completions loaded")

    if args.max_completions is not None:
        records = records[: args.max_completions]
        print(f"  Capped to {len(records)} completions (--max_completions)")

    # ---- 3. Build batches -----------------------------------------------
    batches = build_batches(
        records, args.max_completion_length,
        args.batch_size, args.max_tokens_per_batch,
    )
    print(f"  {len(batches)} batches")

    # ---- 4. Output paths ------------------------------------------------
    token_dump_path  = os.path.join(args.output_dir, f"{pair_tag}_token_probabilities.jsonl.gz")
    run_summary_path = os.path.join(args.output_dir, f"{pair_tag}_run_summaries.jsonl")
    pct_json_path    = os.path.join(args.output_dir, f"{pair_tag}_percentiles.json")
    pct_csv_path     = os.path.join(args.output_dir, f"{pair_tag}_percentiles.csv")

    # ---- 5. Main processing loop ----------------------------------------
    global_logprobs: list[float] = []
    n_problems_seen: set = set()
    n_runs = 0
    n_tokens = 0
    n_empty_completions = 0
    n_truncated_completions = 0

    print(f"Processing {len(records)} completions ...")

    with (
        gzip.open(token_dump_path, "wt", encoding="utf-8") as gz_f,
        open(run_summary_path, "w") as run_f,
    ):
        rec_idx = 0
        for batch_i, batch in enumerate(batches):
            if batch_i % 50 == 0:
                print(f"  Batch [{batch_i}/{len(batches)}] "
                      f"(rec {rec_idx}/{len(records)}, tokens scored={n_tokens})")

            batch_logprobs = compute_batch_token_logprobs(
                model, tokenizer, batch, args.max_completion_length, device
            )

            for rec, token_logps in zip(batch, batch_logprobs):
                qid = rec["question_id"]
                rid = rec["run_id"]
                n_problems_seen.add(qid)
                n_runs += 1

                if len(rec["completion_ids"]) > args.max_completion_length:
                    n_truncated_completions += 1

                if not token_logps:
                    n_empty_completions += 1
                    run_rec: dict = {
                        "question_id": qid,
                        "run_id": rid,
                        "n_tokens": 0,
                        "prob_mean": None, "prob_min": None, "prob_max": None,
                        "logprob_mean": None, "logprob_min": None, "logprob_max": None,
                        "surprisal_mean": None, "surprisal_min": None, "surprisal_max": None,
                    }
                    for p in args.percentiles:
                        tag = _pct_tag(p)
                        run_rec[f"prob_{tag}"] = None
                        run_rec[f"logprob_{tag}"] = None
                        run_rec[f"surprisal_{tag}"] = None
                    run_f.write(json.dumps(run_rec) + "\n")
                    rec_idx += 1
                    continue

                # Write per-token rows to gzipped dump
                comp_ids = rec["completion_ids"][:args.max_completion_length]
                for tok_idx, (tok_id, logp) in enumerate(zip(comp_ids, token_logps)):
                    prob = math.exp(logp)
                    surprisal = -logp
                    tok_row: dict = {
                        "question_id": qid,
                        "run_id": rid,
                        "token_index": tok_idx,
                        "token_id": tok_id,
                    }
                    if args.include_token_text:
                        tok_row["token_text"] = tokenizer.decode([tok_id])
                    tok_row["logprob"] = logp
                    tok_row["prob"] = prob
                    tok_row["surprisal"] = surprisal
                    gz_f.write(json.dumps(tok_row) + "\n")

                global_logprobs.extend(token_logps)
                n_tokens += len(token_logps)

                # Per-run summary
                probs      = [math.exp(lp) for lp in token_logps]
                surprisals = [-lp for lp in token_logps]
                run_pct_prob      = compute_run_percentiles(probs,      args.percentiles)
                run_pct_logprob   = compute_run_percentiles(token_logps, args.percentiles)
                run_pct_surprisal = compute_run_percentiles(surprisals,  args.percentiles)

                run_rec = {
                    "question_id": qid,
                    "run_id": rid,
                    "n_tokens": len(token_logps),
                    "prob_mean":      sum(probs)      / len(probs),
                    "prob_min":       min(probs),
                    "prob_max":       max(probs),
                    "logprob_mean":   sum(token_logps) / len(token_logps),
                    "logprob_min":    min(token_logps),
                    "logprob_max":    max(token_logps),
                    "surprisal_mean": sum(surprisals)  / len(surprisals),
                    "surprisal_min":  min(surprisals),
                    "surprisal_max":  max(surprisals),
                }
                for tag, v in run_pct_prob.items():
                    run_rec[f"prob_{tag}"] = v
                for tag, v in run_pct_logprob.items():
                    run_rec[f"logprob_{tag}"] = v
                for tag, v in run_pct_surprisal.items():
                    run_rec[f"surprisal_{tag}"] = v
                run_f.write(json.dumps(run_rec) + "\n")

                rec_idx += 1

    print(f"Token dump:    {token_dump_path}")
    print(f"Run summaries: {run_summary_path}")

    # ---- 6. Global percentile stats ------------------------------------
    if global_logprobs:
        lp_arr  = np.array(global_logprobs, dtype=np.float64)
        p_arr   = np.exp(lp_arr)
        s_arr   = -lp_arr

        prob_pcts      = compute_percentiles(p_arr,  args.percentiles)
        logprob_pcts   = compute_percentiles(lp_arr, args.percentiles)
        surprisal_pcts = compute_percentiles(s_arr,  args.percentiles)

        prob_mean      = float(p_arr.mean())
        prob_min       = float(p_arr.min())
        prob_max       = float(p_arr.max())
        logprob_mean   = float(lp_arr.mean())
        logprob_min    = float(lp_arr.min())
        logprob_max    = float(lp_arr.max())
        surprisal_mean = float(s_arr.mean())
        surprisal_min  = float(s_arr.min())
        surprisal_max  = float(s_arr.max())
    else:
        prob_pcts = logprob_pcts = surprisal_pcts = {str(p): None for p in args.percentiles}
        prob_mean = prob_min = prob_max = None
        logprob_mean = logprob_min = logprob_max = None
        surprisal_mean = surprisal_min = surprisal_max = None

    summary = {
        "teacher_name":          args.teacher_name,
        "student_name":          args.student_name,
        "rollout_path":          args.rollout_path,
        "model_path":            args.model_path,
        "output_dir":            args.output_dir,
        "max_completion_length": args.max_completion_length,
        "batch_size":            args.batch_size,
        "max_tokens_per_batch":  args.max_tokens_per_batch,
        "percentiles":           args.percentiles,
        "created_at":            datetime.now(timezone.utc).isoformat(),
        "n_problems":            len(n_problems_seen),
        "n_runs":                n_runs,
        "n_tokens":              n_tokens,
        "n_empty_completions":   n_empty_completions,
        "n_truncated_completions": n_truncated_completions,
        "prob": {
            "mean": prob_mean, "min": prob_min, "max": prob_max,
            "percentiles": prob_pcts,
        },
        "logprob": {
            "mean": logprob_mean, "min": logprob_min, "max": logprob_max,
            "percentiles": logprob_pcts,
        },
        "surprisal": {
            "mean": surprisal_mean, "min": surprisal_min, "max": surprisal_max,
            "percentiles": surprisal_pcts,
        },
    }

    with open(pct_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Percentiles JSON: {pct_json_path}")

    # ---- 7. Percentile CSV ----------------------------------------------
    csv_rows = []
    for metric, pcts_dict in [
        ("prob",      prob_pcts),
        ("logprob",   logprob_pcts),
        ("surprisal", surprisal_pcts),
    ]:
        for pct_str, value in pcts_dict.items():
            csv_rows.append({
                "teacher":      args.teacher_name,
                "student":      args.student_name,
                "metric":       metric,
                "percentile":   pct_str,
                "value":        value,
                "n_tokens":     n_tokens,
                "rollout_path": args.rollout_path,
                "model_path":   args.model_path,
            })

    with open(pct_csv_path, "w", newline="") as f:
        fieldnames = ["teacher", "student", "metric", "percentile", "value",
                      "n_tokens", "rollout_path", "model_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Percentiles CSV:  {pct_csv_path}")

    # ---- 8. Console summary -----------------------------------------------
    print("\n" + "=" * 70)
    print("TOKEN PROBABILITY CALIBRATION SUMMARY")
    print(f"  teacher={args.teacher_name}  student={args.student_name}")
    print(f"  n_problems={len(n_problems_seen)}  n_runs={n_runs}  n_tokens={n_tokens}")
    print(f"  n_empty={n_empty_completions}  n_truncated={n_truncated_completions}")
    print()
    for metric, mean_v, min_v, max_v, pcts_dict in [
        ("prob",      prob_mean,      prob_min,      prob_max,      prob_pcts),
        ("logprob",   logprob_mean,   logprob_min,   logprob_max,   logprob_pcts),
        ("surprisal", surprisal_mean, surprisal_min, surprisal_max, surprisal_pcts),
    ]:
        if mean_v is None:
            continue
        print(f"  {metric}:  mean={mean_v:.4f}  min={min_v:.4f}  max={max_v:.4f}")
        pct_line = "  ".join(
            f"p{k}={v:.4f}" for k, v in pcts_dict.items() if v is not None
        )
        print(f"    {pct_line}")
    print("=" * 70)


if __name__ == "__main__":
    main()
