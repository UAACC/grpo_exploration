#!/usr/bin/env python3
"""Parse calibration results for all new eta values and print valid training commands.

Reads:
    outputs/calibration_eta{ETA_TAG}/token_gate_calibration_thresholds.json
for eta in [0.75, 0.5, 0.25, 0.1].

Prints:
  - Valid eta/variant/threshold combinations (with keep rates and projected epochs).
  - The sbatch commands needed for sanity checks and main training runs.
  - A summary table for EXPERIMENT_TRACKER.md.

Usage:
    python SoftDG-Token-Mask-Offline/parse_calibration_for_training.py
"""
from __future__ import annotations

import json
import os
import sys

WORK_DIR = os.path.dirname(os.path.abspath(__file__))

ETA_VALUES = [0.75, 0.5, 0.25, 0.1]

VARIANT_META = {
    "A": {
        "reward_coding": "signed",
        "training_signal": "raw_reward",
        "loss_type": "dr_grpo",
    },
    "B": {
        "reward_coding": "zero_two",
        "training_signal": "advantage",
        "loss_type": "grpo",
    },
    "C": {
        "reward_coding": "signed",
        "training_signal": "advantage",
        "loss_type": "dr_grpo",
    },
}

VALIDITY_MIN_KEEP = 0.10
VALIDITY_MAX_KEEP = 0.90
VALIDITY_MAX_EPOCHS = 20


def eta_tag(eta: float) -> str:
    return str(eta).replace(".", "p")


def _to_finite_float(x) -> float | None:
    """Return x as a finite float, or None if not representable."""
    import math
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def is_valid(detail: dict) -> bool:
    kr  = _to_finite_float(detail.get("token_keep_rate"))
    eff = _to_finite_float(detail.get("effective_tokens_per_epoch"))
    ep  = _to_finite_float(detail.get("projected_epochs_to_budget"))
    if kr is None or eff is None or ep is None:
        return False
    if not (VALIDITY_MIN_KEEP <= kr <= VALIDITY_MAX_KEEP):
        return False
    if eff <= 0:
        return False
    if ep > VALIDITY_MAX_EPOCHS:
        return False
    return True


def find_closest_to_50pct(details: list[dict]) -> dict | None:
    valid = [d for d in details if is_valid(d)]
    if not valid:
        return None
    return min(valid, key=lambda d: abs(d["token_keep_rate"] - 0.50))


def main():
    print("=" * 72)
    print("ETA SWEEP — CALIBRATION RESULTS PARSER")
    print("=" * 72)

    all_valid: list[dict] = []   # {eta, variant, threshold, keep_rate, proj_epochs}
    missing_files: list[str] = []

    for eta in ETA_VALUES:
        etag = eta_tag(eta)
        thr_path = os.path.join(
            WORK_DIR, "outputs", f"calibration_eta{etag}",
            "token_gate_calibration_thresholds.json"
        )
        if not os.path.exists(thr_path):
            missing_files.append(thr_path)
            print(f"\n  eta={eta}: MISSING — {thr_path}")
            continue

        with open(thr_path) as f:
            data = json.load(f)

        print(f"\n  eta={eta}  ({thr_path})")
        for variant, meta in VARIANT_META.items():
            cfg = data.get(variant, {})
            details = cfg.get("threshold_details", [])
            pcts = cfg.get("global_percentiles", {})

            p10 = pcts.get("p10", "?")
            p50 = pcts.get("p50", "?")
            p90 = pcts.get("p90", "?")
            print(f"    Variant {variant}: gate p10={p10}  p50={p50}  p90={p90}")

            valid_details = [d for d in details if is_valid(d)]
            if not valid_details:
                print(f"      -> NO valid thresholds")
            else:
                for d in valid_details:
                    print(f"      -> thr={d['threshold']:.4f}  kr={d['token_keep_rate']:.4f}  "
                          f"eff_tok/ep={d['effective_tokens_per_epoch']}  "
                          f"proj_ep={d['projected_epochs_to_budget']}")
                    all_valid.append({
                        "eta": eta,
                        "eta_tag": etag,
                        "variant": variant,
                        **meta,
                        **d,
                    })

    if missing_files:
        print(f"\n  WARNING: {len(missing_files)} calibration file(s) missing.")
        print("  Run submit_eta_sweep_calibration.sh and wait for completion first.")
        sys.exit(1)

    if not all_valid:
        print("\n  No valid thresholds found across all eta/variant combinations.")
        print("  Consider: lower thresholds, different token budget, or different rollout.")
        sys.exit(0)

    # ---- Print sanity check commands ----------------------------------------
    print("\n" + "=" * 72)
    print("SANITY CHECK COMMANDS")
    print("(one per eta/variant pair, closest threshold to 50% keep rate)")
    print("=" * 72)

    seen_sanity: set = set()
    for entry in all_valid:
        key = (entry["eta"], entry["variant"])
        if key in seen_sanity:
            continue
        seen_sanity.add(key)

        # Find closest-to-50% threshold for this eta/variant
        pair_entries = [e for e in all_valid
                        if e["eta"] == entry["eta"] and e["variant"] == entry["variant"]]
        best = min(pair_entries, key=lambda e: abs(e["token_keep_rate"] - 0.50))

        print(f"\n  eta={entry['eta']} Variant {entry['variant']} "
              f"({entry['reward_coding']} + {entry['training_signal']}):")
        print(f"    DG_ETA={entry['eta']} \\")
        print(f"    REWARD_CODING={entry['reward_coding']} \\")
        print(f"    TRAINING_SIGNAL={entry['training_signal']} \\")
        print(f"    LOSS_TYPE={entry['loss_type']} \\")
        print(f"    SOFTDG_GATE_THRESHOLD={best['threshold']} \\")
        print(f"    TARGET_EFFECTIVE_TOKENS=32 \\")
        print(f"    sbatch SoftDG-Token-Mask-Offline/run_math.sh sanity")

    # ---- Print main training commands ----------------------------------------
    print("\n" + "=" * 72)
    print("MAIN TRAINING COMMANDS")
    print(f"(all {len(all_valid)} valid eta/variant/threshold combinations)")
    print("=" * 72)

    for entry in all_valid:
        etag = entry["eta_tag"]
        thr = entry["threshold"]
        thr_str = f"{thr:.4f}"
        print(f"\n  eta={entry['eta']} Variant {entry['variant']} thr={thr_str} "
              f"(kr={entry['token_keep_rate']:.4f}, proj_ep={entry['projected_epochs_to_budget']}):")
        print(f"    DG_ETA={entry['eta']} \\")
        print(f"    REWARD_CODING={entry['reward_coding']} \\")
        print(f"    TRAINING_SIGNAL={entry['training_signal']} \\")
        print(f"    LOSS_TYPE={entry['loss_type']} \\")
        print(f"    SOFTDG_GATE_THRESHOLD={thr} \\")
        print(f"    sbatch SoftDG-Token-Mask-Offline/run_math.sh train")

    # ---- Tracker table -------------------------------------------------------
    print("\n" + "=" * 72)
    print("EXPERIMENT_TRACKER.md TABLE ROWS (copy-paste into M0/M3)")
    print("=" * 72)
    print("\n### M0: Gate Calibration — New Eta Values\n")
    print("| Run ID | Eta | Variant | Status | Valid Thresholds |")
    print("|--------|-----|---------|--------|-----------------|")
    cal_reported: set = set()
    for eta in ETA_VALUES:
        etag = eta_tag(eta)
        for variant in VARIANT_META:
            pair_entries = [e for e in all_valid if e["eta"] == eta and e["variant"] == variant]
            if pair_entries:
                thrs = ", ".join(f"{e['threshold']:.4f} (kr={e['token_keep_rate']:.2f})"
                                 for e in pair_entries)
                print(f"| CAL-{variant}-eta{etag} | {eta} | {variant} | DONE ✓ | {thrs} |")
            else:
                print(f"| CAL-{variant}-eta{etag} | {eta} | {variant} | DONE ✗ | No valid thresholds |")

    print("\n### M3: Main Method — Eta Sweep\n")
    print("| Run ID | Eta | Variant | Threshold | KR | Proj Ep | MATH Acc | Status |")
    print("|--------|-----|---------|-----------|-----|---------|----------|--------|")
    for i, entry in enumerate(all_valid, 1):
        etag = entry["eta_tag"]
        thr = entry["threshold"]
        kr = entry["token_keep_rate"]
        pe = entry["projected_epochs_to_budget"]
        print(f"| ETA-{i:02d} | {entry['eta']} | {entry['variant']} | {thr:.4f} "
              f"| {kr:.2f} | {pe} | — | TODO |")

    print(f"\nTotal must-run training jobs: {len(all_valid)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
