#!/bin/bash
#
# Submit DG-weighted selected-token PG (wgate) training jobs.
#
# Uses calibrated thresholds from EXPERIMENT_PLAN_2026-06-22.md.
# Does NOT re-run calibration (gate formula unchanged from June 20).
#
# Usage:
#   bash SoftDG-Token-Mask-Offline/submit_wgate_sweep.sh [--sanity-only] [--dry-run]
#
# Options:
#   --sanity-only   Submit only sanity jobs (one per eta/variant, closest-to-50% threshold).
#   --dry-run       Print sbatch commands without submitting.

set -euo pipefail

WORK_DIR="/project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/SoftDG-Token-Mask-Offline"
SANITY_ONLY=false
DRY_RUN=false

for arg in "$@"; do
    case "${arg}" in
        --sanity-only) SANITY_ONLY=true ;;
        --dry-run)     DRY_RUN=true ;;
    esac
done

echo "=== wgate sweep submission ==="
echo "  sanity_only: ${SANITY_ONLY}"
echo "  dry_run:     ${DRY_RUN}"
echo

submit() {
    local cmd="$*"
    if [ "${DRY_RUN}" = "true" ]; then
        echo "  [DRY-RUN] ${cmd}"
    else
        eval "${cmd}"
    fi
}

# Thresholds from EXPERIMENT_PLAN_2026-06-22.md.
# Format: "eta variant reward_coding training_signal loss_type threshold"
# Variant A: signed raw_reward dr_grpo
# Variant B: zero_two advantage grpo
# Variant C: signed advantage dr_grpo

TOTAL_SUBMITTED=0

launch_jobs() {
    local eta="$1"
    local variant="$2"
    local rc="$3"
    local ts="$4"
    local lt="$5"
    shift 5
    local thresholds=("$@")

    if [ ${#thresholds[@]} -eq 0 ]; then
        echo "  eta=${eta} Variant ${variant}: no valid thresholds — skip"
        return
    fi

    if [ "${SANITY_ONLY}" = "true" ]; then
        # Pick threshold closest to 0.50 keep rate.
        # Since we don't have keep rates here directly, use threshold value closest to 0.50.
        local best_thr="${thresholds[0]}"
        local best_dist
        best_dist=$(python3 -c "print(abs(float('${thresholds[0]}') - 0.50))")
        for thr in "${thresholds[@]}"; do
            dist=$(python3 -c "print(abs(float('${thr}') - 0.50))")
            if python3 -c "import sys; sys.exit(0 if float('${dist}') < float('${best_dist}') else 1)"; then
                best_thr="${thr}"
                best_dist="${dist}"
            fi
        done
        echo "  eta=${eta} Variant ${variant} thr=${best_thr} -> sanity"
        submit "DG_ETA=${eta} REWARD_CODING=${rc} TRAINING_SIGNAL=${ts} LOSS_TYPE=${lt} SOFTDG_GATE_THRESHOLD=${best_thr} sbatch ${WORK_DIR}/run_math_wgate.sh sanity"
        TOTAL_SUBMITTED=$((TOTAL_SUBMITTED + 1))
    else
        for thr in "${thresholds[@]}"; do
            echo "  eta=${eta} Variant ${variant} thr=${thr} -> train"
            submit "DG_ETA=${eta} REWARD_CODING=${rc} TRAINING_SIGNAL=${ts} LOSS_TYPE=${lt} SOFTDG_GATE_THRESHOLD=${thr} sbatch ${WORK_DIR}/run_math_wgate.sh train"
            TOTAL_SUBMITTED=$((TOTAL_SUBMITTED + 1))
        done
    fi
}

# ---------------------------------------------------------------------------
# eta=1.0 — Variant A only (weighted anchor)
# ---------------------------------------------------------------------------
echo "--- eta=1.0 ---"
launch_jobs 1.0 A signed raw_reward dr_grpo 0.4845 0.4982

# ---------------------------------------------------------------------------
# eta=0.75
# ---------------------------------------------------------------------------
echo "--- eta=0.75 ---"
launch_jobs 0.75 A signed raw_reward dr_grpo 0.3737 0.4793 0.4977
launch_jobs 0.75 B zero_two advantage grpo 0.4999
launch_jobs 0.75 C signed advantage dr_grpo 0.4999

# ---------------------------------------------------------------------------
# eta=0.5
# ---------------------------------------------------------------------------
echo "--- eta=0.5 ---"
launch_jobs 0.5 A signed raw_reward dr_grpo 0.4690 0.4965

# ---------------------------------------------------------------------------
# eta=0.25
# ---------------------------------------------------------------------------
echo "--- eta=0.25 ---"
launch_jobs 0.25 A signed raw_reward dr_grpo 0.4383 0.4930 0.4991

# ---------------------------------------------------------------------------
# eta=0.1
# ---------------------------------------------------------------------------
echo "--- eta=0.1 ---"
launch_jobs 0.1 A signed raw_reward dr_grpo 0.0204 0.3497 0.4825 0.4977 0.5093

echo
echo "=== Submission complete ==="
echo "  Submitted: ${TOTAL_SUBMITTED}"
if [ "${DRY_RUN}" = "true" ]; then
    echo "  [DRY-RUN] No jobs actually submitted."
fi
