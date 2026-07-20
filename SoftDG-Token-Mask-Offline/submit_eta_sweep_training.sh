#!/bin/bash
#
# Submit full training jobs for the eta sweep, based on calibration results.
#
# Usage:
#   bash SoftDG-Token-Mask-Offline/submit_eta_sweep_training.sh [--sanity-only] [--dry-run]
#
# Options:
#   --sanity-only   Submit only the closest-to-50%-keep-rate sanity job per eta/variant pair.
#   --dry-run       Print commands without submitting.
#
# MUST be run AFTER all calibration jobs from submit_eta_sweep_calibration.sh complete.
# Reads outputs/calibration_eta*/token_gate_calibration_thresholds.json for valid thresholds.

set -euo pipefail

WORK_DIR="/project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/SoftDG-Token-Mask-Offline"
ETA_VALUES=(0.75 0.5 0.25 0.1)
SANITY_ONLY=false
DRY_RUN=false

for arg in "$@"; do
    case "${arg}" in
        --sanity-only) SANITY_ONLY=true ;;
        --dry-run)     DRY_RUN=true ;;
    esac
done

echo "=== eta sweep training submission ==="
echo "  sanity_only: ${SANITY_ONLY}"
echo "  dry_run:     ${DRY_RUN}"
echo

declare -A VARIANT_REWARD_CODING
VARIANT_REWARD_CODING=([A]="signed" [B]="zero_two" [C]="signed")
declare -A VARIANT_TRAINING_SIGNAL
VARIANT_TRAINING_SIGNAL=([A]="raw_reward" [B]="advantage" [C]="advantage")
declare -A VARIANT_LOSS_TYPE
VARIANT_LOSS_TYPE=([A]="dr_grpo" [B]="grpo" [C]="dr_grpo")

eta_tag() {
    printf "%s" "$1" | tr '.' 'p'
}

submit() {
    local cmd="$*"
    if [ "${DRY_RUN}" = "true" ]; then
        echo "  [DRY-RUN] ${cmd}"
    else
        eval "${cmd}"
    fi
}

TOTAL_SUBMITTED=0
TOTAL_SKIPPED=0

for ETA in "${ETA_VALUES[@]}"; do
    ETAG=$(eta_tag "${ETA}")
    THR_FILE="${WORK_DIR}/outputs/calibration_eta${ETAG}/token_gate_calibration_thresholds.json"

    if [ ! -f "${THR_FILE}" ]; then
        echo "  MISSING calibration for eta=${ETA}: ${THR_FILE}"
        echo "  Run submit_eta_sweep_calibration.sh first."
        TOTAL_SKIPPED=$((TOTAL_SKIPPED + 3))
        continue
    fi

    echo "  eta=${ETA} — loading ${THR_FILE}"

    for VARIANT in A B C; do
        RC="${VARIANT_REWARD_CODING[$VARIANT]}"
        TS="${VARIANT_TRAINING_SIGNAL[$VARIANT]}"
        LT="${VARIANT_LOSS_TYPE[$VARIANT]}"

        # Extract valid thresholds using Python (keep rate in [0.10, 0.90], proj_epochs <= 20)
        # Pass values via sys.argv; single-quoted <<'PYEOF' prevents shell expansion inside.
        VALID_THRS=$(python3 - "${THR_FILE}" "${VARIANT}" "${SANITY_ONLY}" <<'PYEOF'
import json, math, sys

def to_finite_float(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None

thr_file, variant, sanity_only = sys.argv[1], sys.argv[2], sys.argv[3]

with open(thr_file) as f:
    data = json.load(f)

cfg = data.get(variant, {})
details = cfg.get("threshold_details", [])

valid = []
for d in details:
    kr  = to_finite_float(d.get("token_keep_rate"))
    eff = to_finite_float(d.get("effective_tokens_per_epoch"))
    ep  = to_finite_float(d.get("projected_epochs_to_budget"))
    if (
        kr is not None and 0.10 <= kr <= 0.90
        and eff is not None and eff > 0
        and ep is not None and ep <= 20
    ):
        valid.append((d["threshold"], kr, d))

if sanity_only == "true":
    if valid:
        best = min(valid, key=lambda t: abs(t[1] - 0.50))
        print(str(best[0]))
else:
    print(" ".join(str(t[0]) for t in valid))
PYEOF
)

        if [ -z "${VALID_THRS}" ]; then
            echo "    Variant ${VARIANT}: no valid thresholds — skip"
            TOTAL_SKIPPED=$((TOTAL_SKIPPED + 1))
            continue
        fi

        for THR in ${VALID_THRS}; do
            CMD_MODE="$( [ "${SANITY_ONLY}" = "true" ] && echo "sanity" || echo "train" )"
            echo "    Variant ${VARIANT} thr=${THR} -> ${CMD_MODE}"
            submit "DG_ETA=${ETA} REWARD_CODING=${RC} TRAINING_SIGNAL=${TS} LOSS_TYPE=${LT} SOFTDG_GATE_THRESHOLD=${THR} sbatch ${WORK_DIR}/run_math.sh ${CMD_MODE}"
            TOTAL_SUBMITTED=$((TOTAL_SUBMITTED + 1))
        done
    done
done

echo
echo "=== Submission complete ==="
echo "  Submitted: ${TOTAL_SUBMITTED}"
echo "  Skipped (no valid thresholds or missing calibration): ${TOTAL_SKIPPED}"
if [ "${DRY_RUN}" = "true" ]; then
    echo "  [DRY-RUN] No jobs actually submitted."
fi
