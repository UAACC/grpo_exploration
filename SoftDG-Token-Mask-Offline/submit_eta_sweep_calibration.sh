#!/bin/bash
#
# Submit gate calibration jobs for the eta sweep: 0.75, 0.5, 0.25, 0.1
#
# Usage:
#   bash SoftDG-Token-Mask-Offline/submit_eta_sweep_calibration.sh
#
# This submits 4 Slurm jobs (one per eta) and prints job IDs.
# Run submit_eta_sweep_training.sh AFTER all calibration jobs complete.
#
# Does NOT rerun eta=1.0 (outputs already in outputs/).

set -euo pipefail

WORK_DIR="/project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/SoftDG-Token-Mask-Offline"
ETA_VALUES=(0.75 0.5 0.25 0.1)

echo "=== Submitting eta sweep calibration jobs ==="
echo "  eta values: ${ETA_VALUES[*]}"
echo

JOB_IDS=()
for ETA in "${ETA_VALUES[@]}"; do
    JOB_ID=$(DG_ETA="${ETA}" sbatch --parsable "${WORK_DIR}/run_calibrate.sh")
    JOB_IDS+=("${JOB_ID}")
    echo "  eta=${ETA}  -> job ${JOB_ID}"
done

echo
echo "All calibration jobs submitted: ${JOB_IDS[*]}"
echo
echo "Monitor with:"
echo "  squeue -u \$USER"
echo
echo "After all jobs complete, run:"
echo "  python SoftDG-Token-Mask-Offline/parse_calibration_for_training.py"
echo "  bash SoftDG-Token-Mask-Offline/submit_eta_sweep_training.sh"
