#!/bin/bash
#
# Gate calibration for SoftDG token-level gating.
#
# Runs calibrate_gate.py on a single GPU (no training).
# Must run BEFORE launching main training experiments.
#
# Usage:
#   sbatch SoftDG-Offline-to-the-bottom/run_calibrate.sh
#
# Key env-var overrides:
#   STUDENT_MODEL=/scratch/shuai14/models/Qwen2.5-0.5B
#   ROLLOUT_PATH=...
#   DG_ETA=1.0
#   MAX_COMPLETIONS=  (empty = all; set to small number for smoke test)
#
#SBATCH --account=aip-szepesva
#SBATCH --job-name=softdg-calibrate
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

SCRATCH="${SCRATCH:-/scratch/shuai14}"
WORK_DIR="/project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/SoftDG-Offline-to-the-bottom"

STUDENT_MODEL="${STUDENT_MODEL:-${SCRATCH}/models/Qwen2.5-0.5B}"
ROLLOUT_PATH="${ROLLOUT_PATH:-${SCRATCH}/rollouts/math_teacher/rollouts_math_Qwen2.5-0.5B-Instruct_0.6_pick4.jsonl}"
DG_ETA="${DG_ETA:-1.0}"
MAX_COMPLETIONS="${MAX_COMPLETIONS:-}"

# ---- Environment ---------------------------------------------------------
module load python/3.11 cuda/12.6 arrow opencv
source /project/aip-szepesva/shuai14/verifiers/.venv/bin/activate
export HF_HOME="/scratch/shuai14"
export HF_DATASETS_CACHE="/scratch/shuai14/datasets/MATH"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "${WORK_DIR}/logs"
mkdir -p "${WORK_DIR}/outputs"

cd "${WORK_DIR}"

if [ ! -f "${ROLLOUT_PATH}" ]; then
    echo "ERROR: rollout file not found: ${ROLLOUT_PATH}" >&2
    exit 1
fi

echo "=== SoftDG Gate Calibration ==="
echo "  Student model:  ${STUDENT_MODEL}"
echo "  Rollout:        ${ROLLOUT_PATH}"
echo "  eta:            ${DG_ETA}"
echo "  Output dir:     ${WORK_DIR}/outputs/"

MAX_COMP_ARGS=()
if [ -n "${MAX_COMPLETIONS}" ]; then
    MAX_COMP_ARGS=(--max_completions "${MAX_COMPLETIONS}")
    echo "  Max completions: ${MAX_COMPLETIONS}"
fi

python calibrate_gate.py \
    --rollout_path "${ROLLOUT_PATH}" \
    --model_path "${STUDENT_MODEL}" \
    --output_dir "${WORK_DIR}/outputs" \
    --max_completion_length 2048 \
    --dg_temperature "${DG_ETA}" \
    "${MAX_COMP_ARGS[@]}"

echo "=== Calibration complete ==="
echo "Results:"
echo "  ${WORK_DIR}/outputs/gate_calibration_v2.jsonl"
echo "  ${WORK_DIR}/outputs/gate_calibration_v2.csv"
echo "  ${WORK_DIR}/outputs/gate_calibration_v2_thresholds.json"
