#!/bin/bash
#
# Token probability calibration for SoftDG-Token-Mask-Offline.
#
# Runs calibrate_token_probabilities.py on a single GPU.
# Recomputes every completion-token probability under the student model.
# MUST run BEFORE any downstream analysis that depends on student log-probs.
#
# Usage:
#   sbatch SoftDG-Token-Mask-Offline/run_probability_calibration.sh
#
# Smoke test (4 completions, batch size 1):
#   MAX_COMPLETIONS=4 BATCH_SIZE=1 bash SoftDG-Token-Mask-Offline/run_probability_calibration.sh
#
# Key env-var overrides:
#   ROLLOUT_PATH=...
#   STUDENT_MODEL=...
#   TEACHER_NAME=...
#   STUDENT_NAME=...
#   OUTPUT_DIR=...
#   MAX_COMPLETIONS=   (empty = all; set to small number for smoke test)
#   BATCH_SIZE=8
#   MAX_TOKENS_PER_BATCH=12000
#   MAX_COMPLETION_LENGTH=2048
#
#SBATCH --account=aip-szepesva
#SBATCH --job-name=softdg-tm-prob-calibrate
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

SCRATCH="${SCRATCH:-/scratch/shuai14}"
WORK_DIR="/project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/SoftDG-Token-Mask-Offline"

ROLLOUT_PATH="${ROLLOUT_PATH:-${SCRATCH}/rollouts/math_teacher/rollouts_math_Qwen2.5-0.5B-Instruct_0.6_pick4.jsonl}"
STUDENT_MODEL="${STUDENT_MODEL:-${SCRATCH}/models/Qwen2.5-0.5B}"
TEACHER_NAME="${TEACHER_NAME:-Qwen2.5-0.5B-Instruct}"
STUDENT_NAME="${STUDENT_NAME:-Qwen2.5-0.5B}"
PAIR_TAG="teacher-${TEACHER_NAME}__student-${STUDENT_NAME}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_DIR}/outputs/token_probability_calibration/${PAIR_TAG}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_TOKENS_PER_BATCH="${MAX_TOKENS_PER_BATCH:-12000}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-2048}"
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
mkdir -p "${OUTPUT_DIR}"

cd "${WORK_DIR}"

if [ ! -f "${ROLLOUT_PATH}" ]; then
    echo "ERROR: rollout file not found: ${ROLLOUT_PATH}" >&2
    exit 1
fi

echo "=== SoftDG-TM Token Probability Calibration ==="
echo "  Teacher name:     ${TEACHER_NAME}"
echo "  Student model:    ${STUDENT_MODEL}"
echo "  Student name:     ${STUDENT_NAME}"
echo "  Rollout:          ${ROLLOUT_PATH}"
echo "  Output dir:       ${OUTPUT_DIR}/"
echo "  Batch size:       ${BATCH_SIZE}"
echo "  Max tokens/batch: ${MAX_TOKENS_PER_BATCH}"
echo "  Max comp length:  ${MAX_COMPLETION_LENGTH}"

MAX_COMP_ARGS=()
if [ -n "${MAX_COMPLETIONS}" ]; then
    MAX_COMP_ARGS=(--max_completions "${MAX_COMPLETIONS}")
    echo "  Max completions: ${MAX_COMPLETIONS}"
fi

python calibrate_token_probabilities.py \
    --rollout_path          "${ROLLOUT_PATH}" \
    --model_path            "${STUDENT_MODEL}" \
    --teacher_name          "${TEACHER_NAME}" \
    --student_name          "${STUDENT_NAME}" \
    --output_dir            "${OUTPUT_DIR}" \
    --max_completion_length "${MAX_COMPLETION_LENGTH}" \
    --batch_size            "${BATCH_SIZE}" \
    --max_tokens_per_batch  "${MAX_TOKENS_PER_BATCH}" \
    "${MAX_COMP_ARGS[@]}"

echo "=== Calibration complete ==="
echo "Results:"
echo "  ${OUTPUT_DIR}/${PAIR_TAG}_token_probabilities.jsonl.gz"
echo "  ${OUTPUT_DIR}/${PAIR_TAG}_run_summaries.jsonl"
echo "  ${OUTPUT_DIR}/${PAIR_TAG}_percentiles.json"
echo "  ${OUTPUT_DIR}/${PAIR_TAG}_percentiles.csv"
