#!/bin/bash
#
# Recompute teacher logprobs aligned with student tokenization, so OG can
# train on R1-Distill rollouts without the silent-corruption case at
# `<think>` (151648) / `</think>` (151649) token IDs.
#
# Output: /scratch/mrli/rollouts/math_deepseek_r1_cleaned/rollouts_full.jsonl
#
#SBATCH --account=aip-szepesva
#SBATCH --job-name=r1-og-prep
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --output=DG-offline/logs/r1ogprep-%j.out
#SBATCH --error=DG-offline/logs/r1ogprep-%j.err

set -euo pipefail

SCRATCH="${SCRATCH:-/scratch/mrli}"
TEACHER_MODEL="${TEACHER_MODEL:-${SCRATCH}/models/DeepSeek-R1-Distill-Qwen-7B}"
STUDENT_MODEL="${STUDENT_MODEL:-${SCRATCH}/models/Qwen2.5-0.5B-Instruct}"
INPUT_PATH="${INPUT_PATH:-${SCRATCH}/rollouts/math_deepseek_r1/rollouts_full.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH}/rollouts/math_deepseek_r1_cleaned}"
OUTPUT_PATH="${OUTPUT_DIR}/rollouts_full.jsonl"

module load python/3.11 cuda/12.6 arrow opencv
source /project/aip-szepesva/mrli/backup_dongheng/.venv/bin/activate

export HF_HOME="${SCRATCH}"
export HF_DATASETS_CACHE="${SCRATCH}/datasets/MATH"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p "${OUTPUT_DIR}"
cd /project/aip-szepesva/mrli/backup_dongheng

echo "=== Cleaned OG rollout prep ==="
echo "  input:    ${INPUT_PATH}"
echo "  output:   ${OUTPUT_PATH}"
echo "  teacher:  ${TEACHER_MODEL}"
echo "  student:  ${STUDENT_MODEL}"
echo

python shared/prepare_cleaned_og_rollouts.py \
    --input_path "${INPUT_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --teacher_model "${TEACHER_MODEL}" \
    --student_model "${STUDENT_MODEL}" \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.90 \
    --max_model_len 33280

echo "=== Prep complete ==="
ls -la "${OUTPUT_PATH}"
