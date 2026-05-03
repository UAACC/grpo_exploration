#!/bin/bash
#
# Recompute teacher logprobs aligned with student tokenization, so OG can
# train on R1-Distill rollouts without the silent-corruption case at
# `<think>` (151648) / `</think>` (151649) token IDs.
#
# Uses a transformers-direct forward pass (gather + chunked-fp32 logsumexp)
# instead of vLLM `prompt_logprobs=1`, because vLLM materialized the full
# [seq_len, vocab_size] log_softmax in fp32 and OOM'd at long context (5
# consecutive failures with vLLM, the last in `compute_logprobs` ->
# `log_softmax(..., dtype=torch.float32)` at ~38K context).
#
# Output: /scratch/mrli/rollouts/math_deepseek_r1_cleaned/rollouts_full.jsonl
#
#SBATCH --account=aip-szepesva
#SBATCH --job-name=r1-og-prep
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
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

mkdir -p "${OUTPUT_DIR}"
cd /project/aip-szepesva/mrli/backup_dongheng

echo "=== Cleaned OG rollout prep (transformers-direct) ==="
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
    --device "cuda:0" \
    --max_seq_len 38000 \
    --logsumexp_chunk 1024

echo "=== Prep complete ==="
ls -la "${OUTPUT_PATH}"
