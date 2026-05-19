#!/bin/bash
#
# Generate teacher rollouts for any dataset.
#
# Usage:
#   DATASET=svamp sbatch --job-name=rollouts-svamp run_generate_rollouts.sh
#   DATASET=asdiv sbatch --job-name=rollouts-asdiv run_generate_rollouts.sh
#   DATASET=acereason sbatch --job-name=rollouts-acereason run_generate_rollouts.sh
#
#SBATCH --account=aip-szepesva
#SBATCH --time=03:00:00
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

DATASET="${DATASET:?Set DATASET env var (svamp, asdiv, etc.)}"
SCRATCH="${SCRATCH:-/home/shuai14/scratch/scratch_dongheng}"
WORK_DIR="/project/aip-szepesva/shuai14/DG_LLM/grpo_exploration"
TEACHER="${TEACHER:-${SCRATCH}/models/Qwen2.5-Math-7B-Instruct}"
NUM_GEN="${NUM_GEN:-5}"
TEMP="${TEMP:-0.7}"

OUTPUT_DIR="${SCRATCH}/rollouts/${DATASET}_teacher"
OUTPUT_PATH="${OUTPUT_DIR}/rollouts_${DATASET}.jsonl"

module load python/3.11 cuda/12.6 arrow opencv
source "/home/shuai14/projects/aip-szepesva/shuai14/verifiers/.venv/bin/activate"
export HF_HOME="${SCRATCH}"
export HF_DATASETS_CACHE="${SCRATCH}/datasets/${DATASET^^}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p "${OUTPUT_DIR}" "${WORK_DIR}/shared/logs"

cd "${WORK_DIR}"

# ASDiv needs manual_split=train since it has no proper train split
EXTRA_ARGS=""
if [ "${DATASET}" = "asdiv" ] || [ "${DATASET}" = "acereason" ]; then
    EXTRA_ARGS="--manual_split train"
fi

echo "=== Generating ${DATASET} teacher rollouts ==="
echo "  Teacher: ${TEACHER}"
echo "  Output: ${OUTPUT_PATH}"
echo "  Num generations: ${NUM_GEN}"

CUDA_VISIBLE_DEVICES=0 python shared/generate_rollouts.py \
    --dataset "${DATASET}" \
    --teacher_model "${TEACHER}" \
    --output_path "${OUTPUT_PATH}" \
    --num_generations "${NUM_GEN}" \
    --temperature "${TEMP}" \
    ${EXTRA_ARGS}
