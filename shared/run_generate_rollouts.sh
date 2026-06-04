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
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

DATASET="${DATASET:?Set DATASET env var (svamp, asdiv, etc.)}"
SCRATCH="${SCRATCH:-/scratch/shuai14}"
WORK_DIR="/project/aip-szepesva/shuai14/DG_LLM/grpo_exploration"
TEACHER_NAME="${TEACHER_NAME:-Qwen2.5-Math-1.5B-Instruct}"
TEACHER="${TEACHER:-${SCRATCH}/models/${TEACHER_NAME}}"
NUM_GEN="${NUM_GEN:-5}"
TEMP="${TEMP:-0.7}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
TP_SIZE="${TP_SIZE:-4}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"

OUTPUT_DIR="${SCRATCH}/rollouts/${DATASET}_teacher"
OUTPUT_PATH="${OUTPUT_DIR}/rollouts_${DATASET}_${TEACHER_NAME}_temp${TEMP}_topp${TOP_P}_topk${TOP_K}.jsonl"

module load python/3.11 cuda/12.6 arrow opencv
source /project/aip-szepesva/shuai14/verifiers/.venv/bin/activate
export HF_HOME="${SCRATCH}"
export HF_DATASETS_CACHE="${SCRATCH}/datasets/${DATASET^^}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p "${OUTPUT_DIR}" "${WORK_DIR}/logs"

cd "${WORK_DIR}"

# ASDiv needs manual_split=train since it has no proper train split
EXTRA_ARGS=""
if [ "${DATASET}" = "asdiv" ] || [ "${DATASET}" = "acereason" ]; then
    EXTRA_ARGS="--manual_split train"
fi

echo "=== Generating ${DATASET} teacher rollouts ==="
echo "  Teacher name: ${TEACHER_NAME}"
echo "  Teacher: ${TEACHER}"
echo "  Output: ${OUTPUT_PATH}"
echo "  Num generations: ${NUM_GEN}"
echo "  Sampling: temp=${TEMP}, top_p=${TOP_P}, top_k=${TOP_K}"
echo "  Tensor parallel size: ${TP_SIZE}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" python shared/generate_rollouts.py \
    --dataset "${DATASET}" \
    --teacher_model "${TEACHER}" \
    --output_path "${OUTPUT_PATH}" \
    --num_generations "${NUM_GEN}" \
    --temperature "${TEMP}" \
    --top_p "${TOP_P}" \
    --top_k "${TOP_K}" \
    --tensor_parallel_size "${TP_SIZE}" \
    ${EXTRA_ARGS}

