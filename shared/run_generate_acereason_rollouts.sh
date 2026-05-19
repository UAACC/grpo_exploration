#!/bin/bash
#
# Generate teacher rollouts for any dataset.
#
# Usage:
#   DATASET=svamp sbatch --job-name=rollouts-svamp run_generate_rollouts.sh
#   DATASET=asdiv sbatch --job-name=rollouts-asdiv run_generate_rollouts.sh
#   DATASET=acereason sbatch --job-name=rollouts-acereason run_generate_acereason_rollouts.sh
#   --account=aip-xt7
#   --account=aip-szepesva
#
#SBATCH --account=aip-xt7
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --mail-user=shuai14@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

DATASET="acereason"
SCRATCH="${SCRATCH:-/home/shuai14/scratch/scratch_dongheng}"
WORK_DIR="/home/shuai14/projects/aip-szepesva/shuai14/DG_LLM/grpo_exploration"
TEACHER="/home/shuai14/scratch/scratch_dongheng/models/Qwen2.5-Math-7B-Instruct"
NUM_GEN="${NUM_GEN:-5}"
TEMP="${TEMP:-0.6}"

OUTPUT_DIR="${SCRATCH}/rollouts/${DATASET}_teacher"
OUTPUT_PATH="${OUTPUT_DIR}/rollouts_${DATASET}_full_${TEMP}.jsonl"

module load python/3.11 cuda/12.6 arrow opencv
source "/home/shuai14/projects/aip-szepesva/shuai14/verifiers/.venv/bin/activate"
export HF_HOME="${SCRATCH}"
# export HF_DATASETS_CACHE="${SCRATCH}/datasets/${DATASET^^}"
# export TRANSFORMERS_OFFLINE=1
# export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# mkdir -p "${OUTPUT_DIR}" "${WORK_DIR}/shared/logs"

cd "/home/shuai14/projects/aip-szepesva/shuai14/DG_LLM/grpo_exploration"

# ASDiv needs manual_split=train since it has no proper train split
EXTRA_ARGS="--manual_split train"

echo "=== Generating acereason teacher rollouts ==="
echo "  Teacher: ${TEACHER}"
echo "  Output: ${OUTPUT_PATH}"
echo "  Num generations: ${NUM_GEN}"

OUTPUT_PATH_WITH_TEMP="${OUTPUT_PATH}_${TEMP}"

# CUDA_VISIBLE_DEVICES=0,1,2,3 python shared/generate_rollouts_by_batch.py \
#     --dataset "${DATASET}" \
#     --teacher_model "${TEACHER}" \
#     --output_path "${OUTPUT_PATH}" \
#     --num_generations "${NUM_GEN}" \
#     --temperature "${TEMP}" \
#     --tensor_parallel_size 4 \
#     --fsync_every_batch \
#     ${EXTRA_ARGS}

CUDA_VISIBLE_DEVICES=0,1,2,3 python shared/generate_rollouts.py \
    --dataset "${DATASET}" \
    --teacher_model "${TEACHER}" \
    --output_path "${OUTPUT_PATH}" \
    --num_generations "${NUM_GEN}" \
    --temperature "${TEMP}" \
    --tensor_parallel_size 4 \
    ${EXTRA_ARGS}

