#!/bin/bash
#
# Generate DeepSeek-R1-Distill-Qwen-7B rollouts on MATH train.
# Uses DeepSeek's recommended sampling: temp=0.6, top_p=0.95, no system prompt,
# max_tokens=32768 (matches their eval methodology / model card).
#
# Outputs: /scratch/mrli/rollouts/math_deepseek_r1/rollouts_full.jsonl
#
#SBATCH --account=aip-szepesva
#SBATCH --job-name=r1-rollouts
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --output=DG-offline/logs/r1rollouts-%j.out
#SBATCH --error=DG-offline/logs/r1rollouts-%j.err

set -euo pipefail

SCRATCH="${SCRATCH:-/scratch/mrli}"
TEACHER_MODEL="${TEACHER_MODEL:-${SCRATCH}/models/DeepSeek-R1-Distill-Qwen-7B}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH}/rollouts/math_deepseek_r1}"
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

echo "=== DeepSeek-R1 rollout generation on MATH train ==="
echo "  Teacher:        ${TEACHER_MODEL}"
echo "  Output:         ${OUTPUT_PATH}"
echo "  Sampling:       temp=0.6, top_p=0.95, no system prompt"
echo "  Lengths:        max_tokens=32768 (DeepSeek recommended)"
echo "  num_generations: 4 per problem"
echo

python shared/generate_rollouts.py \
    --dataset math \
    --teacher_model "${TEACHER_MODEL}" \
    --output_path "${OUTPUT_PATH}" \
    --num_generations 4 \
    --temperature 0.6 \
    --top_p 0.95 \
    --max_tokens 32768 \
    --max_model_len 33280 \
    --no_system_prompt \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.90 \
    --seed 42

echo "=== Rollout generation complete ==="
ls -la "${OUTPUT_PATH}"
wc -l "${OUTPUT_PATH}"
