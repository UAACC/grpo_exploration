#!/bin/bash
#
# Evaluate online GRPO 7B teacher checkpoint on MATH
#
# Usage:
#   sbatch --job-name=eval-online-teacher --gpus-per-node=l40s:1 --cpus-per-task=16 --mem=64G --time=1:00:00 run_eval_teacher_math.sh
#
#SBATCH --account=aip-szepesva
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e

SCRATCH="/scratch/mrli"
WORK_DIR="/project/aip-szepesva/mrli/backup_dongheng/online_grpo"
EVAL_SCRIPT="/project/aip-szepesva/mrli/backup_dongheng/mixture_grpo/evaluate.py"

MODEL_DIR="${SCRATCH}/models/Qwen2.5-Math-7B-Instruct"
CKPT_DIR="${SCRATCH}/checkpoints/online_grpo_math_teacher_7B"
MERGED_DIR="${SCRATCH}/merged/online_grpo_math_teacher_7B_merged"

module load python/3.11 cuda/12.6 arrow opencv
source /project/aip-szepesva/mrli/backup_dongheng/.venv/bin/activate
export HF_HOME="${SCRATCH}"
export HF_DATASETS_CACHE="/scratch/mrli/datasets/MATH"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "=== Evaluating online GRPO 7B teacher on MATH ==="
echo "  Checkpoint: ${CKPT_DIR}"
echo "  Base model: ${MODEL_DIR}"

python "${EVAL_SCRIPT}" \
    --model_path "${CKPT_DIR}" \
    --base_model "${MODEL_DIR}" \
    --merge_lora \
    --merged_output "${MERGED_DIR}" \
    --dataset_type math \
    --temperature 0.0 \
    --runs 5 \
    --max_tokens 2048 \
    --max_model_len 3072

echo "=== Evaluation complete ==="
