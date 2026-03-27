#!/bin/bash
#
# Diagnose Method B offline loss on MATH and GSM8K
#
# Usage:
#   sbatch --job-name=diag-math run_diagnose.sh math
#   sbatch --job-name=diag-gsm8k run_diagnose.sh gsm8k
#
#SBATCH --account=aip-szepesva
#SBATCH --time=1:00:00
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/diagnose/%x-%j.out
#SBATCH --error=logs/diagnose/%x-%j.err

set -e

WORK_DIR="/project/aip-szepesva/mrli/backup_dongheng/mixture_grpo"
SCRATCH="/scratch/mrli"

module load python/3.11 cuda/12.6 arrow opencv
source /project/aip-szepesva/mrli/backup_dongheng/.venv/bin/activate
export HF_HOME="${SCRATCH}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
cd "${WORK_DIR}"

mkdir -p logs/diagnose

TASK="${1:-math}"

if [ "$TASK" = "math" ]; then
    echo "=== Diagnosing Method B on MATH ==="
    # Use baseline model (not the trained one) to see initial state
    python diagnose_method_B.py \
        --model_path "${SCRATCH}/models/Qwen2.5-0.5B-Instruct" \
        --teacher_rollout_path "${SCRATCH}/rollouts/math_teacher/rollouts_full.jsonl" \
        --dataset_type math \
        --num_samples 500 \
        --num_generations 4 \
        --temperature 0.7 \
        --max_tokens 2048 \
        --max_model_len 3072

    echo ""
    echo "=== Now with the trained model ==="
    python diagnose_method_B.py \
        --model_path "${SCRATCH}/merged/mixture_B_math_merged" \
        --teacher_rollout_path "${SCRATCH}/rollouts/math_teacher/rollouts_full.jsonl" \
        --dataset_type math \
        --num_samples 500 \
        --num_generations 4 \
        --temperature 0.7 \
        --max_tokens 2048 \
        --max_model_len 3072

elif [ "$TASK" = "gsm8k" ]; then
    echo "=== Diagnosing Method B on GSM8K ==="
    export HF_DATASETS_CACHE="${SCRATCH}/datasets/GSM8K"
    python diagnose_method_B.py \
        --model_path "${SCRATCH}/models/Qwen2.5-0.5B-Instruct" \
        --teacher_rollout_path "${SCRATCH}/rollouts/gsm8k_teacher/rollouts_gsm8k.jsonl" \
        --dataset_type gsm8k \
        --num_samples 500 \
        --num_generations 4 \
        --temperature 0.7 \
        --max_tokens 1024 \
        --max_model_len 2048

    echo ""
    echo "=== Now with the trained model ==="
    python diagnose_method_B.py \
        --model_path "${SCRATCH}/merged/mixture_B_weighted_merged" \
        --teacher_rollout_path "${SCRATCH}/rollouts/gsm8k_teacher/rollouts_gsm8k.jsonl" \
        --dataset_type gsm8k \
        --num_samples 500 \
        --num_generations 4 \
        --temperature 0.7 \
        --max_tokens 1024 \
        --max_model_len 2048
fi

echo "=== DIAGNOSIS COMPLETE ==="
