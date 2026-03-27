#!/bin/bash
#
# Method B: Weighted Mixture GRPO
#   Online student rollouts + offline teacher rollouts with separate weighted losses.
#
# Configure DATASET_TYPE, MODEL_DIR, TEACHER_DIR before submitting:
#   DATASET_TYPE=math sbatch --job-name=mixture-B-math run.sh train
#   DATASET_TYPE=gsm8k sbatch --job-name=mixture-B-lam01 run.sh train 0.1
#
#SBATCH --account=aip-szepesva
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e

# ── Configurable ─────────────────────────────────────────────────
DATASET_TYPE="${DATASET_TYPE:-gsm8k}"          # gsm8k | math
MODEL_DIR="${MODEL_DIR:-/scratch/mrli/models/Qwen2.5-0.5B-Instruct}"
TEACHER_DIR="${TEACHER_DIR:-/scratch/mrli/models/Qwen2.5-Math-7B-Instruct}"

WORK_DIR="/project/aip-szepesva/mrli/backup_dongheng/mixture_grpo"
SCRATCH="/scratch/mrli"
CONFIG_DIR="/project/aip-szepesva/mrli/backup_dongheng/offline_grpo/configs"
EVAL_SCRIPT="${WORK_DIR}/evaluate.py"

ROLLOUT_PATH="${SCRATCH}/rollouts/${DATASET_TYPE}_teacher/rollouts_${DATASET_TYPE}.jsonl"
OUTPUT_DIR="${SCRATCH}/checkpoints/mixture_B_${DATASET_TYPE}"
MERGED_DIR="${SCRATCH}/merged/mixture_B_${DATASET_TYPE}_merged"

# Dataset-specific defaults
if [ "${DATASET_TYPE}" = "math" ]; then
    MAX_COMPLETION=2048
    MAX_MODEL_LEN=3072
    DATASET_CACHE="/scratch/mrli/datasets/MATH"
else
    MAX_COMPLETION=1024
    MAX_MODEL_LEN=2048
    DATASET_CACHE="/scratch/mrli/datasets/GSM8K"
fi

# ── Activate environment ─────────────────────────────────────────
module load python/3.11 cuda/12.6 arrow opencv
source /project/aip-szepesva/mrli/backup_dongheng/.venv/bin/activate
export HF_HOME="${SCRATCH}"
export HF_DATASETS_CACHE="${DATASET_CACHE}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=$(cat /project/aip-szepesva/mrli/backup_dongheng/offline_grpo/.wandb_key)
export VLLM_WORKER_MULTIPROC_METHOD=spawn
cd "${WORK_DIR}"

mkdir -p logs

case "${1}" in

train)
    LAMBDA="${2:-0.3}"
    echo "=== Method B: Weighted Mixture GRPO on ${DATASET_TYPE} (lambda=${LAMBDA}) ==="
    echo "=== Student: ${MODEL_DIR} ==="
    echo "=== Rollouts: ${ROLLOUT_PATH} ==="
    echo "=== Output: ${OUTPUT_DIR} ==="
    export WANDB_PROJECT="mixture-grpo-B-${DATASET_TYPE}"

    accelerate launch \
        --config_file "${CONFIG_DIR}/accelerate_ddp_4gpu.yaml" \
        method_B_weighted/train.py \
        --teacher_rollout_path "${ROLLOUT_PATH}" \
        --target_model "${MODEL_DIR}" \
        --dataset_type "${DATASET_TYPE}" \
        --output_dir "${OUTPUT_DIR}" \
        --offline_weight "${LAMBDA}" \
        --num_generations 4 \
        --num_teacher_per_prompt 4 \
        --num_train_epochs 5 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 3 \
        --learning_rate 3e-6 \
        --beta 0.01 \
        --max_prompt_length 512 \
        --max_completion_length "${MAX_COMPLETION}" \
        --max_grad_norm 1.0 \
        --save_steps 200 \
        --logging_steps 10 \
        --ref_sync_steps 0 \
        --report_to wandb \
        --resume_from_checkpoint latest

    echo "=== Training complete ==="
    ;;

eval)
    CKPT="${2:-}"
    if [ -n "${CKPT}" ]; then
        EVAL_PATH="${OUTPUT_DIR}/${CKPT}"
    else
        EVAL_PATH="${OUTPUT_DIR}"
    fi
    echo "=== Evaluating Method B on ${DATASET_TYPE}: ${EVAL_PATH} ==="
    python "${EVAL_SCRIPT}" \
        --model_path "${EVAL_PATH}" \
        --base_model "${MODEL_DIR}" \
        --merge_lora \
        --merged_output "${MERGED_DIR}" \
        --dataset_type "${DATASET_TYPE}" \
        --temperature 0.0 \
        --runs 10 \
        --max_tokens "${MAX_COMPLETION}" \
        --max_model_len "${MAX_MODEL_LEN}"
    echo "=== Evaluation complete ==="
    ;;

eval-baseline)
    echo "=== Evaluating baseline on ${DATASET_TYPE}: ${MODEL_DIR} ==="
    python "${EVAL_SCRIPT}" \
        --model_path "${MODEL_DIR}" \
        --dataset_type "${DATASET_TYPE}" \
        --temperature 0.0 \
        --runs 10 \
        --max_tokens "${MAX_COMPLETION}" \
        --max_model_len "${MAX_MODEL_LEN}"
    echo "=== Baseline evaluation complete ==="
    ;;

*)
    echo "Usage: DATASET_TYPE=gsm8k|math sbatch --job-name=<name> run.sh <command> [lambda]"
    echo ""
    echo "  train [lambda] - Weighted Mixture GRPO (default lambda=0.3)"
    echo "  eval [ckpt]    - Evaluate trained model (merge LoRA)"
    echo "  eval-baseline  - Evaluate base model (no training)"
    exit 1
    ;;

esac
