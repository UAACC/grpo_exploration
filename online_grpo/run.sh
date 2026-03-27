#!/bin/bash
#
# Online GRPO with LoRA
#
# Configure MODEL_DIR and DATASET_TYPE before submitting:
#   DATASET_TYPE=math MODEL_DIR=/scratch/mrli/models/Qwen2.5-0.5B-Instruct \
#     sbatch --job-name=online-grpo-math run.sh train
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

WORK_DIR="/project/aip-szepesva/mrli/backup_dongheng/online_grpo"
SCRATCH="/scratch/mrli"
CONFIG_DIR="${WORK_DIR}/configs"
EVAL_SCRIPT="/project/aip-szepesva/mrli/backup_dongheng/mixture_grpo/evaluate.py"

OUTPUT_DIR="${SCRATCH}/checkpoints/online_grpo_${DATASET_TYPE}"
MERGED_DIR="${SCRATCH}/merged/online_grpo_${DATASET_TYPE}_merged"

# Dataset-specific defaults
if [ "${DATASET_TYPE}" = "math" ]; then
    MAX_COMPLETION=2048
    MAX_MODEL_LEN=3072
    BATCH_SIZE=1
    GRAD_ACCUM=10
    DATASET_CACHE="/scratch/mrli/datasets/MATH"
else
    MAX_COMPLETION=1024
    MAX_MODEL_LEN=2048
    BATCH_SIZE=5
    GRAD_ACCUM=2
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
    echo "=== Online GRPO on ${DATASET_TYPE} (4x L40s) ==="
    echo "=== Model: ${MODEL_DIR} ==="
    echo "=== Output: ${OUTPUT_DIR} ==="
    export WANDB_PROJECT="online-grpo-${DATASET_TYPE}"
    export WANDB_RUN_NAME="$(basename ${MODEL_DIR})-${DATASET_TYPE}-$(date +%Y%m%d_%H%M%S)"

    mkdir -p "${OUTPUT_DIR}"

    accelerate launch \
        --config_file "${CONFIG_DIR}/accelerate_ddp_4gpu.yaml" \
        train.py \
        --model "${MODEL_DIR}" \
        --dataset_type "${DATASET_TYPE}" \
        --output_dir "${OUTPUT_DIR}" \
        --num_generations 5 \
        --num_train_epochs 15 \
        --per_device_train_batch_size "${BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRAD_ACCUM}" \
        --learning_rate 3e-6 \
        --beta 0.001 \
        --temperature 0.7 \
        --max_completion_length "${MAX_COMPLETION}" \
        --lora_r 32 \
        --lora_alpha 32 \
        --save_steps 200 \
        --logging_steps 10 \
        --report_to wandb

    echo "=== Training complete ==="
    ;;

eval)
    CKPT="${2:-}"
    if [ -n "${CKPT}" ]; then
        EVAL_PATH="${OUTPUT_DIR}/${CKPT}"
        MERGE_FLAG="--merge_lora --base_model ${MODEL_DIR} --merged_output ${MERGED_DIR}"
    else
        EVAL_PATH="${OUTPUT_DIR}"
        MERGE_FLAG="--merge_lora --base_model ${MODEL_DIR} --merged_output ${MERGED_DIR}"
    fi
    echo "=== Evaluating on ${DATASET_TYPE}: ${EVAL_PATH} ==="
    python "${EVAL_SCRIPT}" \
        --model_path "${EVAL_PATH}" \
        ${MERGE_FLAG} \
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
    echo "Usage: DATASET_TYPE=gsm8k|math MODEL_DIR=/path/to/model sbatch --job-name=<name> run.sh <command>"
    echo ""
    echo "  train          - Online GRPO with LoRA (4x L40s)"
    echo "  eval [ckpt]    - Evaluate trained model (merge LoRA)"
    echo "  eval-baseline  - Evaluate base model (no training)"
    exit 1
    ;;

esac
