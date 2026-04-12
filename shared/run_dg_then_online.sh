#!/bin/bash
#
# Two-stage training: DG-offline (already done) → Online GRPO
# Takes a merged DG-offline checkpoint and runs online GRPO on top.
#
# Usage:
#   DATASET=math sbatch --job-name=dg-then-online-math run_dg_then_online.sh
#   DATASET=gsm8k sbatch --job-name=dg-then-online-gsm8k run_dg_then_online.sh
#
#SBATCH --account=aip-szepesva
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=shared/logs/%x-%j.out
#SBATCH --error=shared/logs/%x-%j.err

set -euo pipefail

DATASET="${DATASET:?Set DATASET (math or gsm8k)}"
SCRATCH="/scratch/mrli"
WORK_DIR="/project/aip-szepesva/mrli/backup_dongheng"

module load python/3.11 cuda/12.6 arrow opencv
source "${WORK_DIR}/.venv/bin/activate"
export HF_HOME="${SCRATCH}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

if [ -f "${WORK_DIR}/offline_grpo/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "${WORK_DIR}/offline_grpo/.wandb_key")
fi
export WANDB_PROJECT="online-grpo-${DATASET}"

mkdir -p "${WORK_DIR}/shared/logs"

case "${DATASET}" in
    math)
        export HF_DATASETS_CACHE="${SCRATCH}/datasets/MATH"
        START_MODEL="${SCRATCH}/merged/dg_offline_math"
        OUTPUT_DIR="${SCRATCH}/checkpoints/dg_then_online_math"
        BATCH=1
        GRAD_ACCUM=10
        MAX_COMP=2048
        EPOCHS=15
        ;;
    gsm8k)
        export HF_DATASETS_CACHE="${SCRATCH}/datasets/GSM8K"
        START_MODEL="${SCRATCH}/merged/dg_offline_gsm8k_eta0.5"
        OUTPUT_DIR="${SCRATCH}/checkpoints/dg_then_online_gsm8k"
        BATCH=5
        GRAD_ACCUM=2
        MAX_COMP=1024
        EPOCHS=15
        ;;
    *)
        echo "Unknown dataset: ${DATASET}. Use math or gsm8k."
        exit 1
        ;;
esac

echo "=== DG-offline → Online GRPO on ${DATASET} ==="
echo "  Starting model: ${START_MODEL}"
echo "  Output: ${OUTPUT_DIR}"

cd "${WORK_DIR}/online_grpo"

accelerate launch --config_file configs/accelerate_ddp_4gpu.yaml \
    train.py \
    --model "${START_MODEL}" \
    --dataset_type "${DATASET}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "dg-then-online-${DATASET}" \
    --learning_rate 3e-6 \
    --beta 0.001 \
    --num_generations 5 \
    --num_train_epochs "${EPOCHS}" \
    --per_device_train_batch_size "${BATCH}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --temperature 0.7 \
    --max_completion_length "${MAX_COMP}" \
    --lora_r 32 \
    --lora_alpha 32 \
    --save_steps 200 \
    --logging_steps 10

echo "=== Training complete ==="
