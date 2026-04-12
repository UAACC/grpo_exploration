#!/bin/bash
#
# Unified training script for all offline methods on any dataset.
#
# Usage:
#   METHOD=bc DATASET=svamp sbatch --job-name=bc-svamp-all run_train_offline.sh
#   METHOD=bc_correct DATASET=svamp sbatch --job-name=bc-svamp-correct run_train_offline.sh
#   METHOD=offline_grpo DATASET=asdiv sbatch --job-name=offline-grpo-asdiv run_train_offline.sh
#   METHOD=dg_offline DATASET=svamp sbatch --job-name=dg-svamp run_train_offline.sh
#
# Env vars:
#   METHOD      Required. One of: bc, bc_correct, offline_grpo, dg_offline
#   DATASET     Required. One of: svamp, asdiv, gsm8k, math
#   EPOCHS      Optional. Default: 5
#   DG_ETA      Optional. Default: 0.5 (only for dg_offline)
#
#SBATCH --account=aip-szepesva
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=shared/logs/%x-%j.out
#SBATCH --error=shared/logs/%x-%j.err

set -euo pipefail

METHOD="${METHOD:?Set METHOD (bc, bc_correct, offline_grpo, dg_offline)}"
DATASET="${DATASET:?Set DATASET (svamp, asdiv, gsm8k, math)}"
EPOCHS="${EPOCHS:-5}"
DG_ETA="${DG_ETA:-0.5}"

SCRATCH="/scratch/mrli"
WORK_DIR="/project/aip-szepesva/mrli/backup_dongheng"
STUDENT="${SCRATCH}/models/Qwen2.5-0.5B-Instruct"
ROLLOUT_PATH="${SCRATCH}/rollouts/${DATASET}_teacher/rollouts_${DATASET}.jsonl"

module load python/3.11 cuda/12.6 arrow opencv
source "${WORK_DIR}/.venv/bin/activate"
export HF_HOME="${SCRATCH}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

if [ -f "${WORK_DIR}/offline_grpo/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "${WORK_DIR}/offline_grpo/.wandb_key")
fi

mkdir -p "${WORK_DIR}/shared/logs"

# Dataset-specific settings
case "${DATASET}" in
    math)
        MAX_LEN=2304
        MAX_COMP=2048
        ROLLOUT_PATH="${SCRATCH}/rollouts/math_teacher/rollouts_full.jsonl"
        ;;
    gsm8k)
        MAX_LEN=1280
        MAX_COMP=1024
        ROLLOUT_PATH="${SCRATCH}/rollouts/gsm8k_teacher/rollouts_gsm8k.jsonl"
        ;;
    svamp|asdiv)
        MAX_LEN=1280
        MAX_COMP=1024
        ;;
    *)
        echo "Unknown dataset: ${DATASET}"
        exit 1
        ;;
esac

echo "=== Training: ${METHOD} on ${DATASET} ==="
echo "  Rollouts: ${ROLLOUT_PATH}"
echo "  Epochs: ${EPOCHS}"

case "${METHOD}" in
    bc)
        export WANDB_PROJECT="bc-${DATASET}"
        cd "${WORK_DIR}/bc"
        accelerate launch --config_file configs/accelerate_ddp_4gpu.yaml \
            train_bc.py \
            --rollout_path "${ROLLOUT_PATH}" \
            --target_model "${STUDENT}" \
            --output_dir "${SCRATCH}/checkpoints/bc_${DATASET}" \
            --run_name "bc-${DATASET}-all" \
            --max_length "${MAX_LEN}" \
            --num_train_epochs "${EPOCHS}" \
            --per_device_train_batch_size 4 \
            --gradient_accumulation_steps 2 \
            --learning_rate 3e-6 \
            --lora_r 32 --lora_alpha 32 \
            --save_steps 500 --logging_steps 10
        ;;

    bc_correct)
        export WANDB_PROJECT="bc-${DATASET}"
        cd "${WORK_DIR}/bc"
        accelerate launch --config_file configs/accelerate_ddp_4gpu.yaml \
            train_bc.py \
            --rollout_path "${ROLLOUT_PATH}" \
            --target_model "${STUDENT}" \
            --output_dir "${SCRATCH}/checkpoints/bc_${DATASET}_correct_only" \
            --run_name "bc-${DATASET}-correct-only" \
            --filter_correct_only \
            --max_length "${MAX_LEN}" \
            --num_train_epochs "${EPOCHS}" \
            --per_device_train_batch_size 4 \
            --gradient_accumulation_steps 2 \
            --learning_rate 3e-6 \
            --lora_r 32 --lora_alpha 32 \
            --save_steps 500 --logging_steps 10
        ;;

    offline_grpo)
        export WANDB_PROJECT="offline-grpo-${DATASET}"
        cd "${WORK_DIR}/offline_grpo"
        # generation_batch_size = per_device_batch * num_gpus * grad_accum = 5*4*2 = 40
        # must be divisible by num_generations=5: 40/5=8 ✓
        accelerate launch --config_file configs/accelerate_ddp_4gpu.yaml \
            train.py \
            --rollout_path "${ROLLOUT_PATH}" \
            --target_model "${STUDENT}" \
            --output_dir "${SCRATCH}/checkpoints/offline_grpo_${DATASET}" \
            --run_name "offline-grpo-${DATASET}" \
            --num_generations 5 \
            --num_train_epochs "${EPOCHS}" \
            --per_device_train_batch_size 5 \
            --gradient_accumulation_steps 2 \
            --learning_rate 3e-6 \
            --beta 0.001 \
            --max_completion_length "${MAX_COMP}" \
            --max_grad_norm 1.0 \
            --lora_r 32 --lora_alpha 32 \
            --save_steps 500 --logging_steps 10
        ;;

    dg_offline)
        export WANDB_PROJECT="dg-offline-${DATASET}"
        cd "${WORK_DIR}/DG-offline"
        # generation_batch_size = per_device_batch * num_gpus * grad_accum = 5*4*2 = 40
        # must be divisible by num_generations=5: 40/5=8 ✓
        accelerate launch --config_file configs/accelerate_ddp_4gpu.yaml \
            train.py \
            --rollout_path "${ROLLOUT_PATH}" \
            --target_model "${STUDENT}" \
            --output_dir "${SCRATCH}/checkpoints/dg_offline_${DATASET}_eta${DG_ETA}" \
            --run_name "dg-${DATASET}-eta${DG_ETA}" \
            --dg_temperature "${DG_ETA}" \
            --dg_gating completion \
            --num_generations 5 \
            --num_train_epochs "${EPOCHS}" \
            --per_device_train_batch_size 5 \
            --gradient_accumulation_steps 2 \
            --learning_rate 3e-6 \
            --beta 0.001 \
            --max_completion_length "${MAX_COMP}" \
            --max_grad_norm 1.0 \
            --lora_r 32 --lora_alpha 32 \
            --save_steps 500 --logging_steps 10
        ;;

    *)
        echo "Unknown method: ${METHOD}"
        echo "Available: bc, bc_correct, offline_grpo, dg_offline"
        exit 1
        ;;
esac

echo "=== Training complete ==="
