#!/bin/bash
#
# Offline BC baseline: train 0.5B student on teacher completions via cross-entropy.
#
# Purpose: Establish whether teacher rollout data contains learnable signal.
# If BC improves the student but offline GRPO degrades it, the problem is in
# the off-policy RL optimization, not the data itself.
#
# Uses the SAME rollout data and hyperparameters as the controlled offline GRPO
# experiment, but replaces the GRPO objective with standard next-token prediction
# loss on teacher completion tokens (prompt masked out).
#
# Usage:
#   sbatch --job-name=bc-math run_bc_math.sh
#
#SBATCH --account=aip-szepesva
#SBATCH --time=06:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=logs/bc_math/%x-%j.out
#SBATCH --error=logs/bc_math/%x-%j.err

set -e

WORK_DIR="/project/aip-szepesva/mrli/backup_dongheng/bc"
SCRATCH="${SCRATCH:-/scratch/mrli}"
MODEL_DIR="${MODEL_DIR:-${SCRATCH}/models/Qwen2.5-0.5B-Instruct}"

ROLLOUT_PATH="${ROLLOUT_PATH:-${SCRATCH}/rollouts/math_teacher/rollouts_full.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH}/checkpoints/bc_math}"
MAX_LENGTH="${MAX_LENGTH:-2304}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
WANDB_RUN_NAME_OVERRIDE="${WANDB_RUN_NAME:-}"

# ── Activate environment ─────────────────────────────────────────
module load python/3.11 cuda/12.6 arrow opencv
source /project/aip-szepesva/mrli/backup_dongheng/.venv/bin/activate
export HF_HOME="${SCRATCH}"
export HF_DATASETS_CACHE="/scratch/mrli/datasets/MATH"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=$(cat /project/aip-szepesva/mrli/backup_dongheng/offline_grpo/.wandb_key)
export WANDB_PROJECT="offline-grpo-math"
if [ -n "${WANDB_RUN_NAME_OVERRIDE}" ]; then
    export WANDB_RUN_NAME="${WANDB_RUN_NAME_OVERRIDE}"
else
    export WANDB_RUN_NAME="bc-0.5B-all-completions-$(date +%m%d)"
fi
cd "${WORK_DIR}"

mkdir -p logs/bc_math

METRICS_FILE="${OUTPUT_DIR}/training_metrics.jsonl"

echo "=== Offline BC: 0.5B Student on Teacher Rollouts ==="
echo "  Model: ${MODEL_DIR}"
echo "  Rollouts: ${ROLLOUT_PATH}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Metrics: ${METRICS_FILE}"
echo "  wandb: ${WANDB_PROJECT} / ${WANDB_RUN_NAME}"
echo ""
echo "  Hyperparameters (aligned with offline GRPO controlled):"
echo "    lr=3e-6, max_grad_norm=1.0, weight_decay=0.01"
echo "    epochs=1, per_device_batch=4, grad_accum=2"
echo "    max_length=2304 (256 prompt + 2048 completion)"
echo "    lora_r=32, lora_alpha=32"
echo "    Loss: cross-entropy on completion tokens only"

accelerate launch \
    --config_file "${WORK_DIR}/configs/accelerate_ddp_4gpu.yaml" \
    train_bc.py \
    --rollout_path "${ROLLOUT_PATH}" \
    --target_model "${MODEL_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${WANDB_RUN_NAME}" \
    --max_length "${MAX_LENGTH}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --learning_rate 3e-6 \
    --max_grad_norm 1.0 \
    --weight_decay 0.01 \
    --lora_r 32 \
    --lora_alpha 32 \
    --save_steps 200 \
    --logging_steps 5 \
    --report_to wandb

echo "=== Training complete ==="
echo ""
echo "=== Metrics saved to: ${METRICS_FILE} ==="
echo "Quick analysis:"
python3 -c "
import json
records = [json.loads(l) for l in open('${METRICS_FILE}')]
print(f'Total logged steps: {len(records)}')
if records:
    first = records[0]
    last = records[-1]
    for key in ['loss', 'train_loss', 'grad_norm']:
        v0 = first.get(key, 'N/A')
        v1 = last.get(key, 'N/A')
        print(f'  {key}: {v0} -> {v1}')
"
