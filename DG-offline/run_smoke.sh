#!/bin/bash
#
# Short smoke run for DG-offline under the new teacher_agnostic_loader.
# 10 optimizer steps on MATH shard 0; verifies the new loader plugs into
# the trainer without crashes and produces sane metric values.
#
set -euo pipefail

WORK_DIR="/project/aip-szepesva/mrli/backup_dongheng/DG-offline"
SCRATCH="${SCRATCH:-/scratch/mrli}"
STUDENT_MODEL="${STUDENT_MODEL:-${SCRATCH}/models/Qwen2.5-0.5B-Instruct}"
ROLLOUT_PATH="${ROLLOUT_PATH:-${SCRATCH}/rollouts/math_teacher/rollouts_shard_0.jsonl}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH}/checkpoints/dg_loader_smoke}"
TRAINING_REGIME="${TRAINING_REGIME:-current}"
LOSS_TYPE="${LOSS_TYPE:-}"
LOSS_TYPE_ARGS=()
if [ -n "${LOSS_TYPE}" ]; then
    LOSS_TYPE_ARGS=(--loss_type "${LOSS_TYPE}")
fi

module load python/3.11 cuda/12.6 arrow opencv
source /project/aip-szepesva/mrli/backup_dongheng/.venv/bin/activate

export HF_HOME="${SCRATCH}"
export HF_DATASETS_CACHE="${SCRATCH}/datasets/MATH"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

mkdir -p "${WORK_DIR}/logs"
cd "${WORK_DIR}"

echo "=== DG-offline loader smoke run ==="
echo "  Student:    ${STUDENT_MODEL}"
echo "  Rollouts:   ${ROLLOUT_PATH}"
echo "  Output:     ${CHECKPOINT_DIR}"
echo "  Loader:     teacher_agnostic_loader (Path A)"
echo "  Regime:     ${TRAINING_REGIME}"
echo "  Loss type:  ${LOSS_TYPE:-train.py default}"
echo "  Steps:      10"
echo

accelerate launch --config_file configs/accelerate_ddp_4gpu.yaml \
    train.py \
    --target_model "${STUDENT_MODEL}" \
    --rollout_path "${ROLLOUT_PATH}" \
    --output_dir "${CHECKPOINT_DIR}" \
    --dg_temperature 0.5 \
    --dg_gating completion \
    --training_regime "${TRAINING_REGIME}" \
    "${LOSS_TYPE_ARGS[@]}" \
    --learning_rate 3e-6 \
    --beta 0.001 \
    --num_generations 4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 1 \
    --max_steps 10 \
    --max_completion_length 2048 \
    --max_grad_norm 1.0 \
    --save_steps 1000 \
    --logging_steps 1 \
    --lora_r 32 \
    --lora_alpha 32 \
    --seed 42 \
    --report_to none

echo "=== Smoke run complete ==="
