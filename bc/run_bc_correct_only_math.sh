#!/bin/bash
#
# BC correct-only: train 0.5B student on CORRECT teacher completions only.
#
# Purpose: Test whether the BC degradation (30.4% -> 27.4%) is caused by
# learning from incorrect completions (29% of data). If correct-only BC
# improves the student, the incorrect completions were the problem.
# If it still degrades, the issue is more fundamental (capacity gap).
#
# Usage:
#   sbatch --job-name=bc-correct run_bc_correct_only_math.sh
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
SCRATCH="/scratch/mrli"
SCRATCH="${SCRATCH:-/scratch/mrli}"
MODEL_DIR="${MODEL_DIR:-${SCRATCH}/models/Qwen2.5-0.5B-Instruct}"

ROLLOUT_PATH="${ROLLOUT_PATH:-${SCRATCH}/rollouts/math_teacher/rollouts_full.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH}/checkpoints/bc_math_correct_only}"
MAX_LENGTH="${MAX_LENGTH:-2304}"
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
    export WANDB_RUN_NAME="bc-0.5B-correct-only-$(date +%m%d)"
fi
cd "${WORK_DIR}"

mkdir -p logs/bc_math

METRICS_FILE="${OUTPUT_DIR}/training_metrics.jsonl"

echo "=== BC Correct-Only: 0.5B Student on Correct Teacher Rollouts ==="
echo "  Model: ${MODEL_DIR}"
echo "  Rollouts: ${ROLLOUT_PATH}"
echo "  Filter: --filter_correct_only"
echo "  Output: ${OUTPUT_DIR}"
echo "  wandb: ${WANDB_PROJECT} / ${WANDB_RUN_NAME}"

accelerate launch \
    --config_file "${WORK_DIR}/configs/accelerate_ddp_4gpu.yaml" \
    train_bc.py \
    --rollout_path "${ROLLOUT_PATH}" \
    --target_model "${MODEL_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${WANDB_RUN_NAME}" \
    --filter_correct_only \
    --max_length "${MAX_LENGTH}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
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
