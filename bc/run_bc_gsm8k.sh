#!/bin/bash
#
# Offline BC on GSM8K: train 0.5B student on GSM8K teacher completions via cross-entropy.
#
# GSM8K rollouts are auto-detected as dataset_type=gsm8k from the JSONL file.
# The BC data loader uses numeric comparison for GSM8K rewards (vs math_verify
# for MATH).
#
# Usage:
#   sbatch --job-name=bc-gsm8k run_bc_gsm8k.sh
#
#SBATCH --account=aip-szepesva
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=logs/bc_gsm8k/%x-%j.out
#SBATCH --error=logs/bc_gsm8k/%x-%j.err

set -e

WORK_DIR="/project/aip-szepesva/mrli/backup_dongheng/bc"
SCRATCH="/scratch/mrli"
MODEL_DIR="${SCRATCH}/models/Qwen2.5-0.5B-Instruct"

ROLLOUT_PATH="${SCRATCH}/rollouts/gsm8k_teacher/rollouts_gsm8k.jsonl"
OUTPUT_DIR="${SCRATCH}/checkpoints/bc_gsm8k"

# ── Activate environment ─────────────────────────────────────────
module load python/3.11 cuda/12.6 arrow opencv
source /project/aip-szepesva/mrli/backup_dongheng/.venv/bin/activate
export HF_HOME="${SCRATCH}"
export HF_DATASETS_CACHE="/scratch/mrli/datasets/GSM8K"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=$(cat /project/aip-szepesva/mrli/backup_dongheng/offline_grpo/.wandb_key)
export WANDB_PROJECT="offline-grpo-gsm8k"
export WANDB_RUN_NAME="bc-0.5B-gsm8k-all-$(date +%m%d)"
cd "${WORK_DIR}"

mkdir -p logs/bc_gsm8k

METRICS_FILE="${OUTPUT_DIR}/training_metrics.jsonl"

echo "=== Offline BC GSM8K: 0.5B Student on Teacher Rollouts ==="
echo "  Model: ${MODEL_DIR}"
echo "  Rollouts: ${ROLLOUT_PATH}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Metrics: ${METRICS_FILE}"
echo "  wandb: ${WANDB_PROJECT} / ${WANDB_RUN_NAME}"
echo ""
echo "  Hyperparameters:"
echo "    lr=3e-6, max_grad_norm=1.0, weight_decay=0.01"
echo "    epochs=5 (GSM8K is smaller than MATH, matches DG-offline GSM8K setup)"
echo "    per_device_batch=4, grad_accum=2"
echo "    max_length=1280 (256 prompt + 1024 completion, GSM8K is shorter than MATH)"
echo "    lora_r=32, lora_alpha=32"

accelerate launch \
    --config_file "${WORK_DIR}/configs/accelerate_ddp_4gpu.yaml" \
    train_bc.py \
    --rollout_path "${ROLLOUT_PATH}" \
    --target_model "${MODEL_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${WANDB_RUN_NAME}" \
    --max_length 1280 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 3e-6 \
    --max_grad_norm 1.0 \
    --weight_decay 0.01 \
    --lora_r 32 \
    --lora_alpha 32 \
    --save_steps 500 \
    --logging_steps 10 \
    --report_to wandb

echo "=== Training complete ==="
echo ""
echo "=== Metrics saved to: ${METRICS_FILE} ==="
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
