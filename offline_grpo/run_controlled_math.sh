#!/bin/bash
#
# Controlled experiment: Offline GRPO on MATH with ONLINE GRPO's hyperparameters.
#
# Purpose: Isolate on-policy vs off-policy effect by removing hyperparameter confounds.
# The original offline GRPO used very different hyperparams from online GRPO:
#   - beta: 0.1 (offline) vs 0.001 (online) — 100x difference
#   - max_grad_norm: 0.1 vs 1.0 — 10x difference
#   - weight_decay: 0.1 vs 0.01 — 10x difference
#   - num_train_epochs: 1 vs 15 — 15x difference
#   - learning_rate: 5e-6 vs 3e-6
# This run matches online GRPO's hyperparameters exactly.
#
# Usage:
#   sbatch --job-name=offline-ctrl-math run_controlled_math.sh
#
#SBATCH --account=aip-szepesva
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=logs/controlled_math/%x-%j.out
#SBATCH --error=logs/controlled_math/%x-%j.err

set -e

WORK_DIR="/project/aip-szepesva/mrli/backup_dongheng/offline_grpo"
SCRATCH="/scratch/mrli"
MODEL_DIR="${SCRATCH}/models/Qwen2.5-0.5B-Instruct"

ROLLOUT_PATH="${SCRATCH}/rollouts/math_teacher/rollouts_full.jsonl"
OUTPUT_DIR="${SCRATCH}/checkpoints/offline_grpo_math_controlled"

# ── Activate environment ─────────────────────────────────────────
module load python/3.11 cuda/12.6 arrow opencv
source /project/aip-szepesva/mrli/backup_dongheng/.venv/bin/activate
export HF_HOME="${SCRATCH}"
export HF_DATASETS_CACHE="/scratch/mrli/datasets/MATH"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=$(cat /project/aip-szepesva/mrli/backup_dongheng/offline_grpo/.wandb_key)
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_PROJECT="offline-grpo-math"
export WANDB_RUN_NAME="controlled-online-hparams-$(date +%m%d)"
cd "${WORK_DIR}"

mkdir -p logs/controlled_math

METRICS_FILE="${OUTPUT_DIR}/training_metrics.jsonl"

echo "=== Controlled Experiment: Offline GRPO with Online GRPO's Hyperparameters ==="
echo "  Model: ${MODEL_DIR}"
echo "  Rollouts: ${ROLLOUT_PATH}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Metrics: ${METRICS_FILE}"
echo "  wandb: ${WANDB_PROJECT} / ${WANDB_RUN_NAME}"
echo ""
echo "  Hyperparameters (matching online GRPO):"
echo "    beta=0.001, lr=3e-6, max_grad_norm=1.0, weight_decay=0.01"
echo "    epochs=1, per_device_batch=4, grad_accum=2, num_gen=4"
echo "    (MATH rollouts have 4 completions/prompt, so num_gen=4)"
echo "    Effective batch: 4*4*2=32 = 8 prompts/step (same as online)"
echo "    lora_r=32, lora_alpha=32"

accelerate launch \
    --config_file "${WORK_DIR}/configs/accelerate_ddp_4gpu.yaml" \
    train.py \
    --rollout_path "${ROLLOUT_PATH}" \
    --target_model "${MODEL_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${WANDB_RUN_NAME}" \
    --num_generations 4 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 3e-6 \
    --beta 0.001 \
    --max_completion_length 2048 \
    --max_grad_norm 1.0 \
    --weight_decay 0.01 \
    --save_steps 200 \
    --logging_steps 5 \
    --ref_sync_steps 0 \
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
    for key in ['loss', 'reward', 'kl', 'grad_norm', 'entropy']:
        v0 = first.get(key, 'N/A')
        v1 = last.get(key, 'N/A')
        print(f'  {key}: {v0} -> {v1}')
"
