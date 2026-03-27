#!/bin/bash
#
# Control experiment: Offline GRPO on MATH with 7B teacher training on its own rollouts.
#
# Purpose: Test whether offline GRPO works when the target model matches the
# behavior policy (both are the 7B teacher). This is still off-policy (rollouts
# are pre-collected, not generated during training), but eliminates the
# distribution mismatch between behavior and target that exists when the 0.5B
# student trains on 7B teacher rollouts.
#
# If the teacher improves on its own rollouts, the off-policy failure (26.48%)
# is caused by the student-teacher distribution mismatch.
# If it also fails, offline GRPO has deeper issues beyond mismatch.
#
# Key differences from run_controlled_math.sh:
#   - target_model: Qwen2.5-Math-7B-Instruct (was 0.5B-Instruct)
#   - per_device_train_batch_size: 2 (was 4) — 7B needs more memory
#   - gradient_accumulation_steps: 4 (was 2) — keeps effective batch at 32
#   - time: 24h (was 12h) — 7B is ~7x slower per step
# All other hyperparameters identical (online GRPO's settings).
#
# Expected: IS ratio near 1.0 at step 0 (same model generated the rollouts),
#           KL near 0, non-zero loss.
#
# Usage:
#   sbatch --job-name=teacher-self-math run_teacher_self_math.sh
#
#SBATCH --account=aip-szepesva
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=logs/teacher_self_math/%x-%j.out
#SBATCH --error=logs/teacher_self_math/%x-%j.err

set -e

WORK_DIR="/project/aip-szepesva/mrli/backup_dongheng/offline_grpo"
SCRATCH="/scratch/mrli"
MODEL_DIR="${SCRATCH}/models/Qwen2.5-Math-7B-Instruct"

ROLLOUT_PATH="${SCRATCH}/rollouts/math_teacher/rollouts_full.jsonl"
OUTPUT_DIR="${SCRATCH}/checkpoints/offline_grpo_math_teacher_self"

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
export WANDB_RUN_NAME="teacher-self-7B-$(date +%m%d)"
cd "${WORK_DIR}"

mkdir -p logs/teacher_self_math

METRICS_FILE="${OUTPUT_DIR}/training_metrics.jsonl"

echo "=== Control: 7B Teacher Offline GRPO on Its Own Rollouts ==="
echo "  Model (target): ${MODEL_DIR}"
echo "  Rollouts (behavior = same model): ${ROLLOUT_PATH}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Metrics: ${METRICS_FILE}"
echo "  wandb: ${WANDB_PROJECT} / ${WANDB_RUN_NAME}"
echo ""
echo "  Hyperparameters (matching online GRPO):"
echo "    beta=0.001, lr=3e-6, max_grad_norm=1.0, weight_decay=0.01"
echo "    epochs=1, per_device_batch=2, grad_accum=4, num_gen=4"
echo "    Effective batch: 2*4*4=32 = 8 prompts/step (same as online)"
echo "    lora_r=32, lora_alpha=32"
echo ""
echo "  Expected: IS ratio ~1.0 (target = behavior), KL ~0 at start"

accelerate launch \
    --config_file "${WORK_DIR}/configs/accelerate_ddp_4gpu.yaml" \
    train.py \
    --rollout_path "${ROLLOUT_PATH}" \
    --target_model "${MODEL_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${WANDB_RUN_NAME}" \
    --num_generations 4 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
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
