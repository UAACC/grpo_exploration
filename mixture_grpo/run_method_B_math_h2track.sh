#!/bin/bash
#
# Method B MATH training with enhanced IS ratio tracking.
# Purpose: Collect per-step evidence of H1/H2 metrics during real training.
#
# Usage:
#   sbatch --job-name=B-math-h2track run_method_B_math_h2track.sh
#
#SBATCH --account=aip-szepesva
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=logs/mixture_B_math/%x-%j.out
#SBATCH --error=logs/mixture_B_math/%x-%j.err

set -e

WORK_DIR="/project/aip-szepesva/mrli/backup_dongheng/mixture_grpo"
SCRATCH="/scratch/mrli"
STUDENT_DIR="${SCRATCH}/models/Qwen2.5-0.5B-Instruct"

ROLLOUT_PATH="${SCRATCH}/rollouts/math_teacher/rollouts_full.jsonl"
OUTPUT_DIR="${SCRATCH}/checkpoints/mixture_B_math_h2track"

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
export WANDB_PROJECT="mixture-grpo-B-math"
export WANDB_RUN_NAME="h2-tracking-math-$(date +%m%d)"
cd "${WORK_DIR}"

mkdir -p logs/mixture_B_math

METRICS_FILE="${OUTPUT_DIR}/training_metrics.jsonl"

echo "=== Method B MATH: H2 IS ratio tracking run ==="
echo "  Output: ${OUTPUT_DIR}"
echo "  Metrics JSONL: ${METRICS_FILE}"
echo "  wandb project: mixture-grpo-B-math"
echo "  wandb run: ${WANDB_RUN_NAME}"

accelerate launch \
    --config_file "${WORK_DIR}/../offline_grpo/configs/accelerate_ddp_4gpu.yaml" \
    method_B_weighted/train.py \
    --teacher_rollout_path "${ROLLOUT_PATH}" \
    --target_model "${STUDENT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --dataset_type math \
    --offline_weight 0.3 \
    --num_generations 4 \
    --num_teacher_per_prompt 4 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-6 \
    --beta 0.1 \
    --max_completion_length 786 \
    --max_grad_norm 0.1 \
    --save_steps 200 \
    --logging_steps 5 \
    --ref_sync_steps 0 \
    --report_to wandb

echo "=== Training complete ==="
echo ""
echo "=== Metrics saved to: ${METRICS_FILE} ==="
echo "Key columns: offline/clip_fraction, offline/is_ratio_mean, offline/effective_signal, frac_all_wrong, frac_all_correct"
echo ""
echo "Quick analysis:"
python3 -c "
import json
records = [json.loads(l) for l in open('${METRICS_FILE}')]
print(f'Total logged steps: {len(records)}')
if records:
    first = records[0]
    last = records[-1]
    for key in ['offline/clip_fraction', 'offline/is_ratio_mean', 'offline/effective_signal',
                'frac_all_wrong', 'frac_all_correct', 'frac_reward_zero_std']:
        v0 = first.get(key, 'N/A')
        v1 = last.get(key, 'N/A')
        print(f'  {key}: {v0} → {v1}')
"
