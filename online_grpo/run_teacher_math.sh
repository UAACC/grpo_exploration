#!/bin/bash
#
# Online GRPO on MATH with 7B teacher (Qwen2.5-Math-7B-Instruct) + LoRA
#
# Purpose: Test if online GRPO works for the 7B teacher when offline GRPO
# degraded it (74.96% -> 71.20%). If online improves it, the issue is
# with stale/offline rollouts, not the GRPO algorithm.
#
# Memory: 7B bf16 (~14GB) + vLLM colocate (~14GB) + LoRA/optimizer (~2GB)
#         ~30GB per GPU — fits in 48GB L40s with small batch
#
# Usage:
#   sbatch --job-name=online-teacher-math run_teacher_math.sh
#
#SBATCH --account=aip-szepesva
#SBATCH --time=48:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e

WORK_DIR="/project/aip-szepesva/mrli/backup_dongheng/online_grpo"
SCRATCH="/scratch/mrli"

MODEL_DIR="${SCRATCH}/models/Qwen2.5-Math-7B-Instruct"
OUTPUT_DIR="${SCRATCH}/checkpoints/online_grpo_math_teacher_7B"

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
export WANDB_PROJECT="online-grpo-math"
export WANDB_RUN_NAME="teacher-7B-online-$(date +%m%d)"
cd "${WORK_DIR}"

mkdir -p logs

echo "=== Online GRPO: 7B Teacher on MATH ==="
echo "  Model: ${MODEL_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "  wandb: ${WANDB_PROJECT} / ${WANDB_RUN_NAME}"
echo ""
echo "  Hyperparameters:"
echo "    lr=3e-6, beta=0.001, temp=0.7"
echo "    per_device_batch=1, grad_accum=10, num_gen=5"
echo "    epochs=1, max_completion=2048"
echo "    lora_r=32, lora_alpha=32"

accelerate launch \
    --config_file "${WORK_DIR}/configs/accelerate_ddp_4gpu.yaml" \
    train.py \
    --model "${MODEL_DIR}" \
    --dataset_type math \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${WANDB_RUN_NAME}" \
    --num_generations 4 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-6 \
    --beta 0.001 \
    --temperature 0.7 \
    --max_prompt_length 512 \
    --max_completion_length 2048 \
    --lora_r 32 \
    --lora_alpha 32 \
    --save_steps 200 \
    --logging_steps 5 \
    --vllm_gpu_memory_utilization 0.4 \
    --report_to wandb

echo "=== Training complete ==="
