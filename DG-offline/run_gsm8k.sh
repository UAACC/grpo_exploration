#!/bin/bash
#
# DG-offline GRPO on GSM8K dataset
#
# Usage:
#   sbatch run_gsm8k.sh                   # default eta=1.0
#   DG_ETA=0.5 sbatch run_gsm8k.sh       # custom eta
#
#SBATCH --account=aip-szepesva
#SBATCH --job-name=dg-offline-gsm8k
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ---- Paths (configurable via env vars) --------------------------------
SCRATCH="${SCRATCH:-/scratch/mrli}"
WORK_DIR="/project/aip-szepesva/mrli/backup_dongheng/DG-offline"
EVAL_SCRIPT="/project/aip-szepesva/mrli/backup_dongheng/Math_Verifier/eval_unified.py"

STUDENT_MODEL="${STUDENT_MODEL:-${SCRATCH}/models/Qwen2.5-0.5B-Instruct}"
TEACHER_MODEL="${TEACHER_MODEL:-${SCRATCH}/models/Qwen2.5-Math-7B-Instruct}"
ROLLOUT_PATH="${ROLLOUT_PATH:-${SCRATCH}/rollouts/gsm8k_teacher/rollouts_gsm8k.jsonl}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH}/checkpoints/dg_offline_gsm8k}"
MERGED_DIR="${MERGED_DIR:-${SCRATCH}/merged/dg_offline_gsm8k}"

# ---- DG hyperparameters -----------------------------------------------
DG_ETA="${DG_ETA:-1.0}"
DG_GATING="${DG_GATING:-completion}"

# ---- Environment ------------------------------------------------------
module load python/3.11 cuda/12.6 arrow opencv
source /project/aip-szepesva/mrli/backup_dongheng/.venv/bin/activate
export HF_HOME="${SCRATCH}"
export HF_DATASETS_CACHE="${SCRATCH}/datasets/GSM8K"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

if [ -f "/project/aip-szepesva/mrli/backup_dongheng/offline_grpo/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat /project/aip-szepesva/mrli/backup_dongheng/offline_grpo/.wandb_key)
fi

mkdir -p "${WORK_DIR}/logs"

cd "${WORK_DIR}"

echo "=== DG Offline GRPO on GSM8K ==="
echo "  Student: ${STUDENT_MODEL}"
echo "  Teacher: ${TEACHER_MODEL}"
echo "  Rollouts: ${ROLLOUT_PATH}"
echo "  DG eta: ${DG_ETA}"
echo "  DG gating: ${DG_GATING}"
echo "  Output: ${CHECKPOINT_DIR}"

CMD="${1:-train}"

if [ "$CMD" = "train" ]; then
    accelerate launch --config_file configs/accelerate_ddp_4gpu.yaml \
        train.py \
        --target_model "${STUDENT_MODEL}" \
        --behavior_model "${TEACHER_MODEL}" \
        --rollout_path "${ROLLOUT_PATH}" \
        --output_dir "${CHECKPOINT_DIR}" \
        --wandb_project "dg-offline-gsm8k" \
        --dg_temperature "${DG_ETA}" \
        --dg_gating "${DG_GATING}" \
        --learning_rate 3e-6 \
        --beta 0.001 \
        --num_generations 5 \
        --per_device_train_batch_size 5 \
        --gradient_accumulation_steps 2 \
        --num_train_epochs 5 \
        --max_completion_length 1024 \
        --max_grad_norm 1.0 \
        --weight_decay 0.01 \
        --warmup_ratio 0.1 \
        --save_steps 500 \
        --logging_steps 5 \
        --lora_r 32 \
        --lora_alpha 32 \
        --seed 42

    echo "=== Training complete ==="

elif [ "$CMD" = "eval" ]; then
    CKPT="${2:-${CHECKPOINT_DIR}}"
    echo "=== Evaluating ${CKPT} on GSM8K ==="
    python "${EVAL_SCRIPT}" \
        --model_path "${CKPT}" \
        --base_model "${STUDENT_MODEL}" \
        --merge_lora \
        --merged_output "${MERGED_DIR}" \
        --dataset_type gsm8k \
        --runs 5 \
        --temperature 0.0 \
        --max_tokens 1024 \
        --max_model_len 2048

elif [ "$CMD" = "eval-baseline" ]; then
    echo "=== Evaluating baseline ${STUDENT_MODEL} on GSM8K ==="
    python "${EVAL_SCRIPT}" \
        --model_path "${STUDENT_MODEL}" \
        --dataset_type gsm8k \
        --runs 5 \
        --temperature 0.0 \
        --max_tokens 1024 \
        --max_model_len 2048
fi

echo "=== Done ==="
