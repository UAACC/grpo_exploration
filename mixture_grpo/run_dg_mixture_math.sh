#!/bin/bash
#
# DG-Mixture GRPO (MATH)
#   Online student rollouts (standard GRPO loss) + offline teacher rollouts
#   weighted by DG-offline's sigmoid gate on advantage * surprisal.
#
# Usage:
#   sbatch --job-name=dg-mixture-math run_dg_mixture_math.sh train
#   sbatch --job-name=dg-mixture-math-eval run_dg_mixture_math.sh eval
#
# Hyperparameter sweeps via env vars (matches DG-offline/run_math.sh pattern):
#   DG_ETA=0.5 DG_LAMBDA=0.3 \
#   CHECKPOINT_DIR=/scratch/$USER/checkpoints/dg_mixture_math_eta0.5_lam0.3 \
#   sbatch --job-name=dg-mix-eta0.5-lam0.3 run_dg_mixture_math.sh train
#
#SBATCH --account=aip-szepesva
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=logs/dg_mixture_math/%x-%j.out
#SBATCH --error=logs/dg_mixture_math/%x-%j.err

set -e

WORK_DIR="/project/aip-szepesva/mrli/backup_dongheng/mixture_grpo"
SCRATCH="${SCRATCH:-/scratch/mrli}"
STUDENT_DIR="${STUDENT_DIR:-${SCRATCH}/models/Qwen2.5-0.5B-Instruct}"
TEACHER_DIR="${TEACHER_DIR:-${SCRATCH}/models/Qwen2.5-Math-7B-Instruct}"

ROLLOUT_PATH="${ROLLOUT_PATH:-${SCRATCH}/rollouts/math_teacher/rollouts_full.jsonl}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH}/checkpoints/dg_mixture_math}"
MERGED_DIR="${MERGED_DIR:-${SCRATCH}/merged/dg_mixture_math_merged}"

# DG hyperparameters (sweepable via env vars)
DG_ETA="${DG_ETA:-0.5}"
DG_LAMBDA="${DG_LAMBDA:-0.3}"
DG_GATING="${DG_GATING:-completion}"

# ── Activate environment ─────────────────────────────────────────
module load python/3.11 cuda/12.6 arrow opencv
source /project/aip-szepesva/mrli/backup_dongheng/.venv/bin/activate
export HF_HOME="${SCRATCH}"
export HF_DATASETS_CACHE="/scratch/mrli/datasets/MATH"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_PROJECT="dg-mixture-math"

if [ -f "/project/aip-szepesva/mrli/backup_dongheng/offline_grpo/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat /project/aip-szepesva/mrli/backup_dongheng/offline_grpo/.wandb_key)
fi

cd "${WORK_DIR}"
mkdir -p logs/dg_mixture_math

CMD="${1:-train}"

case "${CMD}" in

train)
    echo "=== DG-Mixture GRPO (MATH, 1 epoch) ==="
    echo "  Student: ${STUDENT_DIR}"
    echo "  Rollouts: ${ROLLOUT_PATH}"
    echo "  Output: ${CHECKPOINT_DIR}"
    echo "  DG eta: ${DG_ETA}"
    echo "  DG lambda (offline weight): ${DG_LAMBDA}"
    echo "  DG gating: ${DG_GATING}"

    accelerate launch \
        --config_file "${WORK_DIR}/../offline_grpo/configs/accelerate_ddp_4gpu.yaml" \
        dg_mixture/train.py \
        --teacher_rollout_path "${ROLLOUT_PATH}" \
        --target_model "${STUDENT_DIR}" \
        --output_dir "${CHECKPOINT_DIR}" \
        --dataset_type math \
        --dg_temperature "${DG_ETA}" \
        --dg_offline_weight "${DG_LAMBDA}" \
        --dg_gating "${DG_GATING}" \
        --num_generations 4 \
        --num_teacher_per_prompt 4 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --learning_rate 5e-6 \
        --beta 0.01 \
        --max_prompt_length 512 \
        --max_completion_length 786 \
        --max_grad_norm 0.1 \
        --save_steps 500 \
        --logging_steps 10 \
        --lora_r 32 \
        --lora_alpha 32 \
        --ref_sync_steps 0 \
        --report_to wandb
    echo "=== Training complete ==="
    ;;

eval)
    CKPT="${2:-latest}"
    if [ "${CKPT}" = "latest" ]; then
        CKPT=$(ls -d "${CHECKPOINT_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 | xargs basename 2>/dev/null)
        if [ -z "${CKPT}" ]; then
            echo "No checkpoint found in ${CHECKPOINT_DIR}, using merged final dir directly"
            CKPT_PATH="${CHECKPOINT_DIR}"
        else
            CKPT_PATH="${CHECKPOINT_DIR}/${CKPT}"
        fi
    else
        CKPT_PATH="${CHECKPOINT_DIR}/${CKPT}"
    fi
    echo "=== Evaluating DG-Mixture MATH (${CKPT_PATH}) ==="
    python evaluate.py \
        --model_path "${CKPT_PATH}" \
        --base_model "${STUDENT_DIR}" \
        --merge_lora \
        --merged_output "${MERGED_DIR}" \
        --dataset_type math \
        --runs 5 \
        --temperature 0.0 \
        --max_tokens 2048 \
        --max_model_len 3072
    echo "=== Evaluation complete ==="
    ;;

eval-baseline)
    echo "=== Evaluating baseline on MATH ==="
    python evaluate.py \
        --model_path "${STUDENT_DIR}" \
        --dataset_type math \
        --runs 5 \
        --temperature 0.0 \
        --max_tokens 2048 \
        --max_model_len 3072
    ;;

*)
    echo "Usage: sbatch --job-name=<name> run_dg_mixture_math.sh {train|eval [checkpoint]|eval-baseline}"
    exit 1
    ;;

esac
