#!/bin/bash
#
# Method A: Unified Mixture GRPO (MATH)
#   Online student rollouts + offline teacher rollouts in a single unified group.
#   Settings aligned with offline GRPO MATH experiment.
#
# Usage:
#   sbatch --job-name=mixture-A-math run_method_A_math.sh train
#   sbatch --job-name=mixture-A-math-eval run_method_A_math.sh eval
#
#SBATCH --account=aip-szepesva
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=logs/mixture_A_math/%x-%j.out
#SBATCH --error=logs/mixture_A_math/%x-%j.err

set -e

WORK_DIR="/project/aip-szepesva/mrli/backup_dongheng/mixture_grpo"
SCRATCH="/scratch/mrli"
STUDENT_DIR="${SCRATCH}/models/Qwen2.5-0.5B-Instruct"
TEACHER_DIR="${SCRATCH}/models/Qwen2.5-Math-7B-Instruct"

# MATH teacher rollouts (same as offline GRPO)
ROLLOUT_PATH="${SCRATCH}/rollouts/math_teacher/rollouts_full.jsonl"
OUTPUT_DIR="${SCRATCH}/checkpoints/mixture_A_math"
MERGED_DIR="${SCRATCH}/merged/mixture_A_math_merged"

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
export WANDB_PROJECT="mixture-grpo-A-math"
cd "${WORK_DIR}"

mkdir -p logs/mixture_A_math

case "${1}" in

train)
    echo "=== Method A: Unified Mixture GRPO (MATH, 1 epoch) ==="
    accelerate launch \
        --config_file "${WORK_DIR}/../offline_grpo/configs/accelerate_ddp_4gpu.yaml" \
        method_A_unified/train.py \
        --teacher_rollout_path "${ROLLOUT_PATH}" \
        --target_model "${STUDENT_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --dataset_type math \
        --num_generations 4 \
        --num_teacher_per_prompt 4 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --learning_rate 5e-6 \
        --beta 0.1 \
        --max_prompt_length 512 \
        --max_completion_length 786 \
        --max_grad_norm 0.1 \
        --save_steps 500 \
        --logging_steps 10 \
        --ref_sync_steps 0 \
        --report_to wandb
    echo "=== Training complete ==="
    ;;

eval)
    CKPT="${2:-latest}"
    if [ "${CKPT}" = "latest" ]; then
        # Find the latest checkpoint directory
        CKPT=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 | xargs basename)
        if [ -z "${CKPT}" ]; then
            echo "No checkpoint found in ${OUTPUT_DIR}"
            exit 1
        fi
    fi
    echo "=== Evaluating Method A MATH (${CKPT}) ==="
    python evaluate.py \
        --model_path "${OUTPUT_DIR}/${CKPT}" \
        --base_model "${STUDENT_DIR}" \
        --merge_lora \
        --merged_output "${MERGED_DIR}" \
        --dataset_type math \
        --runs 1 \
        --temperature 0.6 \
        --max_tokens 2048 \
        --max_model_len 3072
    echo "=== Evaluation complete ==="
    ;;

eval-baseline)
    echo "=== Evaluating baseline on MATH ==="
    python evaluate.py \
        --model_path "${STUDENT_DIR}" \
        --dataset_type math \
        --runs 1 \
        --temperature 0.6 \
        --max_tokens 2048 \
        --max_model_len 3072
    ;;

*)
    echo "Usage: sbatch --job-name=<name> run_method_A_math.sh {train|eval [checkpoint]|eval-baseline}"
    exit 1
    ;;

esac
