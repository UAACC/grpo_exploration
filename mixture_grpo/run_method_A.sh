#!/bin/bash
#
# Method A: Unified Mixture GRPO (GSM8K)
#   Online student rollouts + offline teacher rollouts in a single unified group.
#
# Usage:
#   sbatch --job-name=mixture-A-rollouts run_method_A.sh rollouts
#   sbatch --job-name=mixture-A run_method_A.sh train
#   sbatch --job-name=mixture-A-eval run_method_A.sh eval
#
#SBATCH --account=aip-szepesva
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=logs/mixture_A/%x-%j.out
#SBATCH --error=logs/mixture_A/%x-%j.err

set -e

WORK_DIR="/project/aip-szepesva/mrli/backup_dongheng/mixture_grpo"
SCRATCH="/scratch/mrli"
STUDENT_DIR="${SCRATCH}/models/Qwen2.5-0.5B-Instruct"
TEACHER_DIR="${SCRATCH}/models/Qwen2.5-Math-7B-Instruct"

ROLLOUT_PATH="${SCRATCH}/rollouts/gsm8k_teacher/rollouts_gsm8k.jsonl"
OUTPUT_DIR="${SCRATCH}/checkpoints/mixture_A_gsm8k"
MERGED_DIR="${SCRATCH}/merged/mixture_A_gsm8k_merged"

# ── Activate environment ─────────────────────────────────────────
module load python/3.11 cuda/12.6 arrow opencv
source /project/aip-szepesva/mrli/backup_dongheng/.venv/bin/activate
export HF_HOME="${SCRATCH}"
export HF_DATASETS_CACHE="/scratch/mrli/datasets/GSM8K"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=$(cat /project/aip-szepesva/mrli/backup_dongheng/offline_grpo/.wandb_key)
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_PROJECT="mixture-grpo-A"
cd "${WORK_DIR}"

mkdir -p logs/mixture_A

case "${1}" in

rollouts)
    echo "=== Generating GSM8K teacher rollouts ==="
    mkdir -p "${SCRATCH}/rollouts/gsm8k_teacher"
    python generate_rollouts.py \
        --model_name "${TEACHER_DIR}" \
        --output_path "${ROLLOUT_PATH}" \
        --num_generations 4 \
        --temperature 0.7 \
        --max_tokens 1024 \
        --max_model_len 2048 \
        --tensor_parallel_size 1 \
        --gpu_memory_utilization 0.8
    echo "=== Rollout generation complete ==="
    ;;

train)
    echo "=== Method A: Unified Mixture GRPO (GSM8K) ==="
    accelerate launch \
        --config_file "${WORK_DIR}/../offline_grpo/configs/accelerate_ddp_4gpu.yaml" \
        method_A_unified/train.py \
        --teacher_rollout_path "${ROLLOUT_PATH}" \
        --target_model "${STUDENT_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --num_generations 4 \
        --num_teacher_per_prompt 4 \
        --num_train_epochs 5 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 3 \
        --learning_rate 3e-6 \
        --beta 0.01 \
        --max_prompt_length 512 \
        --max_completion_length 1024 \
        --max_grad_norm 1.0 \
        --save_steps 200 \
        --logging_steps 10 \
        --ref_sync_steps 0 \
        --report_to wandb \
        --resume_from_checkpoint latest
    echo "=== Training complete ==="
    ;;

eval)
    CKPT="${2:-checkpoint-1200}"
    echo "=== Evaluating Method A (${CKPT}) ==="
    python evaluate.py \
        --model_path "${OUTPUT_DIR}/${CKPT}" \
        --base_model "${STUDENT_DIR}" \
        --merge_lora \
        --merged_output "${MERGED_DIR}" \
        --runs 10 \
        --temperature 0.0
    echo "=== Evaluation complete ==="
    ;;

eval-baseline)
    echo "=== Evaluating baseline ==="
    python evaluate.py \
        --model_path "${STUDENT_DIR}" \
        --runs 10 \
        --temperature 0.0
    ;;

eval-teacher)
    echo "=== Evaluating teacher model on GSM8K test ==="
    python evaluate.py \
        --model_path "${TEACHER_DIR}" \
        --runs 5 \
        --temperature 0.0
    ;;

*)
    echo "Usage: sbatch --job-name=<name> run_method_A.sh {rollouts|train|eval|eval-baseline|eval-teacher}"
    exit 1
    ;;

esac
