#!/bin/bash
#
# Offline GRPO on GSM8K
#   Teacher (behavior model): Qwen2.5-Math-7B-Instruct (generates rollouts)
#   Student (target model):   Qwen2.5-0.5B-Instruct   (trained with offline GRPO)
#   Dataset: GSM8K (7,473 train / 1,319 test)
#
# Usage:
#   sbatch --job-name=gsm8k-rollouts  run_gsm8k_offline.sh rollouts
#   sbatch --job-name=gsm8k-train     run_gsm8k_offline.sh train
#   sbatch --job-name=gsm8k-eval      run_gsm8k_offline.sh eval
#   sbatch --job-name=gsm8k-eval-base run_gsm8k_offline.sh eval-baseline
#
#SBATCH --account=aip-szepesva
#SBATCH --time=4:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=logs/gsm8k_offline/%x-%j.out
#SBATCH --error=logs/gsm8k_offline/%x-%j.err

set -e

WORK_DIR="/project/aip-szepesva/mrli/backup_dongheng/offline_grpo"
SCRATCH="/scratch/mrli"
CONFIG_DIR="${WORK_DIR}/configs"

# Teacher (behavior model) — generates rollouts
TEACHER_DIR="${SCRATCH}/models/Qwen2.5-Math-7B-Instruct"

# Student (target model) — trained with offline GRPO
STUDENT_DIR="${SCRATCH}/models/Qwen2.5-0.5B-Instruct"

ROLLOUT_PATH="${SCRATCH}/rollouts/gsm8k_teacher/rollouts_gsm8k.jsonl"
OUTPUT_DIR="${SCRATCH}/checkpoints/offline_grpo_gsm8k"

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
cd "${WORK_DIR}"

case "${1}" in

# ── Step 1: Generate rollouts from teacher on GSM8K train ─────────
rollouts)
    mkdir -p "$(dirname ${ROLLOUT_PATH})"
    echo "=== Generating rollouts on GSM8K train set ==="
    echo "=== Teacher: ${TEACHER_DIR} ==="
    echo "=== Output: ${ROLLOUT_PATH} ==="

    # num_generations=5: matches online GRPO (VERL 3B script)
    # temperature=0.7: matches online GRPO generation temperature
    python generate_rollouts.py \
        --model_name "${TEACHER_DIR}" \
        --dataset_type gsm8k \
        --split train \
        --num_generations 5 \
        --temperature 0.7 \
        --max_tokens 1024 \
        --max_model_len 2048 \
        --seed 42 \
        --tensor_parallel_size 4 \
        --gpu_memory_utilization 0.9 \
        --output_path "${ROLLOUT_PATH}"

    echo "=== Rollout generation complete ==="
    ;;

# ── Step 2: Train with offline GRPO on GSM8K rollouts ─────────────
train)
    echo "=== Offline GRPO training on GSM8K ==="
    echo "=== Rollouts: ${ROLLOUT_PATH} ==="
    echo "=== Output: ${OUTPUT_DIR} ==="
    export WANDB_PROJECT="offline-grpo-gsm8k"

    accelerate launch \
        --config_file "${CONFIG_DIR}/accelerate_ddp_4gpu.yaml" \
        train.py \
        --rollout_path "${ROLLOUT_PATH}" \
        --target_model "${STUDENT_DIR}" \
        --behavior_model "${TEACHER_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --run_name "offline-grpo-gsm8k-$(date +%Y%m%d_%H%M%S)" \
        --num_generations 5 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 5 \
        --gradient_accumulation_steps 2 \
        --learning_rate 5e-6 \
        --beta 0.1 \
        --max_prompt_length 512 \
        --max_completion_length 1024 \
        --save_steps 500 \
        --logging_steps 10 \
        --ref_sync_steps 0 \
        --report_to wandb

    echo "=== Training complete ==="
    ;;

# ── Step 3: Evaluate trained model on GSM8K test ──────────────────
eval)
    echo "=== Evaluating offline GRPO model on GSM8K ==="
    MERGED_DIR="${SCRATCH}/merged/offline_grpo_gsm8k_merged"

    # Use online_grpo evaluate.py which has GSM8K-specific extraction
    EVAL_SCRIPT="${WORK_DIR}/../online_grpo/evaluate.py"

    python "${EVAL_SCRIPT}" \
        --model_path "${OUTPUT_DIR}" \
        --base_model "${STUDENT_DIR}" \
        --merge_lora \
        --merged_output "${MERGED_DIR}" \
        --dataset "openai/gsm8k" \
        --split test \
        --temperature 0.0 \
        --max_tokens 1024 \
        --max_model_len 2048 \
        --runs 5

    echo "=== Evaluation complete ==="
    ;;

# ── Evaluate baseline (no training) ──────────────────────────────
eval-baseline)
    echo "=== Evaluating baseline on GSM8K ==="
    EVAL_SCRIPT="${WORK_DIR}/../online_grpo/evaluate.py"

    python "${EVAL_SCRIPT}" \
        --model_path "${STUDENT_DIR}" \
        --dataset "openai/gsm8k" \
        --split test \
        --temperature 0.0 \
        --max_tokens 1024 \
        --max_model_len 2048 \
        --runs 5

    echo "=== Baseline evaluation complete ==="
    ;;

*)
    echo "Usage: sbatch --job-name=<name> run_gsm8k_offline.sh <command>"
    echo ""
    echo "  rollouts      - Generate teacher rollouts on GSM8K train set (4x L40s)"
    echo "  train         - Train offline GRPO on GSM8K rollouts (4x L40s DDP)"
    echo "  eval          - Evaluate trained model on GSM8K test (1x L40s)"
    echo "  eval-baseline - Evaluate base student on GSM8K test (1x L40s)"
    exit 1
    ;;

esac
