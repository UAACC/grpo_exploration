#!/bin/bash
#
# Offline GRPO test run:
#   Teacher (behavior model): Qwen2.5-Math-7B-Instruct
#   Student (target model):   Qwen2.5-0.5B-Instruct
#   Data: small MATH subset (~41 problems)
#
# Usage:
#   1. Get a GPU:       salloc --account=aip-szepesva --time=1:00:00 --gpus-per-node=1 --cpus-per-task=6 --mem=48G
#   2. On compute node: bash run_test.sh rollouts    (generate rollouts with teacher via vLLM)
#   3. On compute node: bash run_test.sh train       (train student with offline GRPO)
#   Or run all at once:
#   2. On compute node: bash run_test.sh gpu         (rollouts + train)

set -e

WORK_DIR="/project/aip-szepesva/mrli/backup_dongheng/offline_grpo"
SCRATCH="/scratch/mrli"

# Teacher (behavior model) — generates rollouts
TEACHER_DIR="${SCRATCH}/models/Qwen2.5-Math-7B-Instruct"

# Student (target model) — trained with offline GRPO
STUDENT_DIR="${SCRATCH}/models/Qwen2.5-0.5B-Instruct"

ROLLOUT_PATH="${WORK_DIR}/rollouts_test.jsonl"
OUTPUT_DIR="${WORK_DIR}/outputs/offline_grpo_test"

# ── Activate environment ─────────────────────────────────────────
module load python/3.11 cuda/12.6 arrow opencv
source /project/aip-szepesva/mrli/backup_dongheng/.venv/bin/activate
export HF_HOME="${SCRATCH}"
export HF_DATASETS_CACHE="/scratch/mrli/datasets/MATH"
cd "${WORK_DIR}"

case "${1}" in

rollouts)
    # Run this on a COMPUTE NODE (needs GPU)
    echo "=== Generating rollouts with teacher: ${TEACHER_DIR} ==="
    echo "=== Test mode: ~41 problems, 4 completions each ==="
    python generate_rollouts.py \
        --model_name "${TEACHER_DIR}" \
        --num_generations 4 \
        --test_mode \
        --max_tokens 1024 \
        --max_model_len 2048 \
        --output_path "${ROLLOUT_PATH}"
    echo "=== Rollouts saved to ${ROLLOUT_PATH} ==="
    ;;

train)
    # Run this on a COMPUTE NODE (needs GPU)
    echo "=== Training student: ${STUDENT_DIR} ==="
    python train.py \
        --rollout_path "${ROLLOUT_PATH}" \
        --target_model "${STUDENT_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --num_generations 4 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --max_completion_length 512 \
        --report_to none
    echo "=== Training complete. Model saved to ${OUTPUT_DIR} ==="
    ;;

diagnose)
    # Run diagnostics on rollouts + student model (compute node)
    echo "=== Running diagnostics ==="
    python diagnose.py \
        --rollout_path "${ROLLOUT_PATH}" \
        --target_model "${STUDENT_DIR}" \
        --max_samples 8
    echo "=== Diagnostics complete ==="
    ;;

gpu)
    # Run both rollouts + train on a COMPUTE NODE
    bash "${WORK_DIR}/run_test.sh" rollouts
    bash "${WORK_DIR}/run_test.sh" train
    ;;

*)
    echo "Usage: bash run_test.sh {rollouts|train|gpu|diagnose}"
    echo ""
    echo "  rollouts   - Generate rollouts with teacher (compute node)"
    echo "  train      - Train student with offline GRPO (compute node)"
    echo "  gpu        - Run rollouts + train (compute node)"
    echo "  diagnose   - Run diagnostic checks (compute node)"
    exit 1
    ;;

esac
