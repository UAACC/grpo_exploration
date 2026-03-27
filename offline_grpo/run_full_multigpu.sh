#!/bin/bash
#
# Full-scale offline GRPO with multi-GPU training
#   Teacher (behavior model): Qwen2.5-Math-7B-Instruct
#   Student (target model):   Qwen2.5-0.5B-Instruct
#   Data: full MATH training set (12,000 problems)
#   Training: 4x L40s with DDP
#
# Experiment variants:
#   A) ref_sync=0   — reference = original base model (current behavior)
#   B) ref_sync=16  — reference = student from 16 steps ago
#   C) ref_sync=1   — reference = student from 1 step ago
#
# Usage (from login node):
#   sbatch --job-name=grpo-4gpu-refsync0   run_full_multigpu.sh train-refsync0
#   sbatch --job-name=grpo-4gpu-refsync16  run_full_multigpu.sh train-refsync16
#   sbatch --job-name=grpo-4gpu-refsync1   run_full_multigpu.sh train-refsync1
#   sbatch --job-name=grpo-eval-refsyncN   run_full_multigpu.sh eval-refsyncN
#
#SBATCH --account=aip-szepesva
#SBATCH --time=4:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=logs/exp03_multigpu_refsync/%x-%j.out
#SBATCH --error=logs/exp03_multigpu_refsync/%x-%j.err

set -e

WORK_DIR="/project/aip-szepesva/mrli/backup_dongheng/offline_grpo"
SCRATCH="/scratch/mrli"
CONFIG_DIR="${WORK_DIR}/configs"

# Teacher (behavior model) — generates rollouts
TEACHER_DIR="${SCRATCH}/models/Qwen2.5-Math-7B-Instruct"

# Student (target model) — trained with offline GRPO
STUDENT_DIR="${SCRATCH}/models/Qwen2.5-0.5B-Instruct"

# Reuse existing rollouts from exp02 (already generated)
ROLLOUT_PATH="${SCRATCH}/rollouts/math_teacher/rollouts_full.jsonl"

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
cd "${WORK_DIR}"

# ── Common training function ─────────────────────────────────────
run_train() {
    local ref_sync="$1"
    local output_dir="$2"
    local run_name="$3"

    echo "=== Multi-GPU Training (DDP, 4x L40s) ==="
    echo "=== ref_sync_steps: ${ref_sync} ==="
    echo "=== Rollouts: ${ROLLOUT_PATH} ==="
    echo "=== Output: ${output_dir} ==="
    export WANDB_PROJECT="offline-grpo"
    export WANDB_RUN_NAME="${run_name}"

    accelerate launch \
        --config_file "${CONFIG_DIR}/accelerate_ddp_4gpu.yaml" \
        train.py \
        --rollout_path "${ROLLOUT_PATH}" \
        --target_model "${STUDENT_DIR}" \
        --behavior_model "${TEACHER_DIR}" \
        --output_dir "${output_dir}" \
        --num_generations 4 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --max_completion_length 786 \
        --save_steps 500 \
        --logging_steps 10 \
        --ref_sync_steps "${ref_sync}" \
        --report_to wandb

    echo "=== Training complete. Model saved to ${output_dir} ==="
}

# ── Common eval function ─────────────────────────────────────────
run_eval() {
    local output_dir="$1"
    local merged_dir="$2"

    echo "=== Evaluating trained model (LoRA merged) ==="
    echo "=== Checkpoint: ${output_dir} ==="
    python evaluate.py \
        --model_path "${output_dir}" \
        --base_model "${STUDENT_DIR}" \
        --merge_lora \
        --merged_output "${merged_dir}" \
        --runs 1
    echo "=== Evaluation complete ==="
}

case "${1}" in

# ── Training variants ────────────────────────────────────────────

train-refsync0)
    run_train 0 \
        "${SCRATCH}/checkpoints/offline_grpo_math_refsync0" \
        "qwen05b-4gpu-refsync0-$(date +%Y%m%d_%H%M%S)"
    ;;

train-refsync16)
    run_train 16 \
        "${SCRATCH}/checkpoints/offline_grpo_math_refsync16" \
        "qwen05b-4gpu-refsync16-$(date +%Y%m%d_%H%M%S)"
    ;;

train-refsync1)
    run_train 1 \
        "${SCRATCH}/checkpoints/offline_grpo_math_refsync1" \
        "qwen05b-4gpu-refsync1-$(date +%Y%m%d_%H%M%S)"
    ;;

# ── Evaluation variants ─────────────────────────────────────────

eval-refsync0)
    run_eval \
        "${SCRATCH}/checkpoints/offline_grpo_math_refsync0" \
        "${SCRATCH}/merged/offline_grpo_math_refsync0_merged"
    ;;

eval-refsync16)
    run_eval \
        "${SCRATCH}/checkpoints/offline_grpo_math_refsync16" \
        "${SCRATCH}/merged/offline_grpo_math_refsync16_merged"
    ;;

eval-refsync1)
    run_eval \
        "${SCRATCH}/checkpoints/offline_grpo_math_refsync1" \
        "${SCRATCH}/merged/offline_grpo_math_refsync1_merged"
    ;;

eval-baseline)
    echo "=== Evaluating baseline (no LoRA) ==="
    python evaluate.py \
        --model_path "${STUDENT_DIR}" \
        --runs 1
    echo "=== Baseline evaluation complete ==="
    ;;

# ── Reliable evaluation (30 runs) ─────────────────────────────

eval30-temp06)
    echo "=== Reliable eval: temp=0.6, 30 runs ==="
    for variant in refsync0 refsync16 refsync1; do
        merged="${SCRATCH}/merged/offline_grpo_math_${variant}_merged"
        echo "--- Evaluating ${variant} (merged: ${merged}) ---"
        python evaluate.py \
            --model_path "${merged}" \
            --temperature 0.6 \
            --runs 30
    done
    echo "--- Evaluating baseline ---"
    python evaluate.py \
        --model_path "${STUDENT_DIR}" \
        --temperature 0.6 \
        --runs 30
    echo "=== eval30-temp06 complete ==="
    ;;

eval30-temp0)
    echo "=== Reliable eval: temp=0.0, 30 runs ==="
    for variant in refsync0 refsync16 refsync1; do
        merged="${SCRATCH}/merged/offline_grpo_math_${variant}_merged"
        echo "--- Evaluating ${variant} (merged: ${merged}) ---"
        python evaluate.py \
            --model_path "${merged}" \
            --temperature 0.0 \
            --runs 30
    done
    echo "--- Evaluating baseline ---"
    python evaluate.py \
        --model_path "${STUDENT_DIR}" \
        --temperature 0.0 \
        --runs 30
    echo "=== eval30-temp0 complete ==="
    ;;

*)
    echo "Usage: sbatch --job-name=<name> run_full_multigpu.sh <command>"
    echo ""
    echo "Training (4x L40s, DDP):"
    echo "  train-refsync0   - Reference = original base model (no sync)"
    echo "  train-refsync16  - Reference synced every 16 steps"
    echo "  train-refsync1   - Reference synced every step"
    echo ""
    echo "Evaluation (1x L40s):"
    echo "  eval-refsync0    - Evaluate refsync0 model"
    echo "  eval-refsync16   - Evaluate refsync16 model"
    echo "  eval-refsync1    - Evaluate refsync1 model"
    echo "  eval-baseline    - Evaluate base student (no training)"
    echo "  eval30-temp06    - All models, temp=0.6, 30 runs"
    echo "  eval30-temp0     - All models, temp=0.0, 30 runs"
    exit 1
    ;;

esac
