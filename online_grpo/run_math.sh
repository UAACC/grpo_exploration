#!/bin/bash
#
# Online GRPO on MATH with Qwen2.5-0.5B-Instruct + LoRA
#   Hardware: 4x L40s with DDP + 1 GPU for vLLM generation (colocate mode)
#
# Usage:
#   sbatch --job-name=online-grpo-math run_math.sh train
#   sbatch --job-name=online-grpo-math-eval --gpus-per-node=l40s:1 --cpus-per-task=16 --mem=64G --time=0:30:00 run_math.sh eval
#
#SBATCH --account=aip-szepesva
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e

WORK_DIR="/project/aip-szepesva/mrli/backup_dongheng/online_grpo"
SCRATCH="/scratch/mrli"
CONFIG_DIR="${WORK_DIR}/configs"

MODEL_DIR="${SCRATCH}/models/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR="${SCRATCH}/checkpoints/online_grpo_math"

# ── Activate environment ─────────────────────────────────────────────
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

case "${1}" in

# ── Training ──────────────────────────────────────────────────────────
train)
    echo "=== Online GRPO on MATH (4x L40s) ==="
    echo "=== Model: ${MODEL_DIR} ==="
    echo "=== Output: ${OUTPUT_DIR} ==="
    export WANDB_PROJECT="online-grpo-math"
    export WANDB_RUN_NAME="qwen05b-math-$(date +%Y%m%d_%H%M%S)"

    mkdir -p "${OUTPUT_DIR}"

    accelerate launch \
        --config_file "${CONFIG_DIR}/accelerate_ddp_4gpu.yaml" \
        train.py \
        --model "${MODEL_DIR}" \
        --dataset_type math \
        --output_dir "${OUTPUT_DIR}" \
        --num_generations 5 \
        --num_train_epochs 15 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 10 \
        --learning_rate 3e-6 \
        --beta 0.001 \
        --temperature 0.7 \
        --max_prompt_length 512 \
        --max_completion_length 2048 \
        --lora_r 32 \
        --lora_alpha 32 \
        --save_steps 200 \
        --logging_steps 10 \
        --report_to wandb

    echo "=== Training complete. Model saved to ${OUTPUT_DIR} ==="
    ;;

# ── Evaluation ────────────────────────────────────────────────────────
eval)
    EVAL_SCRIPT="/project/aip-szepesva/mrli/backup_dongheng/Math_Verifier/eval_unified.py"
    echo "=== Evaluating trained model on MATH (LoRA merged) ==="
    MERGED_DIR="${SCRATCH}/merged/online_grpo_math_merged"
    python "${EVAL_SCRIPT}" \
        --model_path "${OUTPUT_DIR}" \
        --base_model "${MODEL_DIR}" \
        --merge_lora \
        --merged_output "${MERGED_DIR}" \
        --dataset_type math \
        --temperature 0.0 \
        --runs 10 \
        --max_tokens 2048 \
        --max_model_len 3072
    echo "=== Evaluation complete ==="
    ;;

eval-baseline)
    EVAL_SCRIPT="/project/aip-szepesva/mrli/backup_dongheng/Math_Verifier/eval_unified.py"
    echo "=== Evaluating baseline on MATH ==="
    python "${EVAL_SCRIPT}" \
        --model_path "${MODEL_DIR}" \
        --dataset_type math \
        --temperature 0.0 \
        --runs 10 \
        --max_tokens 2048 \
        --max_model_len 3072
    echo "=== Baseline evaluation complete ==="
    ;;

eval-checkpoints)
    EVAL_SCRIPT="/project/aip-szepesva/mrli/backup_dongheng/Math_Verifier/eval_unified.py"
    echo "=== Evaluating multiple checkpoints on MATH ==="
    for step in 1000 2000 3000 5000 7000; do
        ckpt="${OUTPUT_DIR}/checkpoint-${step}"
        if [ -d "${ckpt}" ]; then
            merged="${SCRATCH}/merged/online_grpo_math_merged_step${step}"
            echo "--- Checkpoint step ${step} ---"
            python "${EVAL_SCRIPT}" \
                --model_path "${ckpt}" \
                --base_model "${MODEL_DIR}" \
                --merge_lora \
                --merged_output "${merged}" \
                --dataset_type math \
                --temperature 0.0 \
                --runs 10 \
                --max_tokens 2048 \
                --max_model_len 3072
        else
            echo "--- Checkpoint step ${step}: NOT FOUND, skipping ---"
        fi
    done
    echo "--- Baseline ---"
    python "${EVAL_SCRIPT}" \
        --model_path "${MODEL_DIR}" \
        --dataset_type math \
        --temperature 0.0 \
        --runs 10 \
        --max_tokens 2048 \
        --max_model_len 3072
    echo "=== eval-checkpoints complete ==="
    ;;

*)
    echo "Usage: sbatch --job-name=<name> run_math.sh <command>"
    echo ""
    echo "Training (4x L40s):"
    echo "  train              - Online GRPO with LoRA on MATH"
    echo ""
    echo "Evaluation (1x L40s):"
    echo "  eval               - Evaluate trained model (merge LoRA)"
    echo "  eval-baseline      - Evaluate base model (no training)"
    echo "  eval-checkpoints   - Eval multiple checkpoints + baseline"
    exit 1
    ;;

esac
