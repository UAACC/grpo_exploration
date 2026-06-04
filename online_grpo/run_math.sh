#!/bin/bash
#
# Online GRPO on MATH with Qwen2.5-0.5B-Instruct + LoRA
#   Hardware: 4x L40s with DDP + 1 GPU for vLLM generation (colocate mode)
#
# Usage:
#   sbatch run_math.sh train              # train, then eval final model
#   sbatch run_math.sh eval               # eval latest checkpoint
#   sbatch run_math.sh eval checkpoint-500 # eval a named checkpoint
#   sbatch run_math.sh eval-baseline      # eval base model
#
#SBATCH --account=aip-szepesva
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ---- Paths (configurable via env vars) --------------------------------
SCRATCH="${SCRATCH:-/scratch/shuai14}"
WORK_DIR="/project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/online_grpo"
CONFIG_DIR="${WORK_DIR}/configs"
EVAL_SCRIPT="/project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/mixture_grpo/evaluate.py"

MODEL_DIR="${MODEL_DIR:-${SCRATCH}/models/Qwen2.5-0.5B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH}/checkpoints/online_grpo_math}"
MERGED_DIR="${MERGED_DIR:-${SCRATCH}/merged/online_grpo_math}"

# ---- Environment ------------------------------------------------------
module load python/3.11 cuda/12.6 arrow opencv
source /project/aip-szepesva/shuai14/verifiers/.venv/bin/activate
export HF_HOME="${SCRATCH}"
export HF_DATASETS_CACHE="${SCRATCH}/datasets/MATH"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

if [ -f "${WORK_DIR}/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "${WORK_DIR}/.wandb_key")
    echo "Loaded Weights & Biases API key from file."
elif [ -f "/project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/DG-offline/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat /project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/DG-offline/.wandb_key)
    echo "Loaded Weights & Biases API key from DG-offline."
fi

mkdir -p "${WORK_DIR}/logs"
cd "${WORK_DIR}"

resolve_eval_checkpoint() {
    local ckpt="${1:-latest}"
    local ckpt_path

    if [ "${ckpt}" = "latest" ]; then
        ckpt_path=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
        if [ -z "${ckpt_path}" ]; then
            echo "No checkpoint found in ${OUTPUT_DIR}; evaluating ${OUTPUT_DIR} directly" >&2
            ckpt_path="${OUTPUT_DIR}"
        fi
    elif [ -d "${ckpt}" ]; then
        ckpt_path="${ckpt}"
    else
        ckpt_path="${OUTPUT_DIR}/${ckpt}"
    fi

    printf "%s\n" "${ckpt_path}"
}

run_math_eval() {
    local ckpt_path="$1"

    echo "=== Evaluating ${ckpt_path} on MATH ==="
    python "${EVAL_SCRIPT}" \
        --model_path "${ckpt_path}" \
        --base_model "${MODEL_DIR}" \
        --merge_lora \
        --merged_output "${MERGED_DIR}" \
        --dataset_type math \
        --runs 30 \
        --temperature 0.0 \
        --max_tokens 2048 \
        --max_model_len 3072
    echo "=== Evaluation complete ==="
}

case "${1:-train}" in

# ---- Training ---------------------------------------------------------
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
        --num_train_epochs 1 \
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
    run_math_eval "${OUTPUT_DIR}"
    ;;

# ---- Evaluation -------------------------------------------------------
eval)
    CKPT_PATH=$(resolve_eval_checkpoint "${2:-latest}")
    run_math_eval "${CKPT_PATH}"
    ;;

eval-baseline)
    echo "=== Evaluating baseline on MATH ==="
    python "${EVAL_SCRIPT}" \
        --model_path "${MODEL_DIR}" \
        --dataset_type math \
        --temperature 0.0 \
        --runs 30 \
        --max_tokens 2048 \
        --max_model_len 3072
    echo "=== Baseline evaluation complete ==="
    ;;

eval-checkpoints)
    echo "=== Evaluating multiple checkpoints on MATH ==="
    for step in 1000 2000 3000 5000 7000; do
        ckpt="${OUTPUT_DIR}/checkpoint-${step}"
        if [ -d "${ckpt}" ]; then
            merged="${MERGED_DIR}_step${step}"
            echo "--- Checkpoint step ${step} ---"
            python "${EVAL_SCRIPT}" \
                --model_path "${ckpt}" \
                --base_model "${MODEL_DIR}" \
                --merge_lora \
                --merged_output "${merged}" \
                --dataset_type math \
                --temperature 0.0 \
                --runs 30 \
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
        --runs 30 \
        --max_tokens 2048 \
        --max_model_len 3072
    echo "=== eval-checkpoints complete ==="
    ;;

*)
    echo "Usage: sbatch --job-name=<name> run_math.sh <command>"
    echo ""
    echo "Training (4x L40s):"
    echo "  train              - Online GRPO with LoRA on MATH, then eval"
    echo ""
    echo "Evaluation (1x L40s):"
    echo "  eval [ckpt|latest] - Evaluate trained model (merge LoRA)"
    echo "  eval-baseline      - Evaluate base model (no training)"
    echo "  eval-checkpoints   - Eval multiple checkpoints + baseline"
    exit 1
    ;;

esac

echo "=== Done ==="
