#!/bin/bash
#
# BC correct-only: train 0.5B student on CORRECT teacher completions only.
#
# Purpose: Test whether BC degradation is caused by learning from incorrect
# completions. If correct-only BC improves the student, the incorrect
# completions were the problem. If it still degrades, the issue is more
# fundamental.
#
# Usage:
#   sbatch --job-name=bc-correct run_bc_correct_only_math.sh train
#   sbatch --job-name=bc-correct-eval run_bc_correct_only_math.sh eval
#   sbatch --job-name=bc-correct-eval run_bc_correct_only_math.sh eval checkpoint-200
#   sbatch --job-name=bc-correct-baseline run_bc_correct_only_math.sh eval-baseline
#
#SBATCH --account=aip-xt7
#SBATCH --time=06:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=logs/bc_math/%x-%j.out
#SBATCH --error=logs/bc_math/%x-%j.err

set -euo pipefail

SCRATCH="${SCRATCH:-/scratch/shuai14}"
WORK_DIR="/project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/bc"
EVAL_SCRIPT="/project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/mixture_grpo/evaluate.py"

MODEL_DIR="${MODEL_DIR:-${SCRATCH}/models/Qwen2.5-0.5B}"
ROLLOUT_PATH="${ROLLOUT_PATH:-${SCRATCH}/rollouts/math_teacher/rollouts_full.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH}/checkpoints/bc_math_correct_only}"
MERGED_DIR="${MERGED_DIR:-${SCRATCH}/merged/bc_math_correct_only}"
MAX_LENGTH="${MAX_LENGTH:-2304}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
WANDB_RUN_NAME_OVERRIDE="${WANDB_RUN_NAME:-}"

# ---- Environment ------------------------------------------------------
module load python/3.11 cuda/12.6 arrow opencv
source /project/aip-szepesva/shuai14/verifiers/.venv/bin/activate
export HF_HOME="${SCRATCH}"
export HF_DATASETS_CACHE="${SCRATCH}/datasets/MATH"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="/project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/offline_grpo:${PYTHONPATH:-}"
if [ -f "/project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/DG-offline/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat /project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/DG-offline/.wandb_key)
    echo "Loaded Weights & Biases API key from file."
fi
export WANDB_PROJECT="offline-grpo-math"
if [ -n "${WANDB_RUN_NAME_OVERRIDE}" ]; then
    export WANDB_RUN_NAME="${WANDB_RUN_NAME_OVERRIDE}"
else
    export WANDB_RUN_NAME="bc-0.5B-correct-only-$(date +%m%d)"
fi

cd "${WORK_DIR}"
mkdir -p logs/bc_math

METRICS_FILE="${OUTPUT_DIR}/training_metrics.jsonl"

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

echo "=== BC Correct-Only: 0.5B Student on Correct Teacher Rollouts ==="
echo "  Model: ${MODEL_DIR}"
echo "  Rollouts: ${ROLLOUT_PATH}"
echo "  Filter: --filter_correct_only"
echo "  Output: ${OUTPUT_DIR}"
echo "  Merged eval output: ${MERGED_DIR}"
echo "  Metrics: ${METRICS_FILE}"
echo "  wandb: ${WANDB_PROJECT} / ${WANDB_RUN_NAME}"
echo ""
echo "  Hyperparameters (aligned with offline GRPO controlled):"
echo "    lr=3e-6, max_grad_norm=1.0, weight_decay=0.01"
echo "    epochs=1, per_device_batch=4, grad_accum=2"
echo "    max_length=2304 (256 prompt + 2048 completion)"
echo "    lora_r=32, lora_alpha=32"
echo "    Loss: cross-entropy on correct completion tokens only"

CMD="${1:-train}"
if [ "${CMD}" = "train" ]; then
    accelerate launch \
        --config_file "${WORK_DIR}/configs/accelerate_ddp_4gpu.yaml" \
        train_bc.py \
        --rollout_path "${ROLLOUT_PATH}" \
        --target_model "${MODEL_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --run_name "${WANDB_RUN_NAME}" \
        --filter_correct_only \
        --max_length "${MAX_LENGTH}" \
        --num_train_epochs 1 \
        --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
        --gradient_accumulation_steps "${GRAD_ACCUM}" \
        --learning_rate 3e-6 \
        --max_grad_norm 1.0 \
        --weight_decay 0.01 \
        --lora_r 32 \
        --lora_alpha 32 \
        --save_steps 200 \
        --logging_steps 5 \
        --report_to wandb

    echo "=== Training complete ==="
    echo ""
    echo "=== Metrics saved to: ${METRICS_FILE} ==="
    echo "Quick analysis:"
    python3 -c "
import json
records = [json.loads(l) for l in open('${METRICS_FILE}')]
print(f'Total logged steps: {len(records)}')
if records:
    first = records[0]
    last = records[-1]
    for key in ['loss', 'train_loss', 'grad_norm']:
        v0 = first.get(key, 'N/A')
        v1 = last.get(key, 'N/A')
        print(f'  {key}: {v0} -> {v1}')
"
    run_math_eval "${OUTPUT_DIR}"

elif [ "${CMD}" = "eval" ]; then
    CKPT_PATH=$(resolve_eval_checkpoint "${2:-latest}")
    run_math_eval "${CKPT_PATH}"

elif [ "${CMD}" = "eval-baseline" ]; then
    echo "=== Evaluating baseline ${MODEL_DIR} on MATH ==="
    python "${EVAL_SCRIPT}" \
        --model_path "${MODEL_DIR}" \
        --dataset_type math \
        --runs 30 \
        --temperature 0.0 \
        --max_tokens 2048 \
        --max_model_len 3072 \
        --gpu_memory_utilization "${EVAL_GPU_MEMORY_UTILIZATION:-0.8}"
    echo "=== Baseline evaluation complete ==="

else
    echo "Usage: sbatch --job-name=<name> run_bc_correct_only_math.sh {train|eval [checkpoint|latest]|eval-baseline}"
    exit 1
fi

echo "=== Done ==="
