#!/bin/bash
#
# DG-offline GRPO on MATH dataset
#
# Usage:
#   sbatch run_math.sh train              # train, then eval final model
#   sbatch run_math.sh eval               # eval latest checkpoint
#   sbatch run_math.sh eval checkpoint-500 # eval a named checkpoint
#   sbatch run_math.sh eval-baseline      # eval base student model
#   DG_ETA=0.5 sbatch run_math.sh train   # custom eta
#   DG_ETA=2.0 sbatch run_math.sh train   # softer gate
#   TRAINING_REGIME=signed_reward LOSS_TYPE=dr_grpo sbatch run_math.sh train
#   LOSS_TYPE=dr_grpo sbatch run_math.sh train # choose grpo or dr_grpo
#   aip-szepesva
#SBATCH --account=aip-szepesva
#SBATCH --job-name=dg-offline-math
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ---- Paths (configurable via env vars) --------------------------------
SCRATCH="${SCRATCH:-/scratch/shuai14}"
WORK_DIR="/project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/DG-offline"
EVAL_SCRIPT="/project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/mixture_grpo/evaluate.py"

STUDENT_MODEL="${STUDENT_MODEL:-/scratch/shuai14/models/Qwen2.5-0.5B}"
TEACHER_MODEL="${TEACHER_MODEL:-/scratch/shuai14/models/Qwen2.5-Math-7B-Instruct}"
ROLLOUT_PATH="${ROLLOUT_PATH:-/scratch/shuai14/rollouts/math_teacher/rollouts_full.jsonl}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH}/checkpoints/DG_offline_math}"
MERGED_DIR="${MERGED_DIR:-${SCRATCH}/merged/DG_offline_math}"

# ---- DG hyperparameters -----------------------------------------------
#if not set, default to 1.0 (no gating)
DG_ETA="${DG_ETA:-1.0}"
DG_GATING="${DG_GATING:-completion}"
TRAINING_REGIME="${TRAINING_REGIME:-current}"
LOSS_TYPE="${LOSS_TYPE:-}"
LOSS_TYPE_ARGS=()
if [ -n "${LOSS_TYPE}" ]; then
    LOSS_TYPE_ARGS=(--loss_type "${LOSS_TYPE}")
fi
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-2048}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-256}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
WANDB_PROJECT="${WANDB_PROJECT:-dg-offline-math}"

# ---- Environment ------------------------------------------------------
module load python/3.11 cuda/12.6 arrow opencv
source /project/aip-szepesva/shuai14/verifiers/.venv/bin/activate
export HF_HOME="/scratch/shuai14"
export HF_DATASETS_CACHE="/scratch/shuai14/datasets/MATH"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

if [ -f "/project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/DG-offline/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat /project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/DG-offline/.wandb_key)
    echo "Loaded Weights & Biases API key from file."
fi

mkdir -p "${WORK_DIR}/logs"

cd "${WORK_DIR}"

echo "=== DG Offline GRPO on MATH ==="
echo "  Student: ${STUDENT_MODEL}"
echo "  Teacher: ${TEACHER_MODEL}"
echo "  Rollouts: ${ROLLOUT_PATH}"
echo "  DG eta: ${DG_ETA}"
echo "  DG gating: ${DG_GATING}"
echo "  Training regime: ${TRAINING_REGIME}"
echo "  Loss type: ${LOSS_TYPE:-train.py default}"
echo "  Output: ${CHECKPOINT_DIR}"

resolve_eval_checkpoint() {
    local ckpt="${1:-latest}"
    local ckpt_path

    if [ "${ckpt}" = "latest" ]; then
        ckpt_path=$(ls -d "${CHECKPOINT_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
        if [ -z "${ckpt_path}" ]; then
            echo "No checkpoint found in ${CHECKPOINT_DIR}; evaluating ${CHECKPOINT_DIR} directly" >&2
            ckpt_path="${CHECKPOINT_DIR}"
        fi
    elif [ -d "${ckpt}" ]; then
        ckpt_path="${ckpt}"
    else
        ckpt_path="${CHECKPOINT_DIR}/${ckpt}"
    fi

    printf "%s\n" "${ckpt_path}"
}

run_math_eval() {
    local ckpt_path="$1"

    echo "=== Evaluating ${ckpt_path} on MATH ==="
    python "${EVAL_SCRIPT}" \
        --model_path "${ckpt_path}" \
        --base_model "${STUDENT_MODEL}" \
        --merge_lora \
        --merged_output "${MERGED_DIR}" \
        --dataset_type math \
        --runs 30 \
        --temperature 0.0 \
        --max_tokens 2048 \
        --max_model_len 3072
    echo "=== Evaluation complete ==="
}

# ---- Train -------------------------------------------------------------
CMD="${1:-train}"
# /project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/DG-offline/
if [ "$CMD" = "train" ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate_ddp_4gpu.yaml \
        train.py \
        --target_model "${STUDENT_MODEL}" \
        --behavior_model "${TEACHER_MODEL}" \
        --rollout_path "${ROLLOUT_PATH}" \
        --output_dir "${CHECKPOINT_DIR}" \
        --wandb_project "${WANDB_PROJECT}" \
        --dg_temperature "${DG_ETA}" \
        --dg_gating "${DG_GATING}" \
        --training_regime "${TRAINING_REGIME}" \
        "${LOSS_TYPE_ARGS[@]}" \
        --learning_rate 3e-6 \
        --beta 0.001 \
        --num_generations 4 \
        --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
        --gradient_accumulation_steps "${GRAD_ACCUM}" \
        --num_train_epochs 1 \
        --max_prompt_length "${MAX_PROMPT_LENGTH}" \
        --max_completion_length "${MAX_COMPLETION_LENGTH}" \
        --max_grad_norm 1.0 \
        --weight_decay 0.01 \
        --warmup_ratio 0.1 \
        --save_steps 500 \
        --logging_steps 5 \
        --lora_r 32 \
        --lora_alpha 32 \
        --seed 42

    echo "=== Training complete ==="
    run_math_eval "${CHECKPOINT_DIR}"

elif [ "$CMD" = "eval" ]; then
    CKPT_PATH=$(resolve_eval_checkpoint "${2:-latest}")
    run_math_eval "${CKPT_PATH}"

elif [ "$CMD" = "eval-baseline" ]; then
    echo "=== Evaluating baseline ${STUDENT_MODEL} on MATH ==="
    python "${EVAL_SCRIPT}" \
        --model_path "${STUDENT_MODEL}" \
        --dataset_type math \
        --runs 30 \
        --temperature 0.0 \
        --max_tokens 2048 \
        --max_model_len 3072
    echo "=== Baseline evaluation complete ==="

else
    echo "Usage: sbatch --job-name=<name> run_math.sh {train|eval [checkpoint|latest]|eval-baseline}"
    exit 1
fi

echo "=== Done ==="
