#!/bin/bash
#
# RWR-offline GRPO on MATH dataset
#
# Usage:
#   sbatch run_math.sh train              # train, then eval final model
#   sbatch run_math.sh eval               # eval latest checkpoint
#   sbatch run_math.sh eval checkpoint-500 # eval a named checkpoint
#   sbatch run_math.sh eval-baseline      # eval base student model
#   DG_ETA=0.5 sbatch run_math.sh train   # custom eta
#   DG_ETA=2.0 sbatch run_math.sh train   # softer gate
#   TRAINING_REGIME=signed_reward LOSS_TYPE=dr_grpo sbatch run_math.sh train
#   LOSS_TYPE=dr_grpo sbatch run_math.sh train # choose grpo, bnpo, dr_grpo, or dapo
#   TRAINING_SIGNAL=advantage sbatch run_math.sh train # choose reward or advantage
#   LEARNING_RATE=1e-6 sbatch run_math.sh train
#SBATCH --account=aip-xt7
#SBATCH --job-name=rwr-offline-math
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
WORK_DIR="/project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/RWR-offline"
EVAL_SCRIPT="/project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/mixture_grpo/evaluate.py"

STUDENT_MODEL="${STUDENT_MODEL:-/scratch/shuai14/models/Qwen2.5-0.5B}"
TEACHER_MODEL="${TEACHER_MODEL:-/scratch/shuai14/models/Qwen2.5-Math-7B-Instruct}"
ROLLOUT_PATH="${ROLLOUT_PATH:-/scratch/shuai14/rollouts/math_teacher/rollouts_full.jsonl}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH}/checkpoints/RWR_offline_math}"
MERGED_DIR="${MERGED_DIR:-${SCRATCH}/merged/RWR_offline_math}"

# ---- DG hyperparameters -----------------------------------------------
#if not set, default to 1.0 (no gating)
DG_ETA="${DG_ETA:-1.0}"
DG_GATING="${DG_GATING:-completion}"
TRAINING_REGIME="${TRAINING_REGIME:-${REWARD_REGIME:-signed_reward}}"
TRAINING_SIGNAL="${TRAINING_SIGNAL:-reward}"
LOSS_TYPE="${LOSS_TYPE:-}"
LOSS_TYPE_ARGS=()
if [ -n "${LOSS_TYPE}" ]; then
    LOSS_TYPE_ARGS=(--loss_type "${LOSS_TYPE}")
fi
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-2048}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-256}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LEARNING_RATE="${LEARNING_RATE:-3e-6}"
WANDB_PROJECT="${WANDB_PROJECT:-rwr-offline-math}"

# ---- Environment ------------------------------------------------------
module load python/3.11 cuda/12.6 arrow opencv
source /project/aip-szepesva/shuai14/verifiers/.venv/bin/activate
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH}/hf_cache}"
export HF_HOME="${CACHE_ROOT}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SCRATCH}/datasets/MATH}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SCRATCH}/triton_cache}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mkdir -p "${HF_MODULES_CACHE}" "${HUGGINGFACE_HUB_CACHE}" "${TRITON_CACHE_DIR}"

if [ -f "/project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/RWR-offline/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat /project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/RWR-offline/.wandb_key)
    echo "Loaded Weights & Biases API key from file."
fi

mkdir -p "${WORK_DIR}/logs"

cd "${WORK_DIR}"

echo "=== RWR Offline GRPO on MATH ==="
echo "  Student: ${STUDENT_MODEL}"
echo "  Teacher: ${TEACHER_MODEL}"
echo "  Rollouts: ${ROLLOUT_PATH}"
echo "  RWR eta: ${DG_ETA}"
echo "  RWR gating: ${DG_GATING}"
echo "  Training regime: ${TRAINING_REGIME}"
echo "  Training signal: ${TRAINING_SIGNAL}"
echo "  Reward regime: ${TRAINING_REGIME}"
echo "  Loss used/requested: ${LOSS_TYPE:-train.py default}"
echo "  Learning rate: ${LEARNING_RATE}"
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
# /project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/RWR-offline/
if [ "$CMD" = "train" ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file /home/shuai14/projects/aip-szepesva/shuai14/DG_LLM/grpo_exploration/DG-offline/configs/accelerate_ddp_4gpu.yaml \
        train.py \
        --target_model "${STUDENT_MODEL}" \
        --behavior_model "${TEACHER_MODEL}" \
        --rollout_path "${ROLLOUT_PATH}" \
        --output_dir "${CHECKPOINT_DIR}" \
        --wandb_project "${WANDB_PROJECT}" \
        --dg_temperature "${DG_ETA}" \
        --dg_gating "${DG_GATING}" \
        --training_regime "${TRAINING_REGIME}" \
        --training_signal "${TRAINING_SIGNAL}" \
        "${LOSS_TYPE_ARGS[@]}" \
        --learning_rate "${LEARNING_RATE}" \
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
