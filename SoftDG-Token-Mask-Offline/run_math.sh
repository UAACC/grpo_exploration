#!/bin/bash
#
# SoftDG-Token-Mask-Offline on MATH (bad-teacher rollout, per-token PG masking).
#
# Usage:
#   sbatch run_math.sh train    -- full training run
#   sbatch run_math.sh sanity   -- quick sanity check (32 effective tokens)
#   sbatch run_math.sh eval
#   sbatch run_math.sh eval checkpoint-500
#   sbatch run_math.sh eval-baseline
#
# Key env-var overrides:
#   REWARD_CODING=signed|zero_two
#   TRAINING_SIGNAL=raw_reward|advantage
#   LOSS_TYPE=dr_grpo|grpo
#   SOFTDG_GATE_THRESHOLD=0.5
#   TARGET_EFFECTIVE_TOKENS=   (empty = computed from rollout; set for sanity)
#   DG_ETA=1.0
#   STUDENT_MODEL=/scratch/shuai14/models/Qwen2.5-0.5B
#   ROLLOUT_PATH=...
#
# Variant A (primary): REWARD_CODING=signed TRAINING_SIGNAL=raw_reward LOSS_TYPE=dr_grpo
# Variant B:           REWARD_CODING=zero_two TRAINING_SIGNAL=advantage LOSS_TYPE=grpo
# Variant C:           REWARD_CODING=signed TRAINING_SIGNAL=advantage LOSS_TYPE=dr_grpo
#
#SBATCH --account=aip-szepesva
#SBATCH --job-name=softdg-tm-math
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ---- Paths ---------------------------------------------------------------
SCRATCH="${SCRATCH:-/scratch/shuai14}"
WORK_DIR="/project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/SoftDG-Token-Mask-Offline"
EVAL_SCRIPT="/project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/mixture_grpo/evaluate.py"

STUDENT_MODEL="${STUDENT_MODEL:-${SCRATCH}/models/Qwen2.5-0.5B}"
ROLLOUT_PATH="${ROLLOUT_PATH:-${SCRATCH}/rollouts/math_teacher/rollouts_math_Qwen2.5-0.5B-Instruct_0.6_pick4.jsonl}"
BEHAVIOR_MODEL="${BEHAVIOR_MODEL:-Qwen2.5-0.5B-Instruct}"

# SoftDG token-mask hyperparams
REWARD_CODING="${REWARD_CODING:-signed}"
TRAINING_SIGNAL="${TRAINING_SIGNAL:-raw_reward}"
LOSS_TYPE="${LOSS_TYPE:-dr_grpo}"
SOFTDG_GATE_THRESHOLD="${SOFTDG_GATE_THRESHOLD:-0.5}"
DG_ETA="${DG_ETA:-1.0}"
# TARGET_EFFECTIVE_TOKENS: empty = use token budget from rollout
TARGET_EFFECTIVE_TOKENS="${TARGET_EFFECTIVE_TOKENS:-}"

# eta tag for naming: replace '.' with 'p' (e.g. 0.75 -> 0p75)
ETA_TAG=$(printf "%s" "${DG_ETA}" | tr '.' 'p')

# Output paths — include eta in all run identifiers
RUN_TAG="${REWARD_CODING}_${TRAINING_SIGNAL}_${LOSS_TYPE}_eta${ETA_TAG}_thr${SOFTDG_GATE_THRESHOLD}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH}/checkpoints/softdg_tm_math_${RUN_TAG}}"
MERGED_DIR="${MERGED_DIR:-${SCRATCH}/merged/softdg_tm_math_${RUN_TAG}}"

# Training hyperparams
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-3}"
GRAD_ACCUM="${GRAD_ACCUM:-3}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-2048}"
WANDB_PROJECT="${WANDB_PROJECT:-softdg-token-mask-math}"

# ---- Environment ---------------------------------------------------------
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
    echo "Loaded W&B API key."
fi

mkdir -p "${WORK_DIR}/logs"
mkdir -p "${WORK_DIR}/outputs"

cd "${WORK_DIR}"

if [ ! -f "${ROLLOUT_PATH}" ]; then
    echo "ERROR: rollout file not found: ${ROLLOUT_PATH}" >&2
    exit 1
fi

echo "=== SoftDG-Token-Mask-Offline on MATH ==="
echo "  Student:                  ${STUDENT_MODEL}"
echo "  Rollout:                  ${ROLLOUT_PATH}"
echo "  reward_coding:            ${REWARD_CODING}"
echo "  training_signal:          ${TRAINING_SIGNAL}"
echo "  loss_type:                ${LOSS_TYPE}"
echo "  softdg_gate_threshold:    ${SOFTDG_GATE_THRESHOLD}"
echo "  target_effective_tokens:  ${TARGET_EFFECTIVE_TOKENS:-<from rollout>}"
echo "  dg_eta:                   ${DG_ETA}"
echo "  Output:                   ${CHECKPOINT_DIR}"

# ---- Helpers -----------------------------------------------------------
resolve_eval_checkpoint() {
    local ckpt="${1:-latest}"
    if [ "${ckpt}" = "latest" ]; then
        local ckpt_path
        ckpt_path=$(ls -d "${CHECKPOINT_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
        if [ -z "${ckpt_path}" ]; then
            echo "No checkpoint found in ${CHECKPOINT_DIR}; evaluating ${CHECKPOINT_DIR} directly" >&2
            ckpt_path="${CHECKPOINT_DIR}"
        fi
        printf "%s\n" "${ckpt_path}"
    elif [ -d "${ckpt}" ]; then
        printf "%s\n" "${ckpt}"
    else
        printf "%s\n" "${CHECKPOINT_DIR}/${ckpt}"
    fi
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

save_result() {
    local ckpt_path="$1"
    local result_tag="$2"
    local result_file="${WORK_DIR}/outputs/${result_tag}_eval.txt"
    echo "Results tag: ${result_tag}" > "${result_file}"
    echo "Checkpoint: ${ckpt_path}" >> "${result_file}"
    echo "Config: reward_coding=${REWARD_CODING} training_signal=${TRAINING_SIGNAL} threshold=${SOFTDG_GATE_THRESHOLD} eta=${DG_ETA}" >> "${result_file}"
    echo "Date: $(date)" >> "${result_file}"
}

build_token_args() {
    if [ -n "${TARGET_EFFECTIVE_TOKENS}" ]; then
        echo "--target_effective_tokens ${TARGET_EFFECTIVE_TOKENS}"
    fi
}

# ---- Dispatch ----------------------------------------------------------
CMD="${1:-train}"

if [ "$CMD" = "train" ]; then
    TOKEN_ARGS=$(build_token_args)
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
        --config_file configs/accelerate_ddp_4gpu.yaml \
        train.py \
        --target_model "${STUDENT_MODEL}" \
        --behavior_model "${BEHAVIOR_MODEL}" \
        --rollout_path "${ROLLOUT_PATH}" \
        --output_dir "${CHECKPOINT_DIR}" \
        --wandb_project "${WANDB_PROJECT}" \
        --reward_coding "${REWARD_CODING}" \
        --training_signal "${TRAINING_SIGNAL}" \
        --loss_type "${LOSS_TYPE}" \
        --softdg_gate_threshold "${SOFTDG_GATE_THRESHOLD}" \
        --dg_temperature "${DG_ETA}" \
        ${TOKEN_ARGS} \
        --learning_rate 3e-6 \
        --beta 0.001 \
        --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
        --gradient_accumulation_steps "${GRAD_ACCUM}" \
        --num_train_epochs 20 \
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
    save_result "${CHECKPOINT_DIR}" "${RUN_TAG}"

elif [ "$CMD" = "sanity" ]; then
    # Tiny run: max 2 steps or 32 effective tokens — verifies no hang and PG mask is nontrivial
    SANITY_DIR="${SCRATCH}/checkpoints/softdg_tm_sanity_${RUN_TAG}"
    echo "=== Sanity check (32 effective tokens) ==="
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
        --config_file configs/accelerate_ddp_4gpu.yaml \
        train.py \
        --target_model "${STUDENT_MODEL}" \
        --rollout_path "${ROLLOUT_PATH}" \
        --output_dir "${SANITY_DIR}" \
        --report_to none \
        --reward_coding "${REWARD_CODING}" \
        --training_signal "${TRAINING_SIGNAL}" \
        --loss_type "${LOSS_TYPE}" \
        --softdg_gate_threshold "${SOFTDG_GATE_THRESHOLD}" \
        --dg_temperature "${DG_ETA}" \
        --target_effective_tokens 32 \
        --learning_rate 3e-6 \
        --beta 0.001 \
        --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
        --gradient_accumulation_steps "${GRAD_ACCUM}" \
        --num_train_epochs 1 \
        --max_completion_length "${MAX_COMPLETION_LENGTH}" \
        --max_grad_norm 1.0 \
        --weight_decay 0.01 \
        --warmup_ratio 0.0 \
        --save_steps 9999 \
        --logging_steps 1 \
        --lora_r 32 \
        --lora_alpha 32 \
        --seed 42
    echo "=== Sanity check complete — check logs for: PG mask nontrivial, KL active, token counter increments ==="

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

else
    echo "Usage: sbatch run_math.sh {train|sanity|eval [checkpoint|latest]|eval-baseline}"
    exit 1
fi

echo "=== Done ==="
