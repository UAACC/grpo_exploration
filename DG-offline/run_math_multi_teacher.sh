#!/bin/bash
#
# DG-offline GRPO on MATH with multiple teacher rollout files.
#
# Usage:
#   sbatch run_math_multi_teacher.sh train
#   sbatch run_math_multi_teacher.sh eval
#   sbatch run_math_multi_teacher.sh eval checkpoint-500
#   sbatch run_math_multi_teacher.sh eval-baseline
#   ROLLOUT_PATHS="/path/a.jsonl /path/b.jsonl" sbatch run_math_multi_teacher.sh train
#   TRAINING_REGIME=signed_reward LOSS_TYPE=dr_grpo sbatch run_math_multi_teacher.sh train
#
#SBATCH --account=aip-szepesva
#SBATCH --job-name=dg-offline-math-mt
#SBATCH --time=15:00:00
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
TEACHER_MODELS="${TEACHER_MODELS:-/scratch/shuai14/models/Qwen2.5-Math-7B-Instruct /scratch/shuai14/models/Qwen2.5-Math-1.5B-Instruct}"
ROLLOUT_PATHS="${ROLLOUT_PATHS:-${SCRATCH}/rollouts/math_teacher/rollouts_full.jsonl ${SCRATCH}/rollouts/math_teacher/rollouts_math_Qwen2.5-Math-1.5B-Instruct_0.6.jsonl}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH}/checkpoints/DG_offline_math_multi_teacher}"
MERGED_DIR="${MERGED_DIR:-${SCRATCH}/merged/DG_offline_math_multi_teacher}"

# ---- DG hyperparameters -----------------------------------------------
DG_ETA="${DG_ETA:-1.0}"
DG_GATING="${DG_GATING:-completion}"
TRAINING_REGIME="${TRAINING_REGIME:-current}"
LOSS_TYPE="${LOSS_TYPE:-}"
LOSS_TYPE_ARGS=()
if [ -n "${LOSS_TYPE}" ]; then
    LOSS_TYPE_ARGS=(--loss_type "${LOSS_TYPE}")
fi
NUM_GENERATIONS="${NUM_GENERATIONS:-}"
NUM_GENERATIONS_ARGS=()
if [ -n "${NUM_GENERATIONS}" ]; then
    NUM_GENERATIONS_ARGS=(--num_generations "${NUM_GENERATIONS}")
fi
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-2048}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-256}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-3}"
GRAD_ACCUM="${GRAD_ACCUM:-3}"
WANDB_PROJECT="${WANDB_PROJECT:-dg-offline-math-multi-teacher}"

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
read -r -a ROLLOUT_PATH_ARGS <<< "${ROLLOUT_PATHS}"

if [ "${#ROLLOUT_PATH_ARGS[@]}" -lt 2 ]; then
    echo "Warning: ROLLOUT_PATHS has fewer than 2 files; this will behave like single-teacher training." >&2
fi

for rollout_path in "${ROLLOUT_PATH_ARGS[@]}"; do
    if [ ! -f "${rollout_path}" ]; then
        echo "ERROR: rollout file not found: ${rollout_path}" >&2
        exit 1
    fi
done

echo "=== DG Offline GRPO on MATH (multi-teacher rollouts) ==="
echo "  Student: ${STUDENT_MODEL}"
echo "  Teachers: ${TEACHER_MODELS}"
echo "  Rollouts: ${ROLLOUT_PATHS}"
echo "  Num generations: ${NUM_GENERATIONS:-inferred from rollout files}"
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
if [ "$CMD" = "train" ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate_ddp_4gpu.yaml \
        train.py \
        --target_model "${STUDENT_MODEL}" \
        --behavior_model "${TEACHER_MODELS}" \
        --rollout_path "${ROLLOUT_PATH_ARGS[@]}" \
        --output_dir "${CHECKPOINT_DIR}" \
        --wandb_project "${WANDB_PROJECT}" \
        --dg_temperature "${DG_ETA}" \
        --dg_gating "${DG_GATING}" \
        --training_regime "${TRAINING_REGIME}" \
        "${LOSS_TYPE_ARGS[@]}" \
        --learning_rate 3e-6 \
        --beta 0.001 \
        "${NUM_GENERATIONS_ARGS[@]}" \
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
    echo "Usage: sbatch run_math_multi_teacher.sh {train|eval [checkpoint|latest]|eval-baseline}"
    exit 1
fi

echo "=== Done ==="
