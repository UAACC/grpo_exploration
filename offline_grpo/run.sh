#!/bin/bash
#
# Offline GRPO with LoRA
#
# Configure MODEL_DIR, TEACHER_DIR, DATASET_TYPE before submitting:
#   DATASET_TYPE=math sbatch --job-name=offline-grpo-math run.sh train
#
#SBATCH --account=aip-szepesva
#SBATCH --time=4:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e

# ── Configurable ─────────────────────────────────────────────────
DATASET_TYPE="${DATASET_TYPE:-gsm8k}"          # gsm8k | math
MODEL_DIR="${MODEL_DIR:-/scratch/mrli/models/Qwen2.5-0.5B-Instruct}"
TEACHER_DIR="${TEACHER_DIR:-/scratch/mrli/models/Qwen2.5-Math-7B-Instruct}"

WORK_DIR="/project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/offline_grpo"
SCRATCH="/scratch/mrli"
CONFIG_DIR="${WORK_DIR}/configs"
EVAL_SCRIPT="/project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/Math_Verifier/eval_unified.py"

case "${DATASET_TYPE}" in
    math)
        ROLLOUT_PATH="${ROLLOUT:-${SCRATCH}/rollouts/math_teacher/rollouts_full.jsonl}"
        MAX_COMPL="${MAX_COMPL:-2048}"
        ;;
    gsm8k)
        ROLLOUT_PATH="${ROLLOUT:-${SCRATCH}/rollouts/gsm8k_teacher/rollouts_gsm8k.jsonl}"
        MAX_COMPL="${MAX_COMPL:-1024}"
        ;;
    svamp)
        ROLLOUT_PATH="${ROLLOUT:-${SCRATCH}/rollouts/svamp_teacher/rollouts_svamp.jsonl}"
        MAX_COMPL="${MAX_COMPL:-1024}"
        ;;
    asdiv)
        ROLLOUT_PATH="${ROLLOUT:-${SCRATCH}/rollouts/asdiv_teacher/rollouts_asdiv.jsonl}"
        MAX_COMPL="${MAX_COMPL:-1024}"
        ;;
    *)
        echo "Unknown DATASET=${DATASET}"; exit 1
        ;;
esac
OUTPUT_DIR="/scratch/shuai14/checkpoints/offline_grpo_${DATASET_TYPE}"
MERGED_DIR="/scratch/shuai14/merged/offline_grpo_${DATASET_TYPE}_merged}"

# Dataset-specific defaults (overridable via env)
if [ "${DATASET_TYPE}" = "math" ]; then
    MAX_COMPLETION="${MAX_COMPLETION:-2048}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-3072}"
    DATASET_CACHE="${DATASET_CACHE:-/scratch/mrli/datasets/MATH}"
else
    MAX_COMPLETION="${MAX_COMPLETION:-1024}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
    DATASET_CACHE="${DATASET_CACHE:-/scratch/mrli/datasets/GSM8K}"
fi

# ── Activate environment ─────────────────────────────────────────
module load python/3.11 cuda/12.6 arrow opencv
source /project/aip-szepesva/shuai14/verifiers/.venv/bin/activate
export HF_HOME="${SCRATCH}"
export HF_DATASETS_CACHE="${DATASET_CACHE}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=$(cat /project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/RWR-offline/.wandb_key)
export VLLM_WORKER_MULTIPROC_METHOD=spawn
cd "${WORK_DIR}"

mkdir -p logs

case "${1}" in

rollouts)
    echo "=== Generating ${DATASET_TYPE} teacher rollouts ==="
    echo "=== Teacher: ${TEACHER_DIR} ==="
    echo "=== Output: ${ROLLOUT_PATH} ==="
    mkdir -p "$(dirname ${ROLLOUT_PATH})"

    python generate_rollouts.py \
        --model_name "${TEACHER_DIR}" \
        --dataset_type "${DATASET_TYPE}" \
        --split train \
        --num_generations 5 \
        --temperature 0.7 \
        --max_tokens "${MAX_COMPLETION}" \
        --max_model_len "${MAX_MODEL_LEN}" \
        --seed 42 \
        --tensor_parallel_size 4 \
        --gpu_memory_utilization 0.9 \
        --output_path "${ROLLOUT_PATH}"

    echo "=== Rollout generation complete ==="
    ;;

train)
    echo "=== Offline GRPO on ${DATASET_TYPE} ==="
    echo "=== Student: ${MODEL_DIR} ==="
    echo "=== Rollouts: ${ROLLOUT_PATH} ==="
    echo "=== Output: ${OUTPUT_DIR} ==="
    export WANDB_PROJECT="offline-grpo-${DATASET_TYPE}"

    accelerate launch \
        --config_file "${CONFIG_DIR}/accelerate_ddp_4gpu.yaml" \
        train.py \
        --rollout_path "${ROLLOUT_PATH}" \
        --target_model "${MODEL_DIR}" \
        --behavior_model "${TEACHER_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --run_name "offline-grpo-${DATASET_TYPE}-$(date +%Y%m%d_%H%M%S)" \
        --num_generations 5 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 5 \
        --gradient_accumulation_steps 2 \
        --learning_rate 5e-6 \
        --beta 0.1 \
        --max_prompt_length 512 \
        --max_completion_length "${MAX_COMPLETION}" \
        --save_steps 500 \
        --logging_steps 10 \
        --ref_sync_steps 0 \
        --report_to wandb

    echo "=== Training complete ==="
#     ;;

# eval)
    CKPT="${2:-}"
    if [ -n "${CKPT}" ]; then
        EVAL_PATH="${OUTPUT_DIR}/${CKPT}"
    else
        EVAL_PATH="${OUTPUT_DIR}"
    fi
    echo "=== Evaluating on ${DATASET_TYPE}: ${EVAL_PATH} ==="
    python "${EVAL_SCRIPT}" \
        --model_path "${EVAL_PATH}" \
        --base_model "${MODEL_DIR}" \
        --merge_lora \
        --merged_output "${MERGED_DIR}" \
        --dataset_type "${DATASET_TYPE}" \
        --temperature 0.0 \
        --runs 10 \
        --max_tokens "${MAX_COMPLETION}" \
        --max_model_len "${MAX_MODEL_LEN}"
    echo "=== Evaluation complete ==="
    ;;

eval-baseline)
    echo "=== Evaluating baseline on ${DATASET_TYPE}: ${MODEL_DIR} ==="
    python "${EVAL_SCRIPT}" \
        --model_path "${MODEL_DIR}" \
        --dataset_type "${DATASET_TYPE}" \
        --temperature 0.0 \
        --runs 10 \
        --max_tokens "${MAX_COMPLETION}" \
        --max_model_len "${MAX_MODEL_LEN}"
    echo "=== Baseline evaluation complete ==="
    ;;

*)
    echo "Usage: DATASET_TYPE=gsm8k|math sbatch --job-name=<name> run.sh <command>"
    echo ""
    echo "  rollouts       - Generate teacher rollouts (4x L40s TP)"
    echo "  train          - Offline GRPO with LoRA (4x L40s DDP)"
    echo "  eval [ckpt]    - Evaluate trained model (merge LoRA)"
    echo "  eval-baseline  - Evaluate base model (no training)"
    exit 1
    ;;

esac
