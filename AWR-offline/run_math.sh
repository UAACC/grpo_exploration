#!/bin/bash
#
# AWR-offline GRPO on MATH dataset
#
# Usage:
#   sbatch run_math.sh                    # default eta=1.0
#   DG_ETA=0.5 sbatch run_math.sh        # custom eta
#   DG_ETA=2.0 sbatch run_math.sh        # softer gate
#   aip-szepesva aip-xt7
#SBATCH --account=aip-szepesva
#SBATCH --job-name=AWR-offline-math
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
WORK_DIR="/project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/AWR-offline"
EVAL_SCRIPT="/project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/mixture_grpo/evaluate.py"

STUDENT_MODEL="${STUDENT_MODEL:-/scratch/mrli/models/Qwen2.5-0.5B-Instruct}"
TEACHER_MODEL="${TEACHER_MODEL:-/scratch/mrli/models/Qwen2.5-Math-7B-Instruct}"
ROLLOUT_PATH="${ROLLOUT_PATH:-/scratch/mrli/rollouts/math_teacher/rollouts_full.jsonl}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH}/checkpoints/AWR_offline_math}"
MERGED_DIR="${MERGED_DIR:-${SCRATCH}/merged/AWR_offline_math}"

# ---- DG hyperparameters -----------------------------------------------
#if not set, default to 1.0 (no gating)
DG_ETA="${DG_ETA:-1.0}"
DG_GATING="${DG_GATING:-completion}"

# ---- Environment ------------------------------------------------------
module load python/3.11 cuda/12.6 arrow opencv
source /project/aip-szepesva/shuai14/verifiers/.venv/bin/activate
export HF_HOME="/scratch/mrli"
export HF_DATASETS_CACHE="/scratch/mrli/datasets/MATH"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

if [ -f "/project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/RWR-offline/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat /project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/RWR-offline/.wandb_key)
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
echo "  Output: ${CHECKPOINT_DIR}"

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
        --wandb_project "rwr-offline-math" \
        --dg_temperature "${DG_ETA}" \
        --dg_gating "${DG_GATING}" \
        --learning_rate 3e-6 \
        --beta 0.001 \
        --num_generations 4 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 2 \
        --num_train_epochs 1 \
        --max_completion_length 2048 \
        --max_grad_norm 1.0 \
        --weight_decay 0.01 \
        --warmup_ratio 0.1 \
        --save_steps 500 \
        --logging_steps 5 \
        --lora_r 32 \
        --lora_alpha 32 \
        --seed 42

    echo "=== Training complete ==="

elif [ "$CMD" = "eval" ]; then
    CKPT="${2:-${CHECKPOINT_DIR}}"
    echo "=== Evaluating ${CKPT} on MATH ==="
    python "${EVAL_SCRIPT}" \
        --model_path "${CKPT}" \
        --base_model "${STUDENT_MODEL}" \
        --merge_lora \
        --merged_output "${MERGED_DIR}" \
        --dataset_type math \
        --runs 5 \
        --temperature 0.0 \
        --max_tokens 2048 \
        --max_model_len 3072

elif [ "$CMD" = "eval-baseline" ]; then
    echo "=== Evaluating baseline ${STUDENT_MODEL} on MATH ==="
    python "${EVAL_SCRIPT}" \
        --model_path "${STUDENT_MODEL}" \
        --dataset_type math \
        --runs 5 \
        --temperature 0.0 \
        --max_tokens 2048 \
        --max_model_len 3072
fi

echo "=== Done ==="
