#!/bin/bash
#
# BC-all baseline on the bad-teacher rollout (Qwen2.5-0.5B-Instruct pick4).
# Trains student on ALL completions (correct + wrong) for 1 epoch (~48K completions).
#
# Usage:
#   sbatch run_bc.sh train
#   sbatch run_bc.sh eval
#   sbatch run_bc.sh eval-baseline
#
#SBATCH --account=aip-szepesva
#SBATCH --job-name=bc-math-badteacher
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

SCRATCH="${SCRATCH:-/scratch/shuai14}"
BC_SCRIPT="/project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/bc/train_bc.py"
EVAL_SCRIPT="/project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/mixture_grpo/evaluate.py"
WORK_DIR="/project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/SoftDG-Offline-to-the-bottom"

STUDENT_MODEL="${STUDENT_MODEL:-${SCRATCH}/models/Qwen2.5-0.5B}"
ROLLOUT_PATH="${ROLLOUT_PATH:-${SCRATCH}/rollouts/math_teacher/rollouts_math_Qwen2.5-0.5B-Instruct_0.6_pick4.jsonl}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH}/checkpoints/bc_math_badteacher}"
MERGED_DIR="${MERGED_DIR:-${SCRATCH}/merged/bc_math_badteacher}"
WANDB_PROJECT="${WANDB_PROJECT:-softdg-offline-math}"

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

if [ ! -f "${ROLLOUT_PATH}" ]; then
    echo "ERROR: rollout file not found: ${ROLLOUT_PATH}" >&2
    exit 1
fi

echo "=== BC-all baseline on bad-teacher MATH rollout ==="
echo "  Student:  ${STUDENT_MODEL}"
echo "  Rollout:  ${ROLLOUT_PATH}"
echo "  Output:   ${CHECKPOINT_DIR}"

resolve_eval_checkpoint() {
    local ckpt="${1:-latest}"
    if [ "${ckpt}" = "latest" ]; then
        local ckpt_path
        ckpt_path=$(ls -d "${CHECKPOINT_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
        if [ -z "${ckpt_path}" ]; then
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

CMD="${1:-train}"

if [ "$CMD" = "train" ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
        --config_file /project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/DG-offline/configs/accelerate_ddp_4gpu.yaml \
        "${BC_SCRIPT}" \
        --target_model "${STUDENT_MODEL}" \
        --rollout_path "${ROLLOUT_PATH}" \
        --output_dir "${CHECKPOINT_DIR}" \
        --run_name "bc-badteacher-0.5B-all" \
        --learning_rate 3e-6 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 2 \
        --num_train_epochs 1 \
        --max_grad_norm 1.0 \
        --weight_decay 0.01 \
        --warmup_ratio 0.1 \
        --save_steps 200 \
        --logging_steps 5 \
        --lora_r 32 \
        --lora_alpha 32 \
        --seed 42 \
        --report_to wandb

    echo "=== BC training complete ==="
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

else
    echo "Usage: sbatch run_bc.sh {train|eval [checkpoint|latest]|eval-baseline}"
    exit 1
fi

echo "=== Done ==="
