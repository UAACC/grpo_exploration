#!/bin/bash
#
# Train Dr.Mixture-GRPO on one dataset.
#
# The per-qid student baseline r_mean_student(qid) is computed LIVE inside
# the trainer at every step (K_s samples from current LoRA-merged policy,
# scored with Math_Verifier). No precompute step required.
#
# Usage:
#   DATASET=gsm8k sbatch dr_mixture_grpo/run_dr_mixture.sh
#   DATASET=math  sbatch dr_mixture_grpo/run_dr_mixture.sh
#
# Optional env vars: K_S (default 5), BASELINE_TEMP (0.7), MAX_STEPS (-1),
# OUTPUT_DIR, ROLLOUT, MAX_COMPL, MODEL_DIR.
#
#SBATCH --account=aip-xt7
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=dr_mixture_grpo/logs/train-%x-%j.out
#SBATCH --error=dr_mixture_grpo/logs/train-%x-%j.err

set -e

DATASET="${DATASET:?Set DATASET env var (math, gsm8k, svamp, asdiv)}"
SCRATCH="/scratch/mrli"
WORK_DIR="/home/shuai14/projects/aip-szepesva/shuai14/DG_LLM/grpo_exploration"
MODEL_DIR="${MODEL_DIR:-/scratch/mrli/models/Qwen2.5-0.5B-Instruct}"

case "${DATASET}" in
    math)
        ROLLOUT="${ROLLOUT:-${SCRATCH}/rollouts/math_teacher/rollouts_full.jsonl}"
        MAX_COMPL="${MAX_COMPL:-2048}"
        ;;
    gsm8k)
        ROLLOUT="${ROLLOUT:-${SCRATCH}/rollouts/gsm8k_teacher/rollouts_gsm8k.jsonl}"
        MAX_COMPL="${MAX_COMPL:-1024}"
        ;;
    svamp)
        ROLLOUT="${ROLLOUT:-${SCRATCH}/rollouts/svamp_teacher/rollouts_svamp.jsonl}"
        MAX_COMPL="${MAX_COMPL:-1024}"
        ;;
    asdiv)
        ROLLOUT="${ROLLOUT:-${SCRATCH}/rollouts/asdiv_teacher/rollouts_asdiv.jsonl}"
        MAX_COMPL="${MAX_COMPL:-1024}"
        ;;
    *)
        echo "Unknown DATASET=${DATASET}"; exit 1
        ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-/scratch/shuai14/checkpoints/dr_mixture_grpo_${DATASET}}"
K_S="${K_S:-5}"
BASELINE_TEMP="${BASELINE_TEMP:-0.7}"
MAX_STEPS="${MAX_STEPS:--1}"

module load python/3.11 cuda/12.6 arrow opencv
source "/project/aip-szepesva/shuai14/verifiers/.venv/bin/activate"

export HF_HOME="${SCRATCH}"
case "${DATASET}" in
    math)   export HF_DATASETS_CACHE="${SCRATCH}/datasets/MATH" ;;
    gsm8k)  export HF_DATASETS_CACHE="${SCRATCH}/datasets/GSM8K" ;;
    svamp)  export HF_DATASETS_CACHE="${SCRATCH}/datasets/SVAMP" ;;
    asdiv)  export HF_DATASETS_CACHE="${SCRATCH}/datasets/ASDIV" ;;
esac
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=$(cat "/project/aip-szepesva/shuai14/DG_LLM/grpo_exploration/RWR-offline/.wandb_key")
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_PROJECT="mixture-grpo-${DATASET}"
export WANDB_RUN_NAME="dr-mixture-${DATASET}-$(date +%m%d)"

cd "${WORK_DIR}"
mkdir -p dr_mixture_grpo/logs

echo "=== Dr.Mixture-GRPO training: ${DATASET} ==="
echo "  Rollouts:    ${ROLLOUT}"
echo "  Output:      ${OUTPUT_DIR}"
echo "  K_s:         ${K_S}  (baseline_T=${BASELINE_TEMP})"
echo "  max_steps:   ${MAX_STEPS}"
echo "  wandb:       ${WANDB_PROJECT} / ${WANDB_RUN_NAME}"

accelerate launch \
    --config_file "${WORK_DIR}/offline_grpo/configs/accelerate_ddp_4gpu.yaml" \
    dr_mixture_grpo/train.py \
    --rollout_path "${ROLLOUT}" \
    --target_model "${MODEL_DIR}" \
    --dataset "${DATASET}" \
    --K_s "${K_S}" \
    --baseline_temperature "${BASELINE_TEMP}" \
    --max_steps "${MAX_STEPS}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${WANDB_RUN_NAME}" \
    --num_generations 4 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-6 \
    --beta 0.001 \
    --max_completion_length "${MAX_COMPL}" \
    --max_grad_norm 1.0 \
    --weight_decay 0.01 \
    --save_steps 500 \
    --logging_steps 5 \
    --report_to wandb

echo "=== Training complete ==="
ls -la "${OUTPUT_DIR}"
