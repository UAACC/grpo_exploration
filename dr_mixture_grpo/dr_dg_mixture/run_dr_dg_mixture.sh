#!/bin/bash
#
# Train Dr.DG.Mixture (DG gate + Dr.Mixture's live student baseline).
#
# Per-qid student baseline is computed live inside the trainer at every step;
# no precompute step required. Note: DG-offline's eta sweep was calibrated for
# group-normalized advantages in [-1, 1]. Dr.Mixture A is on the raw reward
# scale (~+/-1 numeric, +/-2 MATH), so eta typically needs to be larger.
#
# Usage:
#   DATASET=gsm8k ETA=1.0 sbatch dr_mixture_grpo/dr_dg_mixture/run_dr_dg_mixture.sh train
#   DATASET=math  ETA=1.0 sbatch dr_mixture_grpo/dr_dg_mixture/run_dr_dg_mixture.sh eval
#   DATASET=math  ETA=1.0 sbatch dr_mixture_grpo/dr_dg_mixture/run_dr_dg_mixture.sh eval checkpoint-500
#   DATASET=math  ETA=1.0 sbatch dr_mixture_grpo/dr_dg_mixture/run_dr_dg_mixture.sh eval-baseline
#
#SBATCH --account=aip-szepesva
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=dr_mixture_grpo/logs/dr_dg-%x-%j.out
#SBATCH --error=dr_mixture_grpo/logs/dr_dg-%x-%j.err

set -e

DATASET="${DATASET:?Set DATASET env var (math, gsm8k, svamp, asdiv)}"
ETA="${ETA:-1.0}"
SCRATCH="${SCRATCH:-/scratch/shuai14}"
WORK_DIR="/project/aip-szepesva/shuai14/DG_LLM/grpo_exploration"
MODEL_DIR="${MODEL_DIR:-${SCRATCH}/models/Qwen2.5-0.5B-Instruct}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${WORK_DIR}/mixture_grpo/evaluate.py}"

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

ETA_TAG="$(echo "${ETA}" | tr '.' 'p')"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH}/checkpoints/dr_dg_mixture_${DATASET}_eta${ETA_TAG}}"
MERGED_DIR="${MERGED_DIR:-${SCRATCH}/merged/dr_dg_mixture_${DATASET}_eta${ETA_TAG}}"
K_S="${K_S:-5}"
BASELINE_TEMP="${BASELINE_TEMP:-0.7}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.20}"
LOSS_TYPE="${LOSS_TYPE:-dr_grpo}"
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
export WANDB_API_KEY=$(cat "${WORK_DIR}/offline_grpo/.wandb_key")
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_PROJECT="dr-mixture-grpo"
export WANDB_RUN_NAME="dr-dg-mixture-${DATASET}-eta${ETA}-$(date +%m%d)"

cd "${WORK_DIR}"
mkdir -p dr_mixture_grpo/logs

echo "=== Dr.DG.Mixture training: ${DATASET} eta=${ETA} ==="
echo "  Rollouts:    ${ROLLOUT}"
echo "  Output:      ${OUTPUT_DIR}"
echo "  K_s:         ${K_S}  (baseline_T=${BASELINE_TEMP})"
echo "  vLLM memory: ${VLLM_GPU_MEMORY_UTILIZATION} per GPU"
echo "  eta:         ${ETA}"
echo "  loss_type:   ${LOSS_TYPE}"
echo "  max_steps:   ${MAX_STEPS}"

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

run_eval() {
    local ckpt_path="$1"

    if [ "${DATASET}" != "math" ] && [ "${DATASET}" != "gsm8k" ]; then
        echo "Eval is currently supported only for DATASET=math or DATASET=gsm8k by ${EVAL_SCRIPT}." >&2
        exit 1
    fi

    echo "=== Evaluating ${ckpt_path} on ${DATASET} ==="
    python "${EVAL_SCRIPT}" \
        --model_path "${ckpt_path}" \
        --base_model "${MODEL_DIR}" \
        --merge_lora \
        --merged_output "${MERGED_DIR}" \
        --dataset_type "${DATASET}" \
        --runs "${EVAL_RUNS:-5}" \
        --temperature "${EVAL_TEMPERATURE:-0.0}" \
        --max_tokens "${EVAL_MAX_TOKENS:-${MAX_COMPL}}" \
        --max_model_len "${EVAL_MAX_MODEL_LEN:-3072}"
    echo "=== Evaluation complete ==="
}

CMD="${1:-train}"
if [ "${CMD}" = "train" ]; then
accelerate launch \
    --config_file "${WORK_DIR}/offline_grpo/configs/accelerate_ddp_4gpu.yaml" \
    dr_mixture_grpo/dr_dg_mixture/train.py \
    --rollout_path "${ROLLOUT}" \
    --target_model "${MODEL_DIR}" \
    --dataset "${DATASET}" \
    --K_s "${K_S}" \
    --baseline_temperature "${BASELINE_TEMP}" \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
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
    --loss_type "${LOSS_TYPE}" \
    --max_grad_norm 1.0 \
    --weight_decay 0.01 \
    --save_steps 500 \
    --logging_steps 5 \
    --dg_temperature "${ETA}" \
    --dg_gating "completion" \
    --report_to wandb

    echo "=== Training complete ==="
    ls -la "${OUTPUT_DIR}"
    run_eval "${OUTPUT_DIR}"

elif [ "${CMD}" = "eval" ]; then
    CKPT_PATH=$(resolve_eval_checkpoint "${2:-latest}")
    run_eval "${CKPT_PATH}"

elif [ "${CMD}" = "eval-baseline" ]; then
    if [ "${DATASET}" != "math" ] && [ "${DATASET}" != "gsm8k" ]; then
        echo "Eval is currently supported only for DATASET=math or DATASET=gsm8k by ${EVAL_SCRIPT}." >&2
        exit 1
    fi

    echo "=== Evaluating baseline ${MODEL_DIR} on ${DATASET} ==="
    python "${EVAL_SCRIPT}" \
        --model_path "${MODEL_DIR}" \
        --dataset_type "${DATASET}" \
        --runs "${EVAL_RUNS:-5}" \
        --temperature "${EVAL_TEMPERATURE:-0.0}" \
        --max_tokens "${EVAL_MAX_TOKENS:-${MAX_COMPL}}" \
        --max_model_len "${EVAL_MAX_MODEL_LEN:-3072}"
    echo "=== Baseline evaluation complete ==="

else
    echo "Usage: DATASET=<dataset> sbatch $0 {train|eval [checkpoint|latest]|eval-baseline}"
    exit 1
fi

echo "=== Done ==="
