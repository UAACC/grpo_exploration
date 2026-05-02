#!/bin/bash
#
# Evaluate ALL models on ALL tasks (GSM8K + MATH)
# Uses pre-merged models from /scratch/mrli/merged/ and base models from /scratch/mrli/models/
#
# Usage:
#   sbatch --job-name=eval-all run_eval_all.sh
#
#SBATCH --account=aip-szepesva
#SBATCH --time=6:00:00
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/eval_all/%x-%j.out
#SBATCH --error=logs/eval_all/%x-%j.err

set -e

WORK_DIR="/project/aip-szepesva/mrli/backup_dongheng"
SCRATCH="/scratch/mrli"
EVAL_SCRIPT="${WORK_DIR}/Math_Verifier/eval_unified.py"

# ── Activate environment ─────────────────────────────────────────
module load python/3.11 cuda/12.6 arrow opencv
source "${WORK_DIR}/.venv/bin/activate"
export HF_HOME="${SCRATCH}"
export HF_DATASETS_CACHE="${SCRATCH}/datasets/GSM8K"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
cd "${WORK_DIR}/mixture_grpo"

RUNS=5
TEMP=0.0

# ── Models ────────────────────────────────────────────────────────
BASELINE="${SCRATCH}/models/Qwen2.5-0.5B-Instruct"
TEACHER="${SCRATCH}/models/Qwen2.5-Math-7B-Instruct"
OFFLINE_GSM8K="${SCRATCH}/merged/offline_grpo_gsm8k_merged"
OFFLINE_MATH="${SCRATCH}/merged/offline_grpo_full_4gpu_refsync16_merged"
MIX_A_GSM8K="${SCRATCH}/merged/mixture_A_unified_merged"
MIX_A_MATH="${SCRATCH}/merged/mixture_A_math_merged"
MIX_B_GSM8K="${SCRATCH}/merged/mixture_B_weighted_merged"
MIX_B_MATH="${SCRATCH}/merged/mixture_B_math_merged"

mkdir -p "${WORK_DIR}/logs/eval_all"

log() { printf '\n%s === %s ===\n\n' "$(date '+%F %T')" "$*"; }

run_eval() {
    local label="$1"
    local model="$2"
    local dataset_type="$3"

    log "${label} on ${dataset_type}"

    # Switch dataset cache based on task
    if [ "$dataset_type" = "math" ]; then
        export HF_DATASETS_CACHE="${SCRATCH}/datasets/MATH"
    else
        export HF_DATASETS_CACHE="${SCRATCH}/datasets/GSM8K"
    fi

    python "${EVAL_SCRIPT}" \
        --model_path "${model}" \
        --dataset_type "${dataset_type}" \
        --runs "${RUNS}" \
        --temperature "${TEMP}" \
        --max_tokens 1024 \
        --max_model_len 2048
}

run_eval_math() {
    local label="$1"
    local model="$2"

    log "${label} on math"
    export HF_DATASETS_CACHE="${SCRATCH}/datasets/MATH"

    python "${EVAL_SCRIPT}" \
        --model_path "${model}" \
        --dataset_type math \
        --runs "${RUNS}" \
        --temperature "${TEMP}" \
        --max_tokens 2048 \
        --max_model_len 3072
}

# ══════════════════════════════════════════════════════════════════
# GSM8K evaluations (0.5B models)
# ══════════════════════════════════════════════════════════════════
run_eval "Baseline (0.5B)"          "${BASELINE}"      gsm8k
run_eval "Offline GRPO GSM8K"       "${OFFLINE_GSM8K}" gsm8k
run_eval "Mixture A GSM8K"          "${MIX_A_GSM8K}"   gsm8k
run_eval "Mixture B GSM8K"          "${MIX_B_GSM8K}"   gsm8k

# Cross-task: MATH-trained models on GSM8K
run_eval "Offline GRPO MATH→GSM8K"  "${OFFLINE_MATH}"  gsm8k
run_eval "Mixture A MATH→GSM8K"     "${MIX_A_MATH}"    gsm8k
run_eval "Mixture B MATH→GSM8K"     "${MIX_B_MATH}"    gsm8k

# ══════════════════════════════════════════════════════════════════
# MATH evaluations (0.5B models)
# ══════════════════════════════════════════════════════════════════
run_eval_math "Baseline (0.5B)"          "${BASELINE}"
run_eval_math "Offline GRPO MATH"        "${OFFLINE_MATH}"
run_eval_math "Mixture A MATH"           "${MIX_A_MATH}"
run_eval_math "Mixture B MATH"           "${MIX_B_MATH}"

# Cross-task: GSM8K-trained models on MATH
run_eval_math "Offline GRPO GSM8K→MATH"  "${OFFLINE_GSM8K}"
run_eval_math "Mixture A GSM8K→MATH"     "${MIX_A_GSM8K}"
run_eval_math "Mixture B GSM8K→MATH"     "${MIX_B_GSM8K}"

# ══════════════════════════════════════════════════════════════════
# Teacher model (7B) — needs more memory, lower gpu_memory_utilization
# ══════════════════════════════════════════════════════════════════
log "Teacher (7B) on gsm8k"
export HF_DATASETS_CACHE="${SCRATCH}/datasets/GSM8K"
python "${EVAL_SCRIPT}" \
    --model_path "${TEACHER}" \
    --dataset_type gsm8k \
    --runs "${RUNS}" \
    --temperature "${TEMP}" \
    --max_tokens 1024 \
    --max_model_len 2048 \
    --gpu_memory_utilization 0.90

log "Teacher (7B) on math"
export HF_DATASETS_CACHE="${SCRATCH}/datasets/MATH"
python "${EVAL_SCRIPT}" \
    --model_path "${TEACHER}" \
    --dataset_type math \
    --runs "${RUNS}" \
    --temperature "${TEMP}" \
    --max_tokens 2048 \
    --max_model_len 3072 \
    --gpu_memory_utilization 0.90

log "ALL EVALUATIONS COMPLETE"
