#!/bin/bash
#
# Unified evaluation script for all datasets and models.
#
# Usage:
#   DATASET=math MODEL=/path/to/model sbatch run_eval.sh
#   DATASET=svamp MODEL=/path/to/model MODE=both sbatch run_eval.sh
#   DATASET=aime MODEL=/path/to/lora BASE=/path/to/base MERGE=1 sbatch run_eval.sh
#
# Environment variables:
#   DATASET     Required. One of: math, gsm8k, svamp, aime
#   MODEL       Required. Path to model or LoRA adapter
#   BASE        Optional. Base model for LoRA merging
#   MERGE       Optional. Set to 1 to merge LoRA before eval
#   MERGED_DIR  Optional. Where to save merged model
#   MODE        Optional. "greedy", "best_of_n", or "both" (default: both)
#   RUNS        Optional. Number of greedy runs (default: 5)
#   N_SAMPLES   Optional. Samples for best-of-N (default: 16)
#   TEMP        Optional. Temperature for best-of-N (default: 0.6)
#   MAX_PROBS   Optional. Limit number of problems (for testing)
#
#SBATCH --account=aip-szepesva
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

DATASET="${DATASET:?Set DATASET env var (math, gsm8k, svamp, aime)}"
MODEL="${MODEL:?Set MODEL env var (path to model or LoRA adapter)}"
BASE="${BASE:-}"
MERGE="${MERGE:-0}"
MERGED_DIR="${MERGED_DIR:-}"
MODE="${MODE:-both}"
RUNS="${RUNS:-5}"
N_SAMPLES="${N_SAMPLES:-16}"
TEMP="${TEMP:-0.6}"
MAX_PROBS="${MAX_PROBS:-}"
SEED="${SEED:-42}"

SCRATCH="${SCRATCH:-/scratch/mrli}"
WORK_DIR="/project/aip-szepesva/mrli/backup_dongheng"

module load python/3.11 cuda/12.6 arrow opencv
source "${WORK_DIR}/.venv/bin/activate"
export HF_HOME="${SCRATCH}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Set dataset cache based on dataset type
case "${DATASET}" in
    math)   export HF_DATASETS_CACHE="${SCRATCH}/datasets/MATH" ;;
    gsm8k)  export HF_DATASETS_CACHE="${SCRATCH}/datasets/GSM8K" ;;
    svamp)  export HF_DATASETS_CACHE="${SCRATCH}/datasets/SVAMP" ;;
    asdiv)  export HF_DATASETS_CACHE="${SCRATCH}/datasets/ASDIV" ;;
    *)      export HF_DATASETS_CACHE="${SCRATCH}/datasets/${DATASET}" ;;
esac

mkdir -p "${WORK_DIR}/shared/logs"

cd "${WORK_DIR}"

# Build the command
CMD="python Math_Verifier/eval_unified.py \
    --model_path ${MODEL} \
    --dataset ${DATASET} \
    --mode ${MODE} \
    --runs ${RUNS} \
    --n_samples ${N_SAMPLES} \
    --temperature ${TEMP} \
    --seed ${SEED}"

if [ "${MERGE}" = "1" ] && [ -n "${BASE}" ]; then
    CMD="${CMD} --merge_lora --base_model ${BASE}"
    if [ -n "${MERGED_DIR}" ]; then
        CMD="${CMD} --merged_output ${MERGED_DIR}"
    fi
fi

# ASDiv needs manual test split (no proper test split in HF)
if [ "${DATASET}" = "asdiv" ]; then
    CMD="${CMD} --manual_split test"
fi

if [ -n "${MAX_PROBS}" ]; then
    CMD="${CMD} --max_problems ${MAX_PROBS}"
fi

echo "=== Unified eval ==="
echo "  Dataset: ${DATASET}"
echo "  Model: ${MODEL}"
echo "  Mode: ${MODE}"
echo "  Runs: ${RUNS}, N: ${N_SAMPLES}, Temp: ${TEMP}"
echo ""

eval ${CMD}
