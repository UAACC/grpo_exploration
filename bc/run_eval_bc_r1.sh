#!/bin/bash
#
# Evaluate a BC LoRA checkpoint on MATH-500 via Math_Verifier/eval_unified.py.
# Parameterizable for the two R1 BC runs (bc-all and bc-correct-only).
#
# Usage:
#   ADAPTER=/scratch/mrli/checkpoints/bc_math_r1 \
#   MERGED=/scratch/mrli/merged/bc_math_r1_merged \
#   sbatch --job-name=eval-bc-all-r1 bc/run_eval_bc_r1.sh
#
#SBATCH --account=aip-szepesva
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/bc_math/%x-%j.out
#SBATCH --error=logs/bc_math/%x-%j.err

set -euo pipefail

SCRATCH="${SCRATCH:-/scratch/mrli}"
BASE_MODEL="${BASE_MODEL:-${SCRATCH}/models/Qwen2.5-0.5B-Instruct}"
ADAPTER="${ADAPTER:?ADAPTER env var must be set to the LoRA checkpoint dir}"
MERGED="${MERGED:?MERGED env var must be set to where the merged model goes}"
EVAL_SCRIPT="/project/aip-szepesva/mrli/backup_dongheng/Math_Verifier/eval_unified.py"

# Greedy 5-seed (production cheap eval per the audit funnel).
RUNS="${RUNS:-5}"
# R1-trained students mimic R1's verbose reasoning; default 2048 cuts most of
# them off before the \boxed{} answer. Match the training context window.
MAX_TOKENS="${MAX_TOKENS:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8704}"
# Per-problem records for inspection (set to "" to skip).
SAVE_COMPLETIONS="${SAVE_COMPLETIONS:-${SCRATCH}/eval_outputs/$(basename ${ADAPTER}).jsonl}"

module load python/3.11 cuda/12.6 arrow opencv
source /project/aip-szepesva/mrli/backup_dongheng/.venv/bin/activate
export HF_HOME="${SCRATCH}"
export HF_DATASETS_CACHE="${SCRATCH}/datasets/MATH"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p logs/bc_math
cd /project/aip-szepesva/mrli/backup_dongheng

echo "=== Evaluating BC checkpoint on MATH-500 ==="
echo "  Base model: ${BASE_MODEL}"
echo "  Adapter:    ${ADAPTER}"
echo "  Merged out: ${MERGED}"
echo "  Runs:       ${RUNS} (greedy)"

python3 "${EVAL_SCRIPT}" \
    --model_path "${ADAPTER}" \
    --base_model "${BASE_MODEL}" \
    --merge_lora \
    --merged_output "${MERGED}" \
    --dataset math \
    --mode greedy \
    --runs "${RUNS}" \
    --max_tokens "${MAX_TOKENS}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --save_completions "${SAVE_COMPLETIONS}" \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.85 \
    --seed 42

echo "=== Eval complete ==="
