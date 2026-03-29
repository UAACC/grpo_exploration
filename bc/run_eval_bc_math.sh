#!/bin/bash
#
# Evaluate BC checkpoint on MATH test set.
# Merges LoRA adapter into base model, then evaluates with vLLM.
#
# Usage:
#   sbatch --job-name=eval-bc-math run_eval_bc_math.sh
#
#SBATCH --account=aip-szepesva
#SBATCH --time=03:00:00
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/bc_math/%x-%j.out
#SBATCH --error=logs/bc_math/%x-%j.err

set -e

SCRATCH="/scratch/mrli"
EVAL_SCRIPT="/project/aip-szepesva/mrli/backup_dongheng/offline_grpo/evaluate.py"

module load python/3.11 cuda/12.6 arrow opencv
source /project/aip-szepesva/mrli/backup_dongheng/.venv/bin/activate
export HF_HOME="${SCRATCH}"
export HF_DATASETS_CACHE="/scratch/mrli/datasets/MATH"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

BASE_MODEL="${SCRATCH}/models/Qwen2.5-0.5B-Instruct"
ADAPTER="${SCRATCH}/checkpoints/bc_math"
MERGED="${SCRATCH}/merged/bc_math_final"
DATASET="nlile/hendrycks-MATH-benchmark"

echo "=== Evaluating BC checkpoint on MATH ==="
echo "  Base model: ${BASE_MODEL}"
echo "  Adapter: ${ADAPTER}"
echo "  Merged output: ${MERGED}"
echo "  Runs: 5"

python3 "${EVAL_SCRIPT}" \
    --model_path "${ADAPTER}" \
    --base_model "${BASE_MODEL}" \
    --merge_lora \
    --merged_output "${MERGED}" \
    --dataset "${DATASET}" \
    --split test \
    --runs 5 \
    --temperature 0.6 \
    --max_tokens 2048 \
    --max_model_len 3072 \
    --tensor_parallel_size 1

echo "=== Eval complete ==="
