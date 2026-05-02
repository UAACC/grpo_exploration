#!/bin/bash
#
# Evaluate a single Qwen2.5 Instruct model on GSM8K
# Submit one job per model to avoid vLLM CUDA re-init issue.
#
# Usage:
#   MODEL=Qwen2.5-0.5B-Instruct sbatch --job-name=gsm8k-eval-0.5B run_eval_gsm8k_baselines.sh
#   MODEL=Qwen2.5-1.5B-Instruct sbatch --job-name=gsm8k-eval-1.5B run_eval_gsm8k_baselines.sh
#   MODEL=Qwen2.5-3B-Instruct   sbatch --job-name=gsm8k-eval-3B  run_eval_gsm8k_baselines.sh
#
#SBATCH --account=aip-szepesva
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/bc_math/%x-%j.out
#SBATCH --error=logs/bc_math/%x-%j.err

module load python/3.11 cuda/12.6 arrow opencv
source /project/aip-szepesva/mrli/backup_dongheng/.venv/bin/activate
export HF_HOME=/scratch/mrli
export HF_DATASETS_CACHE=/scratch/mrli/datasets/GSM8K
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

MODEL="${MODEL:?Set MODEL env var, e.g. MODEL=Qwen2.5-0.5B-Instruct}"
EVAL_SCRIPT="/project/aip-szepesva/mrli/backup_dongheng/Math_Verifier/eval_unified.py"
MODEL_PATH="/scratch/mrli/models/${MODEL}"

echo "=== Evaluating ${MODEL} on GSM8K ==="
python3 "${EVAL_SCRIPT}" \
    --model_path "${MODEL_PATH}" \
    --dataset_type gsm8k \
    --split test \
    --runs 5 \
    --temperature 0.0 \
    --max_tokens 1024 \
    --max_model_len 2048 \
    --tensor_parallel_size 1

echo "=== Done ==="
