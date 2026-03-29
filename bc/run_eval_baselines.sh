#!/bin/bash
#
# Evaluate Qwen2.5 Instruct family baselines on MATH and GSM8K
# 0.5B (already known), 1.5B, 3B — all same vocab, same family
#
#SBATCH --account=aip-szepesva
#SBATCH --time=03:00:00
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/bc_math/%x-%j.out
#SBATCH --error=logs/bc_math/%x-%j.err

module load python/3.11 cuda/12.6 arrow opencv
source /project/aip-szepesva/mrli/backup_dongheng/.venv/bin/activate
export HF_HOME=/scratch/mrli
export HF_DATASETS_CACHE=/scratch/mrli/datasets/MATH
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

EVAL_SCRIPT="/project/aip-szepesva/mrli/backup_dongheng/offline_grpo/evaluate.py"
SCRATCH="/scratch/mrli"

# Models to evaluate
MODELS=(
    "${SCRATCH}/models/Qwen2.5-0.5B-Instruct"
    "${SCRATCH}/models/Qwen2.5-1.5B-Instruct"
    "${SCRATCH}/models/Qwen2.5-3B-Instruct"
)

echo "=== Evaluating Qwen2.5 Instruct baselines on MATH ==="
for model in "${MODELS[@]}"; do
    name=$(basename "${model}")
    echo ""
    echo ">>> ${name} on MATH <<<"
    python3 "${EVAL_SCRIPT}" \
        --model_path "${model}" \
        --dataset nlile/hendrycks-MATH-benchmark \
        --split test \
        --runs 5 \
        --temperature 0.6 \
        --max_tokens 2048 \
        --max_model_len 3072 \
        --tensor_parallel_size 1
    echo ""
done

echo "=== Evaluating Qwen2.5 Instruct baselines on GSM8K ==="
export HF_DATASETS_CACHE=/scratch/mrli/datasets/GSM8K

for model in "${MODELS[@]}"; do
    name=$(basename "${model}")
    echo ""
    echo ">>> ${name} on GSM8K <<<"
    python3 "${EVAL_SCRIPT}" \
        --model_path "${model}" \
        --dataset openai/gsm8k \
        --split test \
        --runs 5 \
        --temperature 0.6 \
        --max_tokens 1024 \
        --max_model_len 2048 \
        --tensor_parallel_size 1
    echo ""
done

echo "=== All evaluations complete ==="
