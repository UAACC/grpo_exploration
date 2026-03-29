#!/bin/bash
#SBATCH --account=aip-szepesva
#SBATCH --job-name=best-of-n
#SBATCH --time=01:00:00
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

MODEL="${MODEL:-/scratch/mrli/models/Qwen2.5-0.5B-Instruct}"
N="${N:-16}"
TEMP="${TEMP:-0.6}"
DATASET="${DATASET:-math}"

# Set dataset cache based on dataset type
if [ "$DATASET" = "gsm8k" ]; then
    export HF_DATASETS_CACHE=/scratch/mrli/datasets/GSM8K
    NUM_PROBLEMS="${NUM_PROBLEMS:-1319}"
    MAX_TOKENS=1024
    MAX_MODEL_LEN=2048
else
    NUM_PROBLEMS="${NUM_PROBLEMS:-500}"
    MAX_TOKENS=2048
    MAX_MODEL_LEN=3072
fi

cd /project/aip-szepesva/mrli/backup_dongheng/bc

echo "=== Best-of-N eval: N=${N}, temp=${TEMP}, dataset=${DATASET} ==="
echo "  Model: ${MODEL}"

python eval_best_of_n.py \
    --model_path "${MODEL}" \
    --dataset_type "${DATASET}" \
    --n_samples "${N}" \
    --temperature "${TEMP}" \
    --num_problems "${NUM_PROBLEMS}" \
    --max_tokens "${MAX_TOKENS}" \
    --max_model_len "${MAX_MODEL_LEN}"
