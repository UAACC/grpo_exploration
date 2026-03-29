#!/bin/bash
#SBATCH --account=aip-szepesva
#SBATCH --job-name=analyze-branching
#SBATCH --time=02:00:00
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

SCRATCH=/scratch/mrli
STUDENT="${STUDENT:-${SCRATCH}/models/Qwen2.5-0.5B-Instruct}"
TEACHER="${TEACHER:-${SCRATCH}/models/Qwen2.5-Math-7B-Instruct}"

cd /project/aip-szepesva/mrli/backup_dongheng/bc

python analyze_branching.py \
    --student_model "${STUDENT}" \
    --teacher_model "${TEACHER}" \
    --num_problems 200 \
    --max_tokens 2048 \
    --max_model_len 3072 \
    --top_k_report 20
