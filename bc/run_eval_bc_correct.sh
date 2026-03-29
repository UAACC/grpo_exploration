#!/bin/bash
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

python3 /project/aip-szepesva/mrli/backup_dongheng/offline_grpo/evaluate.py \
    --model_path /scratch/mrli/checkpoints/bc_math_correct_only \
    --base_model /scratch/mrli/models/Qwen2.5-0.5B-Instruct \
    --merge_lora \
    --merged_output /scratch/mrli/merged/bc_math_correct_only \
    --dataset nlile/hendrycks-MATH-benchmark \
    --split test --runs 5 --temperature 0.6 --max_tokens 2048 --max_model_len 3072
