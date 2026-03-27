#!/bin/bash
#SBATCH --account=aip-szepesva
#SBATCH --time=0:10:00
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/check-baseline-%j.out
#SBATCH --error=logs/check-baseline-%j.err

module load python/3.11 cuda/12.6 arrow opencv
source /project/aip-szepesva/mrli/backup_dongheng/.venv/bin/activate
export HF_HOME="/scratch/mrli"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd /project/aip-szepesva/mrli/backup_dongheng/online_grpo
python check_baseline_output.py
