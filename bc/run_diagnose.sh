#!/bin/bash
#SBATCH --account=aip-szepesva
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/bc_math/%x-%j.out
#SBATCH --error=logs/bc_math/%x-%j.err

module load python/3.11 cuda/12.6 arrow opencv
source /project/aip-szepesva/mrli/backup_dongheng/.venv/bin/activate
export HF_HOME="/scratch/mrli"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd /project/aip-szepesva/mrli/backup_dongheng/bc
python3 diagnose_bc.py
