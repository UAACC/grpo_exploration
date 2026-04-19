#!/bin/bash
#SBATCH --account=aip-szepesva
#SBATCH --job-name=diag-svamp
#SBATCH --time=00:15:00
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=shared/logs/%x-%j.out
#SBATCH --error=shared/logs/%x-%j.err

module load python/3.11 cuda/12.6 arrow opencv
source /project/aip-szepesva/mrli/backup_dongheng/.venv/bin/activate
export HF_HOME=/scratch/mrli
export HF_DATASETS_CACHE=/scratch/mrli/datasets/SVAMP
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd /project/aip-szepesva/mrli/backup_dongheng/shared
python check_svamp_baseline.py
