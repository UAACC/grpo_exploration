#!/bin/bash
#SBATCH --account=aip-szepesva
#SBATCH --job-name=multigpu-test
#SBATCH --time=1:00:00
#SBATCH --gpus-per-node=l40s:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --output=multigpu-test-%j.out
#SBATCH --error=multigpu-test-%j.err

bash /project/aip-szepesva/mrli/backup_dongheng/offline_grpo/run_multigpu_test.sh all
