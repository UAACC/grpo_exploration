#!/bin/bash
#
# DG-Mixture GRPO (GSM8K)
#
#SBATCH --account=aip-szepesva
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=logs/dg_mixture_gsm8k/%x-%j.out
#SBATCH --error=logs/dg_mixture_gsm8k/%x-%j.err

set -e

WORK_DIR="/project/aip-szepesva/mrli/backup_dongheng/mixture_grpo"
SCRATCH="${SCRATCH:-/scratch/mrli}"

module load python/3.11 cuda/12.6 arrow opencv
source /project/aip-szepesva/mrli/backup_dongheng/.venv/bin/activate
export HF_HOME="${SCRATCH}"
export HF_DATASETS_CACHE="${SCRATCH}/datasets/GSM8K"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_PROJECT="dg-mixture-gsm8k"
if [ -f "/project/aip-szepesva/mrli/backup_dongheng/offline_grpo/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat /project/aip-szepesva/mrli/backup_dongheng/offline_grpo/.wandb_key)
fi

cd "${WORK_DIR}"
mkdir -p logs/dg_mixture_gsm8k

accelerate launch \
    --config_file "${WORK_DIR}/../offline_grpo/configs/accelerate_ddp_4gpu.yaml" \
    dg_mixture/train.py \
    --teacher_rollout_path "${SCRATCH}/rollouts/gsm8k_teacher/rollouts_gsm8k.jsonl" \
    --target_model "${SCRATCH}/models/Qwen2.5-0.5B-Instruct" \
    --output_dir "${SCRATCH}/checkpoints/dg_mixture_gsm8k" \
    --dataset_type gsm8k \
    --dg_temperature 0.5 \
    --dg_offline_weight 0.3 \
    --num_generations 5 \
    --num_teacher_per_prompt 5 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 5 \
    --gradient_accumulation_steps 2 \
    --learning_rate 3e-6 \
    --beta 0.01 \
    --max_completion_length 1024 \
    --max_grad_norm 1.0 \
    --lora_r 32 --lora_alpha 32 \
    --save_steps 500 --logging_steps 10 \
    --report_to wandb
