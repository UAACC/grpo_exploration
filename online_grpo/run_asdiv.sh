#!/bin/bash
#
# Online GRPO on ASDiv with Qwen2.5-0.5B-Instruct + LoRA
#
# Usage:
#   sbatch --job-name=online-grpo-asdiv run_asdiv.sh train
#   sbatch --job-name=online-grpo-asdiv-eval --gpus-per-node=l40s:1 --cpus-per-task=16 --mem=64G --time=0:30:00 run_asdiv.sh eval
#
#SBATCH --account=aip-szepesva
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e

WORK_DIR="/project/aip-szepesva/mrli/backup_dongheng/online_grpo"
SCRATCH="/scratch/mrli"
CONFIG_DIR="${WORK_DIR}/configs"
MODEL_DIR="${SCRATCH}/models/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR="${SCRATCH}/checkpoints/online_grpo_asdiv"

module load python/3.11 cuda/12.6 arrow opencv
source /project/aip-szepesva/mrli/backup_dongheng/.venv/bin/activate
export HF_HOME="${SCRATCH}"
export HF_DATASETS_CACHE="/scratch/mrli/datasets/ASDIV"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=$(cat /project/aip-szepesva/mrli/backup_dongheng/offline_grpo/.wandb_key)
export VLLM_WORKER_MULTIPROC_METHOD=spawn
cd "${WORK_DIR}"

case "${1}" in

train)
    echo "=== Online GRPO Training on ASDiv (4x L40s) ==="
    export WANDB_PROJECT="online-grpo-asdiv"
    export WANDB_RUN_NAME="qwen05b-asdiv-$(date +%Y%m%d_%H%M%S)"
    mkdir -p "${OUTPUT_DIR}"

    accelerate launch \
        --config_file "${CONFIG_DIR}/accelerate_ddp_4gpu.yaml" \
        train.py \
        --model "${MODEL_DIR}" \
        --dataset_type asdiv \
        --output_dir "${OUTPUT_DIR}" \
        --num_generations 5 \
        --num_train_epochs 15 \
        --per_device_train_batch_size 5 \
        --gradient_accumulation_steps 2 \
        --learning_rate 3e-6 \
        --beta 0.001 \
        --temperature 0.7 \
        --max_prompt_length 512 \
        --max_completion_length 1024 \
        --lora_r 32 \
        --lora_alpha 32 \
        --save_steps 200 \
        --logging_steps 10 \
        --report_to wandb
    echo "=== Training complete. Model saved to ${OUTPUT_DIR} ==="
    ;;

eval)
    echo "=== Evaluating trained model (merge LoRA) ==="
    MERGED_DIR="${SCRATCH}/merged/online_grpo_asdiv_merged"
    DATASET=asdiv MODEL="${OUTPUT_DIR}" BASE="${MODEL_DIR}" MERGE=1 \
        MERGED_DIR="${MERGED_DIR}" \
        bash /project/aip-szepesva/mrli/backup_dongheng/shared/run_eval.sh
    ;;

*)
    echo "Usage: sbatch --job-name=<name> run_asdiv.sh {train|eval}"
    exit 1
    ;;
esac
