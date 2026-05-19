#!/bin/bash
#
# Online GRPO on GSM8K with Qwen2.5-0.5B-Instruct + LoRA
#   Target: match VERL benchmark (54.3% on GSM8K test)
#   Hardware: 4x L40s with DDP + 1 GPU for vLLM generation
#
# Usage:
#   sbatch --job-name=online-grpo-gsm8k run_gsm8k.sh train
#   sbatch --job-name=online-grpo-eval  --gpus-per-node=l40s:1 --cpus-per-task=16 --mem=64G --time=0:30:00 run_gsm8k.sh eval
#   sbatch --job-name=online-grpo-eval  --gpus-per-node=l40s:1 --cpus-per-task=16 --mem=64G --time=0:30:00 run_gsm8k.sh eval-baseline
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

# Model snapshot on local scratch (avoid HF download during training)
MODEL_DIR="${SCRATCH}/models/Qwen2.5-0.5B-Instruct"

OUTPUT_DIR="${SCRATCH}/checkpoints/online_grpo_gsm8k"

# ── Activate environment ─────────────────────────────────────────────
module load python/3.11 cuda/12.6 arrow opencv
source /project/aip-szepesva/mrli/backup_dongheng/.venv/bin/activate
export HF_HOME="${SCRATCH}"
export HF_DATASETS_CACHE="/scratch/mrli/datasets/GSM8K"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=$(cat /project/aip-szepesva/mrli/backup_dongheng/offline_grpo/.wandb_key)
export VLLM_WORKER_MULTIPROC_METHOD=spawn
cd "${WORK_DIR}"

case "${1}" in

# ── Training ──────────────────────────────────────────────────────────
train)
    echo "=== Online GRPO Training (4x L40s) ==="
    echo "=== Model: ${MODEL_DIR} ==="
    echo "=== Output: ${OUTPUT_DIR} ==="
    export WANDB_PROJECT="online-grpo-gsm8k"
    export WANDB_RUN_NAME="qwen05b-gsm8k-$(date +%Y%m%d_%H%M%S)"

    mkdir -p "${OUTPUT_DIR}"

    accelerate launch \
        --config_file "${CONFIG_DIR}/accelerate_ddp_4gpu.yaml" \
        train.py \
        --model "${MODEL_DIR}" \
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

# ── Evaluation ────────────────────────────────────────────────────────
eval)
    echo "=== Evaluating trained model (LoRA merged) ==="
    MERGED_DIR="${SCRATCH}/merged/online_grpo_gsm8k_merged"
    python /project/aip-szepesva/mrli/backup_dongheng/Math_Verifier/eval_unified.py \
        --model_path "${OUTPUT_DIR}" \
        --base_model "${MODEL_DIR}" \
        --merge_lora \
        --merged_output "${MERGED_DIR}" \
        --temperature 0.0 \
        --runs 5
    echo "=== Evaluation complete ==="
    ;;

eval-baseline)
    echo "=== Evaluating baseline (no training) ==="
    python /project/aip-szepesva/mrli/backup_dongheng/Math_Verifier/eval_unified.py \
        --model_path "${MODEL_DIR}" \
        --temperature 0.0 \
        --runs 5
    echo "=== Baseline evaluation complete ==="
    ;;

eval30)
    echo "=== Reliable eval: 30 runs, temp=0.0 ==="
    MERGED_DIR="${SCRATCH}/merged/online_grpo_gsm8k_merged"
    echo "--- Trained model ---"
    python /project/aip-szepesva/mrli/backup_dongheng/Math_Verifier/eval_unified.py \
        --model_path "${MERGED_DIR}" \
        --temperature 0.0 \
        --runs 30
    echo "--- Baseline ---"
    python /project/aip-szepesva/mrli/backup_dongheng/Math_Verifier/eval_unified.py \
        --model_path "${MODEL_DIR}" \
        --temperature 0.0 \
        --runs 30
    echo "=== eval30 complete ==="
    ;;

eval-checkpoints)
    echo "=== Evaluating multiple checkpoints (temp=0.0, 5 runs) ==="
    for step in 5000 6000 7000; do
        ckpt="${OUTPUT_DIR}/checkpoint-${step}"
        merged="${SCRATCH}/merged/online_grpo_gsm8k_merged_step${step}"
        echo "--- Checkpoint step ${step} ---"
        python /project/aip-szepesva/mrli/backup_dongheng/Math_Verifier/eval_unified.py \
            --model_path "${ckpt}" \
            --base_model "${MODEL_DIR}" \
            --merge_lora \
            --merged_output "${merged}" \
            --temperature 0.0 \
            --runs 5
    done
    echo "--- Baseline ---"
    python /project/aip-szepesva/mrli/backup_dongheng/Math_Verifier/eval_unified.py \
        --model_path "${MODEL_DIR}" \
        --temperature 0.0 \
        --runs 5
    echo "=== eval-checkpoints complete ==="
    ;;

*)
    echo "Usage: sbatch --job-name=<name> run_gsm8k.sh <command>"
    echo ""
    echo "Training (4x L40s):"
    echo "  train          - Online GRPO with LoRA on GSM8K"
    echo ""
    echo "Evaluation (1x L40s):"
    echo "  eval           - Evaluate trained model (merge LoRA)"
    echo "  eval-baseline  - Evaluate base model (no training)"
    echo "  eval30         - Both models, temp=0.0, 30 runs"
    echo "  eval-checkpoints - Eval step 5000/6000/7000 + baseline"
    exit 1
    ;;

esac
