#!/bin/bash
#
# Offline GRPO on R1-Distill cleaned rollouts (config #7 of the multi-teacher
# experiment). Uses the cleaned dataset produced by
# shared/prepare_cleaned_og_rollouts.py: completions re-tokenized under the
# student tokenizer + teacher logprobs recomputed via transformers-direct
# forward pass. This avoids the silent-corruption case at R1's `<think>`
# (151648) / `</think>` (151649) special-token IDs, which collide with
# unrelated student-vocab special tokens.
#
# Hyperparameters mirror DG-offline R1 runs (per_device_batch=1, grad_accum=8,
# max_completion_length=8192) except num_generations is dropped from 4 to 2.
# Justification: TRL's GRPOTrainer materializes [batch*num_gen, seq, vocab]
# logits for both current and reference policy. At 4 gen × 8K seq × 152K vocab
# this exceeds 44 GiB even after disabling accelerate's fp32 cast. num_gen=2
# halves the peak. Advantages are precomputed by compute_rewards_and_advantages
# over the full 4-run group per question, so each (qid, rid)'s advantage is
# unaffected by the trainer's per-batch grouping with num_generations=2.
# The 8K completion cap right-truncates ~10% of R1's longest reasoning chains
# during the IS-ratio forward pass — acceptable for OG since the loss is
# advantage-weighted logprob ratio (truncation produces a noisier signal,
# not a wrong target, unlike BC where truncation deletes the \boxed{} answer).
#
# Usage:
#   sbatch --job-name=og-r1 offline_grpo/run_og_r1.sh
#
#SBATCH --account=aip-szepesva
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=offline_grpo/logs/og_r1/%x-%j.out
#SBATCH --error=offline_grpo/logs/og_r1/%x-%j.err

set -e

WORK_DIR="/project/aip-szepesva/mrli/backup_dongheng/offline_grpo"
SCRATCH="/scratch/mrli"
MODEL_DIR="${SCRATCH}/models/Qwen2.5-0.5B-Instruct"

ROLLOUT_PATH="${SCRATCH}/rollouts/math_deepseek_r1_cleaned/rollouts_full.jsonl"
OUTPUT_DIR="${SCRATCH}/checkpoints/offline_grpo_math_r1"

module load python/3.11 cuda/12.6 arrow opencv
source /project/aip-szepesva/mrli/backup_dongheng/.venv/bin/activate
export HF_HOME="${SCRATCH}"
export HF_DATASETS_CACHE="/scratch/mrli/datasets/MATH"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=$(cat /project/aip-szepesva/mrli/backup_dongheng/offline_grpo/.wandb_key)
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_PROJECT="offline-grpo-math"
export WANDB_RUN_NAME="og-r1-$(date +%m%d)"

cd "${WORK_DIR}"
mkdir -p logs/og_r1

METRICS_FILE="${OUTPUT_DIR}/training_metrics.jsonl"

echo "=== Offline GRPO on R1-Distill cleaned rollouts (config #7) ==="
echo "  Model:    ${MODEL_DIR}"
echo "  Rollouts: ${ROLLOUT_PATH}"
echo "  Output:   ${OUTPUT_DIR}"
echo "  wandb:    ${WANDB_PROJECT} / ${WANDB_RUN_NAME}"
echo
echo "  Hyperparameters (mirroring DG-offline R1 config):"
echo "    beta=0.001, lr=3e-6, max_grad_norm=1.0, weight_decay=0.01"
echo "    epochs=1, per_device_batch=1, grad_accum=8, num_gen=2"
echo "    max_completion_length=8192 (R1's 99th percentile is ~32K; 8K cap = ~10% right-truncation, acceptable for IS-ratio loss)"
echo "    lora_r=32, lora_alpha=32"

accelerate launch \
    --config_file "${WORK_DIR}/configs/accelerate_ddp_4gpu.yaml" \
    train.py \
    --rollout_path "${ROLLOUT_PATH}" \
    --target_model "${MODEL_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${WANDB_RUN_NAME}" \
    --num_generations 2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-6 \
    --beta 0.001 \
    --max_completion_length 8192 \
    --max_grad_norm 1.0 \
    --weight_decay 0.01 \
    --save_steps 500 \
    --logging_steps 5 \
    --ref_sync_steps 0 \
    --report_to wandb

echo "=== Training complete ==="
ls -la "${OUTPUT_DIR}"
