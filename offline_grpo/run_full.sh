#!/bin/bash
#
# Full-scale offline GRPO pipeline
#   Teacher (behavior model): Qwen2.5-Math-7B-Instruct
#   Student (target model):   Qwen2.5-0.5B-Instruct
#   Data: full MATH training set (12,000 problems)
#
# Usage (from login node):
#   sbatch --job-name=grpo-rollouts run_full.sh rollouts     (~2.5 hours, 4 GPUs)
#   sbatch --job-name=grpo-train   run_full.sh train         (~1 hour, after rollouts done)
#   sbatch --job-name=grpo-eval    run_full.sh eval          (~30 min, after train done)
#
#SBATCH --account=aip-szepesva
#SBATCH --time=4:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -e

WORK_DIR="/project/aip-szepesva/mrli/backup_dongheng/offline_grpo"
SCRATCH="/scratch/mrli"

# Teacher (behavior model) — generates rollouts
TEACHER_DIR="${SCRATCH}/models/Qwen2.5-Math-7B-Instruct"

# Student (target model) — trained with offline GRPO
STUDENT_DIR="${SCRATCH}/models/Qwen2.5-0.5B-Instruct"

ROLLOUT_PATH="${SCRATCH}/rollouts/math_teacher/rollouts_full.jsonl"
OUTPUT_DIR="${SCRATCH}/checkpoints/offline_grpo_math"

# ── Activate environment ─────────────────────────────────────────
module load python/3.11 cuda/12.6 arrow opencv
source /project/aip-szepesva/mrli/backup_dongheng/.venv/bin/activate
export HF_HOME="${SCRATCH}"
export HF_DATASETS_CACHE="/scratch/mrli/datasets/MATH"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_API_KEY=$(cat /project/aip-szepesva/mrli/backup_dongheng/offline_grpo/.wandb_key)
cd "${WORK_DIR}"

export VLLM_WORKER_MULTIPROC_METHOD=spawn

case "${1}" in

rollouts)
    NUM_SHARDS=4
    echo "=== Generating full rollouts with teacher: ${TEACHER_DIR} ==="
    echo "=== Full MATH train set, 4 completions per problem, ${NUM_SHARDS} GPU shards ==="

    # Launch N parallel vLLM instances, each on its own GPU with its own shard
    for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
        SHARD_OUTPUT="${SCRATCH}/rollouts/math_teacher/rollouts_shard_${SHARD_ID}.jsonl"
        echo "  Launching shard ${SHARD_ID}/${NUM_SHARDS} on GPU ${SHARD_ID} → ${SHARD_OUTPUT}"
        CUDA_VISIBLE_DEVICES=${SHARD_ID} python generate_rollouts.py \
            --model_name "${TEACHER_DIR}" \
            --num_generations 4 \
            --temperature 0.6 \
            --max_tokens 2048 \
            --max_model_len 3072 \
            --gpu_memory_utilization 0.85 \
            --num_shards ${NUM_SHARDS} \
            --shard_id ${SHARD_ID} \
            --output_path "${SHARD_OUTPUT}" &
    done

    echo "=== Waiting for all ${NUM_SHARDS} shards to finish ==="
    wait
    echo "=== All shards complete ==="

    # Merge shards into final file, sorted by question_id
    echo "=== Merging shards into ${ROLLOUT_PATH} ==="
    python -c "
import json, glob
shards = sorted(glob.glob('${SCRATCH}/rollouts/math_teacher/rollouts_shard_*.jsonl'))
records = []
for path in shards:
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
records.sort(key=lambda r: r['question_id'])
with open('${ROLLOUT_PATH}', 'w') as f:
    for r in records:
        f.write(json.dumps(r) + '\n')
print(f'Merged {len(records)} records from {len(shards)} shards')
"
    echo "=== Rollouts saved to ${ROLLOUT_PATH} ==="
    ;;

validate)
    echo "=== Validating rollouts ==="
    python validate_rollouts.py \
        --rollout_path "${ROLLOUT_PATH}" \
        --shard_dir "${SCRATCH}/rollouts/math_teacher" \
        --expected_problems 12000 \
        --num_generations 4
    ;;

train)
    echo "=== Training student (full scale): ${STUDENT_DIR} ==="
    export WANDB_PROJECT="offline-grpo"
    export WANDB_RUN_NAME="qwen05b-full-$(date +%Y%m%d_%H%M%S)"
    python train.py \
        --rollout_path "${ROLLOUT_PATH}" \
        --target_model "${STUDENT_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --num_generations 4 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --max_completion_length 786 \
        --save_steps 500 \
        --logging_steps 10 \
        --report_to wandb \
        --resume_from_checkpoint latest
    echo "=== Training complete. Model saved to ${OUTPUT_DIR} ==="
    ;;

eval)
    echo "=== Evaluating trained model (LoRA merged) ==="
    MERGED_DIR="${SCRATCH}/merged/offline_grpo_math_merged"
    python /project/aip-szepesva/mrli/backup_dongheng/Math_Verifier/eval_unified.py \
        --model_path "${OUTPUT_DIR}" \
        --base_model "${STUDENT_DIR}" \
        --merge_lora \
        --merged_output "${MERGED_DIR}" \
        --runs 1
    echo "=== Evaluation complete ==="
    ;;

eval-baseline)
    echo "=== Evaluating baseline (no LoRA) ==="
    python /project/aip-szepesva/mrli/backup_dongheng/Math_Verifier/eval_unified.py \
        --model_path "${STUDENT_DIR}" \
        --runs 1
    echo "=== Baseline evaluation complete ==="
    ;;

*)
    echo "Usage: sbatch --job-name=<name> run_full.sh {rollouts|validate|train|eval|eval-baseline}"
    echo ""
    echo "  rollouts       - Generate rollouts with 7B teacher (~2.5 hours, 4 GPUs)"
    echo "  validate       - Validate rollouts integrity (run after rollouts)"
    echo "  train          - Train 0.5B student with offline GRPO (~1 hour)"
    echo "  eval           - Evaluate trained model with LoRA merged (~30 min)"
    echo "  eval-baseline  - Evaluate base student without LoRA (~30 min)"
    exit 1
    ;;

esac
