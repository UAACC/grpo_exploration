#!/bin/bash
#
# Submit all 7 R1-rollout training configs in parallel.
#
# Configs:
#   1. BC-all                         on R1 rollouts          (bc/run_bc_math.sh)
#   2. BC-correct-only                on R1 rollouts          (bc/run_bc_correct_only_math.sh)
#   3. Offline-GRPO                   on cleaned R1 rollouts  (offline_grpo/run.sh train)
#   4. DG-offline η=0.1               on R1 rollouts          (DG-offline/run_math.sh)
#   5. DG-offline η=0.5               on R1 rollouts          (DG-offline/run_math.sh)
#   6. DG-offline η=1.0               on R1 rollouts          (DG-offline/run_math.sh)
#   7. DG-offline η=2.0               on R1 rollouts          (DG-offline/run_math.sh)
#
# All configs train at max_completion_length=8192 to match R1's verbose
# reasoning style (avg ~4K tokens, max up to 32K).

set -euo pipefail

SCRATCH="/scratch/mrli"
ROLLOUT_PATH_R1="${SCRATCH}/rollouts/math_deepseek_r1/rollouts_full.jsonl"
ROLLOUT_PATH_R1_CLEANED="${SCRATCH}/rollouts/math_deepseek_r1_cleaned/rollouts_full.jsonl"

if [ ! -f "${ROLLOUT_PATH_R1}" ]; then
    echo "ERROR: R1 rollout file missing: ${ROLLOUT_PATH_R1}" >&2
    exit 1
fi
if [ ! -f "${ROLLOUT_PATH_R1_CLEANED}" ]; then
    echo "WARNING: cleaned R1 rollout file missing: ${ROLLOUT_PATH_R1_CLEANED}" >&2
    echo "         OG config (#3) will be SKIPPED. Run prepare_cleaned_og_rollouts first." >&2
    SKIP_OG=1
fi

# Common student-tokenization budget for R1 rollouts.
MAX_COMPL=8192
MAX_PROMPT=256
MAX_MODEL_LEN=$((MAX_COMPL + MAX_PROMPT + 256))   # 8704

# Memory budget on L40s 44GB at 8K context requires per_device=1 for the 0.5B
# student + LoRA + Adam states.
#
# BC at per_device=1, grad_accum=8 → effective batch 32 (training fine, ~2s/iter).
# DG additionally batches `num_generations` per prompt in TRL's GRPOTrainer, so
# the same grad_accum value blows past memory. DG needs grad_accum=1 (effective
# batch = num_gen=4, the smallest valid setting given num_generations=4).
PER_DEVICE_BATCH=1
BC_GRAD_ACCUM=8
DG_GRAD_ACCUM=1

# Allow skipping configs whose previous launch is still in good shape.
SUBMIT_BC="${SUBMIT_BC:-1}"
SUBMIT_DG="${SUBMIT_DG:-1}"
SUBMIT_OG="${SUBMIT_OG:-1}"

cd /project/aip-szepesva/mrli/backup_dongheng

submit() {
    local jobname="$1"; shift
    echo "==> Submitting ${jobname}"
    sbatch --job-name="${jobname}" "$@"
    echo
}

# 1. BC-all on R1
if [ "${SUBMIT_BC}" = "1" ]; then
    submit "bc-all-r1" \
        --export=ALL,ROLLOUT_PATH="${ROLLOUT_PATH_R1}",OUTPUT_DIR="${SCRATCH}/checkpoints/bc_math_r1",MAX_LENGTH="${MAX_MODEL_LEN}",PER_DEVICE_BATCH="${PER_DEVICE_BATCH}",GRAD_ACCUM="${BC_GRAD_ACCUM}",WANDB_RUN_NAME="bc-all-r1-$(date +%m%d)" \
        bc/run_bc_math.sh

    # 2. BC-correct-only on R1
    submit "bc-cc-r1" \
        --export=ALL,ROLLOUT_PATH="${ROLLOUT_PATH_R1}",OUTPUT_DIR="${SCRATCH}/checkpoints/bc_math_correct_only_r1",MAX_LENGTH="${MAX_MODEL_LEN}",PER_DEVICE_BATCH="${PER_DEVICE_BATCH}",GRAD_ACCUM="${BC_GRAD_ACCUM}",WANDB_RUN_NAME="bc-cc-r1-$(date +%m%d)" \
        bc/run_bc_correct_only_math.sh
fi

# 3. Offline-GRPO on cleaned R1 (only if cleaned dataset exists)
if [ "${SUBMIT_OG}" = "1" ] && [ -z "${SKIP_OG:-}" ]; then
    submit "og-r1" \
        --export=ALL,DATASET_TYPE=math,ROLLOUT_PATH="${ROLLOUT_PATH_R1_CLEANED}",OUTPUT_DIR="${SCRATCH}/checkpoints/offline_grpo_math_r1",MERGED_DIR="${SCRATCH}/merged/offline_grpo_math_r1_merged",MAX_COMPLETION="${MAX_COMPL}",MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
        offline_grpo/run.sh train
fi

# 4-7. DG-offline at four η values
if [ "${SUBMIT_DG}" = "1" ]; then
    for ETA in 0.1 0.5 1.0 2.0; do
        SAFE_ETA="${ETA/./_}"   # 0.5 -> 0_5
        submit "dg-r1-eta${SAFE_ETA}" \
            --export=ALL,ROLLOUT_PATH="${ROLLOUT_PATH_R1}",CHECKPOINT_DIR="${SCRATCH}/checkpoints/dg_offline_math_r1_eta${SAFE_ETA}",MERGED_DIR="${SCRATCH}/merged/dg_offline_math_r1_eta${SAFE_ETA}",DG_ETA="${ETA}",MAX_COMPLETION_LENGTH="${MAX_COMPL}",MAX_PROMPT_LENGTH="${MAX_PROMPT}",PER_DEVICE_BATCH="${PER_DEVICE_BATCH}",GRAD_ACCUM="${DG_GRAD_ACCUM}",WANDB_PROJECT="dg-offline-math-r1" \
            DG-offline/run_math.sh
    done
fi

echo "=== All R1 training configs submitted ==="
squeue -u "$USER" 2>&1 | head
