#!/bin/bash
#
# Submit a lean audit chain (wave-1 + 1 dependency-gated wave-2 job) for a
# single trained checkpoint. Reports greedy + pass@1 only — pass@16 was
# dropped 2026-05-04 to halve audit cost.
#
#   - Wave 1: MERGE=1, MODE=both, RUNS=30 greedy + 30 pass@1 (N=1 sample at
#             T=0.6), SEED=42
#   - Wave 2: MODE=greedy, RUNS=30, SEED=100  (second greedy half -> 60-seed pool)
# Wave 2 depends on wave-1's success (afterok).
#
# Usage:
#   bash shared/submit_audit.sh <ADAPTER_DIR> <MERGED_DIR> <JOB_PREFIX>
#
# Env overrides:
#   BASE          base model path (default: Qwen2.5-0.5B-Instruct)
#   DATASET       dataset name (default: math)
#   MAX_TOKENS    generation max_tokens (default: 16384, fits R1-trained verbose students)
#   MAX_MODEL_LEN vLLM max_model_len (default: 16640, matches BC v4-liger training context)
#   MERGE_FLAG    1 (default) for LoRA adapters: wave-1 merges, wave-2 uses merged dir.
#                 0 for full models (no adapter): no merge step; ADAPTER and MERGED
#                 should be the same path (the model itself).
#   AUDIT_TIME    walltime override for wave-1 (default 06:00:00)

set -euo pipefail

ADAPTER="${1:?Usage: submit_audit.sh ADAPTER MERGED PREFIX}"
MERGED="${2:?Usage: submit_audit.sh ADAPTER MERGED PREFIX}"
PREFIX="${3:?Usage: submit_audit.sh ADAPTER MERGED PREFIX}"

SCRATCH="${SCRATCH:-/scratch/mrli}"
BASE="${BASE:-${SCRATCH}/models/Qwen2.5-0.5B-Instruct}"
DATASET="${DATASET:-math}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16640}"
MERGE_FLAG="${MERGE_FLAG:-1}"
AUDIT_TIME="${AUDIT_TIME:-06:00:00}"

LAUNCHER="shared/run_eval.sh"

COMMON_VARS="DATASET=${DATASET},MAX_TOKENS=${MAX_TOKENS},MAX_MODEL_LEN=${MAX_MODEL_LEN}"

echo "=== Submitting audit chain for ${PREFIX} ==="
echo "  adapter:       ${ADAPTER}"
echo "  merged out:    ${MERGED}"
echo "  base:          ${BASE}"
echo "  dataset:       ${DATASET}"
echo "  max_tokens:    ${MAX_TOKENS}"
echo "  max_model_len: ${MAX_MODEL_LEN}"

# Wave 1: merge + both modes (30-seed greedy + 16-sample best_of_n at SEED=42).
# Bigger walltime since it does both modes plus the merge.
if [ "${MERGE_FLAG}" = "1" ]; then
    W1_EXPORT="ALL,MODEL=${ADAPTER},BASE=${BASE},MERGE=1,MERGED_DIR=${MERGED},MODE=both,RUNS=30,N_SAMPLES=1,TEMP=0.6,SEED=42,${COMMON_VARS}"
else
    # Full-model path: no merge. ADAPTER is the model itself; wave-2 uses
    # the same path (MERGED should be set to the same value by the caller).
    W1_EXPORT="ALL,MODEL=${ADAPTER},MERGE=0,MODE=both,RUNS=30,N_SAMPLES=1,TEMP=0.6,SEED=42,${COMMON_VARS}"
fi
J1=$(sbatch --parsable \
    --time="${AUDIT_TIME}" \
    --job-name="${PREFIX}-w1" \
    --export="${W1_EXPORT}" \
    "${LAUNCHER}")
echo "  wave-1 (greedy 30 + pass@1 30):  ${J1}"

# Wave 2: greedy 30-seed at SEED=100 -> pools with wave-1 greedy for 60-seed.
J2=$(sbatch --parsable \
    --time=04:00:00 \
    --job-name="${PREFIX}-w2g" \
    --dependency="afterok:${J1}" \
    --export="ALL,MODEL=${MERGED},MODE=greedy,RUNS=30,SEED=100,${COMMON_VARS}" \
    "${LAUNCHER}")
echo "  wave-2  (greedy 30-seed @100):   ${J2}"

echo "=== ${PREFIX} chain submitted: ${J1} -> ${J2} ==="
