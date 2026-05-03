#!/bin/bash
#
# Submit one full audit-suite chain (wave-1 + 3 dependency-gated wave-2 jobs)
# for a single trained checkpoint. Implements the eval audit protocol:
#   - Wave 1: MERGE=1, MODE=both, RUNS=30 greedy + 16-sample best_of_n, SEED=42
#   - Wave 2a: MODE=greedy, RUNS=30, SEED=100 (second greedy half -> 60-seed pool)
#   - Wave 2b: MODE=best_of_n, N=16, SEED=43
#   - Wave 2c: MODE=best_of_n, N=16, SEED=44
# All wave-2 jobs depend on wave-1's success (afterok).
#
# Usage:
#   bash shared/submit_audit.sh <ADAPTER_DIR> <MERGED_DIR> <JOB_PREFIX>
#
# Env overrides:
#   BASE          base model path (default: Qwen2.5-0.5B-Instruct)
#   DATASET       dataset name (default: math)
#   MAX_TOKENS    generation max_tokens (default: 16384, fits R1-trained verbose students)
#   MAX_MODEL_LEN vLLM max_model_len (default: 16640, matches BC v4-liger training context)

set -euo pipefail

ADAPTER="${1:?Usage: submit_audit.sh ADAPTER MERGED PREFIX}"
MERGED="${2:?Usage: submit_audit.sh ADAPTER MERGED PREFIX}"
PREFIX="${3:?Usage: submit_audit.sh ADAPTER MERGED PREFIX}"

SCRATCH="${SCRATCH:-/scratch/mrli}"
BASE="${BASE:-${SCRATCH}/models/Qwen2.5-0.5B-Instruct}"
DATASET="${DATASET:-math}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16640}"

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
J1=$(sbatch --parsable \
    --time=06:00:00 \
    --job-name="${PREFIX}-w1" \
    --export="ALL,MODEL=${ADAPTER},BASE=${BASE},MERGE=1,MERGED_DIR=${MERGED},MODE=both,RUNS=30,N_SAMPLES=16,TEMP=0.6,SEED=42,${COMMON_VARS}" \
    "${LAUNCHER}")
echo "  wave-1 (merge + both):           ${J1}"

# Wave 2a: greedy 30-seed at SEED=100 -> pools with wave-1 greedy for 60-seed.
J2A=$(sbatch --parsable \
    --time=04:00:00 \
    --job-name="${PREFIX}-w2g" \
    --dependency="afterok:${J1}" \
    --export="ALL,MODEL=${MERGED},MODE=greedy,RUNS=30,SEED=100,${COMMON_VARS}" \
    "${LAUNCHER}")
echo "  wave-2a (greedy 30-seed @100):   ${J2A}"

# Wave 2b: best_of_n at SEED=43.
J2B=$(sbatch --parsable \
    --time=03:00:00 \
    --job-name="${PREFIX}-w2b1" \
    --dependency="afterok:${J1}" \
    --export="ALL,MODEL=${MERGED},MODE=best_of_n,N_SAMPLES=16,TEMP=0.6,SEED=43,${COMMON_VARS}" \
    "${LAUNCHER}")
echo "  wave-2b (best_of_n @43):         ${J2B}"

# Wave 2c: best_of_n at SEED=44.
J2C=$(sbatch --parsable \
    --time=03:00:00 \
    --job-name="${PREFIX}-w2b2" \
    --dependency="afterok:${J1}" \
    --export="ALL,MODEL=${MERGED},MODE=best_of_n,N_SAMPLES=16,TEMP=0.6,SEED=44,${COMMON_VARS}" \
    "${LAUNCHER}")
echo "  wave-2c (best_of_n @44):         ${J2C}"

echo "=== ${PREFIX} chain submitted: ${J1} -> {${J2A}, ${J2B}, ${J2C}} ==="
