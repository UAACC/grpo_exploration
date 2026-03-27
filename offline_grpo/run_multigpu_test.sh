#!/bin/bash
#
# Multi-GPU LoRA training test matrix
#
# Tests 3 distributed strategies x 2 beta values = 6 combinations
# Uses: 2x L40s GPUs, Qwen2.5-0.5B-Instruct, rollouts_test.jsonl
#
# Usage:
#   1. Get 2 GPUs:
#      salloc --account=aip-szepesva --time=1:00:00 \
#             --gpus-per-node=l40s:2 --cpus-per-task=32 --mem=96G
#   2. On compute node:
#      bash run_multigpu_test.sh all       # run full test matrix
#      bash run_multigpu_test.sh ddp       # run DDP tests only
#      bash run_multigpu_test.sh zero2     # run ZeRO-2 tests only
#      bash run_multigpu_test.sh fsdp      # run FSDP tests only
#
# Expected results:
#   ddp_beta0.0   PASS     ddp_beta0.1   PASS
#   zero2_beta0.0 PASS     zero2_beta0.1 PASS
#   fsdp_beta0.0  PASS     fsdp_beta0.1  FAIL (disable_adapter bug)

set -uo pipefail

WORK_DIR="/project/aip-szepesva/mrli/backup_dongheng/offline_grpo"
SCRATCH="/scratch/mrli"
STUDENT_DIR="${SCRATCH}/models/Qwen2.5-0.5B-Instruct"
ROLLOUT_PATH="${WORK_DIR}/rollouts_test.jsonl"
CONFIG_DIR="${WORK_DIR}/configs"
LOG_DIR="${WORK_DIR}/logs/multigpu_test_$(date +%Y%m%d_%H%M%S)"

# ── Activate environment ─────────────────────────────────────────
module load python/3.11 cuda/12.6 arrow opencv
source /project/aip-szepesva/mrli/backup_dongheng/.venv/bin/activate
export HF_HOME="${SCRATCH}"
export HF_DATASETS_CACHE="/scratch/mrli/datasets/MATH"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=INFO
cd "${WORK_DIR}"

mkdir -p "${LOG_DIR}"

# ── Helper function ──────────────────────────────────────────────
run_test() {
    local strategy="$1"
    local beta="$2"
    local config_file="$3"
    local test_name="${strategy}_beta${beta}"
    local output_dir="${LOG_DIR}/output_${test_name}"
    local log_file="${LOG_DIR}/${test_name}.log"

    echo ""
    echo "============================================================"
    echo "TEST: ${test_name}"
    echo "  Strategy: ${strategy}"
    echo "  Beta: ${beta}"
    echo "  Config: ${config_file}"
    echo "  Output: ${output_dir}"
    echo "============================================================"

    local start_time=$(date +%s)
    set +e

    timeout 600 accelerate launch \
        --config_file "${config_file}" \
        train.py \
        --rollout_path "${ROLLOUT_PATH}" \
        --target_model "${STUDENT_DIR}" \
        --output_dir "${output_dir}" \
        --num_generations 4 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --max_completion_length 512 \
        --beta "${beta}" \
        --report_to none \
        2>&1 | tee "${log_file}"

    local exit_code=$?
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))

    set -e

    if [ ${exit_code} -eq 0 ]; then
        echo ">>> RESULT: ${test_name} PASSED (${elapsed}s)"
    elif [ ${exit_code} -eq 124 ]; then
        echo ">>> RESULT: ${test_name} TIMEOUT after ${elapsed}s"
    else
        echo ">>> RESULT: ${test_name} FAILED (exit=${exit_code}, ${elapsed}s)"
    fi
    echo "${test_name}: exit=${exit_code}, time=${elapsed}s" >> "${LOG_DIR}/summary.txt"
}

# ── Test functions ───────────────────────────────────────────────
run_ddp() {
    echo ""
    echo "######## DDP TESTS (expect both to PASS) ########"
    run_test "ddp" "0.0" "${CONFIG_DIR}/accelerate_ddp_2gpu.yaml"
    run_test "ddp" "0.1" "${CONFIG_DIR}/accelerate_ddp_2gpu.yaml"
}

run_zero2() {
    echo ""
    echo "######## DeepSpeed ZeRO-2 TESTS (expect both to PASS) ########"
    run_test "zero2" "0.0" "${CONFIG_DIR}/accelerate_zero2_2gpu.yaml"
    run_test "zero2" "0.1" "${CONFIG_DIR}/accelerate_zero2_2gpu.yaml"
}

run_fsdp() {
    echo ""
    echo "######## FSDP TESTS (beta=0 should PASS, beta=0.1 likely FAILS) ########"
    run_test "fsdp" "0.0" "${CONFIG_DIR}/accelerate_fsdp_2gpu.yaml"
    run_test "fsdp" "0.1" "${CONFIG_DIR}/accelerate_fsdp_2gpu.yaml"
}

# ── Main dispatch ────────────────────────────────────────────────
echo "Multi-GPU LoRA Training Test Matrix"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Log dir: ${LOG_DIR}"
echo ""
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null
echo ""

echo "=== Test Matrix ===" > "${LOG_DIR}/summary.txt"
echo "Date: $(date)" >> "${LOG_DIR}/summary.txt"
echo "Node: $(hostname)" >> "${LOG_DIR}/summary.txt"
echo "" >> "${LOG_DIR}/summary.txt"

case "${1:-all}" in
    ddp)   run_ddp   ;;
    zero2) run_zero2 ;;
    fsdp)  run_fsdp  ;;
    all)
        run_ddp
        run_zero2
        run_fsdp
        ;;
    *)
        echo "Usage: bash run_multigpu_test.sh {ddp|zero2|fsdp|all}"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "ALL TESTS COMPLETE"
echo "============================================================"
echo ""
cat "${LOG_DIR}/summary.txt"
echo ""
echo "Full logs in: ${LOG_DIR}"
