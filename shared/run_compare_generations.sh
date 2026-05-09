#!/bin/bash
#
# Diagnostic: generate greedy completions on the same 50 MATH-500 problems
# across all 6 R1-trained students (BC×2, DG×4), so we can compare what each
# model emits — completion length distribution, whether `\boxed{}` appears,
# and qualitative differences in reasoning style.
#
# Output: /scratch/mrli/eval_outputs/diag_<model_name>.jsonl per model,
#         each with one JSONL line per (problem, completion) containing
#         {problem, gold, generated_text, completion_token_count, candidates,
#          correct}.
#
# Usage:
#   sbatch shared/run_compare_generations.sh
#
#SBATCH --account=aip-szepesva
#SBATCH --job-name=diag-r1-compare
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/diag-r1-compare-%j.out
#SBATCH --error=logs/diag-r1-compare-%j.err

set -euo pipefail

SCRATCH="${SCRATCH:-/scratch/mrli}"
WORK_DIR="/project/aip-szepesva/mrli/backup_dongheng"

module load python/3.11 cuda/12.6 arrow opencv
source "${WORK_DIR}/.venv/bin/activate"

export HF_HOME="${SCRATCH}"
export HF_DATASETS_CACHE="${SCRATCH}/datasets/MATH"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p "${SCRATCH}/eval_outputs"
cd "${WORK_DIR}"

MODELS=(
    "bc-all:bc_math_r1_v4_liger"
    "bc-cc:bc_math_correct_only_r1_v4_liger"
    "dg-eta0p1:dg_offline_math_r1_eta0_1"
    "dg-eta0p5:dg_offline_math_r1_eta0_5"
    "dg-eta1p0:dg_offline_math_r1_eta1_0"
    "dg-eta2p0:dg_offline_math_r1_eta2_0"
)

echo "=== Diagnostic generation comparison across 6 R1 students ==="
echo "  problems: 50 (fixed seed=42)"
echo "  mode:     greedy (T=0.0)"
echo "  max_tok:  16384"

for spec in "${MODELS[@]}"; do
    short="${spec%%:*}"
    dir="${spec##*:}"
    model="${SCRATCH}/merged/${dir}_merged"
    out="${SCRATCH}/eval_outputs/diag_${short}.jsonl"

    echo
    echo "----- ${short} (${model}) -----"
    python Math_Verifier/eval_unified.py \
        --model_path "${model}" \
        --dataset math \
        --mode greedy \
        --runs 1 \
        --seed 42 \
        --max_problems 50 \
        --max_tokens 16384 \
        --max_model_len 16640 \
        --tensor_parallel_size 1 \
        --gpu_memory_utilization 0.85 \
        --save_completions "${out}"

    echo "  -> ${out}"
done

echo
echo "=== All 6 diagnostics complete ==="
ls -la "${SCRATCH}/eval_outputs/diag_"*.jsonl
