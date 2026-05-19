#!/bin/bash
# Quick re-eval of the old teacher (Qwen2.5-Math-7B-Instruct) on MATH-500
# under the upgraded `is_equiv_multi` comparator. Apples-to-apples with
# our prior 74.96% greedy measurement; the only thing that changed is the
# verifier.
set -euo pipefail
SCRATCH="${SCRATCH:-/scratch/mrli}"
TEACHER_MODEL="${TEACHER_MODEL:-${SCRATCH}/models/Qwen2.5-Math-7B-Instruct}"

module load python/3.11 cuda/12.6 arrow opencv
source /project/aip-szepesva/mrli/backup_dongheng/.venv/bin/activate
export HF_HOME="${SCRATCH}"
export HF_DATASETS_CACHE="${SCRATCH}/datasets/MATH"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd /project/aip-szepesva/mrli/backup_dongheng/mixture_grpo

echo "=== Qwen2.5-Math-7B-Instruct on MATH-500 (upgraded is_equiv_multi) ==="
echo "  Mode: greedy, single run, max_tokens=2048 (matches our prior 74.96% measurement)"

python /project/aip-szepesva/mrli/backup_dongheng/Math_Verifier/eval_unified.py \
    --model_path "${TEACHER_MODEL}" \
    --dataset_type math \
    --runs 1 \
    --temperature 0.0 \
    --max_tokens 2048 \
    --max_model_len 3072 \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.90 \
    --seed 42

echo "=== Eval complete ==="
