#!/bin/bash
# Verify DeepSeek-R1-Distill-Qwen-7B on MATH-500 using DeepSeek's recommended setup:
# no system prompt, directive in user message, max_tokens=16384.
set -euo pipefail
SCRATCH="${SCRATCH:-/scratch/mrli}"
TEACHER_MODEL="${TEACHER_MODEL:-${SCRATCH}/models/DeepSeek-R1-Distill-Qwen-7B}"
SAVE_COMPLETIONS="${SAVE_COMPLETIONS:-${SCRATCH}/r1_eval_final.jsonl}"

module load python/3.11 cuda/12.6 arrow opencv
source /project/aip-szepesva/mrli/backup_dongheng/.venv/bin/activate
export HF_HOME="${SCRATCH}"
export HF_DATASETS_CACHE="${SCRATCH}/datasets/MATH"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd /project/aip-szepesva/mrli/backup_dongheng

echo "=== R1-Teacher eval (FINAL: full DeepSeek pipeline) ==="
echo "  comparator:    extract_math_answer + strip_string + math_equal (DeepSeek-Math port)"
echo "  max_tokens:    32768 (matches DeepSeek's protocol)"
echo "  runs:          16   (4x our prior 4-run; std should be ~0.14pp)"
echo
python Math_Verifier/eval_r1_teacher.py \
    --model_path "${TEACHER_MODEL}" \
    --runs 16 \
    --temperature 0.6 \
    --top_p 0.95 \
    --max_tokens 32768 \
    --max_model_len 33280 \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.90 \
    --seed 42 \
    --save_completions "${SAVE_COMPLETIONS}"

echo "=== Eval complete ==="
