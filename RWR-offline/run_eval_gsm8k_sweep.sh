#!/bin/bash
#
# Evaluate all DG-offline GSM8K checkpoints (greedy + pass@16)
# Run AFTER all training jobs have completed.
#
# Usage: sbatch run_eval_gsm8k_sweep.sh
#
#SBATCH --account=aip-szepesva
#SBATCH --job-name=eval-dg-gsm8k-sweep
#SBATCH --time=03:00:00
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module load python/3.11 cuda/12.6 arrow opencv
source /project/aip-szepesva/mrli/backup_dongheng/.venv/bin/activate
export HF_HOME=/scratch/mrli
export HF_DATASETS_CACHE=/scratch/mrli/datasets/GSM8K
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

SCRATCH=/scratch/mrli
STUDENT="${SCRATCH}/models/Qwen2.5-0.5B-Instruct"
EVAL_SCRIPT="/project/aip-szepesva/mrli/backup_dongheng/mixture_grpo/evaluate.py"
BON_SCRIPT="/project/aip-szepesva/mrli/backup_dongheng/bc/eval_best_of_n.py"

ETAS="0.1 0.5 1.0 2.0"

echo "=== DG-offline GSM8K Evaluation Sweep ==="

for eta in $ETAS; do
    CKPT="${SCRATCH}/checkpoints/dg_offline_gsm8k_eta${eta}"
    MERGED="${SCRATCH}/merged/dg_offline_gsm8k_eta${eta}"

    if [ ! -d "${CKPT}" ]; then
        echo "SKIP eta=${eta}: checkpoint not found at ${CKPT}"
        continue
    fi

    echo ""
    echo "=============================="
    echo "  eta=${eta}"
    echo "=============================="

    # Step 1: Merge LoRA
    echo "--- Merging LoRA ---"
    python3 -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained('${STUDENT}', torch_dtype=torch.bfloat16, device_map='cpu')
model = PeftModel.from_pretrained(base, '${CKPT}')
model = model.merge_and_unload()
model.save_pretrained('${MERGED}')
tok = AutoTokenizer.from_pretrained('${STUDENT}')
tok.save_pretrained('${MERGED}')
print('Merged to ${MERGED}')
"

    # Step 2: Greedy eval (5 runs)
    echo "--- Greedy eval (pass@1) ---"
    python3 "${EVAL_SCRIPT}" \
        --model_path "${MERGED}" \
        --dataset_type gsm8k \
        --runs 5 \
        --temperature 0.0 \
        --max_tokens 1024 \
        --max_model_len 2048

    # Step 3: Best-of-16
    echo "--- Best-of-16 (pass@16) ---"
    python3 "${BON_SCRIPT}" \
        --model_path "${MERGED}" \
        --dataset_type gsm8k \
        --n_samples 16 \
        --temperature 0.6 \
        --num_problems 1319 \
        --max_tokens 1024 \
        --max_model_len 2048

done
echo ""
echo "=== All evaluations complete ==="
