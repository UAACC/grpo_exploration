#!/bin/bash
#
# Measure teacher-rollout surprisal distribution under the student policy,
# across all 4 datasets. Informs whether DG's sigmoid gate is saturating or
# operating in its linear regime.
#
# Usage:
#   sbatch DG-offline/run_measure_surprisal.sh
#
#SBATCH --account=aip-szepesva
#SBATCH --job-name=measure-surprisal
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=DG-offline/logs/%x-%j.out
#SBATCH --error=DG-offline/logs/%x-%j.err

set -euo pipefail

WORK_DIR="/project/aip-szepesva/mrli/backup_dongheng"
SCRATCH="/scratch/mrli"

module load python/3.11 cuda/12.6 arrow opencv
source "${WORK_DIR}/.venv/bin/activate"
export HF_HOME="${SCRATCH}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "${WORK_DIR}/DG-offline/logs"
mkdir -p "${WORK_DIR}/DG-offline/diagnostics"

cd "${WORK_DIR}"

for dataset in math gsm8k svamp asdiv; do
  echo ""
  echo "################################################"
  echo "# Dataset: ${dataset}"
  echo "################################################"
  python DG-offline/measure_surprisal.py \
    --dataset "${dataset}" \
    --student "${SCRATCH}/models/Qwen2.5-0.5B-Instruct" \
    --max_questions 400 \
    --output "${WORK_DIR}/DG-offline/diagnostics/surprisal_${dataset}.json"
done

echo ""
echo "=== Done. Per-dataset JSON dumps in DG-offline/diagnostics/ ==="
