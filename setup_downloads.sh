#!/bin/bash
#
# Download models and datasets to scratch for offline use on compute nodes.
# Run this on the login node (has internet access).
#
# Models are downloaded directly (not via HF cache) to /scratch/mrli/models/
# Datasets are cached to /scratch/mrli/datasets/
#
# Usage:
#   bash setup_downloads.sh models     # Download both models (~16GB)
#   bash setup_downloads.sh datasets   # Download GSM8K and MATH datasets
#   bash setup_downloads.sh all        # Download everything
#

set -e

VENV_DIR="/project/aip-szepesva/mrli/backup_dongheng/.venv"
SCRATCH="/scratch/mrli"
MODELS_DIR="${SCRATCH}/models"
DATASETS_DIR="${SCRATCH}/datasets"

# Activate environment
module load python/3.11 cuda/12.6 arrow opencv
source "${VENV_DIR}/bin/activate"

download_models() {
    echo "=== Downloading Qwen2.5-0.5B-Instruct (student, ~1GB) ==="
    mkdir -p "${MODELS_DIR}"
    huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir "${MODELS_DIR}/Qwen2.5-0.5B-Instruct"

    echo ""
    echo "=== Downloading Qwen2.5-Math-7B-Instruct (teacher, ~15GB) ==="
    huggingface-cli download Qwen/Qwen2.5-Math-7B-Instruct --local-dir "${MODELS_DIR}/Qwen2.5-Math-7B-Instruct"

    echo ""
    echo "=== Models downloaded ==="
    echo "Student: ${MODELS_DIR}/Qwen2.5-0.5B-Instruct"
    echo "Teacher: ${MODELS_DIR}/Qwen2.5-Math-7B-Instruct"
}

download_datasets() {
    echo "=== Downloading GSM8K dataset ==="
    export HF_DATASETS_CACHE="${DATASETS_DIR}/GSM8K"
    python -c "from datasets import load_dataset; ds = load_dataset('openai/gsm8k', 'main'); print(f'GSM8K: {len(ds[\"train\"])} train, {len(ds[\"test\"])} test')"

    echo ""
    echo "=== Downloading MATH dataset ==="
    export HF_DATASETS_CACHE="${DATASETS_DIR}/MATH"
    python -c "from datasets import load_dataset; ds = load_dataset('nlile/hendrycks-MATH-benchmark'); print(f'MATH downloaded: {list(ds.keys())}')"

    echo ""
    echo "=== Datasets downloaded to ${DATASETS_DIR}/ ==="
}

case "${1}" in
models)
    download_models
    ;;
datasets)
    download_datasets
    ;;
all)
    download_models
    download_datasets
    ;;
*)
    echo "Usage: bash setup_downloads.sh {models|datasets|all}"
    echo ""
    echo "  models    - Download Qwen2.5-0.5B-Instruct + Qwen2.5-Math-7B-Instruct"
    echo "  datasets  - Download GSM8K + MATH datasets"
    echo "  all       - Download everything"
    exit 1
    ;;
esac
