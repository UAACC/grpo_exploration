# GRPO Exploration: Off-Policy Reinforcement Learning for LLM Post-Training

This repository implements and compares several variants of **Group Relative Policy Optimization (GRPO)** for improving large language models on mathematical reasoning tasks (MATH and GSM8K).

## Motivation

Standard GRPO is an **on-policy** algorithm: the model generates its own rollouts, scores them, and updates toward higher-reward completions. This is effective but expensive — vLLM generation dominates wall-clock time (~2x overhead per step).

**Offline GRPO** replaces on-policy generation with pre-collected rollouts from a stronger teacher model, using importance-sampling (IS) correction to account for the distribution mismatch. In principle, this should allow a weak student to learn from a strong teacher at half the computational cost.

**The central question**: Can offline GRPO with teacher rollouts improve a student model? If not, why?

## Methods

| Method | Directory | Description |
|--------|-----------|-------------|
| **Online GRPO** | `online_grpo/` | Standard on-policy GRPO. Student generates completions via vLLM at each step. |
| **Offline GRPO** | `offline_grpo/` | Off-policy GRPO on pre-collected teacher rollouts with IS correction (π_student/π_teacher). |
| **Mixture A (Unified)** | `mixture_grpo/method_A_unified/` | Merges online student and offline teacher completions into a single group for joint advantage normalization. |
| **Mixture B (Weighted)** | `mixture_grpo/method_B_weighted/` | Computes separate online and offline losses, combines them with a weighting factor λ. |

All methods use **LoRA** fine-tuning and **PPO-style clipping** (ε=0.2).

## LoRA Configuration

Each method defines its own LoRA defaults. The shell scripts for recent experiments override offline GRPO's defaults to r=32/α=32 for controlled comparison.

| Setting | Online GRPO | Mixture A & B | Offline GRPO (code default) | Offline GRPO (recent experiments) |
|---------|-------------|---------------|----------------------------|-----------------------------------|
| Rank (`r`) | 32 | 32 | 16 | 32 |
| Alpha (`lora_alpha`) | 32 | 32 | 64 | 32 |
| Target modules | `all-linear` | `all-linear` | `q/k/v/o/up/down/gate_proj` | `q/k/v/o/up/down/gate_proj` |
| Dropout | 0.0 | 0.05 | 0.05 | 0.05 |
| Task type | `CAUSAL_LM` | `CAUSAL_LM` | `CAUSAL_LM` | `CAUSAL_LM` |

**Notes**:
- `all-linear` targets all linear layers including `q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `down_proj`, `gate_proj`, plus the `lm_head` embedding layer. The explicit list in offline GRPO excludes `lm_head`.
- The effective LoRA scaling factor is `alpha / r`. With r=32, α=32 the scaling is 1.0; with r=16, α=64 the scaling is 4.0.
- LoRA defaults are defined in `configs.py` (`DEFAULT_LORA_CONFIG`) for offline and mixture methods. Online GRPO defines them as argparse defaults in `train.py`.
- Controlled experiments (`run_controlled_math.sh`, `run_teacher_self_math.sh`) override offline defaults via `--lora_r 32 --lora_alpha 32` to match online GRPO settings.

## Models

| Role | Model | Vocab Size | MATH | GSM8K |
|------|-------|------------|------|-------|
| Student | Qwen2.5-0.5B-Instruct | 151,936 | 27.2% | 48.2% |
| — | Qwen2.5-1.5B-Instruct | 151,936 | 44.0% | 68.3% |
| — | Qwen2.5-3B-Instruct | 151,936 | 58.1% | 82.6% |
| Teacher | Qwen2.5-Math-7B-Instruct | 152,064 | 75.0% | 95.7% |

**Note**: The 7B teacher has a different vocabulary size (128 extra tokens). Rollouts containing out-of-vocabulary tokens are truncated during data loading.

## Repository Structure

```
├── online_grpo/
│   ├── train.py                 # Online GRPO training (self-contained)
│   ├── evaluate.py              # Evaluation on GSM8K
│   ├── configs/                 # Accelerate distributed configs
│   └── run_*.sh                 # SLURM job scripts
│
├── offline_grpo/
│   ├── generate_rollouts.py     # Step 1: Generate teacher rollouts with vLLM
│   ├── validate_rollouts.py     # Step 1.5: Validate rollout integrity
│   ├── train.py                 # Step 2: Train student on pre-collected rollouts
│   ├── trainer.py               # OfflineGRPOTrainer (extends TRL GRPOTrainer)
│   ├── data.py                  # Rollout loading, reward computation, advantage calculation
│   ├── configs.py               # Constants, system prompts, answer extraction
│   ├── evaluate.py              # Evaluation on MATH
│   ├── diagnose.py              # Diagnostic: IS ratios, advantage collapse analysis
│   ├── configs/                 # Accelerate configs (DDP, ZeRO-2, FSDP)
│   ├── grpo_rpg/                # Experimental: GRPO with replay buffer
│   └── run_*.sh                 # SLURM job scripts
│
├── mixture_grpo/
│   ├── method_A_unified/
│   │   ├── train.py             # Training script
│   │   └── trainer.py           # UnifiedMixtureGRPOTrainer
│   ├── method_B_weighted/
│   │   ├── train.py             # Training script
│   │   └── trainer.py           # WeightedMixtureGRPOTrainer
│   ├── configs.py               # Shared constants and utilities
│   ├── data.py                  # Teacher rollout loading and reward computation
│   ├── evaluate.py              # Evaluation on GSM8K or MATH
│   ├── generate_rollouts.py     # Teacher rollout generation
│   ├── diagnose_method_B.py     # Hypothesis testing for Method B failure modes
│   └── run_*.sh                 # SLURM job scripts
│
├── setup_downloads.sh           # Download models and datasets to scratch
├── run_eval_all.sh              # Batch evaluation of all models on all tasks
└── requirements.txt             # Python dependencies
```

## Setup

### Prerequisites

- SLURM cluster with NVIDIA GPUs (tested on 4× L40s 48GB)
- Python 3.11, CUDA 12.6
- Models and datasets pre-downloaded to local scratch storage

### Installation

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Download models and datasets to scratch
#    Edit setup_downloads.sh to set your SCRATCH path, then:
bash setup_downloads.sh all
```

### Environment

All SLURM scripts expect these environment variables / paths:

| Variable | Purpose | Default |
|----------|---------|---------|
| `HF_HOME` | HuggingFace cache root | `/scratch/$USER` |
| `HF_DATASETS_CACHE` | Dataset cache | `/scratch/$USER/datasets/{GSM8K,MATH}` |
| `TRANSFORMERS_OFFLINE=1` | Prevent network access on compute nodes | Required |
| `HF_DATASETS_OFFLINE=1` | Prevent network access on compute nodes | Required |
| `VLLM_WORKER_MULTIPROC_METHOD=spawn` | Required for vLLM on multi-GPU | Required |

**Important**: Download all models and datasets on the login node before submitting jobs. Compute nodes run fully offline.

## Usage

### Online GRPO

The simplest method. The student generates its own rollouts during training.

```bash
# Train on GSM8K (0.5B student, 4 GPUs, ~12h)
cd online_grpo
sbatch run_gsm8k.sh

# Train on MATH (0.5B student, 4 GPUs, ~24h)
sbatch run_math.sh
```

Key hyperparameters (in the shell scripts):

| Parameter | GSM8K | MATH |
|-----------|-------|------|
| `learning_rate` | 3e-6 | 3e-6 |
| `beta` (KL penalty) | 0.001 | 0.001 |
| `num_generations` | 5 | 5 |
| `per_device_train_batch_size` | 5 | 1 |
| `gradient_accumulation_steps` | 2 | 10 |
| `num_train_epochs` | 15 | 15 |
| `temperature` | 0.7 | 0.7 |
| `max_completion_length` | 1024 | 2048 |

### Offline GRPO

Two-step process: (1) generate teacher rollouts, (2) train student on them.

```bash
cd offline_grpo

# Step 1: Generate teacher rollouts (MATH, 12K problems, 4 completions each)
#   Uses 4-GPU tensor parallelism for the 7B teacher
sbatch run_full.sh rollouts

# Step 1.5: Validate rollout integrity
bash run_full.sh validate

# Step 2: Train student on teacher rollouts
sbatch run_full.sh train

# Step 3: Evaluate
sbatch run_full.sh eval
```

### Mixture Methods

Combine online student generation with offline teacher rollouts.

```bash
cd mixture_grpo

# Method A (Unified): Joint advantage normalization over student + teacher completions
sbatch run_method_A_math.sh train
sbatch run_method_A_math.sh eval

# Method B (Weighted): Separate losses combined with weight λ
sbatch run_method_B_math.sh train       # default λ=0.3
sbatch run_method_B_math.sh train 0.5   # custom λ
sbatch run_method_B_math.sh eval
```

### Evaluation

```bash
# Evaluate a single checkpoint on MATH (5 runs, temp=0.0)
python mixture_grpo/evaluate.py \
    --model_path /path/to/checkpoint \
    --dataset_type math \
    --runs 5 \
    --temperature 0.0

# With LoRA merging
python mixture_grpo/evaluate.py \
    --model_path /path/to/lora_adapter \
    --base_model /path/to/base_model \
    --merge_lora \
    --dataset_type math \
    --runs 5
```

### Diagnostics

```bash
# Analyze offline GRPO rollout quality and IS ratios
python offline_grpo/diagnose.py \
    --rollout_path /path/to/rollouts.jsonl \
    --target_model /path/to/student

# Test hypotheses for Method B failure (advantage collapse, IS clipping)
python mixture_grpo/diagnose_method_B.py \
    --model_path /path/to/student \
    --teacher_rollout_path /path/to/rollouts.jsonl \
    --dataset_type math
```

## Rollout Data Format

Teacher rollouts are stored as JSONL. Each line contains one problem with N completions:

```json
{
  "question_id": 0,
  "problem": "Find the value of x...",
  "answer": "42",
  "runs": [
    {
      "run_id": 0,
      "response": "Let me solve this step by step...",
      "extracted_answer": "42",
      "logprobs": [-0.12, -0.34, ...],
      "completion_ids": [1234, 5678, ...]
    },
    ...
  ]
}
```

- `logprobs`: Per-token log-probabilities from the behavior policy (raw, pre-temperature)
- `completion_ids`: Token IDs of the generated completion
- Reward is computed at training time using `math_verify` (MATH) or numeric comparison (GSM8K)

## Key Implementation Details

### Offline GRPO Trainer

`offline_grpo/trainer.py` extends TRL's `GRPOTrainer` by overriding `_generate_and_score_completions()`. Instead of generating online, it retrieves pre-computed completions and sets `old_per_token_logps` to the behavior policy's log-probabilities. TRL then computes the importance-sampling ratio π_student/π_teacher internally.

### Reference Model via LoRA

With LoRA, the reference model (for KL penalty) is obtained by calling `model.disable_adapter()` — no separate model copy needed. This halves memory usage.

### Advantage Computation

GRPO normalizes rewards within each prompt group:

```
advantage_i = (reward_i - mean(rewards)) / (std(rewards) + ε)
```

When all completions in a group receive the same reward (all correct or all incorrect), `std = 0` and the advantage collapses to zero. This is a critical failure mode for high-accuracy models.

### Distributed Training

- **DDP** (default): Each GPU holds a full model copy. Works with all features including `disable_adapter()`.
- **ZeRO-2**: Shards optimizer states across GPUs. Compatible with LoRA reference model.
- **FSDP**: Shards model parameters. Known issue: `disable_adapter()` may fail under FSDP.

Accelerate configs are in `offline_grpo/configs/`. Online GRPO uses vLLM's `colocate` mode (generation on the same GPUs as training).

## Dependencies

Core packages:

| Package | Purpose |
|---------|---------|
| `trl` | GRPOTrainer, GRPOConfig |
| `vllm` | Fast LLM inference for rollout generation and evaluation |
| `transformers` | Model loading, tokenizers |
| `peft` | LoRA adapters |
| `accelerate` | Distributed training orchestration |
| `deepspeed` | ZeRO optimization (optional) |
| `math-verify` | MATH dataset answer verification |
| `wandb` | Experiment tracking |
| `datasets` | HuggingFace dataset loading |
