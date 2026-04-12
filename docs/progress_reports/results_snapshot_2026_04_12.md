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
| **DG-Offline** | `DG-offline/` | Delightful Policy Gradient for offline GRPO. Replaces IS ratios with a sigmoid gate on delight = advantage × surprisal. |
| **Mixture A (Unified)** | `mixture_grpo/method_A_unified/` | Merges online student and offline teacher completions into a single group for joint advantage normalization. |
| **Mixture B (Weighted)** | `mixture_grpo/method_B_weighted/` | Computes separate online and offline losses, combines them with a weighting factor λ. |
| **DG-Mixture** | `mixture_grpo/dg_mixture/` | Online GRPO loss for student rollouts + DG-gated loss for teacher rollouts. Combines Method B's mixture structure with DG-offline's sigmoid gate on delight = advantage × surprisal. |
| **BC** | `bc/` | Behavioral cloning baseline — pure cross-entropy on teacher completions. Also contains analysis scripts. |

All methods train with **PPO-style clipping** (ε=0.2) and use **LoRA** adapters to reduce the trainable parameter count (~3.4% of total) and provide a zero-cost reference model for the KL penalty (via `disable_adapter()`).

## LoRA Adapter Settings

LoRA is used purely for parameter efficiency — only low-rank adapter weights are trained while the base model is frozen. This also gives us a free reference model for KL penalty computation: calling `model.disable_adapter()` recovers the original base model without loading a separate copy.

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
├── DG-offline/
│   ├── train.py                 # DG-offline training script
│   ├── trainer.py               # DGOfflineTrainer (delight gating, no IS ratios)
│   ├── configs/                 # Accelerate configs
│   └── run_*.sh                 # SLURM scripts (MATH and GSM8K, eta configurable)
│
├── bc/
│   ├── train_bc.py              # Behavioral cloning training
│   ├── eval_best_of_n.py        # Best-of-N evaluation (pass@1 and pass@N)
│   ├── analyze_gap.py           # Generation comparison: student vs teacher
│   ├── analyze_branching.py     # Probability distributions at branching points
│   ├── agreement_on_student_ctx.py  # Token agreement on student-generated context
│   ├── diagnose_bc.py           # BC implementation verification
│   └── run_*.sh                 # SLURM scripts
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

# DG-Mixture: online GRPO + DG-gated teacher loss
sbatch run_dg_mixture_math.sh train                            # default η=0.5, λ=0.3
DG_ETA=0.1 DG_LAMBDA=0.5 \
  CHECKPOINT_DIR=/scratch/$USER/checkpoints/dg_mixture_eta0.1_lam0.5 \
  sbatch --job-name=dg-mix-eta0.1-lam0.5 run_dg_mixture_math.sh train
sbatch run_dg_mixture_math.sh eval
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

## Results

The student is Qwen2.5-0.5B-Instruct and the teacher is Qwen2.5-Math-7B-Instruct for all rows below. Three columns per dataset:

- **Greedy (temp=0.0)**: `mixture_grpo/evaluate.py` at `--temperature 0.0`, 5 runs averaged. This is the deployable deterministic accuracy.
- **pass@1 (temp=0.6)**: first sample of `bc/eval_best_of_n.py` at `--temperature 0.6`, `--n_samples 16`. Same script as pass@16, different metric.
- **pass@16 (temp=0.6)**: fraction of problems where at least one of 16 samples is correct. Possible upper bound (requires ground truth to pick the correct sample), NOT deployable on its own. Useful as a diagnostic for what's in the student's distribution.

### MATH (500 problems)

| Method | Greedy (temp=0.0) | pass@1 (temp=0.6) | pass@16 (temp=0.6) |
|--------|-------------------|-------------------|------------------|
| Baseline (untrained 0.5B) | 27.16% | 26.80% | 61.40% |
| BC (all completions) | 27.40% | 24.20% | 61.00% |
| BC (correct only) | 27.20% | 26.40% | 60.40% |
| Offline GRPO (controlled) | 27.64% | 26.40% | 60.20% |
| Mixture A (unified) | 29.28% | 27.20% | 61.20% |
| Mixture B (weighted) | 28.60% | 25.80% | 60.80% |
| DG-offline η=0.1 | 29.04% | 25.00% | 62.40% |
| **DG-offline η=0.5** | **29.00%** | 26.60% | **64.20%** |
| DG-offline η=1.0 | 28.08% | 27.60% | 63.40% |
| DG-offline η=2.0 | 27.88% | 27.80% | 62.20% |
| **Online GRPO** | **32.10%** | **31.80%** | **64.40%** |
| Teacher (7B) | 74.96% | — | — |

### GSM8K (1319 problems)

| Method | Greedy (temp=0.0) | pass@1 (temp=0.6) | pass@16 (temp=0.6) |
|--------|-------------------|-------------------|------------------|
| Baseline (untrained 0.5B) | 48.16% | 43.59% | 86.13% |
| BC (all completions) | 49.67% | 46.55% | 83.55% |
| BC (correct only) | 49.43% | 48.29% | 84.91% |
| Offline GRPO (GSM8K-trained) | 48.11% | 43.06% | 84.15% |
| Mixture A (unified, GSM8K-trained) | 50.30% | 48.22% | 83.55% |
| **Mixture B (weighted, GSM8K-trained)** | **51.11%** | **49.73%** | 83.93% |
| DG-offline η=0.1 | 49.39% | 49.20% | 84.53% |
| DG-offline η=0.5 | 48.73% | 47.16% | 83.17% |
| DG-offline η=1.0 | 48.82% | 46.93% | **85.67%** |
| DG-offline η=2.0 | 48.43% | 45.41% | 84.69% |
| **Online GRPO (MATH ckpt, cross-task)** | **55.82%** | 49.20% | 85.06% |
| Teacher (7B) | 95.74% | — | — |

Caveats worth reading before citing any of these numbers:

- **MATH baseline has a known ~3pp variance across eval runs** (27.16% vs 30.36%) due to vLLM non-determinism across different compute nodes. Offline GRPO, Mixture A, Mixture B on MATH were measured against the 30.36% baseline in their own eval run, so their improvement-over-baseline in their own run is different from what it looks like against the 27.16% row above. The numbers themselves are accurate.
- **Online GRPO GSM8K is cross-task**: the MATH-trained online GRPO checkpoint evaluated directly on GSM8K without retraining. There is no GSM8K-trained online GRPO checkpoint in the current results.
- **pass@1 (temp=0.6) is not the same as greedy (temp=0.0)**. The first column is deterministic argmax decoding; the second column is the first sample out of 16 sampled at temperature 0.6. Models with flatter distributions (BC, DG-offline) show larger gaps between the two columns because temperature sampling more often deviates from the argmax.

Full per-cell source citations (job IDs and log paths) are in `docs/progress_reports/2026_03_28_30.md`.

## Key Findings

### 1. Offline methods don't add knowledge (on MATH)

All standard offline methods (BC, offline GRPO, mixtures) leave MATH pass@16 unchanged at ~60-61%. They shuffle greedy preferences but don't add or remove knowledge. Online GRPO and DG-offline (η=0.5) are the only methods that raise MATH pass@16 above baseline, meaning they actually teach the student new problem-solving approaches.

On GSM8K the picture is different: no method (including DG-offline and online GRPO) raises pass@16 above the already-high baseline of 86.13%. This is likely a headroom effect — GSM8K is easy enough that the baseline's sampling distribution already contains correct paths for most problems.

### 2. The accuracy gap is about strategy selection, not token-level knowledge

Branching point analysis on 82 "gap" problems (teacher correct, student wrong) reveals:
- The teacher's chosen token is in the student's **top-5** at 92.7% of branching points
- Mean rank of teacher's token in student's distribution: **2.0**
- Mean probability: **22.5%**
- Token-level agreement between student and teacher is **92%** regardless of who generated the context

The student and teacher diverge at the **very first line** in 100% of gap problems. The student picks a different problem-solving strategy (e.g., brute force instead of factoring), then writes internally coherent but wrong reasoning. The correct strategy is available in the student's distribution — it's just not the greedy argmax.

### 3. DG-offline is the best offline method on MATH, but not on GSM8K

DG-offline (Delightful Policy Gradient) replaces IS ratios with a sigmoid gate on delight = advantage × surprisal, computed entirely from the learner's own policy.

On MATH, DG-offline η=0.5 is the only offline method that improves both greedy (29.00% vs 27.16% baseline) and pass@16 (64.20% vs 61.40% baseline). It's the clear winner among offline methods, second only to online GRPO.

On GSM8K, Mixture B wins greedy (51.11%) and DG-offline η=1.0 wins pass@16 (85.67%). The η=0.5 sweet spot from MATH does NOT transfer — η=0.5 is actually the worst DG-offline variant on GSM8K pass@16. This suggests the optimal η depends on the dataset's surprisal distribution rather than being a universal constant.

### 4. Best-of-N reveals that most of the student's knowledge gap isn't really a knowledge gap

Sampling 16 completions instead of taking the greedy output nearly triples baseline accuracy on MATH (27.16% → 61.40%) and closes most of the gap to the teacher on GSM8K (48.16% → 86.13%, vs teacher's 95.74%). This happens without any training.

pass@16 is an possible upper bound (it assumes you can pick the correct sample), so the raw number isn't deployable. But it tells us something important about the shape of the student's distribution: the correct reasoning path usually exists in the top-K samples, it just isn't the greedy argmax. The bottleneck for a greedy student is not knowledge, it's selection.

This reframes what offline methods should be doing: rather than trying to teach the student new strategies, they should be shifting the student's greedy preference toward strategies it already considers reasonable. Online GRPO does this naturally (by reinforcing the student's own successful samples); most offline methods don't.

## DG-Offline: Delightful Policy Gradient

Based on Osband (2026), "Delightful Distributed Policy Gradient" (arXiv:2603.20521). See `dg-offline_imp.md` for full technical details.

### How it works

Standard offline GRPO weights each gradient by the IS ratio π_student/π_teacher (clipped PPO-style). DG-offline replaces this with:

```
weight = σ(advantage × surprisal / η)
```

where surprisal = -log π_student(completion) measures how unlikely the teacher's completion is under the **student's own policy**. No behavior policy logprobs are needed.

The gate has asymmetric behavior:
- **Surprising success** (positive advantage, high surprisal) → gate ≈ 1 → amplify
- **Surprising failure** (negative advantage, high surprisal) → gate ≈ 0 → suppress
- **Expected outcome** (low surprisal) → gate ≈ 0.5 → pass through

### Usage

```bash
cd DG-offline

# Train with default η=1.0
sbatch run_math.sh

# Eta sweep (η=0.5 recommended)
DG_ETA=0.5 CHECKPOINT_DIR=/scratch/$USER/checkpoints/dg_eta0.5 sbatch --job-name=dg-eta0.5 run_math.sh

# Evaluate
sbatch run_math.sh eval /path/to/checkpoint

# Best-of-N evaluation
cd ../bc
MODEL=/path/to/merged/model python eval_best_of_n.py \
    --model_path $MODEL --n_samples 16 --temperature 0.6 --dataset_type math
```

## Analysis Scripts (bc/)

The `bc/` directory contains diagnostic and analysis scripts beyond BC training:

| Script | Purpose |
|--------|---------|
| `eval_best_of_n.py` | Best-of-N evaluation (pass@1 and pass@N) for MATH and GSM8K |
| `analyze_gap.py` | Side-by-side generation comparison, quadrant analysis (both correct / teacher only / student only / both wrong), divergence point detection |
| `analyze_branching.py` | Full probability distributions at the first divergence token — where does the teacher's token rank in the student's distribution? |
| `agreement_on_student_ctx.py` | Token agreement measured on student-generated context (vs teacher context) |
| `diagnose_bc.py` | 9-check BC implementation verification |
| `train_bc.py` | BC training with LoRA on teacher completions |

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
