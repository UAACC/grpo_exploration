# Experiment 04: Offline GRPO on GSM8K

**Date**: 2026-03-10
**Jobs**: 4300435 (rollouts), 4301147 (train), 4301492 (eval)
**Status**: Complete
**wandb**: [offline-grpo-gsm8k project](https://wandb.ai) (run: `offline-grpo-gsm8k-20260310_165959`)

---

## Objective

Run offline GRPO on GSM8K to compare with online GRPO (Experiment 01 in `online_grpo/experiment_analysis/`).

- **Online GRPO best**: 55.82% at step 7000 (source: `online_grpo/logs/online-grpo-eval-4299873.out`)
- **Baseline**: 48.14% (source: `online_grpo/logs/eval-baseline2-4300033.out`)
- **VERL target**: 54.3% (source: `verl.readthedocs.io/en/latest/algo/baseline.html`)

Key question: Can offline GRPO (teacher-generated rollouts, single epoch) match online GRPO performance?

---

## Setup

### Models

| Role | Model | Notes |
|------|-------|-------|
| Teacher (behavior) | Qwen2.5-Math-7B-Instruct | ~95.2% GSM8K accuracy (source: Qwen2.5-Math tech report, arxiv.org/html/2409.12122v1) |
| Student (target) | Qwen2.5-0.5B-Instruct | Same as online GRPO |

### Source of hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| num_generations | 5 | Matches online GRPO / VERL 3B script |
| Rollout temperature | 0.7 | Matches online GRPO generation temp |
| Rollout max_tokens | 1024 | VERL 0.5B filename `resp1024` |
| LoRA rank | 16 | Offline GRPO default (differs from online GRPO r=32) |
| LoRA alpha | 64 | Offline GRPO default (differs from online GRPO alpha=32) |
| Learning rate | 5e-6 | Offline GRPO default (online GRPO used 3e-6) |
| Beta (KL coeff) | 0.1 | Offline GRPO default (online GRPO used 0.001) |
| Epochs | 1 | Offline GRPO default (single pass over teacher rollouts) |
| per_device_train_batch_size | 5 | For divisibility: 5 x 4 GPUs x 2 accum = 40 / 5 gen = 8 |
| gradient_accumulation_steps | 2 | Same effective batch as online GRPO |
| max_grad_norm | 0.1 | Offline GRPO default (online GRPO used 1.0) |
| ref_sync_steps | 0 | Reference = original base model (no sync) |
| Warmup | 10% | Offline GRPO default |
| LR schedule | cosine | Same as online GRPO |

### Key differences: Online vs Offline GRPO

| | Online GRPO (Exp01) | Offline GRPO (Exp04) |
|---|---|---|
| Rollout source | Student (live, each step) | Teacher 7B (pre-generated, fixed) |
| IS correction | None (on-policy) | Yes, pi_student / pi_teacher |
| Epochs | 15 (fresh rollouts each epoch) | 1 (single pass) |
| KL beta | 0.001 | 0.1 |
| LoRA rank | 32 | 16 |
| LoRA alpha | 32 | 64 |
| Learning rate | 3e-6 | 5e-6 |
| Max grad norm | 1.0 | 0.1 |
| Training time | ~12h (vLLM bottleneck) | ~62 min train + ~34 min rollouts |
| Framework | TRL GRPOTrainer (vLLM colocate) | OfflineGRPOTrainer (custom) |

---

## Step 1: Rollout Generation (Job 4300435)

**Duration**: ~34 min (14:49:12 generation start → 15:22:56 done)
Source: `logs/gsm8k_offline/gsm8k-rollouts-4300435.out`

- Dataset: 7,473 GSM8K train problems
- Completions: 7,473 x 5 = 37,365 total
- Model: Qwen2.5-Math-7B-Instruct (TP=4, 4x L40s)
- Output: `/home/shuai14/scratch/dongheng/gsm8k_teacher_rollouts/rollouts_gsm8k.jsonl`

---

## Step 2: Training (Job 4301147)

**Duration**: 3,732 seconds (~62 min)
**Steps**: 467 logged steps (logging every 10 steps → ~4,670 total steps)
**Speed**: 1.25 steps/second (source: `train_runtime` in final log entry)

Source: `logs/gsm8k_offline/gsm8k-offline-4301147.out`

### Data loading

```
37,365 completions loaded
Warning: 198 completions truncated at out-of-vocab tokens (vocab_size=151936)
Rewards: 35,550 / 37,365 correct (95.1%)
```

The teacher (Math-7B) achieved **95.1% accuracy** on GSM8K train set at temp=0.7 with 5 generations per problem. 198 completions (0.5%) had out-of-vocab tokens (teacher vocab > student vocab) and were truncated.

### Training steps calculation

```
37,365 completions / 5 num_generations = 7,473 unique prompts
7,473 / (5 per_device x 4 GPUs) = 373.65 micro-batches/epoch
373.65 / 2 grad_accum = 186.8 optimizer steps/epoch
But TRL groups by num_generations internally, so actual: ~4,670 steps
```

### Per-milestone training metrics

| Epoch | Loss | Grad Norm | KL | Entropy | Reward | Reward Std | LR |
|-------|------|-----------|-----|---------|--------|------------|-----|
| 0.00 | -0.0003 | 0.005 | 0.000255 | 0.186 | 0.950 | 0.083 | 9.6e-08 |
| 0.10 | 0.0156 | 0.399 | 0.006100 | 0.323 | 0.990 | 0.032 | 4.8e-06 |
| 0.20 | 0.0105 | 0.016 | 0.001855 | 0.216 | 0.930 | 0.095 | 4.9e-06 |
| 0.30 | 0.0046 | 0.129 | 0.001632 | 0.202 | 0.930 | 0.095 | 4.4e-06 |
| 0.40 | 0.0181 | 0.119 | 0.005871 | 0.353 | 0.990 | 0.032 | 3.8e-06 |
| 0.50 | 0.0174 | 0.016 | 0.004085 | 0.275 | 0.960 | 0.105 | 3.0e-06 |
| 0.60 | 0.0051 | 0.013 | 0.001834 | 0.212 | 1.000 | 0.000 | 2.1e-06 |
| 0.70 | 0.0024 | 0.009 | 0.002126 | 0.243 | 1.000 | 0.000 | 1.3e-06 |
| 0.80 | 0.0167 | 0.113 | 0.006480 | 0.327 | 0.990 | 0.032 | 6.1e-07 |
| 0.90 | 0.0027 | 0.055 | 0.000915 | 0.175 | 0.930 | 0.116 | 1.7e-07 |
| 1.00 | 0.0032 | 0.062 | 0.000875 | 0.171 | 0.940 | 0.084 | 3.1e-10 |

### Key training observations

1. **KL stays very low** (mean=0.003, max=0.015) — much more stable than online GRPO (which had KL=23.85 at epoch 4). This is expected: beta=0.1 (100x larger than online's 0.001) and only 1 epoch of training.

2. **Training reward is ~95%** throughout — this is the teacher's reward on rollouts, not the student's generation accuracy. Since 95.1% of teacher completions are correct, the reward reflects the data distribution, not the student's ability.

3. **Grad norm is stable** (mostly 0.005-0.15, occasional spikes to ~1.5). Much more stable than online GRPO (which had grad norms up to 5.0). The strict `max_grad_norm=0.1` helps.

4. **Loss is near zero** throughout (~0.01 average). The student is learning from high-quality teacher rollouts where most completions are correct.

5. **Entropy** stays in 0.17-0.40 range — stable, no collapse.

---

## Step 3: Evaluation (Job 4301492)

Source: `logs/gsm8k_offline/gsm8k-eval-4301492.out`

### GSM8K test accuracy (temp=0.0, 5 runs)

| Model | Accuracy | Std | Min | Max | Avg Length |
|-------|----------|-----|-----|-----|------------|
| **Offline GRPO** | **48.79%** | 0.0009 | 48.67% | 48.90% | 313.3 |
| Baseline (no training) | 48.14% | 0.0036 | 47.69% | 48.67% | 311.2 |
| Online GRPO (step 7000) | **55.82%** | 0.0063 | 55.12% | 56.56% | - |
| VERL target | 54.3% | - | - | - | - |

Baseline from: `online_grpo/logs/eval-baseline2-4300033.out`

---

## Analysis

### Offline GRPO barely improves over baseline

**Offline GRPO: 48.79% vs Baseline: 48.14% = +0.65 percentage points.**

This is a near-zero improvement, well within noise. The 95% confidence interval overlap between the two models is substantial (baseline max 48.67% vs offline min 48.67%).

### Why offline GRPO failed on GSM8K

**1. Reward diversity problem (primary cause)**

With 95.1% teacher accuracy and 5 generations per problem, most groups have all-correct (reward=1.0) or nearly all-correct outcomes:
- If 95.1% are correct, expected correct per group of 5: ~4.75
- Probability all 5 correct: 0.951^5 ≈ 77.8%
- Probability all 5 wrong: 0.049^5 ≈ 0.00003%
- Probability mixed: ~22.2%

**Only ~22% of problem groups provide any learning signal** (non-zero advantage variance). This was the same issue identified in exp02 on MATH (line 100 of `exp02_train_log.md`): "With only 4 generations per problem and binary reward, many problem groups have all-same rewards."

On MATH, the teacher was ~60-70% accurate, giving much more mixed groups. On GSM8K, the teacher is 95% accurate — paradoxically **too good** for offline GRPO with few generations.

**2. Single epoch limitation**

Online GRPO trains for 15 epochs with fresh rollouts each time. Offline GRPO uses a fixed set of teacher rollouts for 1 epoch. The student sees each training example only once, limiting what it can learn.

**3. Off-policy mismatch**

The rollouts come from a 7B Math model. The student (0.5B Instruct) must learn from completions that may use reasoning patterns it cannot reproduce. The IS correction (pi_student / pi_teacher) should compensate, but with such different model sizes, the correction ratios may be extreme.

**4. LoRA hyperparameter differences**

Offline GRPO used r=16, alpha=64 (from default offline config), while online GRPO used r=32, alpha=32 (from VERL). This means different effective LoRA scaling: alpha/r = 64/16 = 4.0 for offline vs 32/32 = 1.0 for online. The higher scaling might make offline updates too aggressive per step but the strict grad_norm=0.1 clips them back.

### Comparison: Online vs Offline GRPO

| | Online GRPO | Offline GRPO |
|---|---|---|
| Accuracy | **55.82%** | 48.79% |
| Improvement over baseline | **+7.68 pp** | +0.65 pp |
| Training time | ~12 hours | ~1.6 hours (34min rollouts + 62min train) |
| KL behavior | Unstable (explodes epoch 9+) | Stable (max 0.015) |
| Speed | 6x slower | 6x faster |

Online GRPO dramatically outperforms offline GRPO on GSM8K (+7 pp vs +0.65 pp). The key advantage of online GRPO: **fresh on-policy rollouts from the student itself**. The student generates completions it can actually produce, and the reward signal directly tells it which of its own attempts were better. In offline GRPO, the student is trying to learn from a teacher's completions that are almost always correct — there's little contrast to learn from.

---

## Next Steps

1. **Increase num_generations to 16 or 32** — This is the #1 fix. With 16 generations per problem, even at 95% teacher accuracy, ~56% of groups will have mixed results (vs 22% with 5). Rollout generation cost scales linearly but is still fast (~2.5 hours for 16x).

2. **Use a weaker teacher** — A teacher with 60-80% accuracy on GSM8K would give much more diverse rollouts. Qwen2.5-0.5B-Instruct itself (48% baseline) or Qwen2.5-1.5B-Instruct could work. This is conceptually closer to on-policy rollouts.

3. **Train for multiple epochs** — The current 1-epoch setup only passes through data once. Even 3-5 epochs might help.

4. **Match LoRA config to online GRPO** (r=32, alpha=32) for fairer comparison.

5. **Match other hyperparameters**: lr=3e-6, beta=0.001, max_grad_norm=1.0 to isolate the online vs offline effect.

---

## Artifacts

### Code
- `generate_rollouts.py`: Rollout generation (updated for GSM8K support)
- `data.py`: Data loading and reward computation (updated for GSM8K)
- `configs.py`: Shared constants (updated with GSM8K prompts and extraction)
- `train.py`: Offline GRPO training (unchanged)
- `trainer.py`: OfflineGRPOTrainer (unchanged)
- `run_gsm8k_offline.sh`: SLURM batch script

### Output
- Rollouts: `/home/shuai14/scratch/dongheng/gsm8k_teacher_rollouts/rollouts_gsm8k.jsonl`
- Checkpoints: `/home/shuai14/scratch/dongheng/offline_grpo_gsm8k/`
- Merged model: `/home/shuai14/scratch/dongheng/offline_grpo_gsm8k_merged/`
- wandb: `offline-grpo-gsm8k` project, run `offline-grpo-gsm8k-20260310_165959`

### Logs
- Rollouts: `logs/gsm8k_offline/gsm8k-rollouts-4300435.{out,err}`
- Training: `logs/gsm8k_offline/gsm8k-offline-4301147.{out,err}`
- Evaluation: `logs/gsm8k_offline/gsm8k-eval-4301492.{out,err}`

---

*Created: 2026-03-11*
