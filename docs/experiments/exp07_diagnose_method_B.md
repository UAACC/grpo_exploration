# Experiment 07: Diagnose Method B Offline Loss

**Date**: 2026-03-14
**Jobs**: SLURM 4340791 (MATH), 4340792 (GSM8K)
**Status**: Complete

## Objective

Quantify why Method B's offline GRPO loss is ineffective. Two hypotheses:
- **H1 (Advantage collapse)**: With G=4 student rollouts, the student almost always gets all-correct or all-wrong on a problem, so `std_r=0` and teacher advantages collapse to 0.
- **H2 (IS ratio clipping)**: The IS ratio `pi_student/pi_teacher` is far from 1.0 (0.5B student vs 7B teacher), so PPO clipping at [0.8, 1.2] kills most gradient signal.

## Setup

### Script
`mixture_grpo/diagnose_method_B.py` — generates G=4 student rollouts per prompt, computes rewards (H1), then scores teacher completions under the student model to get IS ratios (H2).

### Models tested per task
1. **Baseline**: `Qwen2.5-0.5B-Instruct` (pre-training)
2. **Trained**: merged LoRA checkpoint from Method B training

### Hyperparameters
| Parameter | Value | Source |
|-----------|-------|--------|
| num_samples | 500 | Our choice: large enough for stable statistics |
| num_generations (G) | 4 | Matches Method B training config |
| temperature | 0.7 | Matches Method B training config |
| max_tokens (MATH) | 2048 | Matches training |
| max_tokens (GSM8K) | 1024 | Matches training |
| clip_eps | 0.2 | Standard PPO epsilon, matches training |
| IS ratio scoring | 100 completions per model | Script limits H2 analysis to first 100 qids for speed |

### Log files
- MATH: `mixture_grpo/logs/diagnose/diag-math-4340791.out`
- GSM8K: `mixture_grpo/logs/diagnose/diag-gsm8k-4340792.out`

## Results

### H1: Advantage Collapse (Student Reward Uniformity)

| Metric | MATH Baseline | MATH Trained | GSM8K Baseline | GSM8K Trained |
|--------|:---:|:---:|:---:|:---:|
| All wrong (reward=0 for all G) | 352/500 (70.4%) | 333/500 (66.6%) | 227/500 (45.4%) | 188/500 (37.6%) |
| All correct (reward>0 for all G) | 133/500 (26.6%) | 129/500 (25.8%) | 252/500 (50.4%) | 278/500 (55.6%) |
| Mixed (gradient signal exists) | 15/500 (3.0%) | 38/500 (7.6%) | 21/500 (4.2%) | 34/500 (6.8%) |
| **% teacher completions wasted** | **97.0%** | **92.4%** | **95.8%** | **93.2%** |
| Active teacher advantage (mean) | 0.893 | 0.825 | 0.976 | 1.025 |

Source: `diag-math-4340791.out` lines 38-50, `diag-gsm8k-4340792.out` lines 38-50, 126-138.

**Key finding**: With G=4, the student's 4 rollouts are almost always unanimous (all correct or all wrong). Only 3-8% of prompts produce mixed rewards, meaning 92-97% of teacher completions have `advantage=0` and contribute zero gradient.

This is worse on MATH (70.4% all-wrong vs 45.4% on GSM8K) because the 0.5B student is much weaker on MATH.

### H2: IS Ratio Clipping

| Metric | MATH Baseline | MATH Trained | GSM8K Baseline | GSM8K Trained |
|--------|:---:|:---:|:---:|:---:|
| Total tokens analyzed | 65,141 | 65,141 | 29,401 | 29,401 |
| log(ratio) mean | -0.2215 | -0.2207 | -0.1195 | -0.1112 |
| log(ratio) std | 1.0306 | 1.0473 | 0.4801 | 0.5655 |
| log(ratio) median | 0.0000 | 0.0000 | -0.0001 | 0.0000 |
| ratio median | 1.0000 | 1.0000 | 0.9999 | 1.0000 |
| ratio [5%, 95%] | [0.183, 1.062] | [0.173, 1.069] | [0.463, 1.004] | [0.523, 1.008] |
| Clipped low (ratio < 0.8) | 14.1% | 13.3% | 11.4% | 8.5% |
| In range [0.8, 1.2] | 83.0% | 83.6% | 87.4% | 90.0% |
| Clipped high (ratio > 1.2) | 2.9% | 3.0% | 1.2% | 1.5% |
| **Per-completion clip fraction (mean)** | **14.1%** | **13.5%** | **12.5%** | **9.8%** |
| Completions >50% clipped | 3/100 (3.0%) | 3/100 (3.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| Completions >90% clipped | 0/100 | 0/100 | 0/100 | 0/100 |

Source: `diag-math-4340791.out` lines 58-78, `diag-gsm8k-4340792.out` lines 58-78, 146-166.

**Key finding**: H2 is much milder than hypothesized. The median IS ratio is 1.0 and 83-90% of tokens are within the clip range. Only 10-14% of tokens are clipped. This makes sense in hindsight: the student and teacher share the same tokenizer (Qwen2.5 family) and for most tokens (common math notation, natural language), both models assign similar probabilities. The divergence is concentrated in the long tail.

### Combined Impact

| Metric | MATH Baseline | MATH Trained | GSM8K Baseline | GSM8K Trained |
|--------|:---:|:---:|:---:|:---:|
| H1 wasted | 97.0% | 92.4% | 95.8% | 93.2% |
| H2 clipped (of surviving) | 14.1% | 13.5% | 12.5% | 9.8% |
| **Effective signal** | **2.6%** | **6.6%** | **3.7%** | **6.1%** |

Formula: `effective = (1 - H1_wasted) * (1 - H2_clipped)`

## Analysis

### H1 is the dominant bottleneck, not H2

The original hypothesis (H2) was that IS ratio clipping between the 0.5B student and 7B teacher would kill most gradient. **This is wrong.** The IS ratios are surprisingly well-behaved (median=1.0, 83-90% in range).

The actual bottleneck is H1: advantage collapse. With G=4, the binary nature of math problem correctness means the student almost always produces 4 correct or 4 incorrect answers. When all rewards are equal, `std_r=0`, the normalized advantage for teacher completions is `(tea_reward - mean_r) / std_r = 0/0 → 0`, contributing zero gradient.

### Why H1 is so extreme

The root cause is the **interaction between low G and binary rewards**:
- G=4 is too small for the bimodal reward distribution
- On MATH, the baseline 0.5B student has ~27% accuracy. With G=4, the probability of mixed outcomes is `P(mixed) = 1 - P(all_wrong) - P(all_correct) = 1 - 0.73^4 - 0.27^4 ≈ 1 - 0.284 - 0.005 = 71%` in theory
- But observed mixed is only 3%, suggesting the student's accuracy varies wildly by problem (some are ~0%, others ~100%), not uniformly 27% per problem

### MATH vs GSM8K comparison

As expected, MATH is worse:
- MATH baseline: 97.0% wasted (H1) + 14.1% clipped (H2) → 2.6% effective
- GSM8K baseline: 95.8% wasted (H1) + 12.5% clipped (H2) → 3.7% effective

The difference is driven by H1: MATH has more all-wrong problems (70.4% vs 45.4%) because the student is weaker.

### Training helps slightly but not enough

After Method B training:
- MATH: 97.0% → 92.4% wasted, 2.6% → 6.6% effective (2.5x improvement)
- GSM8K: 95.8% → 93.2% wasted, 3.7% → 6.1% effective (1.6x improvement)

More problems move into the mixed zone after training, but the improvement is marginal (still >92% wasted).

### Implications for fixing Method B

1. **Increasing G** would help H1 significantly: G=16 or G=32 would produce more mixed-reward groups
2. **IS ratio clipping is not the problem**: No need to widen the clip range or use adaptive clipping
3. **Alternative advantage computation**: Could use per-completion reward without group normalization, or use different normalization that doesn't collapse to 0 when all rewards are equal
4. **Curriculum filtering**: Skip problems where student accuracy is near 0% or 100%, focus training on the "boundary" problems

## Artifacts

- Script: `mixture_grpo/diagnose_method_B.py`
- SLURM runner: `mixture_grpo/run_diagnose.sh`
- MATH log: `mixture_grpo/logs/diagnose/diag-math-4340791.out`
- GSM8K log: `mixture_grpo/logs/diagnose/diag-gsm8k-4340792.out`
- Teacher rollouts: `/scratch/mrli/rollouts/{math_teacher/rollouts_full.jsonl, gsm8k_teacher/rollouts_gsm8k.jsonl}`
