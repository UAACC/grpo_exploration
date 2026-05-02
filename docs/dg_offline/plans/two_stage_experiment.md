# Experiment: DG-offline → Online GRPO (two-stage training)

## Motivation

DG-offline (η=0.5) is the best offline method on MATH (29.00% greedy, 64.20% pass@16). Online GRPO is the best overall method (32.10% greedy, 64.40% pass@16). The question: can we get the best of both by running them sequentially?

The hypothesis is that DG-offline gives the student a better initialization for online GRPO. Instead of starting from the raw 0.5B-Instruct baseline (27.16%), online GRPO starts from the DG-offline-improved student (29.00%). If the DG-offline stage shifted the student's greedy preference toward better strategies at branching points, the online stage should find and reinforce correct paths more frequently from the start.

## Design

```
Stage 1 (already done):
  Qwen2.5-0.5B-Instruct → DG-offline (η=0.5, 1 epoch on MATH teacher rollouts) → 29.00% greedy

Stage 2 (this experiment):
  DG-offline checkpoint (merged) → Online GRPO (on MATH train set) → ???% greedy
```

Stage 2 uses the **merged** DG-offline model as the starting point (not the LoRA adapter). Online GRPO applies a fresh LoRA on top and trains with its standard on-policy rollout generation.

## Setup

| Parameter | Value | Notes |
|-----------|-------|-------|
| Starting model | `/scratch/mrli/merged/dg_offline_math` | Merged DG-offline η=0.5 (29.00% greedy) |
| Training method | Online GRPO (standard, `online_grpo/train.py`) | Same script used for the baseline 32.10% result |
| Dataset | MATH train (12K problems) | Same as baseline online GRPO |
| LoRA | r=32, α=32, all-linear | Fresh LoRA on top of the merged DG-offline model |
| learning_rate | 3e-6 | Same as baseline online GRPO |
| beta | 0.001 | Same |
| num_generations | 5 | Same |
| num_train_epochs | 15 | Same |
| temperature | 0.7 | Same |
| max_completion_length | 2048 | Same |

## Expected outcome

| Scenario | What it means |
|----------|---------------|
| Combined > Online GRPO alone (>32.10%) | DG-offline provides a better initialization. Two-stage training is the recommended pipeline. |
| Combined ≈ Online GRPO alone (~32%) | DG-offline's improvement doesn't compound with online GRPO. Online GRPO converges to the same place regardless of starting point. |
| Combined < Online GRPO alone (<32%) | DG-offline hurt the starting distribution in a way that online GRPO can't recover from (unlikely but possible if DG-offline narrowed the distribution too much). |

## Eval

Same protocol as all other experiments:
- Greedy (temp=0.0, 5 runs)
- pass@1 (temp=0.6)
- pass@16 (temp=0.6, oracle)

Compare against:

| Model | Greedy | pass@16 |
|-------|--------|---------|
| Baseline (untrained) | 27.16% | 61.40% |
| DG-offline η=0.5 (stage 1 only) | 29.00% | 64.20% |
| Online GRPO (from scratch) | 32.10% | 64.40% |
| **DG-offline → Online GRPO (this experiment)** | **?** | **?** |

## Artifacts

| Item | Path |
|------|------|
| Starting model (merged DG-offline) | `/scratch/mrli/merged/dg_offline_math` |
| Output checkpoint | `/scratch/mrli/checkpoints/dg_then_online_math` |
| Merged output | `/scratch/mrli/merged/dg_then_online_math` |
