# Experiment 03: Training Log — Multi-GPU + Reference Model Sync

**Date**: 2026-03-09
**Status**: Complete (training + evaluation)

---

## Pipeline Overview

| Stage | Job ID | Status | Duration | Notes |
|-------|--------|--------|----------|-------|
| Train refsync0 | 4295524 | Complete | 1h 40m | 4x L40s DDP, ref = original base model |
| Train refsync16 | 4295525 | Complete | 1h 48m | 4x L40s DDP, ref synced every 16 steps |
| Train refsync1 | 4295526 | Complete | 1h 47m | 4x L40s DDP, ref synced every step |
| Eval refsync0 | 4295710 | Complete | 1m 44s | 1x L40s, LoRA merge + vLLM |
| Eval refsync16 | 4295711 | Complete | 1m 44s | 1x L40s, LoRA merge + vLLM |
| Eval refsync1 | 4295712 | Complete | 1m 43s | 1x L40s, LoRA merge + vLLM |
| Eval baseline | — | Done (exp02) | — | 27.2% (reuse exp02 result) |

## Setup

### Rollout Data (reused from exp02)
- **Teacher**: Qwen2.5-Math-7B-Instruct
- **Data**: 12,000 MATH train problems x 4 generations = 48,000 completions
- **Teacher accuracy**: 70.9% (34,026/48,000 correct)
- **Rollout file**: `rollouts_full.jsonl` (930.8 MB)

### Training (all 3 variants)
- **Student**: Qwen2.5-0.5B-Instruct + LoRA (r=16, alpha=64)
- **GPUs**: 4x L40s with DDP (`accelerate_ddp_4gpu.yaml`)
- **Effective batch size**: 2 per-device x 4 GPUs x 4 grad accum = 32
- **Learning rate**: 5e-6, cosine schedule, 10% warmup
- **Beta (KL penalty)**: 0.1
- **Max completion length**: 786 tokens
- **Epochs**: 1 (600 logged steps, logging every 10 steps = ~6,000 total steps)
- **Checkpoints**: every 500 steps

### What changed from exp02
| Parameter | exp02 | exp03 |
|---|---|---|
| GPUs | 1x L40s | 4x L40s (DDP) |
| gradient_accumulation_steps | 8 | 4 |
| Effective batch size | 16 | 32 |
| Total steps | ~12,000 | ~6,000 |
| ref_sync_steps | 0 (implicit) | 0, 16, or 1 |

## Evaluation Results

| Model | MATH Test Accuracy | Avg Response Length | Delta vs Baseline |
|-------|--------------------|---------------------|-------------------|
| Baseline (no training) | **27.20%** (136/500) | 610.1 tokens | — |
| exp02 refsync0 (1 GPU) | **28.40%** (142/500) | 603.1 tokens | +1.2 pp |
| **refsync0** (4 GPU, no sync) | **28.00%** (140/500) | 596.7 tokens | +0.8 pp |
| **refsync16** (sync every 16) | **24.40%** (122/500) | 586.2 tokens | **-2.8 pp** |
| **refsync1** (sync every step) | **25.20%** (126/500) | 577.3 tokens | **-2.0 pp** |

Evaluation: 500 MATH test problems, temperature=0.6, single run, `math_verify` for answer matching.

## Training Metrics

### Variant A: refsync0 (reference = original base model)

#### Phase Summary (600 logged steps, 60 per phase)

| Phase | Epoch | Loss | Grad Norm | Reward | KL | Entropy | Clip Low |
|-------|-------|------|-----------|--------|-----|---------|----------|
| P1 | 0.00-0.10 | 0.00879 | 0.1459 | 1.430 | 0.000703 | 0.240 | 1.34% |
| P2 | 0.10-0.20 | 0.00838 | 0.1066 | 1.441 | 0.001652 | 0.235 | 1.32% |
| P3 | 0.20-0.30 | 0.00737 | 0.0997 | 1.454 | 0.002238 | 0.236 | 1.33% |
| P4 | 0.30-0.40 | 0.00564 | 0.0776 | 1.428 | 0.002352 | 0.238 | 1.26% |
| P5 | 0.40-0.50 | 0.00792 | 0.0832 | 1.425 | 0.002809 | 0.231 | 1.40% |
| P6 | 0.50-0.60 | 0.00620 | 0.0769 | 1.389 | 0.002960 | 0.245 | 1.31% |
| P7 | 0.60-0.70 | 0.00705 | 0.0850 | 1.416 | 0.002965 | 0.239 | 1.31% |
| P8 | 0.70-0.80 | 0.00646 | 0.0822 | 1.381 | 0.003078 | 0.244 | 1.38% |
| P9 | 0.80-0.90 | 0.00779 | 0.0865 | 1.412 | 0.002906 | 0.237 | 1.34% |
| P10 | 0.90-1.00 | 0.00784 | 0.0954 | 1.465 | 0.002895 | 0.234 | 1.38% |

#### Health Checks
- **NaN losses**: 0
- **Max grad_norm**: 1.43 (step 88)
- **KL range**: 0.000703 → 0.003078 (grew 4.4x, plateaued by epoch 0.3)
- **Entropy range**: 0.231 → 0.245 (stable, no mode collapse)
- **Behavior**: Nearly identical to exp02 — KL grows monotonically, stabilizes mid-training

### Variant B: refsync16 (reference synced every 16 steps)

#### Phase Summary (600 logged steps, 60 per phase)

| Phase | Epoch | Loss | Grad Norm | Reward | KL | Entropy | Clip Low |
|-------|-------|------|-----------|--------|-----|---------|----------|
| P1 | 0.00-0.10 | 0.00869 | 0.1496 | 1.430 | 0.000355 | 0.239 | 1.34% |
| P2 | 0.10-0.20 | 0.00797 | 0.1323 | 1.441 | 0.000442 | 0.236 | 1.36% |
| P3 | 0.20-0.30 | 0.00677 | 0.1084 | 1.454 | 0.000496 | 0.229 | 1.38% |
| P4 | 0.30-0.40 | 0.00488 | 0.0974 | 1.428 | 0.000545 | 0.233 | 1.35% |
| P5 | 0.40-0.50 | 0.00663 | 0.1368 | 1.425 | 0.000596 | 0.240 | 1.56% |
| P6 | 0.50-0.60 | 0.00513 | 0.1241 | 1.389 | 0.000596 | 0.262 | 1.51% |
| P7 | 0.60-0.70 | 0.00581 | 0.1419 | 1.416 | 0.000590 | 0.272 | 1.55% |
| P8 | 0.70-0.80 | 0.00504 | 0.1406 | 1.381 | 0.000582 | 0.264 | 1.60% |
| P9 | 0.80-0.90 | 0.00624 | 0.1656 | 1.412 | 0.000557 | 0.263 | 1.58% |
| P10 | 0.90-1.00 | 0.00609 | 0.1825 | 1.465 | 0.000562 | 0.265 | 1.67% |

#### Health Checks
- **NaN losses**: 0
- **Max grad_norm**: 2.03 (step 88)
- **KL range**: 0.000355 → 0.000596 (stayed 5x lower than refsync0!)
- **Entropy range**: 0.229 → 0.272 (drifted upward — model became less confident)
- **Grad norm**: Did NOT decay like refsync0 — stayed high and increased in later phases (0.15 → 0.18), confirming ongoing gradient signal without KL suppression

### Variant C: refsync1 (reference synced every step)

#### Phase Summary (600 logged steps, 60 per phase)

| Phase | Epoch | Loss | Grad Norm | Reward | KL | Entropy | Clip Low |
|-------|-------|------|-----------|--------|-----|---------|----------|
| P1 | 0.00-0.10 | 0.00869 | 0.1461 | 1.430 | 0.000346 | 0.239 | 1.34% |
| P2 | 0.10-0.20 | 0.00796 | 0.1302 | 1.441 | 0.000401 | 0.233 | 1.36% |
| P3 | 0.20-0.30 | 0.00675 | 0.1044 | 1.454 | 0.000442 | 0.223 | 1.38% |
| P4 | 0.30-0.40 | 0.00488 | 0.0943 | 1.428 | 0.000467 | 0.227 | 1.34% |
| P5 | 0.40-0.50 | 0.00662 | 0.1347 | 1.425 | 0.000526 | 0.237 | 1.56% |
| P6 | 0.50-0.60 | 0.00512 | 0.1204 | 1.389 | 0.000534 | 0.262 | 1.51% |
| P7 | 0.60-0.70 | 0.00582 | 0.1374 | 1.416 | 0.000541 | 0.269 | 1.53% |
| P8 | 0.70-0.80 | 0.00503 | 0.1359 | 1.381 | 0.000571 | 0.260 | 1.58% |
| P9 | 0.80-0.90 | 0.00622 | 0.1602 | 1.412 | 0.000556 | 0.263 | 1.58% |
| P10 | 0.90-1.00 | 0.00609 | 0.1764 | 1.465 | 0.000559 | 0.263 | 1.66% |

#### Health Checks
- **NaN losses**: 0
- **Max grad_norm**: 1.79 (step 88)
- **KL range**: 0.000346 → 0.000571 (lowest of all — effectively no KL penalty)
- **Entropy range**: 0.223 → 0.269 (similar drift to refsync16)
- **Behavior**: Nearly identical to refsync16 — syncing every step vs every 16 made almost no difference

## Wall-Clock Comparison

| Run | GPUs | Batch Size | Steps | Wall Time | Speedup vs exp02 |
|-----|------|------------|-------|-----------|-------------------|
| exp02 | 1x L40s | 16 | ~12,000 | 4h 18m | — |
| refsync0 | 4x L40s | 32 | ~6,000 | 1h 40m | **2.6x** |
| refsync16 | 4x L40s | 32 | ~6,000 | 1h 48m | **2.4x** |
| refsync1 | 4x L40s | 32 | ~6,000 | 1h 47m | **2.4x** |

The ref_sync variants are ~8 minutes slower due to LoRA weight copy overhead (3 deep copies of ~13M params per step).

## Issues Encountered

1. **Log file location**: refsync16 and refsync1 were submitted before the SBATCH output path was updated to `logs/exp03_multigpu_refsync/`. Logs landed in project root. Moved manually after completion.
2. **No other issues**: All 6 jobs (3 train + 3 eval) completed successfully on first attempt.

## Analysis

### KL Behavior Comparison

**Predictions vs actual**:

| Variant | Predicted KL | Actual KL | Prediction correct? |
|---|---|---|---|
| refsync0 | Grows → plateaus by epoch 0.3 | 0.0007 → 0.0031, plateaued by P3 | Yes |
| refsync16 | Sawtooth, stays small | 0.0004 → 0.0006, flat (5x lower than refsync0) | Partially — flat, not sawtooth |
| refsync1 | Near-zero throughout | 0.0003 → 0.0006, flat | Yes |

The sawtooth prediction for refsync16 didn't materialize at the phase-averaged level — the resets every 16 steps are too frequent to see in 60-step phase averages. At the individual step level the pattern may exist but is smoothed out.

**Key finding**: refsync16 and refsync1 are nearly identical in all metrics. Syncing every step vs every 16 steps made virtually no difference — both effectively removed the KL anchor.

### Gradient Variance Hypothesis

| ref_sync | Expected Grad Noise | Actual Avg Grad Norm | Actual Late-Phase Trend | Verdict |
|----------|---------------------|----------------------|------------------------|---------|
| 0 (never) | Low (strong anchor) | 0.09 (decays) | Decreasing with LR | **Confirmed** |
| 16 | Medium | 0.13 (stays high) | Increasing (0.13 → 0.18) | **Partially confirmed** — higher noise, but also more gradient signal |
| 1 | High (no anchor) | 0.13 (stays high) | Increasing (0.13 → 0.18) | **Same as refsync16** — not worse |

The gradient variance hypothesis was partially right: refsync16/1 have higher grad norms than refsync0. But the prediction that refsync1 would be worse than refsync16 was wrong — they behave identically.

### Entropy Drift

A critical observation: refsync16 and refsync1 show **entropy increasing** in the second half of training (0.23 → 0.27), while refsync0 stays flat (0.23 → 0.24).

This means the ref-synced models became **less confident** over time — their output distributions spread out. This is the opposite of mode collapse but still harmful: the models are becoming more uncertain rather than learning clearer preferences.

### Why Lower Train Loss = Worse Eval

| Metric | refsync0 | refsync16/1 | Interpretation |
|---|---|---|---|
| Train loss | 0.0073 (higher) | 0.0063 (lower) | Ref-sync models optimize more aggressively |
| KL penalty | ~0.003 (large) | ~0.0005 (small) | KL was a significant part of refsync0's loss |
| Eval accuracy | 28.0% (better) | 24.4-25.2% (worse) | Aggressive optimization hurt generalization |
| Entropy | Stable (0.24) | Drifting up (0.27) | Ref-sync models lost confidence |
| Response length | 597 tokens | 577-586 tokens | Shorter = potentially degenerate |

The lower train loss came from removing the KL brake, not from learning better. Without KL anchoring to the base model:
- The student drifted away from the base model's general capabilities
- It overfit to the teacher's specific solution patterns on the training problems
- It lost the base model's 27.2% accuracy while trying to mimic the teacher

### Batch Size Effect
- refsync0 (28.0%) closely matched exp02 (28.4%) despite doubling batch size (16→32)
- This confirms the batch size increase did **not** harm performance
- The 0.4 pp difference is within sampling noise of a single eval run

### Answers to Key Questions

1. **Did multi-GPU provide meaningful speedup?** Yes — 2.6x (4h 18m → 1h 40m)
2. **Does refsync16 beat refsync0?** **No** — it performed significantly worse (24.4% vs 28.0%)
3. **Does refsync1 show mode collapse?** No mode collapse (entropy increased, not collapsed), but performance degraded
4. **Is the batch size increase harmful?** No — refsync0 matched exp02

### Root Cause: KL Is Protective, Not Harmful

In exp02, we hypothesized that KL plateauing early was **suppressing learning**. The exp03 results show the opposite: KL was **protecting the base model's capabilities**.

The KL penalty (β=0.1) with the original base model as reference serves as a regularizer that says: "learn from the teacher, but don't forget what you already know." When we removed this by syncing the reference:
- The student was free to deviate arbitrarily from the base model
- It followed teacher-specific patterns that don't generalize to test problems
- The base model's existing 27.2% accuracy was damaged

This is analogous to **catastrophic forgetting** in continual learning: the student forgets its pre-training when the KL anchor is removed.

## Next Steps

1. **Keep refsync0 as default** — reference sync hurts in this setting
2. **Focus on other improvements** from RPG analysis:
   - Switch to `constant_with_warmup` LR schedule (RPG used this exclusively)
   - Higher warmup ratio (0.5 instead of 0.1)
   - Try beta=0.0 explicitly (different from refsync1 — no KL penalty at all, vs KL with moving target)
3. **Increase rollouts** (4→16 or 64 per problem) for better reward diversity
4. **Scale to 1.5B student** — more capacity to learn without forgetting
5. **Multiple eval runs** — single run at temp=0.6 has high variance; use temp=0.0 and 10 runs

## Artifacts

### Output directories (on scratch)
- refsync0: `/home/shuai14/scratch/dongheng/offline_grpo_full_4gpu_refsync0/`
- refsync16: `/home/shuai14/scratch/dongheng/offline_grpo_full_4gpu_refsync16/`
- refsync1: `/home/shuai14/scratch/dongheng/offline_grpo_full_4gpu_refsync1/`
- Merged models: `*_merged/` (created during eval)

### Logs
- Training: `logs/exp03_multigpu_refsync/grpo-4gpu-refsync{0,16,1}-{4295524,4295525,4295526}.out`
- Eval: `logs/exp03_multigpu_refsync/grpo-eval-rs{0,16,1}-{4295710,4295711,4295712}.out`
- wandb: [offline-grpo project](https://wandb.ai/donghengli9-university-of-alberta/offline-grpo)

---

*Created: 2026-03-09*
*Completed: 2026-03-09*
