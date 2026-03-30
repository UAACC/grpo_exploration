# Experiment 02: Full-Scale Training — Qwen2.5-0.5B on 12K MATH Problems

**Date**: 2026-03-05
**Jobs**: 4257279 (rollouts), 4257508 (train v1, crashed), 4257567 (train v2, timed out), 4258328 (train v3, resumed)
**Status**: In progress (training resumed from checkpoint-3500)
**wandb**: [offline-grpo project](https://wandb.ai/donghengli9-university-of-alberta/offline-grpo)

---

## Objective

Run the full offline GRPO pipeline end-to-end: generate teacher rollouts for all 12,000 MATH training problems, train the 0.5B student with LoRA, and evaluate. This validates the pipeline at scale before moving to larger student models.

## Setup

### Rollout Generation
- **Teacher model**: Qwen2.5-Math-7B-Instruct
- **Data**: 12,000 MATH training problems × 4 generations = 48,000 completions
- **Infrastructure**: 4× L40s, data-parallel (4 vLLM shards)
- **Time**: 36 minutes

### Training
- **Student model**: Qwen2.5-0.5B-Instruct with LoRA (r=16, alpha=64)
- **Data**: 48,000 completions from teacher rollouts
- **Hyperparameters**:
  - Learning rate: 5e-6 (cosine schedule, 10% warmup) constant + 0.1 warmup 
  - Batch size: 2 per device × 8 gradient accumulation = effective 16
  - Beta (KL penalty): 0.1
  - Max completion length: 786
  - Epochs: 1
  - Total steps: ~12,000
- **Infrastructure**: 1× L40s
- **Estimated time**: ~7 hours

## Issues Encountered

### Issue 1: Vocab mismatch crash (Job 4257508)
- Teacher (Math-7B) has vocab_size=152064, student (0.5B) has vocab_size=151936
- 128 extra math-specific tokens in teacher → 781/48,000 completions contained out-of-vocab token IDs
- Student embedding layer crashed with CUDA index out of bounds
- **Fix**: Truncate completions at first out-of-vocab token during data loading (`data.py`)
- See `debug_log.md` for full details

### Issue 2: Training timeout (Job 4257567)
- Submitted with 2-hour limit, but training needs ~7 hours
- Reached epoch 0.28 (step 3500/12000) before SLURM killed it
- 7 checkpoints saved (every 500 steps)
- **Fix**: Resubmitted with 8-hour limit and `--resume_from_checkpoint`

## Training Metrics Analysis (Steps 1–3500, epoch 0.00–0.28)

### Summary Table

| Phase (steps) | Avg Loss | Avg Grad Norm | Avg Reward | Avg KL | Avg Clip (low) | Avg Entropy |
|---------------|----------|---------------|------------|--------|----------------|-------------|
| 1–670 | 0.0088 | 0.1284 | 1.4267 | 0.000435 | 0.013217 | 0.2468 |
| 671–1340 | 0.0104 | 0.2322 | 1.4170 | 0.001156 | 0.014516 | 0.2342 |
| 1341–2010 | 0.0080 | 0.1090 | 1.4000 | 0.001516 | 0.012800 | 0.2371 |
| 2011–2680 | 0.0062 | 0.0847 | 1.4136 | 0.001875 | 0.013808 | 0.2382 |
| 2681–3500 | 0.0070 | 0.1514 | 1.4211 | 0.002247 | 0.013823 | 0.2347 |

### Health Indicators

**Stable (good):**
- No NaN losses in 336 logged steps
- Zero `grad_norm=0` steps (0%) — much better than test run (90%), the full dataset has enough reward diversity
- Loss small and stable (~0.008), not exploding or oscillating
- Entropy steady (~0.24) — model not collapsing to deterministic outputs
- Clip ratio low (~1.4%) — IS ratio rarely clipped, student not drifting too far from teacher

**Growing as expected:**
- KL divergence: 0.0004 → 0.0024 (6.4× increase over 28% of training)
  - Interpretation: student is learning to deviate from base model, but at a controlled rate
  - Projection: ~0.006 by end of training if linear — still well within safe range (<0.01)

**Flat (needs watching):**
- Reward: stable at ~1.42 throughout, not increasing yet
  - With 71% teacher accuracy, the maximum mean reward is ~1.42 (= 0.71 × 2.0)
  - The student can't exceed the teacher's reward ceiling in offline GRPO — it can only learn which of the teacher's solutions to prefer
  - The real test of improvement is evaluation accuracy on the test set

### What the metrics tell us

1. **The pipeline is working correctly.** Non-zero gradients, stable loss, growing KL — the student is learning from the teacher's rollouts.

2. **Reward won't increase much during training.** In offline GRPO, the reward metric reflects the teacher's pre-computed rewards weighted by the current batch composition. The student's actual quality improvement shows up in eval accuracy, not training reward.

3. **KL is the key training signal.** The student is developing its own "preferences" among the teacher's solutions (KL growing), which is exactly what GRPO should do — learn to assign higher probability to correct solutions and lower probability to incorrect ones.

4. **No data quality issues at scale.** The zero-gradient problem from the test run (90% all-same rewards) doesn't appear here because the full 12K dataset has enough hard problems where the teacher gets mixed results.

## Rollout Statistics

| Metric | Value |
|--------|-------|
| Total problems | 12,000 |
| Total completions | 48,000 |
| Correct completions | 34,026 (70.9%) |
| File size | 930.8 MB |
| Generation time | 36 min (4× L40s) |
| Avg completion length | 636 tokens |
| Max completion length | 2,048 tokens |
| Out-of-vocab truncations | 781 (1.6%) |

## Next Steps

1. Wait for Job 4258328 to complete training (resume from checkpoint-3500)
2. Run evaluation: `sbatch --job-name=grpo-eval run_full.sh eval`
3. Compare eval accuracy against baseline (student without LoRA)
4. If successful → scale up to 7B student model with ZeRO-2

## Artifacts

- Rollouts: `/home/shuai14/scratch/dongheng/teacher_rollouts/rollouts_full.jsonl`
- Rollout shards: `/home/shuai14/scratch/dongheng/teacher_rollouts/rollouts_shard_{0,1,2,3}.jsonl`
- Checkpoints: `/home/shuai14/scratch/dongheng/offline_grpo_full/checkpoint-{500..3500}`
- Training logs: `grpo-train-4257567.out` (steps 1-3500), `grpo-train-4258328.out` (steps 3500+)
- wandb: project `offline-grpo`
