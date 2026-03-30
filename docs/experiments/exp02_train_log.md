# Experiment 02: Training Log — Qwen2.5-0.5B Offline GRPO

**Date**: 2026-03-05 / 2026-03-06
**Status**: Complete (training + evaluation)

---

## Pipeline Overview

| Stage | Job ID | Status | Duration | Notes |
|-------|--------|--------|----------|-------|
| Rollout generation | 4257279 | Complete | 36 min | 4x L40s, data-parallel |
| Train v1 | 4257508 | Crashed | — | Vocab mismatch (see Issues) |
| Train v2 | 4257567 | Timed out | 2 hours | Reached step 3500/12000 |
| Train v3 (resume) | 4258328 | Failed | — | `--resume_from_checkpoint` not implemented |
| Train v4 (from scratch) | 4258695 | Complete | 4.3 hours | Full 12000 steps, 1 epoch |
| Eval (trained) | 4267687 | Complete | ~2 min | LoRA merged + vLLM |
| Eval (baseline) | 4267688 | Crashed | — | CUDA fork issue in vLLM |
| Eval (baseline retry) | 4267713 | Complete | ~2 min | Fixed with VLLM_WORKER_MULTIPROC_METHOD=spawn |

## Setup

### Rollout Generation
- **Teacher**: Qwen2.5-Math-7B-Instruct
- **Data**: 12,000 MATH train problems x 4 generations = 48,000 completions
- **Teacher accuracy**: 70.9% (34,026/48,000 correct)
- **Out-of-vocab truncations**: 781/48,000 (1.6%) — teacher vocab 152064 vs student vocab 151936

### Training
- **Student**: Qwen2.5-0.5B-Instruct + LoRA (r=16, alpha=64)
- **Effective batch size**: 2 per-device x 8 grad accum = 16
- **Learning rate**: 5e-6, cosine schedule, 10% warmup
- **Beta (KL penalty)**: 0.1
- **Max completion length**: 786 tokens
- **Epochs**: 1 (~12,000 steps)
- **Logging**: every 10 steps (1,186 logged entries total)
- **Checkpoints**: every 500 steps (24 checkpoints saved)

## Evaluation Results

| Model | MATH Test Accuracy | Avg Response Length |
|-------|--------------------|---------------------|
| Qwen2.5-0.5B-Instruct (baseline) | **27.20%** (136/500) | 610.1 tokens |
| + Offline GRPO from 7B teacher | **28.40%** (142/500) | 603.1 tokens |
| **Delta** | **+1.2 pp** (+6 problems) | -7 tokens |

Evaluation: 500 MATH test problems, temperature=0.6, single run, `math_verify` for answer matching.

## Training Metrics (Full Run)

### Phase Summary (10 equal phases across 1 epoch)

| Phase | Epoch | Loss | Grad Norm | Reward | KL | Entropy | Clip Low |
|-------|-------|------|-----------|--------|-----|---------|----------|
| P1 | 0.00-0.10 | 0.0098 | 0.1855 | 1.415 | 0.000708 | 0.240 | 1.39% |
| P2 | 0.10-0.20 | 0.0073 | 0.1069 | 1.412 | 0.001588 | 0.236 | 1.32% |
| P3 | 0.20-0.31 | 0.0074 | 0.1201 | 1.423 | 0.002141 | 0.238 | 1.39% |
| P4 | 0.31-0.41 | 0.0063 | 0.1037 | 1.400 | 0.002194 | 0.237 | 1.31% |
| P5 | 0.41-0.50 | 0.0079 | 0.0858 | 1.422 | 0.002551 | 0.234 | 1.41% |
| P6 | 0.50-0.60 | 0.0065 | 0.0841 | 1.401 | 0.002666 | 0.247 | 1.35% |
| P7 | 0.60-0.70 | 0.0071 | 0.0912 | 1.426 | 0.002643 | 0.239 | 1.32% |
| P8 | 0.70-0.80 | 0.0070 | 0.0853 | 1.424 | 0.002747 | 0.244 | 1.41% |
| P9 | 0.80-0.90 | 0.0072 | 0.0972 | 1.404 | 0.002603 | 0.238 | 1.37% |
| P10 | 0.90-1.00 | 0.0080 | 0.1421 | 1.446 | 0.002588 | 0.235 | 1.41% |

### Health Checks
- **NaN losses**: 0
- **Zero grad_norm steps**: 0 (0%) — full dataset has enough reward diversity
- **Negative loss steps**: 135/1186 (11.4%) — normal for policy gradient
- **Max grad_norm**: 6.17 (single spike, no sustained instability)
- **KL range**: 0.000708 → 0.002747, plateaued by epoch 0.3
- **Entropy range**: 0.234 → 0.247, stable throughout (no mode collapse)

### Key Trends
- **KL divergence**: Grew 3.6x from P1 to P8 (0.0007 → 0.0027), then plateaued. Student learned its preferences in the first ~30% of training and refined them over the remaining 70%.
- **Reward**: Flat at ~1.42 throughout. This is the theoretical ceiling (teacher accuracy 70.9% x max reward 2.0 = 1.42). Cannot increase in offline GRPO.
- **Loss**: Small and stable (~0.007). Initial phase slightly higher (0.0098) then settled.
- **Grad norm**: Decreased from 0.19 to 0.09 as LR decayed, with slight uptick at the end.
- **Learning rate**: Cosine decay from 5e-6 to ~0, fully decayed by epoch 1.0.

## Issues Encountered

1. **Vocab mismatch crash** (Job 4257508): Teacher has 128 extra math tokens. Fixed by truncating completions at first OOV token in `data.py`.
2. **Training timeout** (Job 4257567): 2-hour SLURM limit was too short for ~4.3 hour training.
3. **Resume not implemented** (Job 4258328): `--resume_from_checkpoint` was not a valid argument in `train.py`. Restarted from scratch.
4. **vLLM CUDA fork crash** (Job 4267688): Baseline eval crashed because vLLM forked a subprocess without CUDA spawn. Fixed by setting `VLLM_WORKER_MULTIPROC_METHOD=spawn`.

## Analysis

### Why the improvement is small (+1.2 pp)

1. **KL plateaued very early**. The student stopped diverging from its base policy by epoch 0.3, suggesting the cosine LR schedule decayed too aggressively. The model spent 70% of training with near-zero learning rate making negligible updates.

2. **Offline GRPO has a hard ceiling**. The student can only learn to prefer correct teacher solutions over incorrect ones — it cannot generate new reasoning. With 70.9% teacher accuracy, the signal is: "for problems where teacher got mixed results (some correct, some wrong), upweight the correct ones." Problems the teacher always got right or always wrong provide zero gradient.

3. **0.5B model capacity**. A 0.5B model may not have the capacity to absorb complex mathematical reasoning patterns. The LoRA adapter (r=16) adds ~13M trainable parameters — a small surface for learning nuanced preferences.

4. **Single eval run**. At temperature=0.6, there is meaningful sampling variance. The +1.2 pp could be anywhere from +0 to +3 pp with confidence intervals. Multiple runs would tighten this estimate.

5. **Reward diversity is limited**. With only 4 generations per problem and binary reward (correct/incorrect), many problem groups have all-same rewards (all correct or all incorrect), yielding zero advantage and zero gradient. Only problems with mixed results contribute to learning.

### What the metrics confirm

- The pipeline is **mechanically correct**: non-zero gradients, stable loss, growing KL, valid reference logprobs.
- The student is **not collapsing**: entropy is stable, no degenerate behavior.
- The **offline GRPO objective is being optimized**, but the downstream eval improvement is marginal for this model size and data setup.

## Next Steps: Possible Improvements

### 1. More generations per problem (high priority)
Increase `--num_generations` from 4 to 16 or 32. This directly increases reward diversity per group — more chances for mixed correct/incorrect outcomes, yielding stronger advantage signal. Expected cost: 4-8x rollout generation time.

### 2. Larger student model (high priority)
Move from 0.5B to 7B (Qwen2.5-7B-Instruct) or 1.5B. A larger model has more capacity to learn from the teacher's solutions. Use ZeRO-2 for multi-GPU training (validated in Experiment 01). This is the most impactful single change.

### 3. Tune the learning rate schedule
- Try a **constant LR** or **linear decay** instead of cosine. KL plateaued at epoch 0.3 because cosine already dropped the LR significantly by then.
- Try a **higher peak LR** (1e-5 or 2e-5) to extract more learning before decay kicks in.
- Train for **2-3 epochs** to give the model more passes over the data.

### 4. Multiple eval runs for reliable comparison
Rerun evaluation with `--runs 8` to reduce sampling variance. The current +1.2 pp is within the noise margin of a single run at temperature=0.6.

### 5. Use a stronger teacher
Replace Qwen2.5-Math-7B-Instruct with a 72B model or one with higher MATH accuracy. Higher teacher accuracy means more correct solutions to learn from, and harder problems get solved, providing learning signal on the tail.

### 6. Increase beta or try different KL formulations
Beta=0.1 may be too conservative. Try beta=0.01 or beta=0.0 to let the student deviate more aggressively from its base policy.

### Priority order
1. Multiple eval runs (quick, reduces uncertainty)
2. More generations per problem (moderate cost, directly improves signal quality)
3. Higher LR / more epochs (moderate cost, addresses early KL plateau)
4. Larger student model (highest expected impact, highest cost)

## Artifacts

- Rollouts: `/home/shuai14/scratch/dongheng/teacher_rollouts/rollouts_full.jsonl` (930.8 MB)
- Trained model (LoRA): `/home/shuai14/scratch/dongheng/offline_grpo_full/`
- Merged model: `/home/shuai14/scratch/dongheng/offline_grpo_full_merged/`
- Checkpoints: `/home/shuai14/scratch/dongheng/offline_grpo_full/checkpoint-{500..12000}`
- Training logs: `grpo-train-4257567.out` (steps 1-3500), `grpo-train-4258695.out` (full run)
- Eval logs: `grpo-eval-4267687.out` (trained), `grpo-eval-base-4267713.out` (baseline)
- wandb: [offline-grpo project](https://wandb.ai/donghengli9-university-of-alberta/offline-grpo)

## next step:
- reference mode needs to be update(16 iteration to update) try 1 iteration(to sync)
- x: iterations to sync y: variance:(expected gradient norm)^2 ???
- increase the rollout number for each(4 to 64)
- assumption: iterations to sync -->  high variance --> high noise --> reference model low performance 