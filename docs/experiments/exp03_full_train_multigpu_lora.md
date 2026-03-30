# Experiment 03: Multi-GPU Training + Reference Model Sync

**Date**: 2026-03-09
**Jobs**: 4295524 (rs0), 4295525 (rs16), 4295526 (rs1), 4295710-4295712 (eval)
**Status**: Complete
**wandb**: [offline-grpo project](https://wandb.ai/donghengli9-university-of-alberta/offline-grpo)
**Train log**: [exp03_train_log.md](exp03_train_log.md) (template ready, fill after jobs complete)

---

## Objective

Two goals:

1. **Multi-GPU speedup**: Train on 4× L40s with DDP instead of 1× L40s (exp02 took 4.3 hours)
2. **Reference model sync**: Test whether periodically syncing the KL reference model to the current student prevents the KL plateau observed in exp02 and improves eval accuracy

### Motivation from exp02

In exp02, KL divergence plateaued at epoch 0.3 (~step 3600). The student stopped learning after 30% of training — the growing KL penalty combined with cosine LR decay killed the learning signal. The reference model was always the original base model (`disable_adapter()`), so KL accumulated monotonically.

**Hypothesis**: If we periodically sync the reference model to the current student, KL resets and the student can keep learning. But more frequent syncing may introduce higher variance in gradients.

### Research Question

> x-axis: sync frequency (iterations between syncs)
> y-axis: gradient variance (E[grad_norm²])
>
> Assumption: more frequent sync → higher variance → more noise → potentially worse performance
> But also: more frequent sync → KL doesn't accumulate → more total learning

We want to find the sweet spot.

---

## Experiment Matrix

| Variant | ref_sync_steps | Reference model is... | Expected KL | Job name |
|---|---|---|---|---|
| **A** | 0 (never) | Original base model | Grows → plateaus | `grpo-4gpu-refsync0` |
| **B** | 16 | Student from 16 steps ago | Sawtooth, stays small | `grpo-4gpu-refsync16` |
| **C** | 1 | Student from previous step | Always ~0 | `grpo-4gpu-refsync1` |

All three use: 4× L40s DDP, same rollouts as exp02, same hyperparameters except batch size.

---

## Setup

### What stays the same as exp02

| Parameter | Value |
|---|---|
| Teacher model | Qwen2.5-Math-7B-Instruct |
| Student model | Qwen2.5-0.5B-Instruct + LoRA (r=16, alpha=64) |
| Rollout data | `rollouts_full.jsonl` (48K completions, 12K problems) |
| Learning rate | 5e-6 (cosine schedule, 10% warmup) |
| Beta (KL penalty) | 0.1 |
| Max completion length | 786 tokens |
| Max prompt length | 256 tokens |
| Max grad norm | 0.1 |
| Weight decay | 0.1 |
| Adam betas | (0.9, 0.99) |
| Epochs | 1 |
| bf16 | Yes |
| Seed | 42 |

### What changes from exp02

| Parameter | exp02 | exp03 |
|---|---|---|
| GPUs | 1× L40s | 4× L40s (DDP) |
| Distribution | None (`python train.py`) | `accelerate launch` + DDP |
| gradient_accumulation_steps | 8 | 4 |
| Effective batch size | 2×1×8 = 16 | 2×4×4 = 32 |
| Total steps per epoch | ~12,000 | ~6,000 |
| ref_sync_steps | 0 (implicit) | 0, 16, or 1 |
| Expected time per variant | 4.3 hours | ~1-1.5 hours |

### Batch size note

With 4 GPUs, effective batch size doubles to 32. We reduced `gradient_accumulation_steps` from 8 to 4 to partially compensate. If all variants show lower accuracy than exp02, the batch size is the cause — rerun with `gradient_accumulation_steps=2` (effective batch 16, matching exp02).

---

## How Reference Sync Works

### Current behavior (`ref_sync=0`)

```
KL = log π_student(token) - log π_base(token)
     ↑ changes every step     ↑ never changes (disable_adapter)
```

KL grows monotonically. By epoch 0.3, the penalty is large enough to suppress further learning.

### With reference sync (`ref_sync=N`)

Every N steps, snapshot current LoRA weights → becomes the new reference:

```
Step 0:   ref = base model (LoRA initialized)
Step 16:  ref = student at step 16 (copy LoRA weights)
Step 32:  ref = student at step 32
...

KL = log π_student_current - log π_student_N_steps_ago
     ↑ changes every step     ↑ updates every N steps
```

KL now measures **recent drift**, not total drift. Resets every N steps.

### Implementation (in trainer.py)

1. `_sync_ref_adapter()`: Deep-copies all `lora_*` parameters into `_ref_adapter_state` dict
2. `_get_ref_logprobs()`:
   - If `ref_sync=0`: uses `disable_adapter()` (original behavior)
   - If `ref_sync>0`: temporarily swaps reference LoRA weights in, computes forward pass, swaps current weights back
3. Counter increments each training step, triggers sync when reaching `ref_sync_steps`

---

## Code Changes

| File | Change |
|---|---|
| `trainer.py` | Added `ref_sync_steps` param, `_sync_ref_adapter()`, `_get_ref_logprobs()`, sync counter logic |
| `train.py` | Added `--ref_sync_steps` argument (default 0), logged to wandb |
| `configs/accelerate_ddp_4gpu.yaml` | New: 4-GPU DDP config |
| `configs/accelerate_zero2_4gpu.yaml` | New: 4-GPU ZeRO-2 config |
| `run_full_multigpu.sh` | New: run script with train-refsync{0,16,1} + eval variants |

---

## How to Run

### Submit training (all 3 can run in parallel)

```bash
cd /home/shuai14/projects/aip-szepesva/shuai14/backup_dongheng/offline_grpo

sbatch --job-name=grpo-4gpu-refsync0  run_full_multigpu.sh train-refsync0
sbatch --job-name=grpo-4gpu-refsync16 run_full_multigpu.sh train-refsync16
sbatch --job-name=grpo-4gpu-refsync1  run_full_multigpu.sh train-refsync1
```

### Evaluate (after training, override to 1 GPU)

```bash
sbatch --job-name=grpo-eval-rs0  --gpus-per-node=l40s:1 --cpus-per-task=16 --mem=64G --time=0:30:00 run_full_multigpu.sh eval-refsync0
sbatch --job-name=grpo-eval-rs16 --gpus-per-node=l40s:1 --cpus-per-task=16 --mem=64G --time=0:30:00 run_full_multigpu.sh eval-refsync16
sbatch --job-name=grpo-eval-rs1  --gpus-per-node=l40s:1 --cpus-per-task=16 --mem=64G --time=0:30:00 run_full_multigpu.sh eval-refsync1
```

### Monitor

```bash
squeue -u $USER
tail -f grpo-4gpu-refsync*-*.out
```

---

## Expected Results

### Training speed

```
exp02 (1× L40s, batch=16): 4.3 hours, ~12,000 steps
exp03 (4× L40s, batch=32): ~1-1.5 hours, ~6,000 steps
Expected speedup: ~3-4× wall-clock (not perfect 4× due to GPU communication)
```

### KL behavior predictions

| Variant | KL trajectory |
|---|---|
| refsync0 | Grows to ~0.003, plateaus by step ~1800 (epoch 0.3 compressed) |
| refsync16 | Sawtooth: grows for 16 steps, resets near 0, grows again |
| refsync1 | Near-zero throughout (only measures per-step change) |

### Eval accuracy predictions

| Variant | Expected | Reasoning |
|---|---|---|
| Baseline | 27.2% | Same as exp02 |
| refsync0 | ~28-29% | Same algorithm as exp02, different batch size |
| refsync16 | ~29-31% | More learning — KL doesn't accumulate |
| refsync1 | ~28-32% (uncertain) | Most freedom, but KL penalty is nearly meaningless (always ~0) |

### Gradient variance hypothesis

| ref_sync | Per-step KL penalty | Gradient noise | Expected stability |
|---|---|---|---|
| 0 (never) | Growing, eventually large | Low (strong anchor) | Very stable, but limited learning |
| 16 | Small, periodic resets | Medium | Good balance |
| 1 | Near-zero always | High (effectively β≈0) | May be unstable |

---

## Risks and Mitigations

| Risk | Symptom | Mitigation |
|---|---|---|
| Larger batch hurts all variants | All accuracy < exp02 | Rerun with `gradient_accumulation_steps=2` (batch=16) |
| refsync1 mode collapse | Entropy drops, degenerate outputs | Compare entropy curves; if collapsing, refsync1 is too aggressive |
| refsync1 ≈ beta=0 | KL always ~0, no regularization effect | Compare refsync1 to a separate beta=0 run |
| Weight swap overhead | refsync1 much slower than refsync0 | Monitor wall-clock; swap is CPU copies of ~13M params, should be fast |
| NCCL timeout | Training hangs on multi-GPU | `NCCL_DEBUG=INFO` in env, increase timeout |

---

## Future Directions (from exp02 notes)

1. **Sweep sync frequency**: If refsync16 works well, try 4, 8, 32, 64 to find the sweet spot
2. **Plot**: x = sync frequency, y = E[grad_norm²] to validate the variance hypothesis
3. **Increase rollouts**: 4 → 64 generations per problem to maximize reward diversity
4. **Combine**: best ref_sync + more rollouts + larger student model

---

## Artifacts

### Code changes
- `trainer.py`: ref_sync mechanism
- `train.py`: `--ref_sync_steps` argument
- `configs/accelerate_ddp_4gpu.yaml`, `configs/accelerate_zero2_4gpu.yaml`
- `run_full_multigpu.sh`

### Output directories (on scratch)
- `/home/shuai14/scratch/dongheng/offline_grpo_full_4gpu_refsync0/`
- `/home/shuai14/scratch/dongheng/offline_grpo_full_4gpu_refsync16/`
- `/home/shuai14/scratch/dongheng/offline_grpo_full_4gpu_refsync1/`
- Merged models: `*_merged/` (created during eval)

---

*Created: 2026-03-09*
