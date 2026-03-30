# Debug Log — Offline GRPO Pipeline

---

## 2026-03-05 — Vocab mismatch: CUDA index out of bounds during training

**Job**: 4257508 (grpo-train)

**Symptom**: Training crashed at step 4 with:
```
vectorized_gather_kernel: Assertion `ind >=0 && ind < ind_dim_size` failed
torch.AcceleratorError: CUDA error: device-side assert triggered
```
Traceback pointed to `_get_per_token_logps_and_entropies()` → model forward pass → embedding gather.

**Root cause**: Teacher model (Qwen2.5-Math-7B-Instruct) has `vocab_size=152064`, student model (Qwen2.5-0.5B-Instruct) has `vocab_size=151936`. The Math-7B model has 128 extra math-specific tokens. When the teacher generates completions containing these tokens (IDs 151936–152059), the student model's embedding layer can't index them → out-of-bounds CUDA error.

**Scope**: 781 / 48,000 completions (1.6%) contained at least one out-of-vocab token.

**Fix**: Added `vocab_size` parameter to `load_rollouts()` in `data.py`. Completions are truncated at the first out-of-vocab token during loading. Updated `train.py` to read the student model's `vocab_size` from config and pass it to `load_rollouts()`.

**Files changed**: `data.py`, `train.py`

**Resubmitted as**: Job 4257567

---

## 2026-03-05 — ZeRO-2 config: `gradient_accumulation_steps: auto` not supported

**Job**: 4251937 (multigpu-test, zero2_beta0.0 and zero2_beta0.1)

**Symptom**: Both ZeRO-2 tests failed immediately with:
```
ValueError: invalid literal for int() with base 10: 'auto'
```
at `accelerate/accelerator.py:546`.

**Root cause**: The accelerate YAML config had `gradient_accumulation_steps: auto` in the `deepspeed_config` section. Accelerate 1.10.1's `Accelerator.__init__()` passes this value directly to `int()`, which can't parse the string `"auto"`.

**Fix**: Changed `gradient_accumulation_steps: auto` → `gradient_accumulation_steps: 4` in `configs/accelerate_zero2_2gpu.yaml` to match the training argument.

**Files changed**: `configs/accelerate_zero2_2gpu.yaml`

**Resubmitted as**: Job 4251960 — both ZeRO-2 tests passed.

---

## 2026-03-05 — `.to("cuda")` breaks multi-GPU training

**Job**: N/A (caught during code review before multi-GPU test)

**Symptom**: Predicted OOM or device placement errors when running with `accelerate launch` on multiple GPUs.

**Root cause**: `train.py` line 82 had `.to("cuda")` which forces the entire model onto GPU 0 before `accelerate` can distribute it. With FSDP/ZeRO-3, this means the full model lands on GPU 0 (potentially OOM) before sharding can happen. With DDP, it wastes memory by briefly having two copies on GPU 0.

**Fix**: Removed `.to("cuda")`. The model stays on CPU after `from_pretrained()`, and the Trainer's `accelerator.prepare()` handles device placement correctly for all strategies (DDP, ZeRO, FSDP, single GPU).

**Files changed**: `train.py`

---

## 2026-03-05 — `--resume_from_checkpoint` not a valid argument

**Job**: 4258328 (grpo-train, resumed)

**Symptom**: Job failed in 22 seconds with `train.py: error: unrecognized arguments: --resume_from_checkpoint`

**Root cause**: `train.py` uses a custom `argparse` setup, not HuggingFace's `HfArgumentParser`. The `--resume_from_checkpoint` flag was added to the shell script (`run_full.sh`) but not to `train.py`'s argument parser.

**Fix**: Added `--resume_from_checkpoint` argument to `train.py`'s `parse_args()` and passed it to `trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)`. Also changed the value from the output dir path to `latest` (HF Trainer auto-detects the latest checkpoint in `output_dir`).

**Files changed**: `train.py`, `run_full.sh`

**Resubmitted as**: Job 4258658

---

## 2026-03-05 — Training timed out at 28% (2-hour limit too short)

**Job**: 4257567 (grpo-train)

**Symptom**: Job killed by SLURM with `TIMEOUT` status after 2 hours. Training reached epoch 0.28 (step ~3500 of ~12000).

**Root cause**: Underestimated training time. 48,000 completions with `max_completion_length=786` and `gradient_accumulation_steps=8` takes ~7 hours on 1× L40s, not 1 hour as originally estimated.

**Fix**: Resubmitted with `--time=8:00:00` and added `--resume_from_checkpoint` to `run_full.sh` so training resumes from checkpoint-3500 instead of restarting.

**Checkpoints saved**: 500, 1000, 1500, 2000, 2500, 3000, 3500

**Resubmitted as**: Job 4258328

---

## 2026-03-04 — 90% of training steps have zero gradient

**Job**: diagnose.py run via srun

**Symptom**: `grad_norm: 0.0` on ~90% of training steps. Model barely learning.

**Root cause**: GRPO computes advantages relative to each group: `advantage = (reward - group_mean) / group_std`. With only 4 generations per problem and a strong 7B teacher (73% accuracy), most groups have all-correct or all-wrong answers → `group_std = 0` → `advantage = 0` → zero gradient. diagnose.py confirmed 90.2% of groups had all-same rewards.

**This is not a bug** — it's a property of the algorithm combined with the data. GRPO only learns from prompts where the model produces mixed results.

**Mitigation options** (not yet applied):
- Increase `num_generations` to 16+ per problem
- Use harder problems where the teacher gets more wrong
- Use a weaker teacher or smaller model

**Files changed**: None (data/algorithm issue)

---
