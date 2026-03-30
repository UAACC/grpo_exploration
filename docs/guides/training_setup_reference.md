# Training Setup & Hyperparameter Reference

Quick-reference for our entire offline GRPO pipeline. Everything in one place.

---

## 1. Models

### Teacher (Behavior Model)
| Property | Value |
|---|---|
| Model | Qwen2.5-Math-7B-Instruct |
| Parameters | ~7 billion |
| Vocab size | 152,064 (128 extra math tokens) |
| Role | Generate rollouts (solutions to math problems) |
| Path | `/home/shuai14/scratch/huggingface_cache/hub/models--Qwen--Qwen2.5-Math-7B-Instruct/snapshots/ef9926d75ab1d54532f6a30dd5e760355eb9aa4d` |
| MATH accuracy | 70.9% (34,026/48,000 correct) |
| Used during training? | **No** — logprobs are pre-computed and saved to disk |

### Student (Target Model)
| Property | Value |
|---|---|
| Model | Qwen2.5-0.5B-Instruct |
| Parameters | ~500 million |
| Vocab size | 151,936 |
| Role | Learns from teacher's rollouts via offline GRPO |
| Path | `/home/shuai14/scratch/huggingface_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775` |
| Baseline MATH accuracy | 27.2% (136/500) |
| After training accuracy | 28.4% (142/500) |

### Vocab Mismatch
Teacher has 128 extra math-specific tokens (IDs 151936–152063). 781/48,000 completions (1.6%) contained these tokens. Fix: `data.py` truncates completions at first out-of-vocab token.

---

## 2. Dataset

| Property | Value |
|---|---|
| Dataset | `nlile/hendrycks-MATH-benchmark` |
| Train split | 12,000 problems |
| Test split | 500 problems |
| Cache | `/home/shuai14/scratch/datasets/MATH` |
| Problem format | Math question with step-by-step solution and `\boxed{answer}` |

### System Prompt
```
Please reason step by step, and put your final answer within \boxed{}.
```

---

## 3. Pipeline Stages

### Stage 1: Rollout Generation (`generate_rollouts.py`)

Uses **vLLM** for fast autoregressive generation from the teacher model.

| Parameter | Test | Full | Flag |
|---|---|---|---|
| Num problems | 41 | 12,000 | `--test_mode` |
| Generations per problem | 4 | 4 | `--num_generations` |
| Total completions | 164 | 48,000 | — |
| Temperature | 0.6 | 0.6 | `--temperature` |
| Top-p | 1.0 | 1.0 | `--top_p` |
| Top-k | -1 (disabled) | -1 | `--top_k` |
| Max new tokens | 1024 (test) / 2048 (full) | 2048 | `--max_tokens` |
| Max model context | 2048 (test) / 3072 (full) | 3072 | `--max_model_len` |
| GPU memory utilization | 0.8 (default) | 0.85 | `--gpu_memory_utilization` |
| Tensor parallel | 1 | 1 (data-parallel instead) | `--tensor_parallel_size` |
| Data-parallel shards | 1 | 4 (one GPU each) | `--num_shards` |
| Seed | 42 | 42 | `--seed` |

**Output**: `rollouts.jsonl` — each line is one problem with N completions containing:
- `response`: solution text
- `boxed_answer`: extracted answer
- `completion_ids`: token IDs
- `logprobs`: per-token log-probabilities from teacher

**Timing**: ~36 min on 4× L40s (full), ~2 min on 1× L40s (test)

### Stage 2: Training (`train.py` + `trainer.py`)

Uses **HuggingFace Trainer** (TRL's GRPOTrainer) with standard PyTorch forward/backward passes. No vLLM here — needs gradients.

#### Training Hyperparameters

| Parameter | Test | Full | Flag | Why this value |
|---|---|---|---|---|
| Learning rate | 5e-6 | 5e-6 | `--learning_rate` | Lower than default 5e-5 to prevent instability with IS ratios |
| LR scheduler | cosine | cosine | `--lr_scheduler_type` | Smooth decay to zero |
| Warmup ratio | 0.1 | 0.1 | `--warmup_ratio` | 10% of steps for linear warmup |
| Weight decay | 0.1 | 0.1 | hardcoded in `train.py` | L2 regularization |
| Adam beta1 | 0.9 | 0.9 | hardcoded | Standard |
| Adam beta2 | 0.99 | 0.99 | hardcoded | Slightly lower than default 0.999 |
| Max grad norm | 0.1 | 0.1 | `--max_grad_norm` | Aggressive clipping (default is 1.0) — prevents large IS ratio spikes |
| Precision | bf16 | bf16 | `--bf16` | bfloat16 mixed precision |
| Seed | 42 | 42 | `--seed` | Reproducibility |

#### Batch Size & Steps

| Parameter | Test | Full | Flag |
|---|---|---|---|
| Per-device batch size | 2 | 2 | `--per_device_train_batch_size` |
| Gradient accumulation | 4 | 8 | `--gradient_accumulation_steps` |
| **Effective batch size** | **8** | **16** | batch × accum |
| Epochs | 1 | 1 | `--num_train_epochs` |
| Total samples | 164 | 48,000 | — |
| **Total training steps** | **~41** | **~12,000** | samples / effective_batch × epochs |

#### GRPO-Specific Parameters

| Parameter | Test | Full | Flag | What it does |
|---|---|---|---|---|
| Beta (KL penalty) | 0.1 (default) | 0.1 | `--beta` | Coefficient for KL divergence penalty. Higher = more conservative, stays closer to base model |
| Num generations | 4 | 4 | `--num_generations` | Must match rollout file. Tells TRL that every 4 consecutive rows in the dataset belong to the same problem |
| Max prompt length | 256 | 256 | `--max_prompt_length` | Truncate prompts longer than this |
| Max completion length | 512 (test) | 786 (full) | `--max_completion_length` | Truncate completions longer than this |

#### Sequence Length Budget

```
max_prompt_length (256) + max_completion_length (786) = 1042 tokens per sample

Per training step memory ≈ batch_size (2) × sequence_length (1042) × model_size
```

The `max_completion_length=786` was chosen because the average teacher completion is 636 tokens (from exp02). Setting it to 786 captures ~95% of completions without truncation. Longer completions (up to 2048 tokens from the teacher) are truncated.

#### Checkpointing & Logging

| Parameter | Test | Full | Flag |
|---|---|---|---|
| Save steps | 500 (default) | 500 | `--save_steps` |
| Logging steps | 1 (default) | 10 | `--logging_steps` |
| Report to | none | wandb | `--report_to` |
| Resume | — | latest | `--resume_from_checkpoint` |

**Timing**: ~2 min (test, 1× L40s), ~4.3 hours (full, 1× L40s)

### Stage 3: Evaluation (`evaluate.py`)

Uses **vLLM** again for autoregressive generation from the trained student.

| Parameter | Value | Flag |
|---|---|---|
| Test problems | 500 | `--split test` |
| Temperature | 0.6 | `--temperature` |
| Top-p | 1.0 | `--top_p` |
| Max new tokens | 2048 | `--max_tokens` |
| Max model context | 3072 | `--max_model_len` |
| GPU memory utilization | 0.95 | `--gpu_memory_utilization` |
| Eval runs | 1 | `--runs` |
| LoRA merge | yes | `--merge_lora` |

**Process**: Load base model → merge LoRA adapter → save merged model → load in vLLM → generate → check answers with `math_verify`.

**Timing**: ~2 min on 1× L40s

---

## 4. LoRA Configuration

Defined in `configs.py` as `DEFAULT_LORA_CONFIG`:

| Parameter | Value | Flag | What it does |
|---|---|---|---|
| Rank (r) | 16 | `--lora_r` | Size of low-rank matrices A and B |
| Alpha | 64 | `--lora_alpha` | Scaling factor. Effective scale = alpha/r = 4.0 |
| Dropout | 0.05 | hardcoded | Regularization on LoRA layers |
| Task type | CAUSAL_LM | hardcoded | Language modeling |
| Target modules | q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj | hardcoded | All attention + MLP projections |

### Memory impact (0.5B student)

```
Trainable LoRA params:  ~13M  (vs 500M total)
LoRA optimizer memory:  ~200 MB  (vs ~4 GB for full fine-tune)
Checkpoint size:        ~50 MB  (vs ~1 GB for full model)
```

### Disable with `--no_lora`

Setting `--no_lora` runs full fine-tuning (all 500M parameters trainable). Not recommended for models > 1B due to memory.

---

## 5. Reward & Advantage Computation

Done in `data.py`, runs on CPU during data loading (not during training).

### Reward
```python
reward = 2.0 if verify(parse(extracted_answer), parse(ground_truth)) else 0.0
```

Binary. Uses `math_verify` library for symbolic math comparison (handles equivalent forms like `x=2,3` vs `x=3,2`).

### Advantage (GRPO group normalization)
```python
# Group by question_id (all completions for the same problem)
advantage = (reward - group_mean) / (group_std + eps)    # eps = 1e-4
```

- Mixed group (e.g., 3 correct, 1 wrong): correct ones get positive advantage, wrong one gets large negative
- All-same group (all correct or all wrong): `group_std ≈ 0` → advantage ≈ 0 → zero gradient
- With 4 generations at 70% accuracy: ~25% of groups are all-correct → wasted signal

---

## 6. The Training Loop (What Happens Per Step)

Inside `trainer.py` (`OfflineGRPOTrainer._generate_and_score_completions`):

```
For each batch of 2 samples:

1. LOOK UP teacher data
   - Get completion_ids, behavior_logprobs, advantage from offline_data dict
   - Key: (question_id, run_id)

2. COMPUTE STUDENT LOGPROBS (forward pass)
   - Feed [prompt + teacher's completion] through student model
   - One forward pass, causal mask — not autoregressive
   - Get student's logprob for each token

3. COMPUTE REFERENCE LOGPROBS (if beta > 0)
   - disable_adapter() → turns off LoRA → base model
   - Forward pass again → get base model logprobs
   - Used for KL penalty

4. COMPUTE LOSS (done by TRL's _compute_loss, not our code)
   - ratio = exp(student_logprob - teacher_logprob)       ← IS correction
   - clipped_ratio = clip(ratio, 1-ε, 1+ε)               ← PPO-style
   - token_loss = -advantage × min(ratio, clipped_ratio)
   - KL_penalty = β × (student_logprob - base_logprob)
   - total_loss = mean(token_losses) + KL_penalty

5. BACKWARD PASS
   - Gradients flow through student model (LoRA weights only)

6. ACCUMULATE (repeat steps 1-5 for gradient_accumulation_steps times)

7. OPTIMIZER STEP
   - Update LoRA weights
   - Clip gradients to max_grad_norm (0.1)
```

---

## 7. Infrastructure (Vulcan Cluster)

### Hardware
| Resource | Spec |
|---|---|
| GPU | NVIDIA L40s (48 GB VRAM each) |
| CPUs per GPU | 16 |
| Max walltime | 7 days |
| Account | `aip-szepesva` |

### Resource Usage by Stage

| Stage | GPUs | Time | Memory | SLURM |
|---|---|---|---|---|
| Rollouts (test) | 1× L40s | ~2 min | 48 GB | `salloc --time=1:00:00 --gpus-per-node=l40s:1` |
| Rollouts (full) | 4× L40s | ~36 min | 256 GB | `sbatch run_full.sh rollouts` |
| Training (test) | 1× L40s | ~2 min | 48 GB | `salloc --time=1:00:00 --gpus-per-node=l40s:1` |
| Training (full) | 1× L40s | ~4.3 hours | 64 GB | `sbatch --time=8:00:00 run_full.sh train` |
| Evaluation | 1× L40s | ~2 min | 48 GB | `sbatch run_full.sh eval` |

### Environment Activation
```bash
module load python/3.11 cuda/12.6
source /home/shuai14/projects/aip-szepesva/shuai14/verifiers/.venv/bin/activate
export HF_HOME=/home/shuai14/scratch/huggingface_cache
export HF_DATASETS_CACHE=/home/shuai14/scratch/datasets/MATH
export TRANSFORMERS_OFFLINE=1        # prevent downloads on compute nodes
export HF_DATASETS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn  # fix vLLM CUDA fork crash
```

### Key Packages
| Package | Version |
|---|---|
| PyTorch | 2.8.0 |
| Transformers | 4.56.1 |
| TRL | 0.21.0 |
| PEFT | 0.17.1 |
| vLLM | 0.11.0 |
| Flash Attention | 2.8.3 |
| Accelerate | 1.10.1 |
| math_verify | 0.8.0 |

---

## 8. File Map

### Code
```
offline_grpo/
├── configs.py              ← Constants: SYSTEM_PROMPT, model names, LoRA config
├── generate_rollouts.py    ← Stage 1: teacher generates solutions (vLLM)
├── data.py                 ← Load rollouts, compute rewards & advantages
├── trainer.py              ← OfflineGRPOTrainer (overrides _generate_and_score_completions)
├── train.py                ← Stage 2: orchestrates training (loads data, model, LoRA, calls trainer)
├── evaluate.py             ← Stage 3: evaluate on MATH test set (vLLM)
├── diagnose.py             ← Debug tool: check signal health
├── validate_rollouts.py    ← Verify rollout file integrity
├── run_test.sh             ← Shell script: test run (41 problems)
├── run_full.sh             ← Shell script: full run (12K problems)
├── run_multigpu_test.sh    ← Multi-GPU compatibility tests
└── submit_multigpu_test.sh ← SLURM wrapper for multi-GPU tests
```

### Data & Outputs (on scratch — NOT backed up, 60-day purge)
```
/home/shuai14/scratch/dongheng/
├── teacher_rollouts/
│   ├── rollouts_full.jsonl         ← 48K completions, 930.8 MB
│   └── rollouts_shard_{0,1,2,3}.jsonl
├── offline_grpo_full/              ← training checkpoints
│   └── checkpoint-{500..12000}/
└── offline_grpo_full_merged/       ← merged LoRA model for eval
```

### Configs (for multi-GPU)
```
offline_grpo/configs/
├── accelerate_ddp_2gpu.yaml       ← DDP: full copy per GPU
├── accelerate_zero2_2gpu.yaml     ← ZeRO-2: shard optimizer + gradients
└── accelerate_fsdp_2gpu.yaml      ← FSDP: shard everything
```

---

## 9. Known Issues & Fixes

| Issue | Root Cause | Fix | File |
|---|---|---|---|
| CUDA index out of bounds | Teacher vocab (152064) > student vocab (151936) | Truncate at first OOV token | `data.py` |
| `.to("cuda")` breaks multi-GPU | Forces all weights to GPU 0 before distribution | Removed `.to("cuda")` | `train.py` |
| ZeRO-2 config crash | `gradient_accumulation_steps: auto` not supported | Changed to integer `4` | `configs/accelerate_zero2_2gpu.yaml` |
| `--resume_from_checkpoint` not found | Not in argparse | Added argument + pass to trainer | `train.py` |
| vLLM CUDA fork crash | vLLM forks subprocess without spawn | `VLLM_WORKER_MULTIPROC_METHOD=spawn` | `run_full.sh` |
| 90% zero gradients (test data) | All-same rewards with only 4 gens | Use full dataset (enough diversity) or increase to 16 gens | data issue |
| Training timeout | 2-hour limit too short for ~4.3 hour job | Use `--time=8:00:00` | SLURM |

---

## 10. Experiment Results Summary

### Exp 01: Multi-GPU LoRA Compatibility
- **Result**: 6/6 passed (DDP, ZeRO-2, FSDP × beta=0.0, beta=0.1)
- FSDP + `disable_adapter()` works with PEFT 0.17.1 + `fsdp_use_orig_params: true`

### Exp 02: Full-Scale Training (0.5B Student)
- **Result**: +1.2 pp improvement (27.2% → 28.4%)
- Training is mechanically correct (stable loss, growing KL, no NaNs)
- Improvement is small due to: small model capacity, cosine LR decaying too fast, limited reward diversity with 4 gens

### Proposed Next Steps (from exp02)
1. More eval runs (reduce sampling variance)
2. More generations per problem (4 → 16, improves signal quality)
3. Higher LR / more epochs (address early KL plateau)
4. Larger student model (7B, highest expected impact)
5. Stronger teacher (14B-72B, helps on hard problems)

---

*Created: 2026-03-09*
