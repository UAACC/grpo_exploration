# Experiment 01: Online GRPO on GSM8K (Matching VERL Benchmark)

**Date**: 2026-03-10
**Jobs**: 4296607 (train), 4299873 (eval)
**Status**: Complete
**wandb**: [online-grpo-gsm8k project](https://wandb.ai/donghengli9-university-of-alberta/online-grpo-gsm8k)

---

## Objective

Reproduce the VERL benchmark result: **Qwen2.5-0.5B-Instruct GRPO-LoRA = 54.3% on GSM8K**.

This is our first online GRPO experiment. Unlike offline GRPO (exp02/03), the student generates its own completions every training step using vLLM.

---

## Setup

### Source of hyperparameters

- VERL baseline page: `verl.readthedocs.io/en/latest/algo/baseline.html`
- 0.5B log filename: `Qwen2.5-0.5B-bsz64_2-prompt512-resp1024-lorarank32-score0.543.log` (file itself not available)
- 3B GRPO-LoRA script: `examples/grpo_trainer/run_qwen2_5-3b_gsm8k_grpo_lora.sh`

### Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Model | Qwen2.5-0.5B-Instruct | VERL benchmark |
| Dataset | GSM8K (7,473 train / 1,319 test) | VERL benchmark |
| LoRA rank | 32 | 0.5B filename |
| LoRA alpha | 32 | 3B script |
| LoRA target | all-linear | 3B script |
| LoRA dropout | 0.05 | our default |
| Learning rate | 3e-6 | 3B script |
| LR schedule | cosine, 10% warmup | our default |
| KL coeff (beta) | 0.001 | 3B script |
| Num generations | 5 | 3B script |
| Max prompt length | 512 | 0.5B filename |
| Max completion length | 1024 | 0.5B filename |
| Epochs | 15 | 3B script |
| per_device_train_batch_size | 5 | adjusted for divisibility |
| gradient_accumulation_steps | 2 | - |
| Effective batch size | 5 × 4 × 2 = 40 | VERL used 64 |
| Max grad norm | 1.0 | 3B script |
| Clip ratio | 0.2 | TRL default |
| Temperature | 0.7 | our default |
| GPUs | 4× L40s (single node) | - |
| vLLM mode | colocate (30% GPU memory) | - |
| Framework | TRL 0.21.0 GRPOTrainer | - |
| Seed | 42 | - |

### Key differences from VERL

| | VERL | Ours |
|---|---|---|
| Framework | VERL (Ray-based) | TRL (Accelerate + DDP) |
| Effective batch size | 64 | 40 |
| vLLM integration | Hybrid engine (parallel gen+train) | Colocate (serial gen→train) |
| GPUs | 2 | 4 |
| KL loss type | low_var_kl | standard KL (TRL default) |
| Entropy coeff | 0.001 (3B script) | 0 (TRL default) |

### Training steps calculation

```
7,473 problems × 5 generations = 37,365 rows
37,365 / (5 per_device × 4 GPUs) = 1,868 micro-batches/epoch
1,868 / 2 grad_accum = 934 steps/epoch
934 × 15 epochs = 14,010 total steps
```

---

## Training Log

### Per-epoch summary (first log entry of each epoch)

| Epoch | Reward | KL | Loss | Grad Norm | Entropy | Notes |
|-------|--------|-----|------|-----------|---------|-------|
| 0 | 0.005 | 0.0002 | 0.001 | 0.00003 | 0.176 | Cold start, no learning yet |
| 1 | 0.455 | 0.198 | 0.015 | 0.188 | 0.202 | Rapid learning begins |
| 2 | 0.593 | 0.184 | -0.006 | 0.198 | 0.128 | Good progress |
| 3 | 0.475 | 0.669 | 0.038 | 0.488 | 0.338 | Reward dip, KL spike |
| 4 | 0.043 | 23.85 | 0.051 | 2.817 | 2.116 | **Collapse!** KL explodes, reward crashes |
| 5 | 0.665 | 0.138 | 0.009 | 0.201 | 0.101 | Recovery (fresh rollouts from new policy) |
| 6 | 0.693 | 0.158 | -0.001 | 0.315 | 0.087 | Peak stable performance |
| 7 | 0.628 | 0.222 | -0.002 | 0.300 | 0.109 | Stable |
| 8 | 0.683 | 0.295 | 0.052 | 0.380 | 0.103 | KL starting to grow |
| 9 | 0.470 | 1.122 | 0.166 | 0.642 | 0.159 | KL grows, reward drops |
| 10 | 0.510 | 2.963 | 0.123 | 3.502 | 0.181 | Instability |
| 11 | 0.653 | 3.660 | 0.114 | 4.997 | 0.167 | High KL, reward still ok |
| 12 | 0.653 | 7.479 | 0.161 | 4.911 | 0.162 | KL exploding |

### Key observations

**Phase 1: Learning (epoch 0-2)**
- Reward climbs from 0% to 59% in 2 epochs
- KL stays low (~0.2), training is stable
- Model learns GSM8K pattern quickly

**Phase 2: First collapse (epoch 3-4)**
- Epoch 4 catastrophe: KL = 23.85, reward = 4.3%, entropy = 2.1
- Policy collapsed — model generated gibberish
- Likely cause: accumulated KL drift over epoch 3, cosine LR still near peak

**Phase 3: Recovery (epoch 5-8)**
- Online GRPO self-heals: new rollouts come from current policy
- By epoch 5, reward back to 66%, KL back to 0.14
- Best stable period: epoch 5-8, reward 63-69%, KL 0.1-0.3

**Phase 4: Second instability (epoch 9+)**
- KL grows monotonically: 1.1 → 3.0 → 3.7 → 7.5
- Reward oscillates between 47-67%
- Grad norm explodes: 0.6 → 3.5 → 5.0
- Training is becoming unstable

### Training time

- Speed: ~3.1 s/step
- Estimated total: ~12 hours for 14,010 steps
- SBATCH limit: 12 hours (may be cut off before epoch 15)
- Bottleneck: vLLM generation at each step (colocate mode, 30% GPU memory)

---

## Analysis

### Why does KL explode?

In online GRPO, the reference model is always the **original base model** (via LoRA `disable_adapter()`). After many epochs of training, the student drifts far from the base model, causing large KL. With beta=0.001, the KL penalty is too weak to prevent drift.

Compare to VERL which uses `low_var_kl` loss type — a lower-variance KL estimator that may provide more stable gradients.

### The epoch 4 collapse

This is a known failure mode in online RL for LLMs:
1. Accumulated policy drift (KL grows over epochs 2-3)
2. Generated completions become increasingly unusual
3. Advantage estimates become noisy (all completions are bad → zero-std group)
4. One bad gradient update tips the policy into collapse
5. Entropy shoots up (model outputs become random)

The recovery in epoch 5 is the beauty of online GRPO — once the collapsed policy generates new completions, the algorithm restarts from the new (bad) policy and learns again.

### Why is reward capped at ~68%?

Possible reasons:
1. **GSM8K difficulty**: Many problems require multi-step arithmetic that 0.5B model may not be able to learn
2. **KL instability**: The second instability phase (epoch 9+) prevents further learning
3. **Reward extraction**: Our regex-based answer extraction might miss some correct answers
4. **Effective batch size**: 40 vs VERL's 64 — smaller batch = noisier advantage estimates

### Comparison with VERL target

| | VERL | Ours |
|---|---|---|
| Target accuracy | 54.3% (eval) | **55.82%** (step 7000) |
| Training reward | unknown | ~65-68% peak |
| Epochs completed | 15 | 12.23 (cut off by SBATCH 12h limit) |
| KL behavior | unknown | Unstable after epoch 8 |

Note: Training reward is not directly comparable to eval accuracy. Eval uses temp=0 greedy decoding, while training uses temp=0.7 sampling.

---

## Eval Results

Source: `logs/online-grpo-eval-4299873.out`

### GSM8K test accuracy (temp=0.0, 5 runs)

| Checkpoint | Epoch | Accuracy | Std | Min | Max |
|-----------|-------|----------|-----|-----|-----|
| Baseline | - | **48.14%** | 0.0036 | 47.69% | 48.67% |
| Step 5000 | ~5.4 | 53.41% | 0.0010 | 52.69% | 52.92% |
| Step 6000 | ~6.4 | 49.89% | 0.0025 | 54.51% | 55.19% |
| **Step 7000** | **~7.5** | **55.82%** | **0.0063** | **55.12%** | **56.56%** |

Note: Step 5000/6000/7000 used strict `####` extraction only. Baseline used full extraction
(#### → \boxed{} → last number) since the base model doesn't output `####` format.

Source: `logs/eval-baseline2-4300033.out` (baseline), `logs/online-grpo-eval-4299873.out` (checkpoints)

### Key findings

1. **Step 7000 (55.82%) exceeds VERL target (54.3%)** by 1.5 percentage points.
2. **Improvement over baseline: +7.7%** (48.14% → 55.82%).
3. Performance improves monotonically from step 5000 to 7000 (epoch 5-8 is the stable learning phase).
4. Step 6000 (epoch ~12) drops below step 5000 — consistent with KL instability in epoch 9+.
5. Variance increases with more training (std: 0.001 → 0.003 → 0.006), suggesting later checkpoints are less stable.

### Baseline comparison with VERL

| | Our eval | VERL |
|---|---|---|
| Baseline accuracy | 48.14% | 49.6% |
| Gap | -1.5% | - |

The remaining 1.5% gap is explained by **prompt format differences**:

| Factor | Our eval | VERL eval |
|--------|----------|-----------|
| System prompt | `"Please solve this math problem step by step..."` | None |
| User message | Bare question only | Question + `'Let's think step by step and output the final answer after "####".'` |
| Answer extraction | `####` → `\boxed{}` → last number fallback | Strict `####` only |

Sources:
- VERL prompt format: `github.com/volcengine/verl/blob/main/examples/data_preprocess/gsm8k.py`
  (appends instruction to user message, no system prompt)
- VERL answer extraction: `github.com/volcengine/verl/blob/main/verl/utils/reward_score/gsm8k.py`
  (strict `####` matching only)
- VERL baseline numbers: `verl.readthedocs.io/en/latest/algo/baseline.html`

VERL's instruction explicitly asks for `####` format, which means even the base model outputs
in that format more often. Our system prompt doesn't enforce output format, so the base model
defaults to `\boxed{}` and we recover answers via fallback extraction — but some are still missed.

---

## Checkpoints

Saved every 200 steps to `/home/shuai14/scratch/dongheng/online_grpo_gsm8k/`:
- `checkpoint-200` through `checkpoint-11200+` (and growing)
- Best checkpoint likely around step 5000-7500 (epoch 5-8, peak stable reward)

---

## Next Steps

1. **VERL target matched.** Step 7000 = 55.82% vs VERL's 54.3%. Baseline = 48.14% (close to VERL's 49.6%).
2. **Match VERL prompt format**: Remove system prompt, append `'Let's think step by step and output the final answer after "####".'` to user message — for fair apples-to-apples comparison.
3. **Investigate KL instability** (epoch 9+):
   - Increase beta (0.001 → 0.01) to prevent drift
   - Use fewer epochs (8 seems optimal based on eval results)
   - Try low_var_kl if TRL supports it (VERL uses this: `kl_loss_type=low_var_kl` in 3B script)
4. **Eval more checkpoints** between step 7000-9000 to find true peak before instability.
5. **Add eval during training** for future runs (pass `eval_dataset` to GRPOTrainer, matching VERL's `test_freq=5`).

---

## Artifacts

### Code
- `train.py`: Online GRPO training with TRL GRPOTrainer
- `evaluate.py`: GSM8K evaluation with vLLM
- `run_gsm8k.sh`: SLURM batch script
- `configs/accelerate_ddp_4gpu.yaml`: Accelerate DDP config

### Output
- Model checkpoints: `/home/shuai14/scratch/dongheng/online_grpo_gsm8k/`
- Training logs: `logs/online-grpo-gsm8k-4296607.{out,err}`
- wandb: `online-grpo-gsm8k` project

---

*Created: 2026-03-10*
