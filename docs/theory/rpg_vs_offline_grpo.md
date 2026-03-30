# RPG vs Offline GRPO: Code Analysis

**Files analyzed**:
- `grpo_rpg/grpo_RPG_external.py` — `GRPORPGExternalTrainer` (~1563 lines)
- `rollots_setpu/evaluation.sh` — Evaluation script for RPG experiments
- `trainer.py` — Our `OfflineGRPOTrainer` (~275 lines)

---

## 1. High-Level Comparison

| Feature | Our Offline GRPO (`trainer.py`) | RPG (`grpo_RPG_external.py`) |
|---|---|---|
| Base class | `GRPOTrainer` | `GRPOTrainer` |
| Generation | Offline only (pre-computed rollouts) | **Online + offline hybrid** (live vLLM generation + external rollouts) |
| IS correction | PPO-style clipped ratio: `exp(log_current - log_old)` | **Mixture-denominator** ratio: `π_cur / ((π_cur + π_old) / 2)` |
| Loss structure | Single loss | **Dual loss**: 50% live + 50% replay |
| Replay buffer | None | FIFO `deque(maxlen=buffer_size)` |
| Reference sync | `ref_sync_steps` (our exp03 feature) | `ref_steps` (same concept, already tested) |
| IS granularity | Token-level (implicit via TRL) | Configurable: **token-level** or **sequence-level** |
| Loss types | GRPO only | GRPO, BNPO, DR-GRPO |
| Complexity | ~275 lines, 1 override | ~1563 lines, 5+ overrides |
| Reward computation | Pre-computed offline | **Live** (calls `math_verify` during training) |

---

## 2. Architecture Difference

### Our Offline GRPO

```
Dataset (48K pre-computed rollouts)
    │
    ▼
┌─────────────────────────────┐
│ _generate_and_score_completions │  ← Looks up offline data by (qid, rid)
│   • completion_ids from rollouts │
│   • behavior_logprobs from teacher│
│   • advantages pre-computed       │
│   • ref logprobs via disable_adapter / ref_sync │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ TRL's _compute_loss (unchanged) │  ← PPO-style clipped IS ratio
│   ratio = exp(π_student - π_teacher) │
│   loss = -min(ratio * A, clip(ratio) * A) │
│   loss += β * KL                    │
└─────────────────────────────┘
```

### RPG External

```
┌────────────────────┐     ┌──────────────────────┐
│ Live generation    │     │ External rollouts     │
│ (vLLM on-policy)   │     │ (teacher JSONL file)  │
│ • Generate         │     │ • Load by (qid, rid)  │
│ • Score rewards     │     │ • Get old logprobs    │
│ • Compute advantages│     │ • Recompute rewards   │
└────────┬───────────┘     └──────────┬───────────┘
         │                            │
         ▼                            ▼
┌─────────────────┐        ┌──────────────────────┐
│ compute_loss()  │        │ _process_replay_slice │
│ = super().      │        │ • Forward pass        │
│   compute_loss()│        │ • IS ratio            │
│ (live, on-policy│        │ • Recompute advantage │
│  RPG loss)      │        │ • Backward pass       │
└────────┬────────┘        └──────────┬────────────┘
         │                            │
         ▼                            ▼
    0.5 × live_loss    +    0.5 × replay_loss / N_replay
         │                            │
         └────────────┬───────────────┘
                      ▼
              Total gradient update
```

---

## 3. The RPG Importance Sampling Weight

This is the core algorithmic difference. Standard GRPO (and PPO) use:

```
ratio = π_current(a|s) / π_old(a|s) = exp(log π_current - log π_old)
```

RPG uses a **mixture denominator**:

```
weight = π_current / ((π_current + π_old) / 2)
```

This is the IS weight for a **mixture distribution** `μ = (π_current + π_old) / 2`. Properties:

| Property | Standard IS | RPG Mixture IS |
|---|---|---|
| Formula | `π_cur / π_old` | `π_cur / ((π_cur + π_old) / 2)` |
| Range | `[0, ∞)` | `[0, 2]` (bounded!) |
| When π_cur = π_old | 1.0 | 1.0 |
| When π_cur >> π_old | Explodes | → 2.0 (capped) |
| When π_cur << π_old | → 0 | → 0 |
| Stability | Needs clipping | Self-stabilizing |
| Variance | High | Lower (bounded ratio) |

### In code (token-level, detach_denominator=True):

```python
# grpo_RPG_external.py, lines 701-715
cur_prob = torch.exp(cur_log_prob).clamp(min=1e-9)
old_prob = torch.exp(old_log_prob).clamp(min=1e-9)
cur_prob_detached = cur_prob.detach()

# RPG coefficient: 1 / (old/cur_detached * past_weight + cur_weight) * log_prob
coef_1 = old_prob / cur_prob_detached * self.past_model_weight  # e.g., 0.5
coef_1 = coef_1 + self.cur_model_weight                         # e.g., + 0.5
coef_1 = 1 / coef_1                                              # invert
coef_1 = coef_1 * cur_log_prob                                   # multiply by log prob
```

When `cur_weight = past_weight = 0.5`, this simplifies to:
```
coef = cur_log_prob / (0.5 * old_prob/cur_prob_detached + 0.5)
     = cur_log_prob * cur_prob_detached / (0.5 * old_prob + 0.5 * cur_prob_detached)
     = cur_log_prob * 2 * cur_prob / (old_prob + cur_prob)
```

The `detach_denominator=True` variant stops gradients from flowing through the denominator, which prevents the optimizer from exploiting the ratio by making `π_cur` small.

### In code (token-level, detach_denominator=False):

```python
# Simpler version
coef_1 = cur_prob / ((cur_prob + old_prob) / 2)
coef_1 = torch.clamp(coef_1, max=2.0)
```

This is the straightforward mixture IS weight, hard-clamped at 2.0.

---

## 4. Loss Computation Comparison

### Our Offline GRPO (uses TRL's `_compute_loss` unchanged)

```python
# TRL default (PPO-style clipped objective)
log_ratio = per_token_logps - old_per_token_logps
coef_1 = torch.exp(log_ratio)                          # IS ratio
coef_2 = torch.clamp(coef_1, 1 - ε_low, 1 + ε_high)   # clipped ratio
per_token_loss1 = coef_1 * advantages
per_token_loss2 = coef_2 * advantages
per_token_loss = -torch.min(per_token_loss1, per_token_loss2)  # pessimistic
per_token_loss += β * KL
loss = (per_token_loss * mask).sum(-1) / mask.sum(-1)
```

### RPG (`_compute_loss` override)

```python
# RPG (no clipping, bounded by design)
per_token_loss = -coef_1 * advantages.unsqueeze(1)     # coef_1 = mixture IS weight
if entropy_mask is not None:
    per_token_loss = per_token_loss * entropy_mask      # optional entropy filtering
per_token_loss += β * KL
loss = (per_token_loss * mask).sum(-1) / mask.sum(-1)
```

Key differences:
1. **No clipping** — RPG doesn't need PPO's `min(ratio*A, clip(ratio)*A)` because the mixture IS weight is naturally bounded in `[0, 2]`
2. **Entropy masking** — RPG can optionally mask out low-entropy tokens (where the model is already confident), focusing learning on uncertain tokens
3. **Multiple loss types** — BNPO normalizes over all tokens globally; DR-GRPO divides by `B × max_completion_length`

---

## 5. Dual Loss: Live + Replay

RPG's `compute_loss()` does two things per training step:

```python
def compute_loss(self, model, inputs, ...):
    # 1. Process external (teacher) rollouts
    replay_examples = []
    for qid, gid in zip(question_ids, generation_ids):
        external = self.offline_data.get((qid, gid))
        # ... load completion_ids, old_logprobs, answer from teacher rollouts
        replay_examples.append(external)

    # Backward pass on replay data (gradients accumulate)
    replay_loss = self._process_replay_slice(model, device, replay_examples)
    # Scaled: 0.5 * replay_loss / N_replay

    # 2. Standard live loss (on-policy, model's own generations)
    live_loss = super().compute_loss(model, inputs)

    # Return 0.5 * live_loss (replay gradients already accumulated)
    return 0.5 * live_loss
```

**Why dual loss?**
- **Live loss** (on-policy): The student generates its own solutions and learns from them. This is standard GRPO.
- **Replay loss** (off-policy): The student also learns from the teacher's solutions. This is the offline component.
- Combined: The student gets both on-policy learning signal (explore) and off-policy teacher knowledge (exploit).

**Our approach** is purely offline — the replay part only, no live generation. This is simpler but misses the on-policy exploration.

---

## 6. Replay Buffer Mechanics

RPG maintains a FIFO replay buffer, though in the external variant it's mostly used for staging:

```python
self.buffer = deque(maxlen=buffer_size)  # default 1024

def _stage_batch(self, inputs, per_token_logps):
    """Store samples to pending buffer (CPU)"""
    for i in range(batch_size):
        store = {k: v[i].detach().cpu() for k, v in inputs.items()}
        store["old_per_token_logps"] = per_token_logps[i].detach().cpu()
        self._pending_buffer.append(store)

def _flush_pending_to_buffer(self):
    """Move pending → main buffer"""
    for s in self._pending_buffer:
        self.buffer.append(s)
    self._pending_buffer.clear()
```

In the external variant, the buffer is less central — the external rollouts are loaded directly from the JSONL file each step rather than buffered. The buffer infrastructure exists for the non-external RPG variant where on-policy generations are replayed.

---

## 7. Reward Computation Difference

### Our Offline GRPO
- Rewards are **pre-computed** in `data.py` during rollout loading
- `correctness_reward`: 2.0 if correct, 0.0 if wrong
- Advantages are pre-computed per group: `(reward - group_mean) / group_std`
- Reward is static — never changes during training

### RPG External
- Rewards are **recomputed live** for replay samples:

```python
# In _process_replay_slice:
rewards_per_func[0,0] = correctness_reward_func_original(
    completions_text, replay_sample['answer']
)
# Recompute advantage against the group mean from the live batch
advantages = rewards - mean_grouped_rewards
```

- Uses `math_verify` to parse and verify `\boxed{}` answers
- The `mean_grouped_rewards` comes from the live batch's group mean, not the original group — so advantages are recomputed relative to the current batch context

---

## 8. Reference Model Handling

### Our Offline GRPO (exp03)
```python
# ref_sync_steps=0: disable_adapter() → base model logprobs
# ref_sync_steps>0: swap in snapshot of LoRA weights from N steps ago
def _get_ref_logprobs(self, model, ...):
    if self._ref_sync_steps > 0 and self._ref_adapter_state is not None:
        # Swap in reference LoRA weights → forward → swap back
        ...
    else:
        with model.disable_adapter():
            # Forward with LoRA disabled = original base model
            ...
```

### RPG External
```python
# Standard TRL approach — uses ref_model or disable_adapter()
if self.ref_model is not None:
    ref_per_token_logps = self._get_per_token_logps_and_entropies(self.ref_model, ...)
else:
    with self.accelerator.unwrap_model(self.model).disable_adapter():
        ref_per_token_logps = self._get_per_token_logps_and_entropies(self.model, ...)
```

RPG doesn't implement ref_sync in the trainer — but the evaluation script shows `ref_steps16` and `ref_steps32` in checkpoint paths, meaning reference sync was handled externally or in a different training script variant.

---

## 9. Evaluation Setup Comparison

### Our evaluation (`evaluate.py`)
- 500 MATH test problems
- temperature=0.6
- 1 run (single sample)
- `math_verify` for answer checking

### RPG evaluation (`rollots_setpu/evaluation.sh`)
- Full MATH test set (likely 5000 problems)
- **temperature=0.0** (greedy, deterministic)
- **10 runs** (more statistically reliable)
- `eval_math_boxed.py` with `max_model_len=3072`, `max_tokens=2048`

Key differences:
| Setting | Ours | RPG |
|---|---|---|
| Temperature | 0.6 (sampling) | 0.0 (greedy) |
| Runs | 1 | 10 |
| Test set | 500 problems | Full MATH test |
| Reliability | Low (high variance) | High (deterministic + multiple runs) |

**Takeaway**: We should switch to temperature=0.0 and multiple runs for more reliable eval.

---

## 10. Hyperparameters from RPG Experiments

The evaluation script reveals previously tested configs:

| Model | LR | Beta | Max Grad Norm | ref_steps | Schedule |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 5e-6 | 0.0 | 0.1 | — | constant_with_warmup |
| Qwen2.5-0.5B | 5e-6 | 0.2 | 0.1 | 32 | constant_with_warmup |
| Qwen2.5-1.5B | 5e-6 | 0.0 | 0.1 | — | constant_with_warmup |
| Qwen2.5-1.5B | 5e-7 | 0.4 | 1.0 | 16 | constant_with_warmup |
| Qwen2.5-1.5B | 1e-6 | 0.4 | 0.1 | 16 | constant_with_warmup |
| Qwen2.5-1.5B | 8e-6 | — | 0.1 | — | constant_with_warmup |
| Qwen2.5-1.5B | 9e-6 | — | 0.1 | — | constant_with_warmup |

Notable observations:
1. **`constant_with_warmup`** schedule used everywhere (not cosine like our exp02)
2. **Beta up to 0.4** tested (we use 0.1)
3. **ref_steps=16 and 32** already tested (same as our exp03 concept)
4. **1.5B model** extensively tested (our logical next step)
5. **warm_ratio=0.5** used with ref_steps (50% warmup — much higher than our 10%)
6. RPG with `cur_weight=0.3, past_weight=0.7` variant tested (weighting past more)

---

## 11. What We Can Learn from RPG for Our Experiments

### Immediate improvements
1. **Eval settings**: Switch to temperature=0.0 and 10 runs for reliable comparisons
2. **LR schedule**: Try `constant_with_warmup` instead of cosine — RPG exclusively uses it
3. **Higher warmup ratio**: 0.5 instead of 0.1 — gives the model more time to ramp up

### Future experiments
4. **Mixture IS weight**: Replace PPO-clipped ratio with the bounded `π_cur / ((π_cur + π_old) / 2)` — eliminates the need for clipping hyperparameters
5. **Dual loss**: Add live generation alongside offline learning — lets the student explore while also learning from the teacher
6. **Entropy masking**: Focus learning on uncertain tokens, skip confident ones
7. **Beta=0.4 with ref_steps=16**: The RPG experiments tested much higher KL penalty when using ref_sync — suggests aggressive regularization is viable when the reference refreshes frequently

### Model scaling
8. **Qwen2.5-1.5B**: Well-tested in RPG, natural next step from our 0.5B experiments

---

## 12. Code Quality Comparison

| Aspect | Our Offline GRPO | RPG External |
|---|---|---|
| Lines of code | ~275 | ~1563 |
| Debug artifacts | None | Many (`pdb.set_trace()`, commented blocks, print statements) |
| Commented-out code | Minimal | Extensive (alternative implementations left in) |
| Documentation | Clean docstrings | Sparse |
| Modularity | Single clean override | Multiple intertwined overrides |
| Production readiness | Higher | Research/experimental |

The RPG trainer is clearly a research prototype with many experimental branches left in the code. Our offline GRPO trainer is cleaner and more focused, but also more limited in capabilities.

---

*Created: 2026-03-09*
*Source files: `grpo_rpg/grpo_RPG_external.py`, `rollots_setpu/evaluation.sh`, `trainer.py`*
