# DG-offline: implementation

Quick code walkthrough with line numbers to `DG-offline/trainer.py`. For background reasoning, design decisions, and the experimental plan, see [technical_reference.md](technical_reference.md); for the derivation and side-by-side equations, see [theory.md](theory.md).

The algorithm is a subclass of TRL's `GRPOTrainer` that changes only the gate computation and the IS neutralization. Everything else (KL penalty, PPO surrogate, LoRA, ref model, padding masks) is inherited.

## File map

| File | Purpose | Lines |
|---|---|---|
| `DG-offline/trainer.py` | `DGOfflineTrainer` class; gate computation + IS neutralization | 276 |
| `DG-offline/train.py` | CLI and accelerate launch glue | 249 |
| `DG-offline/run_math.sh` | Single-η MATH runner | 119 |
| `DG-offline/run_gsm8k.sh` | Single-η GSM8K runner | 116 |
| `DG-offline/run_eval_gsm8k_sweep.sh` | Multi-η eval wrapper | 90 |
| `shared/run_train_offline.sh` (`METHOD=dg_offline` branch) | Unified launcher for all 4 datasets | section at lines 137-160 |
| `shared/run_dg_then_online.sh` | Two-stage wrapper: DG-offline → online GRPO | 92 |

## The three changes from `GRPOTrainer`

All localized to `DGOfflineTrainer._compute_loss` (actually `_prepare_inputs`, which is overridden).

### Change 1 — Compute surprisal under the current policy

```python
# DG-offline/trainer.py:169-172
with torch.no_grad():
    current_per_token_logps, _ = self._get_per_token_logps_and_entropies(
        self.model, prompt_completion_ids, attention_mask, logits_to_keep,
    )
```

`current_per_token_logps` has shape `(B, C)` where `B` is the batch and `C` is the max completion length. Surprisal is its negation:

```python
# trainer.py:175
surprisal = -current_per_token_logps  # (B, C), non-negative
```

This is the ONLY place we compute a logprob at this step. We never need the teacher's logprobs. They are not stored in the rollout loader, not read from disk, not used in the loss.

### Change 2 — Build the gate

Two modes: `completion` (one gate per rollout) and `token` (one gate per token, folded to a mean).

```python
# trainer.py:177-193
if self._dg_gating == "completion":
    lengths = completion_mask.sum(dim=1).clamp(min=1).float()
    completion_surprisal = (surprisal * completion_mask).sum(dim=1) / lengths  # (B,)
    delight = advantages * completion_surprisal
    gate = torch.sigmoid(delight / self._dg_temperature)  # (B,)
    gated_advantages = gate * advantages
elif self._dg_gating == "token":
    per_token_delight = advantages.unsqueeze(1) * surprisal  # (B, C)
    per_token_gate = torch.sigmoid(per_token_delight / self._dg_temperature)  # (B, C)
    mean_gate = (per_token_gate * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)
    gated_advantages = mean_gate * advantages
```

`dg_temperature` is the η in `sigmoid(delight / η)`. Lower η → sharper (more aggressive filtering). All our headline runs use `dg_gating="completion"`, which corresponds to a single gate per rollout based on the rollout's average surprisal.

`gated_advantages` replaces the plain `advantages` everywhere downstream in the loss.

### Change 3 — Neutralize the IS ratio in the PPO surrogate

TRL's `GRPOTrainer` uses a PPO-style loss of the form `min(r * A, clip(r) * A)`, where `r = exp(new_logp - old_logp)`. To make the gate the only weighting, we set `old_logp = current_logp.detach()` so that at the first gradient step `r = exp(0) = 1`, the clip becomes a no-op, and the loss reduces to `-gated_advantage × log π_current(completion)` plus the KL term:

```python
# trainer.py:199
old_per_token_logps = current_per_token_logps.detach()
```

The `.detach()` is critical: without it, the gradient flows through both the "new" and "old" logprobs and cancels. Detaching the old copy makes it a constant, so the gradient flows only through `log π_current` in the surrogate.

With `num_ppo_epochs > 1` the ratio diverges from 1 on later inner epochs and the clipping kicks in, which is the desired PPO behavior. We currently run `num_ppo_epochs=1`, which makes this effectively pure gated REINFORCE.

## What goes into the rollout loader

`DG-offline/train.py` loads teacher rollouts via `offline_grpo/data.py:load_rollouts`. Required fields:

| Field | Used for | Required? |
|---|---|---|
| `completion_ids` | input_ids for the forward pass | yes |
| `response` | reward computation via the dataset's `check_answer` | yes |
| `ground_truth_answer` | reward | yes |
| `logprobs` (per-token teacher logprobs) | would be IS ratio; we ignore | **no, ignored by DG** |

This makes DG-offline portable to any teacher, including black-box APIs that don't expose logprobs. It also means our bf16-rounded stored teacher logprobs never contribute numerical drift.

## Hyperparameters

| Parameter | Default in `shared/run_train_offline.sh` | Role |
|---|---|---|
| `DG_ETA` (env var, maps to `--dg_temperature`) | 0.5 | gate sharpness; swept ∈ {0.1, 0.5, 1.0, 2.0} |
| `--dg_gating` | `completion` | completion-level gate (vs per-token) |
| `--ref_sync_steps` | 0 | never sync ref LoRA; use base model via `disable_adapter()` |
| `--beta` | 0.001 | KL coefficient against ref |
| `--learning_rate` | 3e-6 | matches VERL 3B GRPO-LoRA reference |
| `--num_train_epochs` | 5 | same as BC and Offline GRPO for fair comparison |
| `--num_generations` | 5 | group size for advantage normalization (matches what teacher rollouts were generated with) |
| `--per_device_train_batch_size × gradient_accumulation_steps` | 5 × 2 | generation_batch_size = 5 × 4 × 2 = 40; 40 % 5 = 0 ✓ |
| `--lora_r`, `--lora_alpha` | 32, 32 | applied to all-linear target modules |

## Invocation examples

```bash
# Single run, η=0.5, on MATH
DG_ETA=0.5 METHOD=dg_offline DATASET=math \
  sbatch --job-name=dg-math-eta0.5 shared/run_train_offline.sh

# Full η sweep on SVAMP (what we ran on 2026-04-18)
for eta in 0.1 0.5 1.0 2.0; do
  DG_ETA=$eta METHOD=dg_offline DATASET=svamp \
    sbatch --job-name=dg-svamp-eta${eta} shared/run_train_offline.sh
done

# Two-stage: DG-offline stage already merged, now run online GRPO on top
DATASET=math sbatch --job-name=dg-then-online-math shared/run_dg_then_online.sh
```

## What to check when debugging

| Symptom | Likely cause | Fix |
|---|---|---|
| Loss oscillates wildly between steps | `num_ppo_epochs > 1` and gate is flipping | lower η or stay at `num_ppo_epochs=1` |
| Gate saturates near 0.5 for every rollout | η too large relative to typical `advantage × surprisal` | lower η |
| Gate saturates at 0 or 1 binary for every rollout | η too small | raise η |
| `ref_per_token_logps` is `None` and KL = 0 | `beta=0` in the config; KL disabled | verify `beta=0.001` was passed |
| Training crashes with `IndexError` on embedding lookup | completion includes a teacher-padding token id ≥ 151936 | ensure `offline_grpo/data.py` truncation is active; `vocab_size=model_config.vocab_size` must be passed to `load_rollouts`. Note: this constraint comes from the *shared loader* carrying teacher token IDs, not from DG itself. A string-input loader path (planned for the multi-teacher experiment, see [plans/multi_teacher_experiment.md](plans/multi_teacher_experiment.md)) re-tokenizes under the student tokenizer and removes the issue entirely. |
| Two-stage run diverges or gradient explodes | the merged base model is the checkpoint-era base, not base Qwen; any LoRA delta is relative to that | verify `base_model_name_or_path` in `adapter_config.json` points to the merged DG-offline dir |

## Related code: DG-Mixture (not yet trained)

`mixture_grpo/dg_mixture/trainer.py` applies the same sigmoid-gate idea to a mixture of online student rollouts + offline teacher rollouts in a single step. Design doc at `docs/dg_offline/plans/dg_mixture_design.md`. Prototype, not yet trained. Lower priority than closing the DG-offline audit.
