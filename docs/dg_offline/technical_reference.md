# DG-Offline: Technical Implementation Document

## 1. Background

### 1.1 The Problem with Standard Offline GRPO

Standard offline GRPO trains a student model on pre-collected teacher rollouts using importance-sampling (IS) correction. The loss for each completion is:

```
L = -min(r * A, clip(r, 1-ε, 1+ε) * A)
```

where r = π_student(completion) / π_teacher(completion) is the IS ratio and A is the GRPO advantage.

This has three failure modes in our setting (0.5B student learning from 7B teacher):

1. **IS ratio instability.** Per-token ratios compound over 500+ token completions. A single low-probability token under the teacher but high-probability under the student (or vice versa) makes the sequence-level ratio explode or collapse. PPO clipping truncates these, but clipped samples contribute zero gradient — wasted compute.

2. **Symmetric treatment of surprisal.** A teacher completion that the student finds surprising (high -log π_student) gets the same IS treatment whether it succeeded (positive advantage) or failed (negative advantage). But these two cases have very different learning value: surprising successes reveal strategies the student hasn't discovered; surprising failures reveal strategies the student has already learned to avoid.

3. **Dependency on behavior logprobs.** The IS ratio requires π_teacher(a_t | context) for every token, stored during rollout generation. These are computed in bf16 with vLLM, introducing quantization noise (mean diff 0.003, max diff 0.105 from our diagnostics). The ratio amplifies these errors multiplicatively.

### 1.2 The Delightful Policy Gradient

Osband (2026) proposes replacing IS ratios with a sigmoid gate based on "delight":

```
delight = advantage × surprisal
```

where surprisal = -log π_learner(action) is computed under the **learner's current policy** (not the behavior policy).

The gate σ(delight/η) has four regimes:

| Advantage | Surprisal | Delight | Gate | Interpretation |
|-----------|-----------|---------|------|----------------|
| Positive (success) | High (unexpected) | Large positive | ≈ 1 | Amplify: student should learn from this surprising success |
| Positive (success) | Low (expected) | Small positive | ≈ 0.5 | Pass through: student already does this |
| Negative (failure) | High (unexpected) | Large negative | ≈ 0 | Suppress: student wouldn't do this anyway |
| Negative (failure) | Low (expected) | Small negative | ≈ 0.5 | Pass through: student needs to learn to avoid this |

The key theoretical result (Proposition 3): no action-only reweighting (including exact IS correction) can reproduce this asymmetric behavior. IS weights f(a) = π(a)/μ(a) are sign-blind — they scale successes and failures identically for the same action. DG's gate depends on the **product** of advantage and surprisal, which is sign-dependent.

## 2. Mapping DG to Our Setting

### 2.1 Paper vs Our Setting

| Aspect | Paper | Our Setting |
|--------|-------|-------------|
| Behavior policy | Same model, delayed D steps | Different model (7B teacher) |
| Learner | Same architecture as actor | 0.5B student (different capacity) |
| Distribution gap | Small (stale checkpoint) | Large (cross-model) |
| Actions | Single token (bandit) or short sequence | 500+ token completions |
| Advantage | Binary (±0.5) or shaped reward | GRPO group-normalized |
| Behavior logprobs | Available (same model) | Available but noisy (bf16 quantized) |

The largest difference is the distribution gap. In the paper, surprisal is moderate because the behavior policy is a recent copy of the learner. In our setting, surprisal can be very high because the teacher generates tokens the student assigns near-zero probability.

This actually **strengthens** DG's filtering: the gate becomes more decisive (closer to 0 or 1) when surprisal is high, which is exactly when filtering matters most.

### 2.2 Design Decisions

**Surprisal aggregation.** The paper uses per-episode delight (one scalar per trajectory). Our completions contain hundreds of tokens. We compute:

```
completion_surprisal = mean(-log π_student(a_t | context_t))  over completion tokens
```

Using the mean (not sum) normalizes for completion length. Without this, longer completions would have systematically higher surprisal and thus more extreme gates, biasing training toward shorter completions.

**Gating granularity.** We implement two modes:

- `completion` (default): One gate per completion. All tokens in a completion share the same weight. This matches the paper's per-episode gating.
- `token` (experimental): Per-token delight χ_t = A × (-log π_student(a_t)). Each token gets its own gate. Tokens the student finds particularly surprising within a successful completion are upweighted; within a failed completion, they're downweighted.

**No behavior logprobs.** The trainer reads `completion_ids`, `advantage`, and `reward` from the offline data. `behavior_logprobs` are loaded by the data pipeline (for compatibility with existing rollout format) but are never passed to the DG trainer.

**Teacher-agnosticism (method-level).** As a *method*, DG-offline needs only two things from the teacher: (i) the completion as text, (ii) a reward signal. Surprisal is computed under the student's own policy on the student's own tokenization, so the teacher's tokenizer, vocab, architecture, and logprob availability are all irrelevant. Cross-tokenizer teachers, different-architecture teachers, and closed-API teachers (text-only) are supported by the method. See [`theory.md`](theory.md) §6 for the comparison with IS-based offline GRPO, which requires shared tokenization and teacher logprobs.

The current code path does not yet exercise this fully. `DG-offline/train.py` reuses `load_rollouts` from `offline_grpo/data.py`, which expects pre-tokenized teacher `completion_ids` on disk and applies the 128-row Qwen2.5 vocab truncation (see §6 below). This is a code-sharing artifact, not a DG requirement. A string-input loader path that re-tokenizes the teacher's response text under the student tokenizer at training time would remove the constraint entirely. That loader is the planned vehicle for the multi-teacher experiment ([`plans/multi_teacher_experiment.md`](plans/multi_teacher_experiment.md)).

## 3. Implementation

### 3.1 Architecture

```
DG-offline/
├── trainer.py          DGOfflineTrainer (extends TRL GRPOTrainer)
├── train.py            Training entry point with argparse
├── configs/            Accelerate distributed training configs
├── run_math.sh         SLURM script for MATH experiments
└── run_gsm8k.sh        SLURM script for GSM8K experiments

Shared dependencies (imported, not duplicated):
├── offline_grpo/data.py      Rollout loading, reward computation, advantage calculation
└── mixture_grpo/evaluate.py  Evaluation with vLLM
```

### 3.2 DGOfflineTrainer (trainer.py)

The trainer extends `GRPOTrainer` and overrides a single method: `_generate_and_score_completions()`. This is the same extension point used by `OfflineGRPOTrainer`, but the internal logic differs.

#### Data Flow

```
_generate_and_score_completions(inputs):
    │
    ├─ 1. Tokenize prompts (identical to OfflineGRPOTrainer)
    │
    ├─ 2. Look up offline completions by (question_id, run_id)
    │     → completion_ids, advantages  (behavior_logprobs NOT used)
    │     [Cross-tokenizer path (planned, not yet implemented):
    │      look up completion *text* → re-tokenize under student tokenizer
    │      → student-vocab completion_ids. No vocab_size truncation needed.]
    │
    ├─ 3. Pad completions and build attention masks
    │
    ├─ 4. Forward pass: compute current policy logprobs
    │     → current_per_token_logps = log π_θ(a_t | context)
    │
    ├─ 5. Compute DG gate:
    │     surprisal = -current_per_token_logps              # (B, C)
    │     completion_surprisal = mean(surprisal, over tokens) # (B,)
    │     delight = advantages * completion_surprisal         # (B,)
    │     gate = sigmoid(delight / eta)                       # (B,)
    │     gated_advantages = gate * advantages                # (B,)
    │
    ├─ 6. Neutralize IS ratio:
    │     old_per_token_logps = current_per_token_logps.detach()
    │
    ├─ 7. Compute reference logprobs (for KL penalty, same as before)
    │
    └─ 8. Return {completion_ids, completion_mask, gated_advantages,
              old_per_token_logps, ref_per_token_logps, ...}
```

#### Step 5: The DG Gate in Detail

```python
# Per-token surprisal under current student policy
surprisal = -current_per_token_logps  # (batch, seq_len), non-negative

# Aggregate to per-completion scalar
lengths = completion_mask.sum(dim=1).clamp(min=1).float()
completion_surprisal = (surprisal * completion_mask).sum(dim=1) / lengths

# Delight = advantage × surprisal
delight = advantages * completion_surprisal

# Sigmoid gate (eta controls temperature)
gate = torch.sigmoid(delight / self._dg_temperature)

# Apply gate to advantages
gated_advantages = gate * advantages
```

Concrete example: A teacher completion with advantage = +0.8 (correct, above group mean) and mean surprisal = 3.5 (student assigns avg probability e^{-3.5} ≈ 0.03 per token):

```
delight = 0.8 × 3.5 = 2.8
gate = σ(2.8 / 1.0) = 0.943
gated_advantage = 0.943 × 0.8 = 0.754  (nearly full strength)
```

Same completion but advantage = -0.8 (incorrect):

```
delight = -0.8 × 3.5 = -2.8
gate = σ(-2.8 / 1.0) = 0.057
gated_advantage = 0.057 × (-0.8) = -0.046  (nearly fully suppressed)
```

#### Step 6: Neutralizing the IS Ratio

TRL's `_compute_loss` computes:

```python
ratio = exp(new_per_token_logps - old_per_token_logps)
clipped_ratio = clip(ratio, 1-ε, 1+ε)
loss = -min(ratio * advantages, clipped_ratio * advantages)
```

By setting `old_per_token_logps = current_per_token_logps.detach()`:

- At the start of each training step, `new ≈ old` (same model weights), so `ratio ≈ 1.0`.
- `clip(1.0, 0.8, 1.2) = 1.0`, so clipping is a no-op.
- The loss becomes `-1.0 × gated_advantages × per_token_loss_terms`.

Within PPO mini-batches (if `num_ppo_epochs > 1`), the model weights update, causing `new` to drift from `old`. The ratio will deviate from 1.0, and PPO clipping will engage — this is desirable, as it prevents the model from taking too-large steps even within a single outer step.

The effective gradient is:

```
∇L = -σ(delight/η) × advantage × ∇ log π_θ(completion)
```

This matches the DG paper's Equation 1 exactly.

### 3.3 Training Script (train.py)

The training script follows the same structure as `offline_grpo/train.py` with these additions:

- `--dg_temperature` (eta): Controls gate sharpness. Default 1.0 (paper's recommendation).
- `--dg_gating`: `completion` (default) or `token`.
- All model/dataset paths via argparse — nothing hardcoded.
- Imports `load_rollouts`, `compute_rewards_and_advantages`, `build_training_dataset`, `build_offline_lookup` from `offline_grpo/data.py` to avoid code duplication.

### 3.4 SLURM Scripts

Both `run_math.sh` and `run_gsm8k.sh` accept environment variable overrides:

```bash
# Default: 0.5B student, 7B teacher, eta=1.0
sbatch run_math.sh

# Same-tokenizer alternate teacher (works with the current shared loader)
STUDENT_MODEL=/scratch/mrli/models/Qwen2.5-1.5B-Instruct \
TEACHER_MODEL=/scratch/mrli/models/Qwen2.5-3B-Instruct \
ROLLOUT_PATH=/scratch/mrli/rollouts/math_3b/rollouts.jsonl \
DG_ETA=0.5 \
sbatch run_math.sh

# Cross-tokenizer teacher (planned; needs the string-input loader path)
# DG-offline as a method handles this fine — only needs (text, reward) from
# the teacher. See plans/multi_teacher_experiment.md for the implementation
# lift and the experiment design.
#
# STUDENT_MODEL=/scratch/mrli/models/Qwen2.5-0.5B-Instruct \
# TEACHER_MODEL=/scratch/mrli/models/Llama-3.1-8B-Instruct \
# ROLLOUT_PATH=/scratch/mrli/rollouts/math_llama3/rollouts.jsonl \
# ROLLOUT_FORMAT=text \
# DG_ETA=0.5 \
# sbatch run_math.sh

# Eta sweep
for eta in 0.1 0.5 1.0 2.0 5.0; do
    DG_ETA=$eta CHECKPOINT_DIR=/scratch/mrli/checkpoints/dg_eta${eta} \
    sbatch run_math.sh
done
```

## 4. Comparison with Standard Offline GRPO

### 4.1 Side-by-Side

| Aspect | Offline GRPO | DG Offline |
|--------|-------------|------------|
| **Gradient weight** | IS ratio π_student/π_teacher, PPO-clipped | σ(advantage × surprisal / η) |
| **Requires behavior logprobs** | Yes (stored during rollout generation) | No |
| **Treatment of surprising failures** | Same as surprising successes (ratio is large either way) | Suppressed (gate ≈ 0) |
| **Treatment of surprising successes** | Same as surprising failures | Amplified (gate ≈ 1) |
| **Zero-advantage completions** | Zero gradient (no learning) | Gate = 0.5 (half-strength KL gradient still flows) |
| **Robustness to noisy logprobs** | Sensitive (ratio amplifies errors) | Immune (behavior logprobs not used) |
| **Cross-tokenizer compatibility** | Requires the teacher to share the student's tokenizer (IS ratio is undefined otherwise) | Only needs the student to consume the text; teacher tokenizer/architecture/closed-API status are all irrelevant |
| **Per-sample compute** | 1 forward pass (current model) | 1 forward pass (same — surprisal comes from current model logprobs already computed) |

### 4.2 What We Expect

**Optimistic scenario (DG helps):** DG suppresses the high-surprisal failures that dominate standard offline GRPO's gradient. Teacher completions that the student finds surprising AND that are correct (positive advantage) get amplified. This selective pressure lets the student learn from the best teacher demonstrations without being dragged by the worst ones.

**Neutral scenario (DG is flat):** The advantage collapse problem (82-93% of groups have zero reward variance) means most completions have advantage ≈ 0, giving delight ≈ 0 and gate ≈ 0.5 regardless of surprisal. DG's filtering only acts on the 7-18% of completions with non-zero advantage. If this subset is too small, DG can't do much.

**Pessimistic scenario (DG hurts):** For completions where the student assigns extremely low probability (surprisal >> 1), the gate becomes nearly binary. Positive-advantage completions get gate ≈ 1 and the gradient pushes the student hard toward tokens it currently considers nearly impossible. This could destabilize training — the student is asked to make a large policy change in a single step. The temperature η controls this: higher η softens the gate and prevents extreme upweighting.

## 5. Experimental Plan

### 5.1 Controlled Comparison

Use the same rollout data as the standard offline GRPO experiments (7B teacher on MATH, 12K problems × 4 completions):

| Experiment | Method | Key Setting |
|------------|--------|-------------|
| Baseline | No training | — |
| Offline GRPO (existing) | IS ratio + PPO clip | beta=0.001, lr=3e-6 |
| DG offline η=1.0 | DG gate, completion-level | Same lr, beta, LoRA |
| DG offline η=0.5 | Harder gate | Same |
| DG offline η=2.0 | Softer gate | Same |

All other hyperparameters (lr, beta, LoRA config, batch size, epochs) are held constant.

### 5.1.1 Multi-teacher follow-up (planned)

The same-teacher η-sweep above is a controlled comparison against Offline-GRPO on a teacher that *both* methods can run on. The natural follow-up exercises DG's teacher-agnosticism: vary the teacher pool (including cross-tokenizer teachers Offline-GRPO cannot use) and ask (a) whether widening the surprisal distribution makes the four-quadrant gate fire on more than 1–6% of samples (the Apr 21 surprisal diagnostic established the current-setup baseline), and (b) what DG-offline produces in regimes Offline-GRPO is undefined in. Design draft, candidate teachers, pre-check protocol, and staging are in [`plans/multi_teacher_experiment.md`](plans/multi_teacher_experiment.md). Stage 2 of that plan is the string-input loader implementation.

### 5.2 Diagnostic Metrics

The trainer logs DG-specific metrics to wandb and the JSONL metrics file:

- `dg/gate_mean`: Mean gate value across the batch. Should be ~0.5 for zero-advantage samples, higher when positive advantages dominate.
- `dg/gate_min`, `dg/gate_max`: Range of gates. If min is always ~0 and max always ~1, the gate is acting decisively.
- `dg/delight_mean`: Mean delight. Sign indicates whether the batch is dominated by surprising successes (+) or surprising failures (−).
- `dg/surprisal_mean`: Mean per-token surprisal. Measures the distribution gap between student and teacher. Expected to be high (3-5 nats) initially and decrease if the student learns.

### 5.3 What to Look For

1. **Does reward improve during training?** Standard offline GRPO showed flat-to-declining reward on MATH.
2. **Does KL stay controlled?** DG doesn't use IS ratios for stability, so KL divergence is the main safety valve.
3. **Does surprisal decrease?** If the student is learning from teacher completions, it should assign higher probability to them over time, reducing surprisal.
4. **Does the gate distribution change?** Early in training, gates should be concentrated near 0.5 (low advantage magnitude). If DG is working, we should see gates spreading toward 0 and 1 as the model develops stronger preferences.

## 6. Known Limitations

1. **DG was designed for small distribution shifts.** The paper's theory assumes contamination around the learner's own policy. Our 7B→0.5B gap is orders of magnitude larger than any delay D tested in the paper. The theoretical guarantees (Propositions 1-3) may not hold.

2. **Advantage collapse is not addressed by DG.** When all completions for a prompt get the same reward, advantage = 0 and delight = 0 regardless of surprisal. DG cannot create learning signal where none exists. This is the same bottleneck as standard offline GRPO.

3. **One forward pass overhead.** Computing current-policy logprobs in `_generate_and_score_completions` adds one forward pass per step compared to standard offline GRPO (which only looks up stored logprobs). This is the same cost as computing reference logprobs for KL, so the overhead is ~50% per step. We could amortize this if TRL exposed the current-policy logprobs computed during loss computation, but the API doesn't support this cleanly.

4. **The gate is not differentiable w.r.t. the policy.** We stop gradients through the gate (via `old_per_token_logps.detach()`). The paper treats the gate as a constant weight. A differentiable gate (where gradients flow through σ(delight/η)) would be a different algorithm — potentially interesting but not what the paper proposes.

5. **Current loader inherits offline-GRPO's tokenizer constraint, not a DG limitation.** `DG-offline/train.py` reuses `load_rollouts` from `offline_grpo/data.py`, which expects pre-tokenized teacher `completion_ids` and applies the 128-row Qwen2.5 vocab truncation. As a method, DG-offline is teacher-tokenizer-agnostic (see §2.2 and [`theory.md`](theory.md) §6); the current code path simply does not exercise that capability yet. The string-input loader required to lift this is tracked in [`plans/multi_teacher_experiment.md`](plans/multi_teacher_experiment.md).
