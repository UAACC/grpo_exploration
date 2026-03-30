# Offline GRPO: Algorithm and Implementation

This document explains the full algorithm from first principles — what GRPO is, how offline GRPO differs, and exactly how our code implements each step.

---

## Table of Contents

1. [Background: RLHF and Policy Optimization](#1-background-rlhf-and-policy-optimization)
2. [GRPO: Group Relative Policy Optimization](#2-grpo-group-relative-policy-optimization)
3. [Offline GRPO: Off-Policy Extension](#3-offline-grpo-off-policy-extension)
4. [Our Pipeline: End-to-End](#4-our-pipeline-end-to-end)
5. [Code Architecture](#5-code-architecture)
6. [Mathematical Details](#6-mathematical-details)

---

## 1. Background: RLHF and Policy Optimization

### The setup

We have a language model (the "policy") that generates text in response to prompts. We want to improve it on a specific task — in our case, solving math problems. The standard approach:

1. **Supervised fine-tuning (SFT)**: Train on (question, correct answer) pairs.
2. **Reinforcement learning from human feedback (RLHF)**: Further optimize the model using a reward signal.

RLHF treats text generation as a sequential decision process:
- **State**: the prompt + tokens generated so far
- **Action**: the next token
- **Reward**: a score for the complete generation (e.g., did it solve the math problem?)
- **Policy**: the language model itself, π(token | context)

### PPO for language models

The standard RLHF approach uses PPO (Proximal Policy Optimization):

```
L_PPO = E[ min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A) ]
```

where:
- `r(θ) = π_θ(a|s) / π_old(a|s)` is the importance sampling ratio
- `A` is the advantage (how much better was this action than average?)
- The `clip` prevents the policy from changing too much in one step

PPO requires a **value function** (a separate neural network) to estimate advantages. This adds significant memory and compute overhead — the value network is often the same size as the policy itself.

### The problem PPO creates

For large language models, PPO is expensive:
- Need to maintain 4 models in memory: policy, reference policy, reward model, value function
- Value function training is unstable and adds hyperparameters
- On-policy: must generate fresh completions every training step

---

## 2. GRPO: Group Relative Policy Optimization

GRPO (introduced in DeepSeek-Math, 2024) eliminates the value function entirely by using **group-relative advantages**.

### Core idea

Instead of estimating "how good is this action compared to what we expect?" with a learned value function, GRPO asks: "how good is this completion compared to **other completions for the same prompt**?"

For each prompt q:
1. Sample G completions: o_1, o_2, ..., o_G ~ π_θ(· | q)
2. Score each: r_1, r_2, ..., r_G
3. Normalize within the group:

```
A_i = (r_i - mean(r_1..G)) / (std(r_1..G) + ε)
```

This is the **group-relative advantage**. No value function needed — we just compare completions against each other.

### The GRPO objective

```
L_GRPO = (1/G) Σ_i [ min(ρ_i * A_i, clip(ρ_i, 1-ε, 1+ε) * A_i) ] - β * KL(π_θ || π_ref)
```

where:
- `ρ_i = π_θ(o_i | q) / π_old(o_i | q)` — importance sampling ratio between current and old policy
- `A_i` — group-relative advantage
- `β * KL(π_θ || π_ref)` — KL penalty to prevent the policy from drifting too far from a reference model
- The `clip` keeps the ratio in [1-ε, 1+ε], preventing destructively large updates

### Why group-relative works

Consider a math problem where 3 out of 4 completions are correct (reward = 2.0) and 1 is wrong (reward = 0.0):
- Mean = 1.5, Std = 0.866
- Correct completions get advantage: (2.0 - 1.5) / 0.866 = +0.577
- Wrong completion gets advantage: (0.0 - 1.5) / 0.866 = -1.732

The model learns to increase probability of the correct solutions and decrease probability of the wrong one, *relative to each other*. This is the key insight: we don't need an absolute value estimate, just relative comparisons within a group.

### When advantage is zero

If all completions in a group get the same reward (all correct or all incorrect):
- std = 0 (or near zero with ε)
- All advantages ≈ 0
- No gradient from this group

This is correct behavior: if the model always succeeds or always fails on a problem, there's no signal about which solution style to prefer.

---

## 3. Offline GRPO: Off-Policy Extension

### The online GRPO bottleneck

Standard (online) GRPO generates fresh completions at every training step:

```
for each training step:
    1. Sample prompts from dataset
    2. Generate G completions per prompt using current policy π_θ     ← SLOW
    3. Score completions
    4. Compute group advantages
    5. Update π_θ with GRPO objective
```

Step 2 is extremely expensive — it requires running inference with the full model for every batch. For large models, generation dominates training time.

### The offline solution

Offline GRPO decouples generation from training:

```
Phase 1 (once): Generate all completions with a behavior policy π_β
    - For each prompt, sample G completions
    - Record completions, rewards, AND per-token logprobs from π_β

Phase 2 (iterate): Train target policy π_θ on the fixed dataset
    - Use pre-computed completions (no generation needed)
    - Correct for distribution mismatch with importance sampling
```

### The distribution mismatch problem

In online GRPO, the completions come from the current policy π_θ, so the ratio is:

```
ρ = π_θ(o | q) / π_old(o | q) ≈ 1  (when old = θ from last step)
```

In offline GRPO, the completions come from a **different** model (the behavior policy π_β):

```
ρ = π_θ(o | q) / π_β(o | q)
```

This is an **importance sampling correction**. It reweights each completion by how much more or less likely it is under the target policy compared to the behavior policy:
- If π_θ assigns higher probability than π_β → ρ > 1 → amplified gradient
- If π_θ assigns lower probability than π_β → ρ < 1 → diminished gradient
- The PPO clip constrains ρ to [1-ε, 1+ε], preventing extreme reweighting

### Why this works with TRL's GRPOTrainer

TRL's `_compute_loss` already computes:

```python
ratio = exp(current_logps - old_per_token_logps)
```

In online GRPO, `old_per_token_logps` is from the policy at the previous step. In our offline version, we set `old_per_token_logps` to the **behavior policy's** logprobs. The math is identical — TRL computes the IS-corrected clipped loss without any changes to its loss function.

### The teacher-student setup

Our specific offline GRPO setup uses a **teacher-student** paradigm:
- **Behavior policy (teacher)**: A larger, more capable model (Qwen2.5-Math-7B-Instruct) that generates high-quality rollouts
- **Target policy (student)**: A smaller model (Qwen2.5-0.5B-Instruct) that learns from the teacher's solutions

The student learns to assign higher probability to the teacher's correct solutions and lower probability to incorrect ones. This is a form of **knowledge distillation through RL** — the student doesn't just mimic the teacher's outputs, it learns which outputs are good through the reward signal.

### The KL penalty in offline GRPO

The KL penalty term uses a **reference policy**, which is distinct from the behavior policy:

```
KL penalty = β * KL(π_θ || π_ref)
```

- **π_β (behavior policy)**: The teacher model that generated the rollouts. Its logprobs appear in the IS ratio.
- **π_ref (reference policy)**: The student model's *initial weights* (before training). The KL penalty prevents the student from drifting too far from its starting point.

With LoRA, π_ref is obtained cheaply by disabling the adapter — the base model weights serve as the reference.

---

## 4. Our Pipeline: End-to-End

### Phase 1: Rollout Generation (`generate_rollouts.py`)

```
Input:  MATH training set (12,000 problems)
Model:  Qwen2.5-Math-7B-Instruct (teacher)
Output: rollouts_full.jsonl (48,000 completions with logprobs)
```

For each problem:
1. Format prompt with system message: "Please reason step by step, and put your final answer within \boxed{}."
2. Generate G=4 completions with temperature=0.6
3. Record for each completion:
   - The response text
   - The extracted `\boxed{...}` answer
   - The token IDs of the completion
   - The **per-token log-probabilities** from the teacher (these are π_β logprobs, critical for IS correction)

### Phase 2: Data Processing (`data.py`)

```
Input:  rollouts_full.jsonl
Output: HuggingFace Dataset + offline lookup dict
```

Steps:
1. **Load rollouts** — parse JSONL, truncate any completions with out-of-vocab tokens (teacher has 128 extra math tokens)
2. **Compute rewards** — binary: 2.0 if `\boxed{answer}` matches ground truth (via `math_verify`), 0.0 otherwise
3. **Compute group-relative advantages** — for each problem's G completions, normalize rewards to zero mean and unit variance
4. **Build training dataset** — sorted by (question_id, run_id) so consecutive rows form groups
5. **Build offline lookup** — dict keyed by (question_id, run_id) mapping to pre-computed behavior logprobs, completion IDs, and advantages

### Phase 3: Training (`train.py` + `trainer.py`)

```
Input:  Dataset + offline lookup + student model
Output: LoRA adapter weights
```

The student model (Qwen2.5-0.5B-Instruct) is trained with LoRA, using TRL's GRPOTrainer with our `OfflineGRPOTrainer` override.

Each training step:
1. Sample a batch of prompt groups from the dataset
2. Look up pre-computed completions, behavior logprobs, and advantages (no generation!)
3. Forward pass: compute current policy's log-probabilities for the teacher's completions
4. Compute IS ratio: `ρ = exp(π_θ logprobs - π_β logprobs)`
5. Compute reference logprobs by disabling LoRA adapter (for KL penalty)
6. TRL's `_compute_loss` applies the clipped objective with the IS-corrected ratio

### Phase 4: Evaluation (`evaluate.py`)

```
Input:  Trained LoRA adapter + base model
Output: MATH test accuracy
```

1. Merge LoRA weights into base model
2. Load merged model into vLLM for fast inference
3. Generate completions for 500 MATH test problems
4. Score with `math_verify` (symbolic equivalence checking)

---

## 5. Code Architecture

### File overview

```
generate_rollouts.py   Phase 1: Teacher generates completions with logprobs
data.py                Phase 2: Load rollouts, compute rewards/advantages
train.py               Phase 3: Entry point — wires model, data, trainer
trainer.py             Phase 3: OfflineGRPOTrainer (core algorithm)
evaluate.py            Phase 4: Merge LoRA + evaluate on test set
configs.py             Shared constants, prompts, LoRA config
run_full.sh            SLURM orchestration for all phases
```

### `generate_rollouts.py` — Behavior policy rollouts

This file runs the teacher model (behavior policy) over the MATH training set.

Key implementation details:

- **vLLM for fast inference** (line 43-51): Uses vLLM's batched generation engine for high throughput.
- **`logprobs=1`** (line 94): Requests top-1 log-probability at each token position. This gives us `log π_β(token | context)` — the behavior policy's per-token logprobs needed for the IS ratio.
- **`n=num_generations`** (line 93): Generates G completions per prompt in a single call.
- **Per-token logprob extraction** (lines 118-122): Flattens vLLM's logprob dicts into a simple float list aligned with token IDs.
- **Data-parallel sharding** (lines 63-65): Supports splitting the dataset across multiple GPUs for faster generation.

Output format (one JSON line per problem):
```json
{
  "question_id": 42,
  "original_problem": "Find the value of x...",
  "ground_truth_answer": "7",
  "runs": [
    {
      "run_id": 0,
      "response": "Step 1: ...",
      "boxed_answer": "7",
      "logprobs": [-0.12, -0.05, -0.31, ...],
      "completion_ids": [1234, 5678, 91011, ...]
    },
    ...  // G runs total
  ]
}
```

### `data.py` — Reward computation and advantage normalization

This file transforms raw rollouts into training-ready data.

**`load_rollouts()`** (lines 16-62):
- Parses the JSONL file into a flat list of per-completion records
- Handles vocab mismatch: if any token ID exceeds the student's vocab size, the completion is truncated at that point (lines 41-47). This prevents CUDA index-out-of-bounds errors during training.

**`compute_rewards_and_advantages()`** (lines 79-102):
- Computes binary reward: 2.0 for correct, 0.0 for incorrect (line 88)
- Uses `math_verify` for symbolic answer matching — handles equivalent forms like `\frac{1}{2}` vs `0.5` (lines 70-76)
- Groups completions by `question_id` and normalizes rewards within each group (lines 91-100):

```python
# For each group of G completions of the same prompt:
mean_r = sum(rewards) / len(rewards)
std_r = sqrt(sum((r - mean_r)^2) / len(rewards))
advantage = (reward - mean_r) / (std_r + eps)
```

This is the **group-relative advantage** — the core of GRPO. Examples:
- All 4 correct → advantages all 0.0 (no learning signal)
- 3 correct, 1 wrong → correct gets +0.58, wrong gets -1.73
- 1 correct, 3 wrong → correct gets +1.73, wrong gets -0.58
- All 4 wrong → advantages all 0.0 (no learning signal)

**`build_training_dataset()`** (lines 109-131):
- Creates an HF Dataset sorted by (question_id, run_id)
- This ordering is critical: TRL expects consecutive `num_generations` rows to belong to the same prompt group

**`build_offline_lookup()`** (lines 138-150):
- Creates a dict keyed by `(question_id, run_id)` for O(1) lookup during training
- Each entry contains: behavior logprobs, completion token IDs, pre-computed advantage, reward

### `trainer.py` — The core algorithm

This is the heart of the offline GRPO implementation. It subclasses TRL's `GRPOTrainer` and overrides exactly one method: `_generate_and_score_completions`.

In standard (online) GRPOTrainer, this method:
1. Takes a batch of prompts
2. **Generates** completions using the current policy
3. **Scores** them with reward functions
4. Returns completions, logprobs, advantages

Our `OfflineGRPOTrainer` replaces generation and scoring with **offline lookup**:

**Step 1: Tokenize prompts** (lines 51-68)
- Same as upstream — tokenize and left-pad the prompts

**Step 2: Look up offline completions** (lines 71-91)
- For each (question_id, run_id) in the batch, retrieve pre-computed data from `self._offline_data`
- This replaces the expensive generation step entirely

**Step 3: Pad completions and build tensors** (lines 93-128)
- Pad completion token IDs, masks, and behavior logprobs to uniform length
- `old_per_token_logps` is set to the **behavior policy's logprobs** (line 128)
- This is the key insight: TRL will compute `ratio = exp(current - old)`, which becomes `π_θ / π_β` — the IS correction

**Step 4: Compute reference logprobs for KL penalty** (lines 131-154)
- Only computed when `beta > 0`
- With LoRA: disables the adapter to get base model (π_ref) logprobs (line 148)
- Without LoRA: uses a separate reference model
- These logprobs are used in the KL divergence term: `KL = π_θ * log(π_θ / π_ref)`

**Step 5-7: Decode, log metrics, and return** (lines 156-206)
- Decode completions for logging
- Track reward, completion length, and other metrics
- Return the dict that TRL's `_compute_loss` expects

What happens after `_generate_and_score_completions` returns (in TRL's `_compute_loss`, which we do NOT override):

```python
# TRL computes current policy logprobs via forward pass
current_logps = get_logps(model, prompt_ids, completion_ids)

# IS ratio (becomes π_θ / π_β because old_logps = π_β logprobs)
ratio = exp(current_logps - old_per_token_logps)

# Clipped surrogate objective
surr1 = ratio * advantages
surr2 = clip(ratio, 1-ε, 1+ε) * advantages
policy_loss = -min(surr1, surr2)

# KL penalty
kl = exp(ref_logps - current_logps) - (ref_logps - current_logps) - 1
total_loss = policy_loss + beta * kl
```

### `train.py` — Training entry point

Wires everything together:
1. Loads rollouts and computes rewards/advantages (lines 63-77)
2. Loads the student model with LoRA (lines 80-102)
3. Configures GRPOConfig (lines 110-133) — note this reuses TRL's config class
4. Creates `OfflineGRPOTrainer` with the offline data lookup (lines 140-148)
5. Calls `trainer.train()` — from here, TRL handles the training loop

The dummy reward function (line 137) is required by TRL's API but never called — all rewards are pre-computed.

### `configs.py` — Shared configuration

- **`SYSTEM_PROMPT`**: "Please reason step by step, and put your final answer within \boxed{}." — used in both rollout generation and evaluation
- **`extract_boxed_answer()`**: Parses the last `\boxed{...}` from generated text, handling nested braces
- **`DEFAULT_LORA_CONFIG`**: r=16, alpha=64, targeting all attention + MLP projections

### `evaluate.py` — Test-time evaluation

1. **LoRA merge** (lines 39-53): Loads base model + LoRA adapter, merges weights, saves full model. This is necessary because vLLM cannot load LoRA adapters directly for offline inference.
2. **vLLM generation** (lines 67-99): Generates one completion per test problem at temperature=0.6
3. **Answer verification** (lines 108-113): Extracts `\boxed{...}`, parses with `math_verify`, and checks symbolic equivalence against ground truth

---

## 6. Mathematical Details

### The full offline GRPO objective

For a batch of prompts {q_j} with G completions each {o_{j,1}, ..., o_{j,G}} generated by behavior policy π_β:

```
L(θ) = (1/|B|) Σ_j Σ_i [
    min(
        ρ_{j,i}(θ) * A_{j,i},
        clip(ρ_{j,i}(θ), 1-ε, 1+ε) * A_{j,i}
    )
    - β * KL_token(π_θ || π_ref)[j,i]
]
```

where:

**Importance sampling ratio** (per-token, then aggregated):
```
ρ_{j,i}(θ) = Π_t [ π_θ(o_{j,i,t} | q_j, o_{j,i,<t}) / π_β(o_{j,i,t} | q_j, o_{j,i,<t}) ]
            = exp( Σ_t [ log π_θ(o_{j,i,t} | ...) - log π_β(o_{j,i,t} | ...) ] )
```

In practice, the ratio is computed in log-space for numerical stability.

**Group-relative advantage**:
```
A_{j,i} = (r_{j,i} - μ_j) / (σ_j + ε)

where μ_j = (1/G) Σ_i r_{j,i}
      σ_j = sqrt( (1/G) Σ_i (r_{j,i} - μ_j)^2 )
```

**Per-token KL divergence** (approximated):
```
KL_token ≈ exp(log π_ref - log π_θ) - (log π_ref - log π_θ) - 1
```

This is the Schulman KL estimator, which is always non-negative and equals zero when π_θ = π_ref.

### Why clipping matters more in offline GRPO

In online GRPO, the ratio starts near 1.0 (policy hasn't changed much since the last generation). In offline GRPO, the ratio can be far from 1.0 from the start because π_θ (student) and π_β (teacher) are different models. The clip bounds [1-ε, 1+ε] prevent the loss from being dominated by a few completions with extreme ratios.

Our implementation uses `max_grad_norm=0.1` for aggressive gradient clipping as an additional safety measure.

### The reward structure

We use a binary reward:
```
r(o) = 2.0  if math_verify(extract_boxed(o), ground_truth) is True
     = 0.0  otherwise
```

The scale (0 and 2 vs. 0 and 1) affects the magnitude of advantages and thus gradients. With G=4 completions, the possible advantage distributions are:

| Correct/Total | Correct Advantage | Wrong Advantage | Gradient Signal |
|:---:|:---:|:---:|:---|
| 4/4 | 0.0 | — | None (all same reward) |
| 3/4 | +0.58 | -1.73 | Moderate |
| 2/4 | +1.0 | -1.0 | Strongest (balanced) |
| 1/4 | +1.73 | -0.58 | Moderate |
| 0/4 | — | 0.0 | None (all same reward) |

Problems with 2/4 correct provide the strongest gradient signal. Problems where the teacher always succeeds or always fails provide zero signal — this is why teacher accuracy around 50% would be ideal for learning, and why 70.9% teacher accuracy means many all-correct groups contribute nothing.

---

## Summary

| Concept | Online GRPO | Our Offline GRPO |
|---------|------------|-----------------|
| Completions from | Current policy π_θ | Teacher π_β (pre-computed) |
| IS ratio | π_θ / π_old ≈ 1 | π_θ / π_β (can be far from 1) |
| Advantages | Computed on-the-fly | Pre-computed from teacher rewards |
| KL reference | Frozen copy of π_θ | Base student model (LoRA disabled) |
| Generation cost | Every training step | Once (Phase 1 only) |
| Can exceed teacher? | Yes (explores new solutions) | No (limited to teacher's outputs) |
| Code change | — | Override `_generate_and_score_completions` |
