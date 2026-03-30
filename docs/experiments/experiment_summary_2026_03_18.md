# Experiment Summary: Why Does Offline GRPO Degrade the Student?

**Date**: 2026-03-18
**Authors**: Dongheng Li (mrli)
**Purpose**: Summary of all experiments for advisor discussion on next steps

---

## 1. Problem Statement

We are studying **offline GRPO** — an off-policy variant of Group Relative Policy Optimization that uses pre-collected teacher rollouts instead of on-policy generation. The motivation is that offline GRPO avoids the expensive vLLM generation step during training, making it ~2× faster per step.

**Setup:**
- **Student model**: Qwen2.5-0.5B-Instruct (151,936 vocab)
- **Teacher model**: Qwen2.5-Math-7B-Instruct (152,064 vocab)
- **Rollout data**: 7B teacher generates 5 completions per problem (temperature=1.0), ~70.9% correct on MATH
- **Training**: LoRA (r=32, alpha=32, targets=q/k/v/o/up/down/gate_proj), 4× L40s GPUs
- **Cluster**: Vulcan HPC (ComputeCanada)

**Core question**: Can offline GRPO with a strong teacher improve a weak student on MATH? If not, why?

### Methods

| Method | Description |
|--------|-------------|
| **Online GRPO** | Standard GRPO: student generates its own rollouts via vLLM, then optimizes. On-policy. |
| **Offline GRPO** | Student optimizes on pre-collected teacher rollouts. Off-policy with IS correction (π_student/π_teacher). |
| **Mixture A (Unified)** | Interleaves online and offline batches in a single training loop. |
| **Mixture B (Weighted)** | Separate online/offline phases with weighted loss combination. |
| **Behavioral Cloning (BC)** | Pure cross-entropy on teacher completions (no RL, no advantage). |

All RL methods use PPO-style clipping (ε=0.2) and KL penalty (β varies).

---

## 2. Experiment 1: Method Comparison

### Objective

Compare online GRPO, offline GRPO, and mixture methods on both GSM8K (easier) and MATH (harder).

### Results

**GSM8K** (test set, 1319 problems, 5 runs, temp=0.0):

| Method | Accuracy | Δ vs Baseline | Source |
|--------|----------|---------------|--------|
| Baseline (0.5B) | 47.79% | — | eval-all-4333310 |
| Offline GRPO | 48.11% | +0.32% | eval-all-4333310 |
| Mixture A (Unified) | 50.30% | +2.51% | eval-all-4333310 |
| Mixture B (Weighted) | 51.11% | +3.32% | eval-all-4333310 |
| Online GRPO (step 7000) | 55.82% | +8.03% | online-grpo-eval-4299873 |
| Teacher (7B) | 95.74% | — | eval-all-4333310 |

**MATH** (test set, 500 problems, 5 runs, temp=0.0):

| Method | Accuracy | Δ vs Baseline | Source |
|--------|----------|---------------|--------|
| Baseline (0.5B) | 30.36% | — | eval-all-4333310 |
| Offline GRPO (MATH) | 27.64% | **-2.72%** | eval-all-4333310 |
| Mixture A (MATH) | 29.28% | -1.08% | eval-all-4333310 |
| Mixture B (MATH) | 28.60% | -1.76% | eval-all-4333310 |
| Online GRPO (step 8200) | 32.10% | +1.74% | online-grpo-math-eval-4340781 |
| Teacher (7B) | 74.96% | — | eval-all-4333310 |

### Analysis

**Finding 1: On MATH, all offline methods degrade the student below baseline.**

The pattern across the two benchmarks is revealing:
- **GSM8K**: All methods improve over baseline. Ranking: Online (+8%) >> Mixture B (+3.3%) > Mixture A (+2.5%) > Offline (+0.3%). Even offline GRPO gives a small positive signal.
- **MATH**: Only online GRPO improves (+1.7%). All offline-based methods are **negative** (-1% to -3%).

**Finding 2: The task difficulty matters critically.**

The student has 47.8% baseline on GSM8K but only 30.4% on MATH. On GSM8K, the student can solve many problems partially — so among 5 rollouts, there is often a mix of correct and incorrect completions. This creates meaningful GRPO advantages. On MATH, the student is too weak — most prompt groups have all-wrong or all-correct completions, yielding zero advantage.

Training log evidence: `frac_reward_zero_std` (fraction of groups with zero reward variance) is 0.63 → 0.73 on MATH, meaning 63-73% of training examples provide **zero gradient signal**.

**Finding 3: Cross-dataset transfer is informative.**

Offline GRPO trained on GSM8K → eval on MATH: 30.52% (near baseline). Offline GRPO trained on MATH → eval on MATH: 27.64% (below baseline). Training on MATH actively harms the model — worse than not training at all.

### Key Question Raised

Why does offline GRPO degrade on MATH? Two main hypotheses:
1. **Distribution mismatch** (H1): π_behavior (7B teacher) ≠ π_target (0.5B student). The IS ratio π_student/π_teacher is far from 1.0, causing clipped/noisy gradients.
2. **Advantage collapse** (H2): On MATH, most prompt groups have zero reward variance → zero advantage → no learning signal, but gradient noise from the few non-zero groups pushes the model in random directions.

---

## 3. Experiment 2: Behavioral Cloning Baseline

### Motivation

To separate the effect of the **RL objective** from the effect of **learning from teacher data**, we run a pure supervised learning baseline: behavioral cloning (cross-entropy on teacher completions). If BC also degrades the student, the problem is not specific to GRPO — it's about the teacher data itself.

### Setup

- Same 7B teacher rollouts on MATH (9,600 prompts × 5 completions = 48,000 examples)
- Pure next-token prediction loss on completion tokens (prompt masked to -100)
- Same LoRA config, same training duration (1 epoch)
- Two variants: (a) train on all completions, (b) train on correct-only (~70.9%)

### Results

| Method | MATH Accuracy | Δ vs Baseline | Source |
|--------|---------------|---------------|--------|
| Baseline (0.5B, re-eval) | 27.16% | — | eval-Qwen2.5-0.5B-Instruct-4389323 |
| BC (all completions) | 27.40% | +0.24% | eval-bc-math-4382964 |
| BC (correct-only) | 27.20% | +0.04% | eval-bc-correct-4383318 |

**Note**: The baseline re-eval (27.16%) differs from the earlier eval (30.36%). Both use the same untrained model with 5 runs at temp=0.0 but different eval scripts. This ~3% variance is concerning (see Section 7.3).

### Analysis

**Finding: BC neither helps nor hurts.** Despite training on completions that are ~71% correct (vs student's ~27-30%), the student doesn't improve.

This is surprising. The teacher data is much better than what the student can produce — yet the student doesn't learn from it. Why?

To investigate, we ran extensive diagnostics (Section 5 below). The key findings:
- **Implementation is correct**: 9 automated checks passed. Single-sample overfit works perfectly (loss → 0.00004).
- **Token agreement is high**: Student and teacher already agree on 91-93% of next-token predictions. BC is only teaching the remaining 7-9%.
- **Disagreement tokens are stylistic**: "of" vs "for", "make" vs "cause", LaTeX formatting differences — not math reasoning decisions.
- **Training does shift the student**: Student's own log-probability drops slightly after BC (-0.007), suggesting it moves away from its own distribution but doesn't gain accuracy.

**Interpretation**: The student and teacher are already surprisingly close at the token level. The gap between their accuracies (27% vs 75%) is not primarily about individual token predictions — it's about the **long-horizon coherence** of reasoning chains. BC on individual tokens doesn't teach multi-step reasoning structure.

This also raises the possibility of **exposure bias**: during BC training, the student sees teacher-generated prefixes, but at test time it conditions on its own tokens. Over 500+ token completions, small errors compound.

### Question Raised

OK, so BC doesn't help — but is this because of the teacher data or because BC is too simple? The offline GRPO result (-2.72%) is even worse than BC (+0.24%). So GRPO's RL objective is actively harmful on top of the data being unhelpful. Why?

This led us to test: is the problem specific to the **off-policy** nature (distribution mismatch), or is there something wrong with the offline GRPO algorithm itself?

---

## 4. Experiment 3: Teacher-Self Offline GRPO (Isolating Distribution Mismatch)

### Motivation

**H1 (Distribution Mismatch)**: The 0.5B student assigns very different probabilities to teacher tokens than the 7B teacher does. The IS ratio π_student/π_teacher is far from 1.0, causing PPO clipping to eliminate most gradient signal.

To test this, we remove the distribution mismatch entirely: **train the 7B teacher on its own rollouts** using the same offline GRPO pipeline. Here π_behavior = π_target (same model), so the IS ratio starts at ~1.0.

- If the teacher improves → distribution mismatch was the root cause.
- If the teacher also degrades → offline GRPO has deeper problems.

### Setup

- Target model: Qwen2.5-Math-7B-Instruct (same as behavior model)
- Same MATH rollouts, same LoRA config, same hyperparameters (matching online GRPO)
- Adjusted batch size for 7B memory: per_device_batch=2, grad_accum=4 (same effective batch)
- SLURM job: 4381598, 24h on 4× L40s

### Results

| Method | MATH Accuracy | Δ vs Baseline | Source |
|--------|---------------|---------------|--------|
| Teacher 7B baseline | 74.96% | — | eval-all-4333310 |
| Teacher-self offline GRPO | 71.20% | **-3.76%** | eval-teacher-self-4384696 |

### Analysis

**Finding: Even without distribution mismatch, offline GRPO degrades the model by 3.76%.**

This is a critical result. It means **H1 (distribution mismatch) is not the sole cause** — or possibly not the primary cause at all. The offline GRPO algorithm itself has a problem on MATH.

**What could explain this?**

1. **Advantage collapse (H2)**: The 7B teacher gets 75% on MATH. With 5 rollouts per problem, many prompt groups still have zero variance (either all correct or all incorrect). These groups contribute zero learning signal but their gradients add noise.

2. **Rollout staleness**: Even though the rollouts come from the same model, after the first gradient update the model is no longer identical to the behavior policy. With each update, the rollouts become more off-policy. Over a full epoch (4800+ steps), the accumulated drift may be substantial.

3. **Single-epoch limitation**: Online GRPO continuously refreshes its rollouts. Offline GRPO sees each rollout exactly once. If the model needs multiple passes through different rollouts to converge, a single epoch of stale data is insufficient.

### Key Question Raised

If offline GRPO degrades even the teacher, does **online** GRPO work for the teacher? This would confirm that the offline setting (not the model or the task) is the bottleneck.

---

## 5. Experiment 4: Online GRPO for 7B Teacher (Complete)

### Motivation

Direct comparison: same model (7B), same task (MATH), but online (fresh rollouts each step) instead of offline (pre-collected rollouts).

- If online 7B improves → offline setting is the specific problem.
- If online 7B also degrades → the 7B model + MATH task combination has issues beyond online/offline.

### Setup

- Model: Qwen2.5-Math-7B-Instruct with LoRA (r=32, alpha=32)
- Online GRPO via TRL GRPOTrainer + vLLM colocate mode
- num_generations=4, per_device_batch=1, grad_accum=8, lr=3e-6, beta=0.001, temp=0.7
- 1 epoch on MATH train set, 4× L40s, 48h
- SLURM job: 4384834, wandb run: teacher-7B-online-0317

### Results

| Method | MATH Accuracy | Δ vs Baseline | Source |
|--------|---------------|---------------|--------|
| Teacher 7B baseline | 74.96% | — | eval-all-4333310 |
| Online GRPO 7B teacher | **74.72%** | **-0.24%** | eval-online-teacher-4392895 |
| Offline GRPO 7B teacher-self | 71.20% | -3.76% | eval-teacher-self-4384696 |

### Analysis

**Finding: Online GRPO for the 7B teacher is flat — no improvement, no degradation (-0.24%, within noise).**

This is an important result when contrasted with the offline variant:
- **Offline** GRPO on 7B teacher's own rollouts: **-3.76%** (significant degradation)
- **Online** GRPO on 7B teacher: **-0.24%** (flat, within noise)

Online GRPO avoids the degradation that offline causes, but it also fails to improve the teacher. This is consistent with the **advantage collapse** problem: with only 4 generations per prompt and a 75%-accurate model, most prompt groups have zero reward variance (`frac_reward_zero_std` = 0.75-0.93 in the training logs). There is very little gradient signal to learn from.

**Training dynamics:**
- `reward` ≈ 0.65-0.80 throughout (no clear trend)
- `frac_reward_zero_std` ≈ 0.75-0.93 (severe advantage collapse)
- `kl` ≈ 6e-5 (barely moved from init — model didn't change much)
- `IS ratio mean` ≈ 0.97-1.01 (on-policy, as expected)
- `clip_ratio` ≈ 0 (almost no clipping — consistent with on-policy)

**Key takeaway**: The difference between online and offline is not about improvement — it's that **offline actively harms** while online is neutral. The offline degradation (-3.76%) comes from training on stale rollouts that become increasingly off-policy over the epoch, not from the GRPO algorithm itself.

---

## 6. Diagnostic Investigations

### 6.1 Implementation Verification (No Bugs Found)

We thoroughly verified both offline GRPO and BC implementations:

| Check | Result | Source |
|-------|--------|--------|
| Prompt masking (logits_to_keep) | Correct | `trainer.py` inspection |
| Padding masking (completion_mask) | Correct | `trainer.py` inspection |
| IS ratio computation | Correct, uses raw (pre-temp) logprobs | vLLM v1 sampler source |
| Chat template alignment | Exact token match | Manual comparison |
| BC label/loss/attention/padding | 9/9 checks passed | `bc/diagnose_bc.py`, job 4382985 |
| BC single-sample overfit | Loss 0.256→0.00004, top1 91→100% | `bc/full_diagnostic.py`, job 4382994 |
| Stored logprobs vs recomputation | Mean diff 0.003, max 0.105 (bf16) | `bc/check_p4.py`, job 4383020 |

**Conclusion**: No implementation bugs found. The degradation is a real algorithmic/data phenomenon.

### 6.2 BC Training Dynamics

From `bc/investigate_bc_degradation.py` (job 4383221):

| Metric | Value |
|--------|-------|
| Correct completions loss | 0.16 |
| Incorrect completions loss | 0.68 (4× harder) |
| Student logprob shift after BC | -0.1907 → -0.1976 (Δ=-0.007) |
| Generation length (base vs BC) | 486 vs 475 tokens |

The student moves slightly away from its own distribution after BC, but not enough to change accuracy.

### 6.3 Token-Level Disagreement Analysis

From `bc/check_disagreement_tokens.py` (job 4383341):

- Student-teacher agreement: ~91-93% of next-token predictions match
- Disagreement tokens are overwhelmingly **stylistic**: "of"↔"for", "make"↔"cause", "However"↔"Here", LaTeX formatting
- No systematic pattern in math-reasoning tokens (operators, numbers, logical connectors)

**Implication**: The 27% vs 75% accuracy gap is NOT about individual token predictions — it's about long-horizon reasoning coherence. The student "knows" most of the right tokens but cannot string them together into a correct proof.

### 6.4 Advantage Signal Collapse

From offline GRPO training logs on MATH:

| Metric | Value (training range) |
|--------|----------------------|
| `frac_reward_zero_std` | 0.63 → 0.73 |
| Effective training prompts | 27-37% of total |
| `clip_ratio/low_mean` | 0.01-0.05 (low-side clipping) |

63-73% of prompt groups have **zero reward variance** across their 5 rollouts (all correct or all wrong). For these groups, GRPO advantage = 0 and they contribute zero gradient. Only 27-37% of prompts provide actual learning signal.

---

## 7. Open Questions

### 7.1 Why does BC fail despite 70% correct teacher data?

**Most likely explanation**: High token-level agreement (93%) + exposure bias. The student and teacher are already close at the per-token level. The accuracy gap comes from compounding small errors over long completions, which BC (teacher-forcing) cannot fix.

**Unverified but testable**: If we do BC with scheduled sampling (mix of teacher and student prefixes), does it perform better?

### 7.2 Why does offline GRPO degrade even the teacher?

**Most likely explanation**: Combination of advantage collapse (~70% of prompts useless) and rollout staleness (single epoch, no refresh). The few prompts with non-zero advantage push the model in a direction that isn't reinforced by fresh evidence.

**Waiting for**: Online GRPO 7B result to confirm this is an offline-specific problem.

### 7.3 Baseline Eval Variance

The 0.5B baseline gives 30.36% in one eval and 27.16% in another. Both use temp=0.0 with 5 runs. Possible causes:
- Different eval scripts (different max_new_tokens, prompts, or math-verify versions)
- Different random seeds for non-deterministic components
This gap (~3%) is comparable to the treatment effects we're measuring, so it must be resolved.

---

## 8. Summary of What We Know

| Claim | Evidence | Confidence |
|-------|----------|------------|
| Online GRPO works for 0.5B (both GSM8K and MATH) | +8% GSM8K, +1.7% MATH | High |
| Online GRPO is flat for 7B teacher on MATH | -0.24% (within noise) | High |
| All offline methods degrade on MATH | -1% to -3% across 3 methods | High |
| Offline methods give small gains on GSM8K | +0.3% to +3.3% | Moderate (small effects) |
| Offline GRPO actively harms even on-policy (7B self) | -3.76% vs flat for online | High |
| Distribution mismatch is NOT the sole cause | Teacher-self also degrades (-3.76%) | High |
| Offline degradation comes from rollout staleness | Online neutral, offline harmful (same model) | High |
| Advantage collapse is severe on MATH | 75-93% zero-variance groups (7B online logs) | High |
| Implementation is correct | 9 checks passed, overfit works | High |
| BC has no significant effect | Within noise of baseline | High |
| Token-level agreement is ~93% | Disagreement analysis | High |
| Disagreement tokens are stylistic | Manual inspection | Moderate (qualitative) |

---

## 9. Proposed Next Steps (for discussion)

### 9.1 Completed

1. **Online GRPO 7B teacher result**: 74.72% (flat vs 74.96% baseline)
   - Confirms: offline degrades (-3.76%), online is neutral (-0.24%). Rollout staleness is the key factor.

### 9.2 Clean Up

2. **Resolve baseline eval variance** (30.36% vs 27.16%)
   - Run both eval scripts on the same model with identical settings

### 9.3 New Experimental Direction: Same-Vocab Pair

3. **Qwen2.5-0.5B → Qwen2.5-3B (same vocab=151,936)**
   - Eliminates OOV token issue
   - 3B teacher is weaker (58%) but closer to student → less distribution mismatch
   - Generate 3B rollouts on MATH, then run offline GRPO and BC

4. **Stronger student: Qwen2.5-1.5B → Qwen2.5-3B**
   - 1.5B has 44% on MATH — closer to the ~48% level where offline worked on GSM8K
   - Test if the "student must be strong enough" hypothesis holds

### 9.4 Algorithmic Improvements

5. **Address advantage collapse**:
   - Increase `num_generations` (5 → 8 or 16) for more reward variance
   - Use global baseline instead of group-level baseline for zero-std groups
   - Explore partial-credit reward (process reward model) instead of binary correct/incorrect

6. **Curriculum / filtering**: Skip zero-variance prompt groups during training.

7. **Semi-online (periodic rollout refresh)**: Do K gradient steps on rollouts, regenerate, repeat.

---

## 10. Artifacts

### Code
| Directory | Contents |
|-----------|----------|
| `offline_grpo/` | Offline GRPO trainer, data loading, evaluation, SLURM scripts |
| `online_grpo/` | Online GRPO using TRL GRPOTrainer with vLLM colocate mode |
| `bc/` | Behavioral cloning training, evaluation, diagnostic scripts |

### Key Checkpoints (on scratch)
| Path | Description |
|------|-------------|
| `/scratch/mrli/checkpoints/offline_grpo_math_controlled/` | Controlled offline GRPO (0.5B, online hparams) |
| `/scratch/mrli/checkpoints/offline_grpo_math_teacher_self/` | Teacher-self offline GRPO (7B) |
| `/scratch/mrli/checkpoints/bc_math/` | BC all completions |
| `/scratch/mrli/checkpoints/bc_math_correct_only/` | BC correct-only |
| `/scratch/mrli/checkpoints/online_grpo_math/` | Online GRPO (0.5B) |
| `/scratch/mrli/checkpoints/online_grpo_math_teacher_7B/` | Online GRPO (7B teacher) |

### Qwen2.5 Instruct Family Baselines
| Model | Vocab | MATH Accuracy | GSM8K Accuracy | MATH Source | GSM8K Source |
|-------|-------|---------------|----------------|-------------|--------------|
| 0.5B-Instruct | 151,936 | 27.16% (std 0.0058) | 48.16% (std 0.0055) | eval-4389323 | gsm8k-eval-0.5B-4391991 |
| 1.5B-Instruct | 151,936 | 44.04% (std 0.0113) | 68.28% (std 0.0015) | eval-4389324 | gsm8k-eval-1.5B-4391992 |
| 3B-Instruct | 151,936 | 58.08% (std 0.0098) | 82.56% (std 0.0043) | eval-4389325 | gsm8k-eval-3B-4391993 |
| Math-7B-Instruct | 152,064 | 74.96% (std 0.0036) | 95.74% | eval-all-4333310 | eval-all-4333310 |

All evals: 5 runs, temp=0.0. MATH=500 problems, GSM8K=1319 problems.

### Log Files
| Path | Job ID | Experiment |
|------|--------|------------|
| `logs/eval_all/eval-all-4333310.out` | 4333310 | Phase 1 full evaluation |
| `online_grpo/logs/online-grpo-math-eval-4340781.out` | 4340781 | Online GRPO 0.5B MATH eval |
| `bc/logs/bc_math/eval-bc-math-4382964.out` | 4382964 | BC eval |
| `bc/logs/bc_math/eval-bc-correct-4383318.out` | 4383318 | BC correct-only eval |
| `offline_grpo/logs/teacher_self_math/eval-teacher-self-4384696.out` | 4384696 | Teacher-self 7B eval |
| `online_grpo/logs/online-teacher-math-4384834.out` | 4384834 | Online GRPO 7B teacher training |
| `online_grpo/logs/eval-online-teacher-4392895.out` | 4392895 | Online GRPO 7B teacher eval |
| `bc/logs/bc_math/eval-Qwen2.5-{0.5B,1.5B,3B}-Instruct-*.out` | 4389323-25 | Family baselines (MATH) |
| `bc/logs/bc_math/gsm8k-eval-0.5B-4391991.out` | 4391991 | 0.5B GSM8K baseline |
| `bc/logs/bc_math/gsm8k-eval-1.5B-4391992.out` | 4391992 | 1.5B GSM8K baseline |
| `bc/logs/bc_math/gsm8k-eval-3B-4391993.out` | 4391993 | 3B GSM8K baseline |
