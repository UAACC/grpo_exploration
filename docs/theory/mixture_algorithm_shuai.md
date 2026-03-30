# Hybrid Online–Offline GRPO for LLM Post-Training

## Overview

This document describes a **hybrid online–offline GRPO (Group Relative Policy Optimization)** algorithm for post-training large language models such as Qwen on reasoning datasets (e.g., GSM8K). The algorithm combines:

- **Online reinforcement learning** from student-generated rollouts
- **Offline guidance** from a stronger teacher model
- **Stability regularization** via a lagged reference policy

The method is designed to retain the exploration benefits of RL while incorporating reliable signals from a precomputed teacher trajectory.

---

# Algorithm Summary

For each prompt in a training batch:

1. **Online sampling**
   - Sample a group of completions from the previous student policy.

2. **Reward computation**
   - Assign a binary reward to each completion:
     - `1` if the final answer is correct
     - `0` otherwise.

3. **Group-relative normalization**
   - Compute the mean reward across the group.
   - Compute the scale from centered rewards.

4. **Online policy gradient**
   - Compute normalized advantages.
   - Apply a clipped PPO/GRPO surrogate loss.

5. **Offline teacher guidance**
   - Retrieve the teacher rollout for the same prompt.
   - Normalize the teacher reward using the **online group statistics**.
   - Apply a clipped off-policy PPO surrogate.

6. **KL regularization**
   - Compute KL divergence against a lagged reference model.

7. **Optimization**
   - Combine online loss, offline loss, and KL penalty.
   - Perform one gradient update.

---

# Training Signals

The algorithm combines three learning signals:

| Signal | Source | Purpose |
|------|------|------|
| Online exploration | Student rollouts | Discover new reasoning strategies |
| Offline supervision | Teacher rollout | Provide strong reference answers |
| KL regularization | Reference model | Prevent policy collapse |

---

# Mathematical Formulation

## Online Rollouts

For prompt \(x_b\), sample a group of size \(G\):

\[
a_{b,1}, \dots, a_{b,G} \sim \pi_{old}(\cdot | x_b)
\]

Binary rewards:

\[
r_{b,g} \in \{0,1\}
\]

---

## Group Baseline

Mean reward:

\[
\mu_b = \frac{1}{G} \sum_{g=1}^{G} r_{b,g}
\]

Centered reward scale:

\[
s_b =
\sqrt{
\frac{1}{G}
\sum_{g=1}^{G}
(r_{b,g}-\mu_b)^2
}
\]

---

## Online Advantages

\[
A_{b,g}^{on}
=
\frac{r_{b,g}-\mu_b}{\max(s_b, c)}
\]

Where \(c\) is a small stability constant.

---

## Offline Advantage

Teacher rollout reward \(r_b^T\).

Normalized using the same baseline and scale:

\[
A_b^{off}
=
\frac{r_b^T-\mu_b}{\max(s_b, c)}
\]

This ties the teacher signal to the student’s current performance.

---

## Importance Ratios

### Online

\[
\rho_{b,g}^{on}
=
\frac{\pi_\theta(a_{b,g}|x_b)}
{\pi_{old}(a_{b,g}|x_b)}
\]

### Offline

\[
\rho_b^{off}
=
\frac{\pi_\theta(a_b^T|x_b)}
{\pi_T(a_b^T|x_b)}
\]

---

## PPO Clipped Surrogate

Online:

\[
\ell_{b,g}^{on}
=
\min
(
\rho_{b,g}^{on} A_{b,g}^{on},
\text{clip}(\rho_{b,g}^{on},1-\epsilon,1+\epsilon) A_{b,g}^{on}
)
\]

Offline:

\[
\ell_b^{off}
=
\min
(
\rho_b^{off} A_b^{off},
\text{clip}(\rho_b^{off},1-\epsilon,1+\epsilon) A_b^{off}
)
\]

---

## Total Objective

Surrogate loss:

\[
L_{sur}
=
-
\frac{1}{m}
\sum_{b=1}^{m}
\frac{1}{G}
\sum_{g=1}^{G}
\ell_{b,g}^{on}
-
\lambda
\frac{1}{m}
\sum_{b=1}^{m}
\ell_b^{off}
\]

Add KL regularization:

\[
L_{total}
=
L_{sur}
+
\beta L_{KL}
\]

Where \(L_{KL}\) is computed against a lagged reference policy.

---

# Training Procedure

## Offline Precomputation

Before training:

For every prompt in the dataset:

1. Generate one teacher rollout.
2. Compute its reward.
3. Store:
   - prompt
   - teacher completion
   - reward
   - teacher log probabilities.

---

## Training Loop
sample minibatch prompts

# Online rollouts
generate G student completions per prompt

compute rewards

compute group mean and scale

compute online advantages

retrieve teacher rollout

compute offline advantage using online stats

evaluate student log probabilities

compute importance ratios

compute clipped PPO losses

compute shared KL

combine losses

update student parameters

refresh old policy

periodically update reference policy

---

# Implementation Guidance

## Store Teacher Log Probabilities

Offline importance ratios require:

\[
\log \pi_T(a^T | x)
\]

Prefer storing **token-level log probabilities** to allow later analysis.

---

## Use Sequence Log Probabilities

Compute completion probability as:

\[
\log \pi(a|x)
=
\sum_t
\log \pi(a_t|x,a_{<t})
\]

Then compute ratios using sequence log probabilities.

---

## Handle Zero Variance

Binary rewards often produce zero variance.

Use: denominator = max(scale, epsilon)

Example: epsilon = 1e-4

---

## Hyperparameters

Typical starting values:

| Parameter | Typical value |
|------|------|
| Online group size | 4–8 |
| PPO clip ε | 0.2 |
| Offline weight λ | 0.1–0.5 |
| KL coefficient β | small (tuned) |
| Reference sync steps | 50–200 |

---

# Monitoring Metrics

Track these during training:

### Online metrics

- mean reward
- fraction of correct rollouts
- online advantage magnitude
- PPO clip fraction

### Offline metrics

- teacher reward
- offline advantage
- offline ratio

### Stability metrics

- KL to reference
- KL to old policy
- gradient norms
- model perplexity

---

# Failure Modes

## Teacher Dominance

Large λ causes the model to imitate the teacher rather than explore.

Mitigation:

- reduce λ
- increase exploration.

---

## No Online Reward Variance

If all rollouts are incorrect, advantages collapse.

Mitigation:

- increase sampling temperature
- increase group size
- start from stronger SFT checkpoint.

---

## Ratio Explosion

Off-policy ratio can become large.

Mitigation:

- PPO clipping
- KL regularization.

---

# Recommended Ablations

To validate the algorithm, compare:

1. Online-only GRPO
2. Offline-only teacher training
3. Hybrid algorithm
4. Hybrid without KL
5. Hybrid with varying λ

This identifies whether improvement comes from exploration, teacher supervision, or their combination.

---

# Key Design Insight

The most important property of the algorithm:

> The offline teacher sample is normalized using the **online group baseline**, linking the teacher signal to the student’s current performance.

This prevents teacher supervision from overpowering reinforcement learning.

---

# One-Sentence Description

Hybrid Online–Offline GRPO trains a student model using grouped PPO updates on its own rollouts while incorporating a clipped off-policy teacher trajectory normalized by the student’s group statistics, with KL regularization against a lagged reference model.

---