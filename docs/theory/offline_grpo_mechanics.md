# Offline GRPO Mechanics — How Everything Connects

A detailed walkthrough of how rewards, advantages, ratios, and losses work in our offline GRPO pipeline, and why scaling decisions matter.

---

## 1. Reward Computation

Rewards are computed **once** during rollout generation. The student is not involved.

```
1. Teacher generates: "Factor (x-2)(x-3), answer is \boxed{x=2,3}"
2. Extract boxed answer: "x=2,3"
3. Compare to ground truth: "x=2,3" → match → reward = 2.0

If no match → reward = 0.0
```

No partial credit. No style points. Just right or wrong. The reward is saved to `rollouts.jsonl` and never recomputed.

---

## 2. Advantage: Why GRPO Needs Contrast

GRPO computes advantages **relative to each group** of solutions for the same problem:

```
advantage = (reward - group_mean) / group_std
```

### Example: Mixed results (learning happens)

Problem: "Solve x² - 5x + 6 = 0", teacher gets 3 out of 4 right:

| Solution | Answer | Reward | Advantage |
|---|---|---|---|
| A: "Factor (x-2)(x-3) → x=2,3" | correct | 2.0 | +0.58 |
| B: "Quadratic formula → x=2,3" | correct | 2.0 | +0.58 |
| C: "Factor (x-2)(x-3) → x=2,3" | correct | 2.0 | +0.58 |
| D: "Factor (x-1)(x-6) → x=1,6" | wrong | 0.0 | -1.73 |

```
group_mean = 1.5, group_std = 0.87
A: (2.0 - 1.5) / 0.87 = +0.58  → "be MORE like this"
D: (0.0 - 1.5) / 0.87 = -1.73  → "be LESS like this"
```

The student gets a clear signal: increase probability of correct reasoning, decrease probability of wrong reasoning.

### Example: All correct (zero signal)

Problem: "What is 2+3?", teacher gets all 4 right:

| Solution | Reward | Advantage |
|---|---|---|
| A: "2+3 = 5, so \boxed{5}" | 2.0 | 0 |
| B: "Adding 2 and 3 gives \boxed{5}" | 2.0 | 0 |
| C: "5 is the answer \boxed{5}" | 2.0 | 0 |
| D: "2+3 = \boxed{5}" | 2.0 | 0 |

```
group_mean = 2.0, group_std = 0.0
advantage = (2.0 - 2.0) / 0.0 → set to 0
```

All solutions scored the same. GRPO only learns from **differences within a group**. No contrast → no signal → zero gradient.

### Example: All wrong (zero signal)

Problem: hard olympiad integral, teacher gets all 4 wrong:

| Solution | Reward | Advantage |
|---|---|---|
| A: wrong | 0.0 | 0 |
| B: wrong | 0.0 | 0 |
| C: wrong | 0.0 | 0 |
| D: wrong | 0.0 | 0 |

```
group_mean = 0.0, group_std = 0.0
advantage = (0.0 - 0.0) / 0.0 → set to 0
```

All solutions are equally bad. GRPO can't say "do less of this" because there's no correct solution to contrast against. Zero gradient again.

### Key takeaway

GRPO is a **relative** method — it ranks solutions within a group. It can only say "this one is better *than that one*." Without variation in the group, there's nothing to learn. This is why:

- A 99% accurate teacher is actually **worse** for GRPO — almost every group is all-correct
- A 10% accurate teacher is also bad — almost every group is all-wrong
- ~50-70% accuracy is the sweet spot for maximum reward diversity
- More generations per problem (16 instead of 4) drastically reduces the chance of all-same groups

---

## 3. How the Student Participates

The student **never generates its own solutions**. It reads the teacher's solutions and computes how likely it would be to produce those same tokens.

### The forward pass: one shot, not autoregressive

The teacher's solution is a known token sequence. The student feeds it all in at once and gets logprobs for every position in **one forward pass**.

This works because Qwen is a decoder-only model with a **causal attention mask**. Even though all tokens are fed in simultaneously, position 5 can only attend to positions 0–4:

```
Teacher's solution tokens: [F, a, c, t, o, r, (, x, -, 2, )]

Student forward pass (one shot):

Input position:    [prompt]  F       a       c       t       o       r       (
                      ↓      ↓       ↓       ↓       ↓       ↓       ↓       ↓
Model predicts:    P(F|prompt) P(a|F) P(c|a) P(t|c) P(o|t) P(r|o) P((|r) P(x|()
                                 ↑ each prediction only sees tokens to its left

Attention mask (causal):

         sees →   F  a  c  t  o  r  (  x
predict F:        ✓  ✗  ✗  ✗  ✗  ✗  ✗  ✗   ← sees only prompt
predict a:        ✓  ✓  ✗  ✗  ✗  ✗  ✗  ✗   ← sees prompt + F
predict c:        ✓  ✓  ✓  ✗  ✗  ✗  ✗  ✗   ← sees prompt + F, a
...
predict x:        ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓   ← sees everything before it
```

The prediction at each position is identical to what the model would have produced if it had generated tokens autoregressively up to that point. But instead of 100 forward passes for 100 tokens, it's just 1.

This gives the student's logprobs: "what probability would I have assigned to each of the teacher's tokens, given everything before it."

---

## 4. The Importance Sampling Ratio

The ratio corrects for the fact that solutions came from a different model (teacher, not student):

```
ratio = π_student(token) / π_teacher(token)
      = exp(student_logprob - teacher_logprob)
```

What the ratio means:
- **ratio ≈ 1.0**: student and teacher agree on this token. Full advantage signal flows through.
- **ratio < 1.0**: student assigns lower probability than teacher. Signal is dampened — don't force the student to drastically change a token it was already unlikely to produce.
- **ratio > 1.0**: student assigns higher probability than teacher. Signal is amplified.

The ratio is clipped to `[1-ε, 1+ε]` (PPO-style) to prevent wild updates from extreme ratios.

---

## 5. The Loss Function

For each token in the completion:

```
ratio = exp(student_logprob - teacher_logprob)
clipped_ratio = clip(ratio, 1-ε, 1+ε)
token_loss = -advantage × min(ratio, clipped_ratio)
```

Then:

```
total_loss = mean(token_losses) + β × KL_penalty
```

### Concrete example

Problem: "What is 5x + 3 if 5x - 3 = 12?" (ground truth: 18)

**Solution D (wrong, advantage = -1.73):**

```
Token:           "5x"   "="    "12"   "+"    "3"    "="    "9"      ← wrong
Teacher logprob: -0.10  -0.05  -0.30  -0.20  -0.08  -0.05  -0.90
Student logprob: -0.40  -0.10  -0.50  -0.30  -0.15  -0.10  -1.20

ratio per token: 0.74   0.95   0.82   0.90   0.93   0.95   0.74

token_loss:      -(-1.73) × 0.74 = +1.28    ← positive loss
                 -(-1.73) × 0.95 = +1.64    ← gradient pushes student
                 ...                            AWAY from these tokens
```

**Solution A (correct, advantage = +0.58):**

```
Token:           "5x"   "="    "15"   "+"    "3"    "="    "18"     ← right
Teacher logprob: -0.10  -0.05  -0.15  -0.08  -0.05  -0.03  -0.10
Student logprob: -0.30  -0.08  -0.35  -0.20  -0.12  -0.08  -0.50

ratio per token: 0.82   0.97   0.82   0.89   0.93   0.95   0.67

token_loss:      -(+0.58) × 0.82 = -0.48    ← negative loss
                 -(+0.58) × 0.97 = -0.56    ← gradient pushes student
                 ...                            TOWARD these tokens
```

- **Negative loss** → gradient descent minimizes it → student probability goes **up**
- **Positive loss** → gradient descent minimizes it → student probability goes **down**

The KL penalty (`β × KL`) adds a small cost for deviating too far from the base model, preventing overcorrection.

---

## 6. Why "Offline" — Teacher and Student Are Decoupled

| | Online GRPO | Offline GRPO (ours) |
|---|---|---|
| Who generates solutions? | Student | Teacher |
| Who computes logprobs for ratio? | Student (old snapshot) | Teacher (pre-computed) |
| Ratio meaning | π_student_new / π_student_old | π_student / π_teacher |
| Rewards reflect | Student's ability | Teacher's ability |
| Teacher loaded during training? | N/A | No — logprobs read from disk |

Because the teacher's logprobs are saved to `rollouts.jsonl` during generation, the teacher model is **never loaded during training**. This means:

- The teacher can be arbitrarily large (72B+) — it only needs GPU memory once during rollout generation
- Rollout generation is pure inference (no gradients, no optimizer)
- Training only loads the student

The student's actual quality is only measured **after training** during evaluation, when it generates its own solutions.

---

## 7. Scaling: Teacher vs Student vs Both

### Why a larger teacher helps

The 7B teacher at 70.9% accuracy leaves ~25% of problems with zero learning signal (all-correct groups). A stronger teacher solves harder problems → more problems get mixed results → more gradient signal.

But increasing **generations per problem** (4 → 16) is often a better fix:
- 4 gens at 70%: P(all correct) = 0.7⁴ = 24% → ~25% wasted
- 16 gens at 70%: P(all correct) = 0.7¹⁶ = 0.3% → almost nothing wasted

### Why a larger student helps

A 0.5B model has limited capacity for complex math reasoning. A 7B student has:
- Deeper representations
- Higher baseline accuracy to build on
- More capacity to distinguish correct from incorrect reasoning patterns

### Why not both

You can scale both. They run in separate stages so they don't compete for GPU memory. Practical limits on Vulcan (4× L40s, 48 GB each):

| Component | Practical Max | Strategy |
|---|---|---|
| Teacher (inference only) | ~72B | 4× L40s, vLLM tensor parallel |
| Student (LoRA training) | ~32B | 4× L40s, ZeRO-2 + gradient checkpointing |

### Recommended priority

1. **Increase generations** from 4 → 16 with existing 7B teacher (cheapest, directly fixes signal quality)
2. **Scale student** to 7B (highest expected impact on eval accuracy)
3. **Scale teacher** to 14B-32B (helps on hard problems the 7B can't solve at all)
4. **Both at max**: 72B teacher → rollouts → 14-32B student (ceiling on Vulcan)

A 99% accurate teacher would actually be **worse** — almost all groups would be all-correct with no contrastive signal. The ~50-70% range is ideal for GRPO.

---

*Created: 2026-03-06*
