# DG-offline: intuition

Written for a reader who knows policy gradient and PPO at a casual level but has not read the DG paper.

## The setup we are in

We have a small student (Qwen2.5-0.5B-Instruct) and a larger math-tuned teacher (Qwen2.5-Math-7B-Instruct). We have a pile of teacher completions for math problems, each labeled correct or wrong. We want to use those completions to train the student with something better than behavioral cloning.

Standard offline GRPO does this by reweighting each token's gradient by the importance ratio `π_student(token) / π_teacher(token)`, then clipping to `[0.8, 1.2]` to control variance. It's the "correct" policy-gradient recipe on paper.

In practice, it breaks.

## Why the standard recipe breaks here

1. **Teacher and student are very different models**. Over a 500-token completion, per-token IS ratios compound. Even if each ratio is close to 1, the product across hundreds of tokens can be `1e-5` or `1e+5`. After clipping most of the rollout contributes nothing.

2. **Teacher logprobs are stored in bf16**. Per-token ratios close to ties get noisy. In practice many of our stored `π_teacher` values are approximate.

3. **Clipping is symmetric but the signal isn't**. A "surprising success" (correct answer from a student-unlikely trajectory) is very informative. A "surprising failure" (wrong answer from a student-unlikely trajectory) is usually the student being asked to do something it cannot. IS treats both identically.

## The DG insight

Instead of asking "how much more likely was the teacher to emit this trajectory than the student?" (which is what IS measures), DG asks a simpler and better-posed question:

> **How informative is this rollout to the student right now?**

Informativeness combines two things:
- **Sign and size of the advantage** (did this rollout do better than average within its group?)
- **How surprising the rollout is under the student's current policy** (measured by the student's own negative log-probability of the rollout)

Their product is called **delight**:

```
delight = advantage × surprisal
```

Delight is large and positive for **surprising successes** (good outcome, the student would not have picked this rollout on its own → very informative). It is large and negative for **surprising failures** (bad outcome, the student would not have picked this rollout → probably a teacher-specific quirk the student cannot execute → skip this gradient step). It is near zero for rollouts the student already produces reliably (nothing new to learn there).

## The gate

DG passes delight through a sigmoid:

```
gate = sigmoid(delight / η)
```

- For surprising-positive rollouts, `gate → 1` and the full gradient is applied.
- For surprising-negative rollouts, `gate → 0` and the gradient is zeroed out.
- For low-surprisal or low-advantage rollouts, `gate ≈ 0.5` and the gradient gets roughly half-weight.

η is a temperature. Small η makes the gate nearly binary (aggressive filtering). Large η makes the gate soft and DG converges toward unweighted REINFORCE on the advantage.

## Four quadrants (the memorable picture)

|                    | Expected (low surprisal) | Surprising (high surprisal) |
|--------------------|--------------------------|-----------------------------|
| **Success** (A>0)  | Gate ≈ 0.5 (small update) | **Gate → 1** (big amplification) |
| **Failure** (A<0)  | Gate ≈ 0.5 (small update) | **Gate → 0** (suppressed)  |

DG keeps the "surprising success" quadrant, suppresses the "surprising failure" quadrant, and damps the two routine quadrants. That asymmetry is the whole point and is exactly what PPO's symmetric clip fails to do.

## Why the student uses its own surprisal

Surprisal in DG is `−log π_student(token)`, **not** `−log π_teacher(token)`. This is intentional. It means:

- We don't need the teacher's logprobs at all. Any teacher, including a closed API or a completely different architecture, works.
- The "what counts as surprising" definition updates as the student learns, which is the right thing.
- No numerical drift from bf16 stored teacher logprobs.

The cost is that we cannot claim IS-style unbiasedness. DG trades statistical guarantees for practical stability, which was the right trade in our setting.

## One catch we ran into

DG-offline still runs inside TRL's `GRPOTrainer`, which expects an IS ratio in its PPO loss. We handle this by setting `old_per_token_logps = current_per_token_logps.detach()` so the ratio becomes `exp(0) = 1`, the PPO clip becomes a no-op, and the whole thing reduces to `loss = −gate × advantage × log π_student(completion)` plus a KL penalty against the reference. See [implementation.md](implementation.md) for the specific lines.

## When to expect DG to help vs not

- **DG helps** when teacher and student are far apart (so IS ratios are unstable), when reward signal is clean (so advantage sign is trustworthy), and when the student has enough capacity to benefit from informative updates but not enough to blindly imitate the teacher.
- **DG helps less** when BC already works well (teacher is close enough to the student that naive imitation is fine), or when the dataset is small enough that the student memorizes regardless of gradient shaping.
- **DG's η becomes a dataset-dependent knob**. Our sweep shows no universal best η: 0.1 wins on GSM8K and SVAMP, 0.5 on MATH, 2.0 on ASDiv. Dataset-specific tuning is currently unavoidable.
