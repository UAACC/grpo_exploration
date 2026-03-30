# Experiment 08: Capacity vs Data — Why Does the Student Fail?

**Date**: 2026-03-28
**Status**: Planning — hypotheses formed, experiments designed, awaiting empirical results
**Depends on**: experiment_summary_2026_03_18.md (all prior results)

**Important**: The hypotheses below are unverified. We document them to design targeted experiments. We do NOT know the answer yet. DG-offline has not been fully trained. The capacity explanation may be wrong.

---

## 1. The Puzzle

Our student (Qwen2.5-0.5B-Instruct) agrees with the teacher (Qwen2.5-Math-7B-Instruct) on 91-93% of next-token predictions when conditioned on teacher-generated context. Yet the student gets 27% on MATH while the teacher gets 75%.

**Source of the 93% claim**:
- Script: `bc/check_p4.py`, job 4383020
- Log: `bc/logs/bc_math/check-p4-4383020.out`
- Exact result: "mean_top1=0.9306" (base model before BC), measured on 200 teacher completions
- Method: For each token in a teacher completion, check if the student's argmax prediction matches
- **Critical caveat**: This is measured on **teacher-generated context** (teacher prefix → student predicts next token). We do NOT have evidence for the same agreement on student-generated context.

A 7-9% per-token disagreement rate over a 500-token completion means roughly 35-45 tokens differ. These disagreements are mostly stylistic:

**Source of the "stylistic" claim**:
- Script: `bc/check_disagreement_tokens.py`, job 4383341
- Log: `bc/logs/bc_math/disagree-4383341.out`
- Examples: "left" vs "red", "We" vs "The", "lamps" vs "red", "number" vs "probability"
- **Caveat**: This was qualitative inspection, not systematic categorization. Some disagreements like "number" vs "probability" could be reasoning-relevant.

**So where does the 48 percentage point accuracy gap come from?**

## 2. Hypothesis: Compounding Errors Under Self-Generated Context

### 2.1 The Exposure Bias Argument

The 93% agreement is measured under **teacher-generated context**: given the prefix the teacher wrote, the student predicts the same next token 93% of the time. But at test time, the student conditions on **its own** tokens.

After the first divergent token, the student's context differs from the teacher's. The second prediction is now conditioned on a slightly different history. By token 100, the contexts may have diverged substantially. By token 300, the student may be in a completely different region of the reasoning chain.

This is a well-known problem in sequence modeling called **exposure bias**: the model is evaluated (or analyzed) under teacher-forcing conditions but tested autoregressively on its own outputs.

### 2.2 Error Recovery as the Real Gap

The teacher, with 7B parameters, has enough capacity to recognize when an intermediate step is going wrong and course-correct. The 0.5B student, with 14x fewer parameters, may lack this ability. Once it takes a wrong turn in a proof, it can't recover.

A 500-token mathematical proof is a sequential decision process. One wrong algebraic step leads to a dead end. The question is not "can the student predict the next token?" but "can the student maintain a coherent multi-step argument for 500 tokens without a fatal error?"

### 2.3 How This Would Explain Prior Results (If True)

| Result | Explanation under this hypothesis |
|--------|----------------------------------|
| BC doesn't help (27.4% vs 27.2% baseline) | BC trains on teacher context, but at test time the student sees its own context. The skill transfer doesn't bridge the context gap. |
| Offline GRPO degrades (27.6% → 26.5%) | Pushes student toward teacher-style completions it can't sustain autoregressively. Disrupts the student's own (weak but functional) reasoning patterns. |
| Online GRPO helps slightly (+1.7%) | Trains on student's own completions — no context mismatch. But at 30% accuracy, reward signal is sparse. |
| 7B teacher online GRPO is flat | Teacher already recovers from most errors. Little room to improve with binary rewards. |
| 7B teacher offline GRPO degrades (-3.8%) | Even on own rollouts, stale data + advantage collapse prevents learning. Not a capacity issue — an algorithm issue. |
| Token disagreements are stylistic | The 7% token-level gap isn't where the accuracy gap lives. The gap is in long-horizon coherence, not individual token quality. |

## 3. Testable Predictions

If the hypothesis is correct:

**P1**: Stronger students should benefit more from offline teacher data, because they have enough capacity to maintain coherence even when nudged toward teacher-style reasoning.

**P2**: On shorter reasoning tasks (or easy MATH problems requiring <100 tokens), the compounding effect is smaller, so even the 0.5B student should benefit from teacher data.

**P3**: The student's accuracy should degrade with completion length — problems requiring longer solutions should have disproportionately lower accuracy.

**P4**: Token-level agreement measured on **student-generated context** (not teacher context) should be substantially lower than the 93% measured on teacher context.

**P5**: DG-offline (Delightful Policy Gradient) results are unknown — a full eta sweep is needed before drawing conclusions. If DG helps, it suggests gradient quality matters; if it doesn't, it supports the capacity hypothesis.

## 4. Proposed Experiments

### Experiment 8A: Stronger Student (1.5B) with Offline GRPO

**Objective**: Test P1 — does a stronger student benefit from offline teacher data?

**Setup**:
- Student: Qwen2.5-1.5B-Instruct (44% baseline on MATH, same vocab as 0.5B)
- Teacher: Qwen2.5-3B-Instruct (58% on MATH, same vocab — eliminates OOV issue)
- Generate 3B teacher rollouts on MATH (same pipeline as before)
- Run offline GRPO, BC, and DG-offline on 1.5B with 3B rollouts
- Compare against 1.5B online GRPO

**Prediction**: If capacity is the bottleneck, 1.5B should improve where 0.5B didn't, because it has enough parameters to maintain coherence under teacher-style guidance.

**Control**: Also run 0.5B on 3B rollouts (same data, weaker student) to confirm that the data alone isn't the differentiator.

| Experiment | Student | Teacher | Method | Expected |
|------------|---------|---------|--------|----------|
| 8A-1 | 1.5B | 3B | Offline GRPO | Improvement if P1 holds |
| 8A-2 | 1.5B | 3B | BC (correct-only) | Improvement if P1 holds |
| 8A-3 | 1.5B | 3B | DG-offline (best η from 8D) | Depends on 8D results |
| 8A-4 | 1.5B | 3B | Online GRPO | Positive control |
| 8A-5 | 0.5B | 3B | Offline GRPO | No improvement (capacity too low) |
| 8A-6 | 0.5B | 3B | BC (correct-only) | No improvement (capacity too low) |

### Experiment 8B: Accuracy vs Completion Length

**Objective**: Test P3 — does accuracy drop with solution length?

**Setup**:
- Use the existing 0.5B baseline eval runs (5 runs, 500 MATH problems)
- For each problem, record: (a) whether the model got it right, (b) the length of the generated completion, (c) the ground-truth difficulty level if available
- Plot accuracy as a function of completion length (binned)
- Also plot accuracy vs MATH difficulty level

**Prediction**: Accuracy should be high for short completions and decay for longer ones, following a compounding error curve.

**Implementation**: Can be done by modifying evaluate.py to log per-problem results, or by post-processing existing eval logs.

### Experiment 8C: Agreement on Student Context vs Teacher Context

**Objective**: Test P4 — is token agreement lower on the student's own context?

**Setup**:
- Take 200 MATH problems
- For each problem, generate a student completion (0.5B, greedy)
- For each position in the student's completion, compute: does the teacher assign highest probability to the same token?
- Compare this "student-context agreement" against the "teacher-context agreement" (93%) from our prior analysis

**Prediction**: Student-context agreement should be substantially lower than 93%, confirming that the student's own context diverges into regions where teacher-student alignment breaks down.

**Implementation**: Load student completion, run teacher model on each prefix, check top-1 agreement. Similar to `check_disagreement_tokens.py` but using student-generated context instead of teacher-generated context.

### Experiment 8D: DG-Offline — Delightful Policy Gradient for Offline GRPO

**Objective**: Test whether replacing IS-ratio weighting with DG's delight gating can fix offline GRPO's degradation in the cross-model (7B→0.5B) setting.

**Background — Why DG**:
Standard offline GRPO weights gradients by the IS ratio π_student/π_teacher, clipped PPO-style. This has known problems: ratio instability over 500+ token sequences, symmetric treatment of surprising successes and failures, and dependence on noisy bf16 behavior logprobs. DG (Osband, 2026, arXiv:2603.20521) replaces IS ratios with a sigmoid gate on "delight" = advantage × surprisal, computed entirely from the learner's own policy. It suppresses surprising failures (gate→0) while amplifying surprising successes (gate→1). See `dg-offline_imp.md` for full technical details.

**Key concern from test run**:
Our test run (η=1.0, 1130 steps before timeout) revealed that per-token surprisal in our setting is low (~0.1-0.3 nats), meaning the student already assigns ~82% probability per token to teacher completions. With such low surprisal, delight values are small, and the gate barely moves from 0.5 (neutral). This is fundamentally different from the paper's setting (stale actors with high surprisal). The gate may need a much lower η to be effective, or the mechanism may not apply to our cross-model setting at all.

**Setup**:
- Implementation: `DG-offline/` folder (trainer.py, train.py, run_math.sh)
- Student: Qwen2.5-0.5B-Instruct, Teacher rollouts: 7B on MATH (existing `rollouts_full.jsonl`)
- Same hyperparameters as controlled offline GRPO (lr=3e-6, beta=0.001, LoRA r=32/α=32)
- Eta sweep: η ∈ {0.5, 1.0, 2.0} (submitted as jobs 4543658, 4543660, 4543662)
- 1 epoch, 4×L40s, DDP, completion-level gating
- Each run saves to separate checkpoint dir: `/scratch/mrli/checkpoints/dg_offline_math_eta{η}`

**Jobs submitted**:

| Job ID | η | Checkpoint | Status |
|--------|---|------------|--------|
| 4543658 | 0.5 | `dg_offline_math_eta0.5` | Resubmit needed (wrong rollout path, now fixed) |
| 4543660 | 1.0 | `dg_offline_math_eta1.0` | Resubmit needed |
| 4543662 | 2.0 | `dg_offline_math_eta2.0` | Resubmit needed |

**Note**: First submission crashed due to wrong default rollout path (`rollouts_math_full.jsonl` → `rollouts_full.jsonl`). Fixed in run_math.sh.

**What to monitor during training** (logged to wandb and training_metrics.jsonl):
- `dg/gate_mean`: Should be ~0.5 for zero-advantage batches. When non-zero advantages exist, lower η should push gates further from 0.5 (more decisive filtering).
- `dg/gate_min`, `dg/gate_max`: Range of gates. Wider range = more active filtering.
- `dg/surprisal_mean`: Distribution gap indicator. If it decreases during training, the student is learning to match teacher tokens.
- `dg/delight_mean`: Sign indicates batch composition — positive means surprising successes dominate.
- `reward`: Does it trend up (learning) or stay flat (no signal)?
- `kl`: Does the model move from base? Our test run showed kl~0.007, much higher than online GRPO 7B (kl~6e-5).

**Expected behavior per η**:

| η | Gate behavior | Expected effect |
|---|---------------|-----------------|
| 0.5 | More decisive (gates spread further from 0.5) | Stronger filtering, may help or may be too aggressive |
| 1.0 | Moderate (gates ~0.42-0.56 from test run) | Paper's default, may be too gentle for our low-surprisal setting |
| 2.0 | Softer (gates very close to 0.5) | Nearly equivalent to standard REINFORCE, weak filtering |

**Open questions**:
1. Does any η value produce improvement over the 27% baseline on MATH?
2. Does any η value avoid the degradation seen in standard offline GRPO (26.5%)?
3. How does the gate distribution evolve during training — does surprisal decrease as the student learns?
4. Is the advantage collapse problem (82-93% zero-std groups) still the dominant bottleneck, making the DG gate irrelevant for most samples?
5. Should we try much smaller η (0.01, 0.1) to force more decisive gating despite low surprisal?

**After eval, compare against**:

| Method | MATH Accuracy | Source |
|--------|---------------|--------|
| Baseline (0.5B) | 27.16% | eval-4389323 |
| Offline GRPO (standard IS) | 27.64% | eval-all-4333310 |
| Offline GRPO (controlled hparams) | TBD | Not yet evaluated separately |
| BC (all completions) | 27.40% | eval-bc-math-4382964 |
| BC (correct-only) | 27.20% | eval-bc-correct-4383318 |
| DG-offline η=0.5 | **TBD** | Pending |
| DG-offline η=1.0 | **TBD** | Pending |
| DG-offline η=2.0 | **TBD** | Pending |

**This is the highest-priority experiment. All hypotheses about capacity vs gradient quality depend on these results.**

### Experiment 8E: Easy vs Hard MATH Subsets

**Objective**: Test P2 — does offline GRPO help on easy problems?

**Setup**:
- Split MATH test set by difficulty level (levels 1-5)
- Evaluate all methods (baseline, offline GRPO, BC, online GRPO) per difficulty level
- Check if offline GRPO improves on easy problems (where solutions are shorter and the student's baseline is higher) while degrading on hard problems

**Prediction**: Offline GRPO may show improvement on Level 1-2 problems (short solutions, student baseline ~50%+) while degrading on Level 4-5 (long solutions, student baseline <15%).

**Implementation**: MATH dataset includes difficulty levels. Modify eval script to report per-level accuracy.

## 5. Priority Order

1. **8D (DG-offline eta sweep)** — already coded, needs full training runs. Must complete before we can distinguish gradient-quality vs capacity explanations. Without these results, the capacity hypothesis is speculation.
2. **8B (accuracy vs length)** — cheapest analysis, pure post-processing of existing data, directly tests the compounding hypothesis
3. **8C (agreement on student context)** — one eval job, directly measures the exposure bias gap
4. **8E (easy vs hard subsets)** — analysis of existing eval data, tests prediction P2
5. **8A (1.5B student)** — most expensive (need to generate 3B rollouts + multiple training runs), but the most conclusive test of the capacity hypothesis

## 6. What Would Change Our Mind

- If **8A shows 1.5B also doesn't improve** with offline teacher data → capacity isn't the bottleneck, something else is wrong (maybe GRPO advantage collapse is the real issue regardless of capacity)
- If **8B shows no correlation between accuracy and length** → compounding errors isn't the mechanism
- If **8C shows agreement is still ~93% on student context** → exposure bias isn't the explanation, the gap must be elsewhere
- If **8D (DG) significantly outperforms standard offline GRPO at some η** → gradient quality matters more than we thought, and the capacity hypothesis is weakened or wrong
- If **8E shows uniform degradation across all difficulty levels** → the problem isn't specific to hard/long problems

Each of these would force us to revise the hypothesis. We document them now so we can update honestly when results arrive.

## 7. Artifacts

- DG-offline implementation: `DG-offline/` (committed to git)
- Prior analysis: `experiment_summary_2026_03_18.md`
- Disagreement analysis: `bc/check_disagreement_tokens.py`, `bc/logs/bc_math/disagree-*.out`
- Token agreement data: `bc/check_p4.py`, `bc/logs/bc_math/check-p4-*.out`
