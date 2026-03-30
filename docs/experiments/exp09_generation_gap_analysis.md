# Experiment 09: Generation Gap Analysis — Where Does the Student Actually Fail?

**Date**: 2026-03-28
**Status**: In Progress
**Depends on**: exp08_capacity_vs_data.md (hypothesis), check_p4.py (93% claim)

---

## 1. Motivation

We claimed "student and teacher agree on 93% of next-token predictions" based on `check_p4.py`. But this was measured on **teacher-generated context** — the student is asked to predict the next token given a prefix the teacher wrote.

At test time, the student generates from its own prefix. If the student's actual generations diverge from the teacher's, then the 93% number is misleading — it measures a condition that never occurs in practice.

**Before we can theorize about WHY the student fails (capacity, exposure bias, compounding errors), we need to understand WHERE and HOW it fails.**

## 2. What We Know

| Fact | Source | Confidence |
|------|--------|------------|
| 93% top-1 agreement on teacher context | `bc/check_p4.py`, job 4383020 | High (but narrow — only teacher context) |
| Disagreement tokens look stylistic | `bc/check_disagreement_tokens.py`, job 4383341 | Low (qualitative, 5 samples, teacher context only) |
| 5-problem side-by-side looks similar | `bc/show_comparison.py`, job 4383334 | Very low (5 easy problems, too small) |
| Student: 27% on MATH | eval-4389323 | High |
| Teacher: 75% on MATH | eval-all-4333310 | High |

**What we don't know:**
- What do the student's generations look like on problems where the teacher succeeds and the student fails?
- Does the student diverge early or late in the generation?
- Is the student's reasoning structure different, or does it follow the same steps but make a mistake?
- What is the token-level agreement when measured on the student's own context?

## 3. Analysis Plan

### 3A: Large-Scale Disagreement Comparison

**What**: Generate completions from both models on 200+ MATH problems. For each problem, categorize into four quadrants:

| | Teacher Correct | Teacher Wrong |
|---|---|---|
| **Student Correct** | Both succeed (easy) | Student better (rare) |
| **Student Wrong** | **THE GAP** — focus here | Both fail (hard) |

**Focus on the "gap" quadrant**: teacher correct, student wrong. These are the problems that create the accuracy difference. We want to understand:

1. **Does the student's reasoning start correctly then derail?** If so, at what point?
2. **Does the student attempt a completely different approach?**
3. **Does the student make a specific mathematical error (wrong computation, wrong formula)?**
4. **Does the student's generation degenerate (repetition, incoherence, truncation)?**

**Implementation**: `bc/analyze_gap.py`
- Generate student completions (greedy, temp=0) on 200 MATH test problems
- Generate teacher completions (greedy, temp=0) on same problems
- Check correctness of both using math_verify
- For each "gap" problem (teacher correct, student wrong):
  - Print both completions side by side
  - Compute token-level metrics (length, where first divergence occurs)
  - Classify failure mode (will start with manual inspection, may automate later)

**Resource**: 1 GPU, ~30 min (200 problems × 2 models sequentially, vLLM)

### 3B: Token Agreement on Student Context

**What**: Measure the same "top-1 agreement" as check_p4.py, but on the student's own generated prefix instead of the teacher's.

For each problem:
1. Student generates a full completion autoregressively
2. For each position t in the student's completion, ask: what token does the teacher assign highest probability to, given the student's prefix up to position t?
3. Compute agreement rate = fraction of positions where teacher's argmax matches student's actual token

**Why this matters**: If agreement drops from 93% (teacher context) to, say, 70% (student context), it proves the student's own generations push both models into regions of disagreement. The 93% number would be an artifact of teacher-forcing, not a real property of the student-teacher relationship.

**Implementation**: `bc/agreement_on_student_ctx.py`
- Generate 100 student completions (greedy)
- For each completion, run teacher model on the student's token sequence
- At each position, check if teacher's argmax == student's actual token
- Report: mean agreement, agreement vs position (does it decay over the completion?)

**Resource**: 1 GPU, ~20 min. Need to load both student and teacher sequentially (7B teacher needs ~14GB, fits on L40s after student is deleted).

### 3C: Divergence Point Analysis

**What**: For the "gap" problems from 3A, find exactly where the student's generation first diverges from the teacher's, and what happens after.

For each gap problem:
1. Align student and teacher completions token by token (longest common prefix)
2. Record the divergence point (token index where they first differ)
3. Check: after divergence, does the student ever recover and produce correct reasoning?
4. Compute: what fraction of the total completion occurs after the divergence point?

**Why this matters**: If divergence happens early (first 10-20 tokens) and the student never recovers, it suggests the student lacks the capacity to find a correct reasoning path independently. If divergence happens late (token 300+), it suggests the student follows the right approach but makes a computational error.

**Implementation**: Part of `bc/analyze_gap.py` (extend 3A script)

## 4. Expected Outcomes and What They Would Mean

### Scenario A: Student generations are structurally similar to teacher's, with localized errors
- Agreement on student context stays high (~85%+)
- Divergence happens late in completions
- Student follows the same approach but makes algebraic/arithmetic mistakes
- **Implication**: The capacity gap is about computation precision, not reasoning strategy. Targeted interventions (calculator tools, chain-of-thought verification) might help more than RL training.

### Scenario B: Student generations diverge early and follow different paths
- Agreement on student context drops significantly (<75%)
- Divergence happens in the first 50-100 tokens
- Student attempts different (often wrong) approaches
- **Implication**: The 93% teacher-context agreement is an illusion. The student can predict teacher tokens given teacher context, but cannot reproduce the teacher's reasoning strategy independently. This is a real capacity gap.

### Scenario C: Student generations degenerate (repetition, truncation, incoherence)
- Many gap completions show repetitive text, premature stopping, or loss of structure
- Agreement drops sharply after a certain position
- **Implication**: The student's autoregressive generation is unstable for long sequences. The problem is generation stability, not mathematical knowledge.

### Scenario D: Mixed — depends on problem difficulty
- Easy gap problems look like Scenario A (small errors)
- Hard gap problems look like Scenario B (different approaches)
- **Implication**: There's a capacity threshold — the student can handle simple multi-step reasoning but breaks down on complex problems. Curriculum-based training would be the right approach.

## 5. Implementation Order

### Step 1: `bc/analyze_gap.py` (3A + 3C combined)

```
Input:  200 MATH test problems
Output:
  - Per-problem: student completion, teacher completion, both correctness
  - Quadrant counts (both correct, teacher only, student only, both wrong)
  - For "gap" problems: side-by-side printout, divergence point, completion lengths
  - Summary statistics: how many gap problems, mean divergence point, length distribution
```

SLURM: 1×L40s, 1h, 64GB

### Step 2: `bc/agreement_on_student_ctx.py` (3B)

```
Input:  100 MATH test problems
Output:
  - Mean token agreement on student context
  - Agreement vs position plot data (does it decay?)
  - Comparison: 93% (teacher context) vs X% (student context)
```

SLURM: 1×L40s, 1h, 64GB (loads 7B teacher, needs memory)

### Step 3: Analysis and Documentation

Read the outputs, classify failure modes, update exp08 and exp09 with findings.

## 6. Artifacts

Scripts to create:
- `bc/analyze_gap.py` — 3A + 3C: large-scale generation comparison + divergence analysis
- `bc/agreement_on_student_ctx.py` — 3B: token agreement on student-generated context
- `bc/run_analyze_gap.sh` — SLURM script
- `bc/run_agreement_student_ctx.sh` — SLURM script

Results will go in:
- `bc/logs/bc_math/analyze-gap-*.out`
- `bc/logs/bc_math/agreement-student-ctx-*.out`
