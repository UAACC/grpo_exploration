# Math evaluation methodology

**Status**: Authoritative as of 2026-05-02. All MATH-format evals (MATH-500, ASDiv `\boxed{}` outputs) and reward computation for MATH-format training data go through `shared/math_eval.py`. Older `math_verify`-based code paths are deprecated.

## TL;DR

The project's MATH equivalence checker is `shared/math_eval.py`, a faithful port of DeepSeek-Math's `evaluation/eval/eval_utils.py:math_equal` and `evaluation/data_processing/answer_extraction.py:strip_string` + `extract_math_answer`. Comparator pipeline (in order):

1. Multi-candidate `\boxed{}` extraction from the model's response (every `\boxed{}` is a candidate, not just the last).
2. `strip_string` LaTeX canonicalization on each candidate AND the gold answer (`\dfrac→\frac`, drop `\left/\right`, strip `\text{...}`, normalize whitespace, collapse degree/percent symbols, etc.).
3. `math_equal` between each canonicalized candidate and the canonicalized gold:
   - direct string match, then
   - numeric equality (handles `42` vs `42.0`, `50%` vs `0.5`), then
   - bracketed-list element-wise recursion (`(a,b)` vs `[a,b]`), then
   - matrix element-wise recursion (`\begin{pmatrix}…\end{pmatrix}`), then
   - equation-form symbolic equality (move both sides to LHS, compare to zero), then
   - sympy `parse_latex` + `simplify(a-b) == 0`.

Any candidate matching makes the response correct. Source URLs are documented at the top of `shared/math_eval.py`.

## Critical venv dependency

`shared/math_eval.py` requires:

```
pip install antlr4-python3-runtime==4.11
```

Sympy's `parse_latex` will silently no-op (raise `ImportError` that the caller swallows) without exactly this version. **This was the root cause of our pre-2026-05-01 eval underestimate.** The Compute Canada–bundled `antlr4_python3_runtime 4.13.2+computecanada` does NOT work with sympy's LaTeX grammar.

The package is installed once in the project venv. If the venv ever gets rebuilt, this dependency must be re-pinned.

## The story: how we discovered the old verifier was broken

### What we used to use

Every MATH eval and every MATH reward computation in the project went through the `math_verify` package:

```python
from math_verify import parse, verify
verify(parse(extracted), parse(gold)) is True
```

This was wired through `offline_grpo/configs.py:check_math`, used by `mixture_grpo/evaluate.py`, `bc/data.py`, `offline_grpo/data.py`, `online_grpo/evaluate.py`, and the early version of `DG-offline/teacher_agnostic_loader.py`.

### What was actually wrong (two layers)

**Layer 1 — silent dependency failure.** `math_verify.parse()` calls sympy's `parse_latex()`, which requires `antlr4-python3-runtime==4.11`. Our venv had `antlr4_python3_runtime 4.13.2+computecanada` (Compute Canada's bundled version, which is API-incompatible). Every `parse_latex(...)` call raised:

```
ImportError: LaTeX parsing requires the antlr4 Python package, provided by
pip (antlr4-python3-runtime) or conda (antlr-python-runtime), version 4.11
```

`math_verify` catches this exception silently and returns `False` from `verify(...)`. From the outside, the code "worked": no errors, no warnings, just a stream of "wrong" verdicts on inputs that should have been correct.

**Layer 2 — `math_verify` is the wrong library for our needs.** Even with antlr4 fixed, `math_verify` lacks DeepSeek's pre-normalization layer (`strip_string`) and multi-candidate extraction logic (`extract_math_answer`). After installing antlr4, `math_verify` still missed:

- `\dfrac` vs `\frac` (sympy's `parse_latex` doesn't auto-canonicalize)
- `\text{Evelyn}` vs `Evelyn`
- `\left( ... \right)` size markers vs plain parentheses
- whitespace differences like `6 + 9i` vs `6+9i`
- multi-`\boxed{}` responses where the final box is wrong but an earlier one is right

### How we found it

We were verifying our chosen multi-teacher experiment teacher (DeepSeek-R1-Distill-Qwen-7B) against its model-card claim of 92.8% MATH-500 pass@1. Our pipeline measured **73.0%** under `math_verify` — a 19.8pp gap, way outside any reasonable noise margin.

Inspection of failing cases on saved completions showed:

| Ground truth | Extracted | Verdict | Why |
|---|---|---|---|
| `\frac{14}{3}` | `\dfrac{14}{3}` | wrong | parse_latex broken; even after fix, no `\dfrac→\frac` normalization |
| `\sqrt{51}` | `\sqrt{51}` | wrong | parse_latex broken; literally identical strings |
| `\pi` | `\pi` | wrong | parse_latex broken; literally identical strings |
| `p - q` | `p - q` | wrong | parse_latex broken; literally identical strings |
| `\text{Evelyn}` | `Evelyn` | wrong | no `\text{}` stripping; sympy can't parse `\text{Evelyn}` even with antlr4 fixed |
| `\left(3, \frac{\pi}{2}\right)` | `\left( 3, \frac{\pi}{2} \right)` | wrong | no `\left/\right` stripping, no whitespace normalization |
| `6+9i` | `6 + 9i` | wrong | sympy parse divergence on whitespace tokenization |

86 problems out of 500 flipped wrong→correct after we ported DeepSeek's full pipeline. Final result on the same data: **92.6% under `is_equiv` (single-candidate), 93.0% under `is_equiv_multi` (multi-candidate)** — both within sampling noise of the model card's 92.8%.

### The diagnostic sequence

| Comparator | MATH-500 acc | Notes |
|---|---|---|
| `math_verify` only (broken antlr4) | 73.00% | What the project used through 2026-05-01 |
| Hand-rolled lenient comparator (math_verify + LaTeX-normalized string equality) | 90.20% | Diagnostic-only, not the production fix |
| `is_equiv` (DeepSeek `strip_string` + `math_equal`, single-candidate) | 92.60% | First pass under proper pipeline |
| **`is_equiv_multi` (DeepSeek full pipeline, multi-candidate)** | **93.00%** | Production. Within noise of card 92.8%. |

## Why we didn't notice for months

Three reasons compounded:

1. **No exception ever surfaced.** Both layers fail silently — antlr4's ImportError gets caught inside `math_verify.parse`, which then returns gracefully.
2. **Plausibly low absolute numbers.** Our student is Qwen2.5-0.5B-Instruct, expected to score 25-35% on MATH. We measured numbers in that range and didn't question them.
3. **The bug hit all methods symmetrically.** BC, OG, DG, two-stage all used the same comparator and all got the same broken signal. *Relative* differences between methods (which we paid most attention to in progress reports) were preserved within a few percentage points. The bug only became visible when we measured a known-strong external teacher (R1, claimed 92.8%) and the gap was too large to explain away.

## What `math_verify` was NOT breaking

Some MATH cases came out right under the broken pipeline. `math_verify` did still work for:

- Direct numeric matches like `42` vs `42` (no LaTeX parsing needed).
- Some non-LaTeX symbolic cases via sympy's `parse_expr` fallback.
- `42.0` vs `42`-style numeric tolerance.

This is why our existing GSM8K numbers were less affected than MATH (GSM8K answers are mostly bare integers). The damage is concentrated in MATH-style outputs where formatted LaTeX answers dominate. ASDiv outputs that include `\boxed{}` from the 7B-Math teacher are likely affected similarly to MATH; SVAMP less so since answers are mostly numeric.

## Implications for prior project results

Every MATH-related accuracy number in `docs/progress_reports/*.md` predating 2026-05-01 was measured under the broken comparator and is **undercounted by approximately 5-15pp**. The exact magnitude varies by method (more LaTeX-formatted outputs → more underflow).

Both the EVAL and the TRAINING REWARD signal were affected:

- **Eval underestimate**: the trained checkpoint accuracies we reported are lower than the checkpoints' true accuracy under proper measurement.
- **Reward-signal undercounting during training**: ~17% of correct teacher rollouts were stamped reward=0 instead of reward=2. This shifted group statistics → some groups had advantage=0 → DG's gate stuck at 0.5 on those groups → BC-correct-only's filter dropped genuinely correct rollouts → all methods saw weaker training signal.

### What needs re-running

| Action | Cost | Required? |
|---|---|---|
| Re-eval existing trained checkpoints under `is_equiv_multi` | ~30 min per checkpoint × ~32 checkpoints across datasets ≈ 16 GPU-hrs | Yes; mandatory before any paper-grade comparison |
| Re-train one anchor config (e.g. DG-η=0.5 on Qwen-Math-MATH) under upgraded reward computation | ~15 GPU-hrs | Strongly recommended as a sanity check; tells us whether broken reward signal materially hurt training |
| Re-train all configs | ~100-150 GPU-hrs | Only if the sanity re-train shows training was substantially compromised |

The multi-teacher experiment we are running starting 2026-05-02 is **clean from the start** — R1 rollouts generated under DeepSeek's recommended sampling, training with `is_equiv_multi`-based rewards, eval with `is_equiv_multi`. No re-runs needed for that result.

## How to use the production pipeline

### From training code (reward computation)

`DG-offline/teacher_agnostic_loader.py:_compute_correctness_math` already routes through `is_equiv_multi`. New training scripts using teacher-agnostic data should import:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
from math_eval import is_equiv_multi  # multi-candidate, full pipeline
# or:
from math_eval import is_equiv         # single-candidate, when only the extracted answer is on hand
```

### From eval code

`mixture_grpo/evaluate.py:check_math` is the canonical MATH eval entry. It dispatches to `is_equiv_multi` and accepts the question (used for question-aware extraction in `extract_math_answer`).

### From verification scripts

`DG-offline/eval_r1_teacher.py` is a standalone verification eval following DeepSeek's protocol exactly (no system prompt, temp=0.6, top_p=0.95, max_tokens=32768, multi-candidate `is_equiv_multi`). Use as a template when verifying any new teacher's claimed MATH accuracy on our pipeline.

## Source attribution

`shared/math_eval.py` is a faithful port of:

- `evaluation/eval/eval_utils.py:math_equal, symbolic_equal, is_digit, parse_digits` — from <https://raw.githubusercontent.com/deepseek-ai/DeepSeek-Math/main/evaluation/eval/eval_utils.py>
- `evaluation/data_processing/answer_extraction.py:strip_string, _fix_fracs, _fix_a_slash_b, _fix_sqrt, _fix_tan, extract_answer, extract_boxed_answers, extract_math_answer` — from <https://raw.githubusercontent.com/deepseek-ai/DeepSeek-Math/main/evaluation/data_processing/answer_extraction.py>

Both fetched 2026-05-01. Behavior is unchanged from upstream modulo:

- The `timeout=True` branch of `math_equal` (which uses `multiprocessing` + `call_with_timeout`) is omitted; we always run synchronously. Pathological inputs that hang sympy could in principle wedge a worker; we have not seen this on MATH-500.

## Maintenance notes

- If sympy is upgraded, re-verify that `parse_latex` still works with antlr4 4.11. Sympy occasionally adjusts its grammar dependency.
- If we add a new teacher whose vocab introduces special tokens with IDs colliding with student-vocab special-token slots (cf. R1's `<think>`/`</think>` at 151648/151649), the eval pipeline is unaffected (it reads `response` text, not token IDs). Training reward computation in `teacher_agnostic_loader.py` is also unaffected for the same reason. Only IS-based offline GRPO is at risk; see `docs/dg_offline/plans/multi_teacher_experiment.md` §5.1.1 for the workaround discussion.
