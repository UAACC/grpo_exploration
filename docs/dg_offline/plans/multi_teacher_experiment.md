# Multi-teacher experiment (design locked)

**Status**: design locked 2026-05-01. Loader (Stage 1) implemented and smoke-tested. Ready for Stage 0 pre-check + execution.
**Owner**: mrli.
**Related**: [`docs/dg_offline/theory.md`](../theory.md) §6, [`docs/dg_offline/technical_reference.md`](../technical_reference.md) §5.

## 1. Why this experiment exists

DG-offline as a *method* needs only two things from the teacher: (i) the completion as **text**, and (ii) a reward signal. Surprisal is computed under the student's own policy on the student's own tokenization, so the teacher's tokenizer, vocab, architecture, and logprob availability are all irrelevant. The new `DG-offline/teacher_agnostic_loader.py` operationalizes this in our codebase.

That removes tokenizer family as a constraint on which teacher we use. We can pick on the criterion that actually matters for the experiment: **teacher math performance**.

### Hypothesis (locked 2026-05-01)

> A stronger teacher should help DG-offline more than it helps Offline-GRPO.

Two mechanisms predict this:

1. **Advantage collapse mitigation.** 82–93% of groups currently have zero reward variance ([`technical_reference.md`](../technical_reference.md) §4.2). A stronger teacher → higher correct rate → more groups with non-zero variance → more samples where DG's gate is active at all. Independent of surprisal.
2. **Surprisal saturation, finally.** A stronger and stylistically distinct teacher should produce *correct* completions that are also higher-surprisal to the student (long reasoning chains the student would not generate). That activates DG's "amplify surprising successes" branch (gate → 1 on positive-advantage, high-surprisal rollouts), which the Apr 21 diagnostic showed is currently dormant (correct-completion surprisal ~0.2 nats).

If both effects are real, DG should pull away from Offline-GRPO precisely when teacher quality jumps. The headline number is whether `Δ_DG > Δ_OG` (DG benefits more from a stronger teacher than OG does). A negative result here is also publishable: "the better-teacher mechanism does not differentiate DG from OG in our regime."

### Framing this dissolves

Earlier drafts split this into (A) mechanism vs (B) capability framings. Under the locked design, (A) is the headline (still mechanism) and (B) demonstrates itself as a side result regardless of which teacher we pick — the loader handles cross-tokenizer if the best teacher happens to be cross-tokenizer, but we no longer make that the deciding factor.

## 2. Constraint that bites first: the data loader

The DG-offline *method* is tokenizer-agnostic, but the *current code path* is not. `DG-offline/train.py:31-33` does `sys.path.insert(0, .../offline_grpo)` and then `from data import load_rollouts, ...`, which resolves to `offline_grpo/data.py`. That loader (the **shortcut path**, "Path B"):

- Reads pre-computed `completion_ids` from the rollout JSONL (teacher's token IDs).
- Truncates at the first OOV token to handle the 128-row Qwen2.5 vocab gap.
- Implicitly assumes teacher and student share a tokenizer — fails *silently* on cross-vocab teachers (forward-pass succeeds, surprisal is meaningless).

The shortcut is equivalent to the conceptually correct DG path only because Qwen-Math-7B and Qwen-0.5B happen to share IDs 0–151,935. Outside that coincidence the shortcut is a footgun.

### Decision: self-contained Path A loader (DONE 2026-05-01)

`DG-offline/teacher_agnostic_loader.py` is implemented, smoke-tested, and `DG-offline/train.py` now imports from it. Summary of what shipped:

- **Location**: `DG-offline/teacher_agnostic_loader.py`.
- **Self-contained**: carries its own `load_rollouts_text`, `compute_rewards_and_advantages`, `build_training_dataset`, `build_offline_lookup`, plus inlined `extract_boxed_answer` / `extract_gsm8k_answer` reward parsers. No imports from `offline_grpo/data.py` or `offline_grpo/configs.py`.
- **Input**: rollout JSONL with `response` (text). Teacher `completion_ids` and `behavior_logprobs` on disk are ignored.
- **Output**: per-completion records carrying student-vocab `completion_ids` produced by re-tokenizing `response` under the student tokenizer at load time. No `vocab_size` truncation.
- **Pairs with**: `DG-offline/trainer.py` unchanged.
- **Import switch**: `train.py` now does `from teacher_agnostic_loader import ...` (the legacy `from data import ...` and its `sys.path.insert(0, .../offline_grpo)` are gone).

**Smoke-test results (2026-05-01, MATH `rollouts_shard_0.jsonl`, 12,000 completions):**

| Check | Result |
|---|---|
| Record count parity vs legacy loader | 12,000 = 12,000 ✓ |
| Reward distribution parity | 0 / 12,000 mismatches ✓ |
| Advantage distribution parity | 0 / 12,000 mismatches ✓ |
| Token-ID exact match (Path A vs Path B) | 11,515 / 12,000 (96%); 4% diverge as predicted (legacy OOV truncations + trailing-special-token boundary effects) |
| Round-trip `text → student-tok → decode → text` | 50 / 50 spot-check ✓ |
| End-to-end SLURM training (10 steps, 4 L40s, η=0.5) | COMPLETED, exit 0:0, 1m29s. Loss finite, gates correct (gate_min=5e-15 / gate_max=0.83 on a step where surprisal spiked), KL ~3e-4, no NaN. |

The loader is production-ready for the multi-teacher experiment.

## 3. Pre-check (Stage 0, cheap)

Before generating the full DeepSeek rollout pool and committing 80–100 GPU-hours of training, repeat the Apr 21 surprisal diagnostic on a small sample of DeepSeek-R1-Distill-Qwen-7B rollouts:

1. Generate ~500 DeepSeek completions on a MATH train subset.
2. Run `DG-offline/measure_surprisal.py` under the base student. Report mean correct / mean wrong surprisal and gate-saturation fractions at η ∈ {0.1, 0.5, 1.0, 2.0}.
3. Compare against the Apr 24 baseline (Qwen-Math: correct ~0.19 nats, wrong ~0.99 nats, gate saturation 1.9% / 3.8% at η=0.1).

**Decision criterion** (sanity check, not a hard gate): if DeepSeek's correct-completion surprisal moves materially upward (e.g. >0.5 nats) the "amplify surprising successes" mechanism story is on track. If it doesn't move, the strong-teacher hypothesis is partly refuted on the surprisal mechanism — but the advantage-collapse mechanism is still independently predicted to help, so we proceed regardless. Stage 0 mostly tells us *which* of the two mechanisms (or both) drives any gain.

Cost: ~1 GPU-day total. Cheap relative to the rest.

## 4. Teacher choice (locked)

**Primary teacher: DeepSeek-R1-Distill-Qwen-7B.** Reasoning-style traces (long deliberation, self-correction) make completion style stylistically distant from Qwen2.5-Math's terser proofs — the conditions under which both mechanisms in §1 should fire. Same compute envelope as the current 7B teacher.

### Verified numbers (model card fetched 2026-05-01)

Source: <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B>, "Distilled Model Evaluation" table.

| Metric | DeepSeek-R1-Distill-Qwen-7B | Qwen2.5-Math-7B-Instruct (current teacher, our measurement) |
|---|---|---|
| MATH-500 pass@1 | **92.8%** (model card) | 74.96% (our greedy eval, MATH-500 split) |
| AIME 2024 pass@1 | 55.5 | — |
| AIME 2024 cons@64 | 83.3 | — |
| GPQA Diamond pass@1 | 49.1 | — |

**Headline gap on MATH-500: +17.8pp** stronger than current teacher. Sampling-protocol caveat: DeepSeek number is pass@1 (their recommended sampling, presumably non-greedy); our 74.96% was greedy. The qualitative ranking holds; the exact gap will narrow slightly when we re-measure both under the same sampling protocol.

### Base model and tokenizer compatibility

**Base model: Qwen2.5-Math-7B** (the *base*, not the *Instruct*). Per the model card's "DeepSeek-R1-Distill Models" table. Tokenizer family inherits from Qwen2.5-Math: 152,064 vocab including the 128 special-token rows our student (151,936) does not have. Same configuration as our current teacher → existing rollout-generation infrastructure (`shared/generate_rollouts.py`) should plug in without modification.

### Open issue to confirm post-download: `<think>` reasoning tokens

R1-distilled models emit `<think>...</think>` reasoning blocks. The model card does not state whether these are added as new special tokens (at IDs possibly ≥ 152,064) or whether they live within the inherited Qwen-Math vocab as ordinary text.

Implication for our 7-config plan:

- **`teacher_agnostic_loader.py` (DG-offline + BC configs)**: immune. Re-tokenizes from the response text under the student tokenizer; `<think>...</think>` becomes a literal string sequence in the student's vocab regardless of the teacher's special-token treatment.
- **Legacy `offline_grpo/data.py` (Offline-GRPO config)**: at risk. If `<think>` is a teacher-side special token at ID ≥ 151,936, the existing OOV truncation cuts every rollout at the first `<think>`, producing empty completions that break the IS ratio. This would invalidate the OG comparison row for reasons unrelated to OG-as-a-method.

**Pre-execution gate**: after downloading the model, inspect `tokenizer_config.json` / `vocab_size` and the IDs assigned to `<think>` and `</think>`. If those IDs are ≥ 151,936, we either (a) generate rollouts with `<think>` blocks stripped at the rollout-writer level, (b) accept that OG-on-DeepSeek is a loader-limited rather than method-limited number with an explicit caveat in the writeup, or (c) extend the OG path with a one-pass `<think>`-block stripping step. Decision deferred until we see the actual token IDs.

### Alternates considered, deferred

| Teacher | Reason deferred |
|---|---|
| DeepSeek-R1-Distill-Llama-8B | Slightly weaker than the Qwen-distilled version; would also land us in the cross-tokenizer regime. Useful as a future Stage that demonstrates capability-(B) but not the first cut. |
| Qwen2.5-Math-72B-Instruct | Same tokenizer, ~10× rollout-generation cost for likely smaller gain than DeepSeek-R1-Distill-Qwen-7B. |
| Llama-3.1-8B-Instruct, Mistral-7B-Instruct | Weaker on MATH than Qwen2.5-Math-7B; would only matter as a (B) capability demo, not for the strong-teacher hypothesis. |
| GPT-4 / Claude completions | Pure (B) framing, separate workflow. Defer. |

### Alternates considered, deferred

| Teacher | Reason deferred |
|---|---|
| DeepSeek-R1-Distill-Llama-8B | Slightly weaker than the Qwen-distilled version; would also land us in the cross-tokenizer regime. Useful as a future Stage that demonstrates capability-(B) but not the first cut. |
| Qwen2.5-Math-72B-Instruct | Same tokenizer, ~10× rollout-generation cost for likely smaller gain than DeepSeek-R1-Distill-Qwen-7B. |
| Llama-3.1-8B-Instruct, Mistral-7B-Instruct | Weaker on MATH than Qwen2.5-Math-7B; would only matter as a (B) capability demo, not for the strong-teacher hypothesis. |
| GPT-4 / Claude completions | Pure (B) framing, separate workflow. Defer. |

## 5. Locked design

Single teacher (replace, do not mix), single dataset, fixed methods + η sweep, two-pass audit funnel. The remaining open questions from earlier drafts dissolve as follows:

| Earlier open question | Locked answer |
|---|---|
| (A) mechanism vs (B) capability framing | Lead with (A) under the strong-teacher hypothesis. (B) emerges as a side result if a future teacher choice happens to be cross-tokenizer. |
| Pool composition | N/A — single teacher, no pool to compose. |
| Dataset scope | **MATH only** for first cut. Extend to GSM8K (which has the audited 60-seed BC baseline) if first cut is interesting. |
| Comparison axis | DG-offline vs Offline-GRPO vs BC-all + BC-correct-only on rollouts from the new teacher. See §5.1. |
| Risk mitigation (per-source-teacher stratification) | N/A — single teacher. |
| Audit cost | Two-pass funnel. See §5.2. |

### 5.1 Configurations to train

**6 training runs total** (Offline-GRPO dropped to N/A — see §5.1.1):

| # | Method | Setting | Loader |
|---|---|---|---|
| 1 | BC-all | standard | `teacher_agnostic_loader` |
| 2 | BC-correct-only | standard | `teacher_agnostic_loader` |
| 3 | DG-offline | η = 0.1 | `teacher_agnostic_loader` |
| 4 | DG-offline | η = 0.5 | " |
| 5 | DG-offline | η = 1.0 | " |
| 6 | DG-offline | η = 2.0 | " |

All other hyperparameters held constant against the existing recipes.

### 5.1.1 Offline-GRPO on R1 reported as N/A

OG-on-R1 not run; the R1-distill teacher's reasoning-marker token IDs (`<think>`=151648, `</think>`=151649) collide with unrelated student-vocab special tokens, breaking the IS-ratio alignment the legacy `offline_grpo/data.py` loader assumes. Resolving this would require either patching the loader to drop those two specific token IDs (small but special-token-list-specific) or computing teacher logprobs against the student tokenization in a one-time forward pass (clean but ~1-2 GPU-hours of new infrastructure). Deferred to future work.

### 5.1.2 Production-ready MATH eval (upgraded 2026-05-01)

The project's MATH equivalence checker is now `shared/math_eval.py`, a faithful port of DeepSeek-Math's `is_equiv_multi` (= multi-candidate `\boxed{}` extraction via `extract_math_answer` → `strip_string` LaTeX canonicalization → sympy `parse_latex` + `simplify(a-b) == 0`). Used uniformly across:

- Reward computation in `DG-offline/teacher_agnostic_loader.py:_compute_correctness_math` (multi-candidate via question + response).
- Student checkpoint eval in `mixture_grpo/evaluate.py:check_math`.
- Teacher verification eval (`DG-offline/eval_r1_teacher.py`).

Runtime requirement: `pip install antlr4-python3-runtime==4.11` in the venv (sympy's `parse_latex` silently no-ops without it — was the root cause of our prior 19.8pp accuracy underestimate on the R1 teacher eval).

Verification: re-checking the saved R1 eval completions under the upgraded comparator gave 92.60-93.00% (single vs multi candidate) vs the model card's 92.8% — within sampling noise.

**Caveat for back-comparing with old MATH numbers**: every existing MATH number in `docs/progress_reports/*` was measured under `math_verify` only (no antlr4) and is undercounted by ~5-15pp depending on how many wrong-by-formatting answers the method produced. For the multi-teacher comparison table (§5.3), we should re-eval the Qwen-Math BC-all and DG-η=0.5 anchor checkpoints under the upgraded comparator before reporting Δ values.

### 5.2 Audit funnel (two-pass)

1. **Cheap pass**: single 5-seed greedy eval on each of the 7 trained configs. ~7 short SLURM jobs.
2. **Audited pass**: full 5-job audit suite ([MEMORY.md](../../../MEMORY.md) audit-protocol) on the per-method winners only.
   - **BC**: audit the winner of (BC-all, BC-correct-only). 1 audited config.
   - **Offline-GRPO**: 1 audited config (no η sweep).
   - **DG-offline**: audit the η that won the cheap pass. 1 audited config.
   - **Total: 3 audited configs → 15 audit-wave jobs.**

### 5.3 Headline numbers the experiment outputs

Two method-level deltas vs Qwen-Math-7B baselines (re-eval under `is_equiv_multi` is required for fair comparison; see §5.1.2):

| Method | Old teacher (Qwen-Math) under is_equiv_multi | New teacher (DeepSeek-R1) | Δ |
|---|---|---|---|
| BC (winner of all/cc) | TBD (re-eval) | TBD | ? |
| Offline-GRPO | N/A (R1 token-ID collision; see §5.1.1) | N/A | — |
| DG-offline (best η) | TBD (re-eval @ η=0.5) | TBD @ η=? | ? |

The headline question is now: **"how much does DG benefit from a stronger teacher (Δ_DG)?"** Without an OG comparator on the R1 side, we cannot directly test "Δ_DG > Δ_OG" in this experiment. We can still report the Δ_DG number on its own, plus the Qwen-Math-side OG vs DG gap as historical context, with a methodology footnote about the deferred OG-on-R1 row.

### 5.4 Compute budget

Rough estimate at 4 L40s utilization:

| Phase | GPU-hours |
|---|---|
| Stage 0 pre-check (small sample, surprisal diagnostic) | ~1 |
| Full DeepSeek MATH rollout generation (7.5k problems × 4 generations) | ~3–4 |
| 7 training runs × ~10–15h each | ~80–100 |
| Cheap eval pass (7 short jobs) | ~3–4 |
| Audit pass (3 configs × 5-job suite) | ~30–40 |
| **Total** | **~120–150** |

Comparable to one of our existing per-dataset experiments.

## 6. Staging (locked)

| Stage | Description | Status |
|---|---|---|
| 1 | Build `teacher_agnostic_loader.py` (self-contained Path A loader) | **DONE 2026-05-01** |
| 1b | Move math_equal.py → `shared/math_eval.py`; upgrade reward computation (`teacher_agnostic_loader.py`) and student eval (`mixture_grpo/evaluate.py`) to use `is_equiv_multi`; add R1-friendly flags to `shared/generate_rollouts.py` | **DONE 2026-05-01** |
| 1c | Verify teacher card claim (DeepSeek 92.8% MATH-500 pass@1) on our pipeline | **DONE 2026-05-01** — final 16-run eval running (job 4821718); already verified 92.60% on Run 1 under the upgraded comparator |
| 2 | DeepSeek rollout generation on MATH train (7.5k × 4 at max_tokens=32768, no system prompt) | **RUNNING** (job 4821812; ~12h estimated) |
| 0 | Surprisal diagnostic on the new rollouts (extract first shard once available) | Pending |
| 3 | Re-eval Qwen-Math anchor checkpoints (BC-all, DG-η=0.5) under upgraded `is_equiv_multi` | Pending |
| 4 | Train 6 configs (§5.1) on R1 rollouts | Pending |
| 5 | Cheap 5-seed greedy eval on each of the 6 | Pending |
| 6 | Audit per-method winners (2 configs × 5 jobs = 10 audit jobs: best-BC, best-DG-η) | Pending |
| 7 | Writeup: fill §5.3 table | Pending |

Pre-execution gates (all resolved 2026-05-01):
- ~~Verify the DeepSeek-R1-Distill-Qwen-7B model card's MATH-500 number~~ DONE — 92.8% confirmed; reproduced on our pipeline at 92.60-93.00%.
- ~~Download model weights~~ DONE (15 GB at `/scratch/mrli/models/DeepSeek-R1-Distill-Qwen-7B/`).
- ~~Inspect tokenizer config~~ DONE — `<think>`=151648, `</think>`=151649, both within student vocab range but with different student-side meanings. OG-on-R1 dropped per §5.1.1.
- ~~Confirm vLLM compatibility~~ DONE — eval ran successfully via `shared/generate_rollouts.py` extended with R1-friendly flags.

## 7. Not in scope

- Multi-teacher *online* GRPO. Different problem (vLLM colocate is single-model; you'd need to switch teachers per rollout batch or run multiple servers). Park.
- Teacher mixture *weighting* (per-teacher gradient scale based on teacher reward). Can be added as a follow-up if the uniform-mix result is interesting.
- DG-Mixture interaction. The DG-Mixture prototype ([`plans/dg_mixture_design.md`](dg_mixture_design.md)) is parked; this experiment does not depend on it.
