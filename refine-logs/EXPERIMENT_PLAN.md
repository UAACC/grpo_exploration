# Experiment Plan

**Problem**: Offline policy learning from a bad-teacher rollout (35.1% correct) on MATH. BC baseline is 0.3191. Completion-level delight gating (V2) reached 0.3247 at best — +0.6 pp, not statistically significant.

**Method Thesis**: Per-token delight-guided PG masking (SoftDG-TM) focuses the policy gradient on surprising, informative tokens while preserving KL regularization across the full completion, outperforming both BC and completion-level gating by extracting signal from partially-correct demonstrations.

**Date**: 2026-06-19

**Implementation Spec**: `refine-logs/EXPERIMENT_PLAN_2026-06-19.md` (code layout, token budget, calibration protocol, output format — not duplicated here)

---

## Prior Results (Context)

| System | MATH Acc | Δ vs BC | Status |
|--------|----------|---------|--------|
| BC-all (48K completions) | 0.3191 ± 0.0081 | — | Baseline |
| SoftDG completion-level, best (C002, zero_two, thr=0.5392) | 0.3247 ± 0.0089 | +0.6 pp | Not significant |
| SoftDG completion-level, signed+thr=0.5 (M003) | 0.3109 ± 0.0095 | −0.8 pp | Below BC |

**Key prior finding**: `zero_two + raw_reward` outperforms `signed + raw_reward`; negative signal on wrong completions is actively harmful; at threshold=0.5 correct completions are selected with kr_correct=1.0, kr_wrong=0.0.

**Target**: ≥ 0.3391 (BC + 2 pp)

---

## Claim Map

| # | Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|-------|----------------|-----------------------------|---------------|
| C1 | Token-mask SoftDG beats BC by ≥ 2 pp | Primary contribution — offline DG is useful beyond imitation | Best variant ≥ 0.3391 | B1, B4 |
| C2 | Token-level masking extracts more signal than completion-level gating | Novelty of token-mask over prior SoftDG work | Best token-mask variant > C002 (0.3247) | B2 |
| Anti-C | The gain is not just from selecting correct completions | Token masking must do more than correct-only BC | Best token-mask variant > BC-correct | B3 |

---

## Paper Storyline

**Main paper must prove**:
- C1: Table 1 — BC vs token-mask variants across best thresholds
- C2: Table 1 (add C002 as prior SoftDG row) or Section 4.2
- Anti-C: Table 1 (add BC-correct row) — refutes the trivial explanation

**Appendix can support**:
- Threshold sensitivity curve (B4) — shows the method is robust, not threshold-hacked
- Reward coding breakdown (B5) — explains which variant works and why
- Failure case analysis (B6) — qualitative diagnosis

**Experiments intentionally cut**:
- Multi-seed runs before any variant beats BC-level significance threshold
- Comparison against online GRPO (out of scope: this paper is about offline methods)
- Alternative backbone sizes (only Qwen2.5-0.5B)

---

## Experiment Blocks

### Block 1: Main Anchor Result — Token-Mask SoftDG vs BC
- **Claim tested**: C1
- **Why this block exists**: Primary paper table; establishes whether token-mask DG is worth reporting
- **Dataset / split / task**: MATH, fixed rollout `rollouts_math_Qwen2.5-0.5B-Instruct_0.6_pick4.jsonl`, 500-problem eval set
- **Compared systems**:
  - BC-all (existing, 0.3191) — no rerun
  - SoftDG-TM Variant A: dr_grpo + signed + raw_reward
  - SoftDG-TM Variant B: grpo + zero_two + advantage *(run only if calibration finds valid thresholds)*
  - SoftDG-TM Variant C: dr_grpo + signed + advantage *(run only if calibration finds valid thresholds)*
- **Metrics**: MATH accuracy (primary), Δ vs BC, avg response length, final KL, effective token budget hit
- **Setup details**: Qwen2.5-0.5B, LoRA r=32 α=32 all-linear, LR=3e-6, 4×L40S, max 20 epochs, early stop at token budget, best calibrated threshold per variant
- **Success criterion**: Any variant ≥ 0.3391 (C1 defended). Partial success: any variant > 0.3247 (C2 defended, C1 not)
- **Failure interpretation**: If no variant beats C002 (0.3247), token-mask provides no benefit over completion-level gating → paper claim must be revised to "negative result" or method must be redesigned
- **Table / figure target**: Table 1 (main result)
- **Priority**: MUST-RUN

### Block 2: Novelty Isolation — Token-Level vs Completion-Level Gating
- **Claim tested**: C2
- **Why this block exists**: Establishes that token-mask is a meaningful step beyond prior SoftDG (V2)
- **Dataset / split / task**: Same as B1
- **Compared systems**:
  - Best token-mask variant from B1
  - C002 (completion-level gating, zero_two+raw_reward, thr=0.5392, kr_correct=0.70) — **existing result, no new run**
  - M003 (completion-level, signed+raw_reward, thr=0.5, kr_correct=1.0) — existing
- **Metrics**: MATH accuracy, Δ vs BC, token keep rate (for comparability)
- **Setup details**: Already done for completion-level; B1 provides token-level
- **Success criterion**: Best token-mask > 0.3247 (C2 defended); ideally ≥ 0.3391 (C1 also defended)
- **Failure interpretation**: Token-mask ≤ completion-level gating → fine-grained selection has no benefit; paper must explain why and revise claims
- **Table / figure target**: Table 1 (same table, add "Prior SoftDG (completion-level)" row)
- **Priority**: MUST-RUN (zero additional GPU cost — uses existing C002 result)

### Block 3: Simplicity Check — BC-Correct Baseline
- **Claim tested**: Anti-C (refutation)
- **Why this block exists**: The obvious alternative to token masking is just training BC on correct completions only. If BC-correct ≥ token-mask, then the complexity of gating is unnecessary.
- **Dataset / split / task**: Same rollout, correct completions only (35.1% of 48K = ~16.8K completions)
- **Compared systems**:
  - BC-correct (new run)
  - BC-all (existing)
  - Best token-mask from B1
- **Metrics**: MATH accuracy, avg response length
- **Setup details**: Use `SoftDG-Token-Mask-Offline/run_bc.sh`, filter rollout to correct completions only, 1 epoch (equivalent data exposure to BC-all)
- **Success criterion**: Token-mask > BC-correct AND token-mask > BC-all → token masking adds value beyond filtering
- **Failure interpretation**: If BC-correct ≥ token-mask → token masking is needlessly complex; simpler correct-only BC dominates; revise paper to show this
- **Table / figure target**: Table 1 (add "BC-correct" row between BC-all and SoftDG-TM)
- **Priority**: MUST-RUN

### Block 4: Threshold Sensitivity for Best Variant
- **Claim tested**: C1 (robustness)
- **Why this block exists**: Shows the method is not sensitive to threshold; prevents claim that the result is cherry-picked
- **Dataset / split / task**: Same as B1
- **Compared systems**: Best variant from B1 at all calibrated candidate thresholds (expected 3–6 thresholds)
- **Metrics**: MATH accuracy vs threshold; token keep rate vs threshold
- **Setup details**: Same as B1 but sweep thresholds from calibration output; 1 seed each
- **Success criterion**: Best threshold ≥ 0.3391 and plateau visible (not a single-point spike)
- **Failure interpretation**: If accuracy peaks sharply at one threshold and drops elsewhere, the method requires careful tuning and the claim is weakened
- **Table / figure target**: Appendix Figure (threshold vs accuracy curve)
- **Priority**: MUST-RUN (needed to select the final threshold used in B1 main table)

### Block 5: Reward Coding Ablation — Variant Comparison
- **Claim tested**: Supporting (explains which component drives the gain)
- **Why this block exists**: If B1 works, we need to explain why Variant A, B, or C wins
- **Dataset / split / task**: Same as B1
- **Compared systems**: A vs B vs C at matched threshold (thr=0.5 or closest valid)
- **Metrics**: MATH accuracy, token keep rate by signal sign (correct/wrong), final KL
- **Setup details**: Only run variants not already covered by B1
- **Success criterion**: One variant clearly better than others, consistent with theory (signed+raw_reward best because non-zero signal without normalization collapse)
- **Failure interpretation**: If all three variants perform similarly, reward coding doesn't matter; if signed+raw_reward is worst, the theory needs revision
- **Table / figure target**: Appendix Table (reward coding ablation)
- **Priority**: NICE-TO-HAVE (run after B1/B3 confirm any variant is promising)

### Block 6: Failure Analysis and Qualitative Diagnosis
- **Claim tested**: Scope limitation (anti-oversell)
- **Why this block exists**: Identifies what token masking still misses; shows scientific rigor
- **Dataset / split / task**: MATH eval problems where best variant fails but BC succeeds
- **Compared systems**: Best token-mask vs BC-all on problem subsets (by difficulty, problem type)
- **Metrics**: Per-problem accuracy breakdown; average completion length on failure cases
- **Setup details**: Post-hoc analysis on saved eval outputs; no additional training
- **Success criterion**: Clear qualitative pattern (e.g., token masking underperforms on long-form proofs vs short arithmetic)
- **Failure interpretation**: No clear pattern → just variance; still reportable but less insightful
- **Table / figure target**: Appendix Section (qualitative analysis)
- **Priority**: NICE-TO-HAVE

---

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M0: Gate calibration | Calibrate thresholds for A, B, C | Calibrate 3 variants, no training | Any variant with valid thresholds → proceed; if zero valid → redesign gating formula | ~30 min, 1×L40S | Variant B (zero_two+advantage) likely has no valid thresholds based on prior calibration |
| M1: Sanity check | Verify no DDP hang; PG mask is nontrivial | 3 tiny runs (max_steps=1 or target=32) | PG mask nonzero, KL on all tokens, no hang → proceed | ~15 min each on 4×L40S | None expected |
| M2: BC-correct baseline | Measure BC on correct completions only | 1 run, 1 epoch | — (no decision gate; used later in B3 comparison) | ~2 GPU-hours on 4×L40S | Low |
| M3: Main method (1 seed) | Full training for each valid variant at best calibrated threshold | 1–3 runs depending on valid variants | Any variant > 0.3247 → run threshold sweep; all below 0.3247 → STOP, report negative result | ~4 GPU-hours per run on 4×L40S | Method may not beat completion-level gating |
| M4: Threshold sweep | Find the best threshold for the best M3 variant | 3–5 runs at calibrated thresholds | Best threshold ≥ 0.3391 → report C1; best < 0.3391 → report partial improvement or negative | ~4 GPU-hours per threshold on 4×L40S | Accuracy may plateau below target |
| M5: Second seed (optional) | Confirm best M4 result is not a lucky seed | 1–2 runs, same config as best M4 | Only run if M4 best ≥ 0.3391 and variance is high | ~4 GPU-hours per seed | Low priority |
| M6: Ablations and failure analysis | Appendix material | Reward coding ablation + eval breakdown | No gate; run after M3/M4 are done | ~4 GPU-hours for ablation | Low — appendix only |

**Must-run in order**: M0 → M1 → M2 (can overlap with M1) → M3 → M4
**Nice-to-have**: M5, M6

---

## Stop/Go Gates

| Gate | Condition | Action if PASS | Action if FAIL |
|------|-----------|----------------|----------------|
| G0 | M0: ≥1 variant has valid threshold | Proceed to M1 | Redesign gating formula or lower eta |
| G1 | M1: No DDP hang; PG mask nontrivial | Proceed to M2+M3 | Debug trainer; check DDP reduce + mask |
| G2 | M3: Best variant > C002 (0.3247) | Run full threshold sweep (M4) | Report negative: token-mask ≤ completion-level gating |
| G3 | M4: Best threshold ≥ 0.3391 | Report C1 defended; optionally run M5 | Report partial: token-mask > completion-level but < BC+2pp |

---

## Compute and Data Budget

- **Calibration**: ~30 min, 1×L40S (~0.5 GPU-hours)
- **Sanity checks**: 3 × 15 min × 4 GPUs = ~3 GPU-hours
- **BC-correct**: ~1 epoch × 4 GPUs = ~2 GPU-hours
- **M3 main sweep**: up to 3 variants × 4 GPU-hours = ~12 GPU-hours
- **M4 threshold sweep**: 5 thresholds × 4 GPU-hours = ~20 GPU-hours
- **M5 extra seeds**: 2 seeds × 4 GPU-hours = ~8 GPU-hours (only if needed)
- **M6 ablations**: ~4 GPU-hours
- **Total estimated GPU-hours**: ~35–45 GPU-hours (M0–M4) without M5/M6
- **Data preparation needs**: None — rollout file exists at `/scratch/shuai14/rollouts/math_teacher/rollouts_math_Qwen2.5-0.5B-Instruct_0.6_pick4.jsonl`
- **Human evaluation needs**: None — automated MATH eval
- **Biggest bottleneck**: M0 calibration results determine which variants to run; must complete before M3

---

## Risks and Mitigations

- **Variant B (zero_two+advantage) has no valid thresholds**: Prior calibration found no valid thresholds for this config at the token level. Calibration at M0 will confirm. If no valid thresholds, drop Variant B — the paper story does not require it.
- **Token-mask does not beat completion-level gating (C002)**: If M3 results fall below 0.3247, the contribution is negative. Paper becomes a negative result / analysis paper. Decision at G2.
- **BC-correct outperforms token-mask**: If BC-correct ≥ best token-mask, the complexity of gating is unwarranted. This is informative (BC-correct is a strong simple baseline) and should be reported as the B3 finding.
- **Signed negative signal still harmful in token-mask context**: The V2 result showed negative signal (signed) hurt at completion level. In token-mask mode, wrong completions would contribute a negative PG signal on low-surprisal tokens. This might still be harmful. Variant A (raw_reward=±1, no normalization) is the highest-risk variant for collapse; monitor avg response length.
- **DDP token-count tensor sync issues**: Per-token masking with variable keep rates across GPUs requires careful DDP reduction. Sanity check (M1) validates this before full training.

---

## Final Checklist

- [ ] Main paper table covers BC-all, BC-correct, SoftDG completion-level (C002), SoftDG token-mask best variant
- [ ] Novelty is isolated: token-level vs completion-level comparison is in the main table
- [ ] Simplicity is defended: BC-correct row is in the main table
- [ ] Frontier contribution: This method is intentionally non-frontier (no LLM-in-the-loop, no large teacher model); frontier justification block is skipped
- [ ] Nice-to-have runs are separated from must-run runs
- [ ] Calibration is gating M3 — no training before G0 passes
- [ ] KL regularization on all tokens is verified at M1 sanity stage
