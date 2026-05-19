# Results Snapshot: May 19, 2026

Phase-wrap snapshot of all current experimental results.

**Why this exists.** Prior reports had numbers measured under a broken comparator (`math_verify` with a missing `antlr4` dependency that silently no-op'd `sympy.parse_latex`). Those numbers undercounted MATH accuracy by 5-15pp project-wide and slightly affected the rest. This snapshot replaces all comparable numbers with their calibrated versions under `Math_Verifier` (DeepSeek-Math port). It also folds in the new methods (DG-MADz, AWBC, two-stage with corrected merge base) and the multi-teacher experiment (DeepSeek-R1-Distill-Qwen-7B) added this phase.

**Audit protocol** (where labeled "calibrated"): 60-seed pooled greedy (wave-1 RUNS=30 + wave-2 RUNS=30 at SEED=42 and 100, different nodes) and single-seed pass@1 at T=0.6.

## Calibration status

| Dataset | Calibrated | Notes |
|---|---|---|
| MATH-500 (Qwen-Math era) | Yes, all rows | Most rows shifted +3-8pp from broken-comparator era |
| MATH-500 (R1-Distill era) | Yes, all rows | New experiment, calibrated from the start |
| GSM8K | Yes, all rows | Most rows shifted ±0.7pp (numeric extraction, less affected by `parse_latex` bug) |
| SVAMP | AWBC only | Other rows still pre-calibration (from `reeval-*` logs, 2026_04_17). Numeric-extraction dataset, expected shift small |
| ASDiv | AWBC only | Same status as SVAMP |

## Theory: DG-MADz logic chain and derivation

### Starting point

DG-offline replaces the IS ratio in offline GRPO with a sigmoid gate on χ:

```
gate = σ(η · χ),   where   χ = advantage · surprisal
advantage A = (r − μ_group) / (σ_group + ε),       group-normalized within rollout group
surprisal s = −log π_student(token | context),     non-negative
loss_token = − gate · A · log π_student(token | context)
```

The intent is a four-quadrant filter on `(sign(A), sign(s))`:
- A > 0 and s large: positive correct rollout, surprising token. Want gate → 1 (amplify).
- A < 0 and s large: negative wrong rollout, surprising token. Want gate → 0 (suppress).
- A or s near zero: no clear signal. Gate ≈ 0.5 (no opinion).

For the filter to *act*, σ(η · χ) needs to **saturate** on the strong-signal cases. Saturation requires |η · χ| ≳ 2, so that σ(2) ≈ 0.88 and σ(−2) ≈ 0.12. The implicit assumption: strong-signal χ values have magnitude near O(1) on a natural scale, and η is a knob tuning that scale into the saturating regime.

### Empirical diagnostic that breaks the assumption

Measured the χ distribution across all 4 teacher rollout datasets (MATH, GSM8K, SVAMP, ASDiv). Two pathologies:

**(i) Sparsity at zero.** 85-94% of tokens have χ = 0, because their rollout group has zero reward variance. With binary rewards r ∈ {0, R_max} and a group of K rollouts, either all K are correct (μ_group = R_max, σ_group = 0) or all K are wrong (μ_group = 0, σ_group = 0) or a mix. The all-correct and all-wrong cases give A = 0 by construction. These tokens contribute nothing to either the gate or the loss.

**(ii) Heavy tail on the active subset.** The remaining 6-15% "active" subset (|A| > 0) has std/MAD ratios of 11-22× across the 4 datasets. For reference: a Gaussian distribution has std/MAD ≈ 1.48. An 11-22× ratio means the std is dominated by a thin outlier tail, not by the bulk of the distribution. The active χ is heavy-tailed and negatively skewed.

### Why (ii) breaks the gate

Suppose η is chosen by anchoring to the empirical std of χ_active, the natural calibration choice. Call this η_std. Then for the typical (bulk) active sample whose |χ| is at the *scale of MAD*, we have |η_std · χ| ≈ MAD/std. With std/MAD = 11-22×, this is **|η · χ| in the range 0.045 to 0.091**, deep in the sigmoid's linear regime where σ(x) ≈ 0.5 + x/4.

Empirical measurement under the published recipe (η ∈ {0.1, 0.5, 1.0, 2.0}): 88-99% of active samples land in the sigmoid's linear regime within ±0.1 of 0.5. The four-quadrant filter does not fire. On the active subset, the gate had degenerated into roughly a constant 0.5× multiplier, with occasional saturation only from the rare tail.

**Diagnosis**: DG's analysis implicitly assumed a χ distribution where the typical case sits near saturation distance from zero. The empirical χ distribution is heavy-tailed, so any scaling that uses std (the natural choice for "what's a typical magnitude") is anchored to the tail rather than the bulk. The bulk ends up in the linear regime, the gate degenerates.

### The MAD-z fix

Replace χ with its **robust z-score**:

```
z = (χ − median(χ_active)) / (1.4826 · MAD(χ_active))
MAD = median(|χ_i − median(χ_active)|)
```

The factor 1.4826 = 1/0.6745 is the consistency constant. For Gaussian-distributed data, MAD = 0.6745 · σ, so 1.4826 · MAD ≈ σ exactly. The factor makes the MAD-based scale comparable to the std-based scale on Gaussian inputs while remaining robust to outliers on non-Gaussian inputs.

**Why MAD specifically**: MAD is the prototypical robust scale estimator with a 50% breakdown point (one would need to perturb 50% of the data to corrupt MAD), while std has a 0% breakdown point (a single arbitrarily large value can blow it up without bound). For heavy-tailed χ, MAD measures the typical spread of the bulk and std measures the tail.

After MAD-z, the typical active sample has |z| ≈ 1 in MAD units, by construction: median(|z|) = 1 because median(|χ − median|) = MAD. The gate becomes σ(z/η). With η ≈ 1, the typical active sample lands at σ(±1) ≈ 0.27 or 0.73, comfortably in the nonlinear regime. With slightly higher η (sharper gate), the typical strong-signal case saturates.

**Empirical confirmation**: under DG-MADz with η=1.0, the gate distribution post-normalization is bimodal near 0.2 and 0.8 instead of unimodal near 0.5 (as it was under raw scaling). The four-quadrant filter is firing.

### Result and what it implies

DG-MADz η=1.0 on GSM8K: 49.85 ± 0.29% greedy (60-seed pool).
Prior DG-offline best (η=0.1) on GSM8K under same calibration: 50.16 ± 0.56%.
BC-correct-only on GSM8K under same calibration: 50.21 ± 0.38%.

All three within 0.5pp of each other, all three ~2pp above untrained baseline 47.92%.

**Fixing the gate did not move the headline number.** Combined with the AWBC ablation (drop the gate entirely, keep only the group-normalized advantage; same cluster on all 4 datasets), this is direct evidence that the gate is not the bottleneck. It does not matter whether the gate is degenerated to flat 0.5× (raw scaling) or saturating as designed (MAD-z). Accuracy stays in the BC/DG/OG/AWBC cluster either way.

### Logical closure

1. DG-offline's nominal contribution is the sigmoid gate on χ.
2. Under raw scaling the gate is degenerate (not firing on 88-99% of active samples). DG cannot have been winning by virtue of the gate.
3. MAD-z makes the gate fire as designed. Accuracy does not move.
4. AWBC strips the gate entirely. Accuracy does not move.
5. The advantage signal, with or without the gate, with or without IS or PPO clip (OG vs AWBC), pulls the same modest amount of training signal out of the active subset on all 4 datasets.
6. Therefore the bottleneck is upstream of the gate, in (i): with 85-94% of tokens at A = 0, any advantage-based reweighting on the active 6-15% has limited room to help. **The bottleneck is signal density, not gating.**

MAD-z's narrow contribution is this proof by elimination. It removes (ii) (the gate not firing) as an explanation for DG's behavior, isolating (i) (the A = 0 deadweight) as the load-bearing constraint.

## MATH-500, Qwen-Math era teacher (all calibrated, 60-seed greedy unless noted)

| Method | Greedy ± std | Pass@1 |
|---|---|---|
| Untrained baseline | 33.18 ± 0.74 | 30.80 |
| BC-all | 33.21 ± 0.43 | 30.20 |
| BC-correct-only | 31.13 ± 0.34 | 28.80 |
| Offline GRPO | 30.96 ± 0.50 | 27.20 |
| DG-offline η=0.1 | 33.21 ± 0.18 (n=30) | 30.80 |
| DG-offline η=0.5 | 31.60 ± 0.27 | 31.80 |
| DG-offline η=1.0 | 33.08 ± 0.25 (n=30) | 31.60 |
| DG-offline η=2.0 | 31.35 ± 0.42 | 31.80 |
| **Two-stage (DG → Online)** | **37.75 ± 0.34** | 36.40 |
| Online GRPO | 36.66 ± 0.56 | 36.60 |
| **AWBC** (new this phase) | 32.12 ± 0.45 | 31.20 |
| Teacher (Qwen2.5-Math-7B-Instruct) | 83.18 ± 0.28 | 80.80 |

**Best 0.5B method**: Two-stage at 37.75 (+4.57pp over baseline). Online GRPO at 36.66 (+3.48pp). Every other trained method sits within ±2pp of the 33.18 untrained baseline. The DG-vs-BC gap that the original methodology relied on does not survive under the corrected comparator.

## MATH-500, DeepSeek-R1-Distill-Qwen-7B teacher (all calibrated)

| Method | Greedy ± std | Pass@1 |
|---|---|---|
| Untrained baseline (same as above) | 33.18 ± 0.74 | 30.80 |
| BC-all R1-Distill | 10.29 ± 0.79 (n=25) | 13.80 |
| BC-correct-only R1-Distill | 11.45 ± 0.46 (n=24) | 13.20 |
| DG-offline η=0.1 R1-Distill | 11.81 ± 0.51 (n=23) | 13.20 |
| DG-offline η=0.5 R1-Distill | 5.89 ± 0.24 (n=21) | 10.00 |
| DG-offline η=1.0 R1-Distill | 11.58 ± 0.77 (n=24) | 15.20 |
| **DG-offline η=2.0 R1-Distill** | **34.30 ± 0.27** | 29.20 |
| **OG-on-R1-Distill** | **33.57 ± 0.87** | 32.40 |
| Teacher (R1-Distill-Qwen-7B) | 92.6-93.0 (verified, 16-seed) | n/a |

**Best 0.5B method**: DG η=2.0 at 34.30 (+1.12pp over baseline). OG-on-R1-Distill at 33.57 (≈ baseline). Five of seven trained methods collapse to 6-12% greedy (well below the 33.18 baseline) via verbosity-induced truncation: they imitate R1-Distill's long chain-of-thought, run out of `max_tokens=16384` budget before emitting `\boxed{}`, and never produce a final answer.

## GSM8K (all calibrated, 60-seed greedy)

| Method | Greedy ± std | Pass@1 |
|---|---|---|
| Untrained baseline | 47.92 ± 0.44 | 43.44 |
| BC-all | 49.55 ± 0.27 | 47.16 |
| BC-correct-only | **50.21 ± 0.38** | 47.54 |
| Offline GRPO | 48.66 ± 0.55 | 43.67 |
| DG-offline η=0.1 | **50.16 ± 0.56** | 48.52 |
| DG-offline η=0.5 | 48.58 ± 0.23 | 47.01 |
| DG-offline η=1.0 | 48.61 ± 0.42 | 46.17 |
| DG-offline η=2.0 | 47.73 ± 0.60 | 46.47 |
| **Two-stage (DG → Online)** | 48.59 ± 0.32 | 47.16 |
| Online GRPO (MATH ckpt, cross-task) | **50.32 ± 0.16** | 48.45 |
| **AWBC** (new) | 50.21 ± 0.44 | 44.96 |
| DG-MADz η=0.5 (new) | 49.34 ± 0.10 | 49.66 |
| **DG-MADz η=1.0** (new) | **49.85 ± 0.29** | 47.01 |
| DG-MADz η=2.0 (new) | 48.70 ± 0.12 | 45.79 |
| DG-MADz η=5.0 (new) | 48.70 ± 0.22 | 45.26 |
| Teacher (Qwen2.5-Math-7B-Instruct) | 95.65 ± 0.09 | 95.22 |

**Best 0.5B trained methods cluster within 0.5pp**: BC-cc 50.21, DG-η=0.1 50.16, Online cross-task 50.32, AWBC 50.21, DG-MADz η=1.0 49.85. All ~2pp above the 47.92 untrained baseline. The advantage signal does what work there is to do on GSM8K; DG's gate, OG's IS/clip, and MAD-z's normalization all land in the same cluster.

## SVAMP, ASDiv (mixed calibration: AWBC new, rest pre-calibration)

| Method | SVAMP greedy | ASDiv greedy | Calibrated |
|---|---|---|---|
| Untrained baseline | 58.92 | 73.58 | No |
| BC-all | 65.67 | 74.36 | No |
| BC-correct-only | 65.32 | 74.27 | No |
| Offline GRPO | 65.62 | 75.49 | No |
| DG-offline (best η) | 65.93 (η=0.1) | 76.23 (η=2.0) | No |
| **AWBC** (new this phase) | **64.01 ± 0.15** | **74.88 ± 0.74** | Yes |
| Online GRPO (on-task) | 69.62 | 81.04 | No |
| Teacher (Qwen2.5-Math-7B-Instruct) | 90.66 | 93.32 | No |

AWBC on these two datasets sits in the BC/DG cluster, same shape as on MATH and GSM8K. The mixed-calibration caveat: the 1-2pp gap between AWBC (calibrated) and BC / DG-best (pre-calibration) on SVAMP could be a real method difference or could be the small comparator shift seen on GSM8K. SVAMP/ASDiv full recalibration is on the next-steps list.

## Best 0.5B method per dataset (single-view)

| Dataset | Best method | Greedy | Gain over baseline |
|---|---|---|---|
| MATH (Qwen-Math) | Two-stage | 37.75 | +4.57pp |
| MATH (R1-Distill) | DG-offline η=2.0 | 34.30 | +1.12pp |
| GSM8K | Online GRPO (cross-task) | 50.32 | +2.40pp |
| SVAMP | Online GRPO (on-task) | 69.62 (uncalib) | +10.70pp |
| ASDiv | Online GRPO (on-task) | 81.04 (uncalib) | +7.46pp |

**Across all 5 dataset/teacher combinations, the best trained 0.5B method is either Online GRPO or a two-stage pipeline that incorporates online GRPO.** Pure offline supervised methods (BC, OG, DG) never exceed baseline by more than 1-2pp on any dataset.

## What changed vs prior snapshot (2026_04_12)

| Change | Impact |
|---|---|
| Comparator fix (`Math_Verifier` replaces broken `math_verify`) | MATH shifted +3-8pp; GSM8K ±0.7pp; SVAMP/ASDiv likely small (not yet recalibrated) |
| DG-vs-BC gap on MATH | Dissolves: pre-calibration showed DG-best (29.00) > BC-all (27.40) by +1.60pp; calibrated they tie at 33.21 |
| Two-stage correction (right merge base) | MATH 37.56 → 37.75 (negligible shift); GSM8K 45.00 → 48.59 (recovered 3.6pp; remaining 2pp gap to old 50.61 is eval-pipeline drift) |
| Online cross-task GSM8K label corrected | Old reports' 55.82 was actually `online_grpo_gsm8k_merged_step7000` (on-task, checkpoint wiped in March 12 path migration). The genuine MATH-trained cross-task number is 50.32. |
| New methods this phase | DG-MADz (4 etas on GSM8K), AWBC (4 datasets), all calibrated |
| New experiment | Multi-teacher with R1-Distill (7 student configs trained on MATH, all calibrated) |
| Audit-suite simplification | Dropped pass@16; current protocol is 2 jobs per chain (60-seed greedy + 1-sample pass@1) |
| Eval pipeline robustness | SIGALRM timeout in `check_math` to handle pathological untrained-baseline outputs |

## Open caveats

- SVAMP and ASDiv non-AWBC rows are still pre-calibration. Numeric extraction makes the expected shift small (~0-1pp based on GSM8K behavior), but until those audits run the SVAMP/ASDiv ranking is uncertain at the 1pp level.
- Two MATH DG rows (η=0.1, η=1.0) are 30-seed only because wave-2g never ran (wave-1 hung in the dropped pass@16 phase). Within-job std on those rows is ≤0.25pp so 30 seeds is adequate precision; just not 60.
- Two-stage on R1-Distill has not been trained yet. The natural follow-up is to take DG-η=2.0-R1-Distill (34.30, the only R1-Distill method that meaningfully exceeds baseline) and continue with online GRPO from that warm start, to test whether online RL escapes the verbosity trap that all the supervised R1-Distill methods fall into.
- Online GRPO on MATH uses a merged-only checkpoint (`/scratch/mrli/merged/online_grpo_math_merged`) whose source LoRA was deleted before the path migration. The model is intact and reproducible from this merge, but not re-trainable.
- The "online GRPO on-task" numbers on SVAMP and ASDiv are post-cap-fix (max_tokens 512 → 1024) but still pre-calibration.
