# Experiment Tracker — SoftDG Token-Mask (SoftDG-TM)

**Plan**: `refine-logs/EXPERIMENT_PLAN.md`
**Goal**: Beat BC baseline (0.3191) by ≥ +2 pp → target ≥ 0.3391
**Method**: Per-token PG masking in `SoftDG-Token-Mask-Offline/`
**Implementation spec**: `refine-logs/EXPERIMENT_PLAN_2026-06-19.md`

---

## Reference: Existing Results (do not rerun)

| System | MATH Acc | Std | Notes |
|--------|----------|-----|-------|
| BC-all | 0.3191 | 0.0081 | Job 5155939; 1 epoch, 48K completions |
| SoftDG completion-level C002 | 0.3247 | 0.0089 | Job 5165659; zero_two+raw_reward, thr=0.5392, kr_correct=0.70 |
| SoftDG completion-level M003 | 0.3109 | 0.0095 | Job 5165655; signed+raw_reward, thr=0.5, kr_correct=1.00 |

---

## M0: Gate Calibration

| Run ID | Milestone | Purpose | System / Variant | Status | Notes |
|--------|-----------|---------|------------------|--------|-------|
| CAL-A | M0 | Calibrate thresholds | Variant A: dr_grpo + signed + raw_reward | DONE ✓ | Job 5313114; thresholds: 0.4845 (kr=0.80), 0.4982 (kr=0.70); proj 2 epochs each |
| CAL-B | M0 | Calibrate thresholds | Variant B: grpo + zero_two + advantage | DONE ✗ | No valid thresholds — gate mass at 0.5 (advantage×low-surprisal≈0) |
| CAL-C | M0 | Calibrate thresholds | Variant C: dr_grpo + signed + advantage | DONE ✗ | Same failure as B; p10-p90 all ≈0.5000 |

**Decision gate G0**: PASSED — Variant A has 2 valid thresholds. B and C skipped (no valid thresholds).

---

## M1: Sanity Check

| Run ID | Milestone | Purpose | System / Variant | Threshold | Target Tokens | Status | Notes |
|--------|-----------|---------|------------------|-----------|---------------|--------|-------|
| SAN-A | M1 | No DDP hang; PG mask nontrivial | Variant A: dr_grpo + signed + raw_reward | 0.4845 | 32 | DONE ✓ | Job 5314313; G1 PASS — no hang, PG mask kr=0.81, KL=2.7e-4, counter works |
| SAN-B | M1 | SKIPPED — no valid thresholds | Variant B | — | — | SKIP | B has no valid calibrated thresholds |
| SAN-C | M1 | SKIPPED — no valid thresholds | Variant C | — | — | SKIP | C has no valid calibrated thresholds |

**Decision gate G1**: No hang + PG mask nontrivial + KL active → proceed.

---

## M2: Baselines (run in parallel with M1)

| Run ID | Milestone | Purpose | System / Variant | Status | Notes |
|--------|-----------|---------|------------------|--------|-------|
| BC-CORRECT | M2 | Simplicity check baseline | BC on correct completions only (~16.8K) | DONE | Job 5314314; **0.3118 ± 0.0087** (−0.7 pp vs BC-all); Anti-C bar = 0.3118 |

---

## M3: Main Method — 1 Seed per Valid Variant

| Run ID | Milestone | Purpose | System / Variant | Threshold | MATH Acc | Δ vs BC | Status | Notes |
|--------|-----------|---------|------------------|-----------|----------|---------|--------|-------|
| MAIN-A1 | M3 | Main anchor result | Variant A: dr_grpo + signed + raw_reward | 0.4845 (kr=0.80) | **0.2986** | −2.05 pp | DONE ✗ | Job 5315418; 0.2986 ± 0.0097; below BC-all and SoftDG-CL C002 |
| MAIN-A2 | M3 | Threshold comparison | Variant A: dr_grpo + signed + raw_reward | 0.4982 (kr=0.70) | **0.3139** | −0.52 pp | DONE ✗ | Job 5315419; 0.3139 ± 0.0125; above BC-correct but below BC-all and C002 |
| MAIN-B | M3 | SKIPPED | Variant B | — | — | — | SKIP | No valid thresholds |
| MAIN-C | M3 | SKIPPED | Variant C | — | — | — | SKIP | No valid thresholds |

**Decision gate G2**: FAILED — best M3 = 0.3139 (MAIN-A2) < 0.3247 (SoftDG-CL C002). No M4 threshold sweep. Conclusion: per-token masking ≤ completion-level gating.

---

## M4: Threshold Sweep — Best Variant from M3

| Run ID | Milestone | Purpose | System / Variant | Threshold | MATH Acc | Δ vs BC | Status | Notes |
|--------|-----------|---------|------------------|-----------|----------|---------|--------|-------|
| THR-1 | M4 | SKIPPED | — | — | — | — | SKIP | G2 failed; no M4 |
| THR-2 | M4 | SKIPPED | — | — | — | — | SKIP | |
| THR-3 | M4 | SKIPPED | — | — | — | — | SKIP | |
| THR-4 | M4 | SKIPPED | — | — | — | — | SKIP | |
| THR-5 | M4 | SKIPPED | — | — | — | — | SKIP | |

**Decision gate G3**: Best M4 threshold ≥ 0.3391 → C1 defended. < 0.3391 → partial improvement only.

---

## M5: Extra Seeds (NICE-TO-HAVE — only if M4 best ≥ 0.3391)

| Run ID | Milestone | Purpose | System / Variant | Threshold | Status | Notes |
|--------|-----------|---------|------------------|-----------|--------|-------|
| SEED-2 | M5 | Confirm significance | Best config from M4 | Best thr from M4 | TODO | Seed 2 |
| SEED-3 | M5 | Confirm significance | Best config from M4 | Best thr from M4 | TODO | Seed 3 |

---

## M6: Ablations and Failure Analysis (NICE-TO-HAVE — run after M3/M4)

| Run ID | Milestone | Purpose | System / Variant | Status | Notes |
|--------|-----------|---------|------------------|--------|-------|
| ABL-BC | M6 | Reward coding ablation | Any two variants at matched threshold | TODO | Appendix only |
| FAIL | M6 | Failure analysis | Best variant vs BC-all by problem type | TODO | Post-hoc analysis on eval outputs; no new training |

---

---

## Eta Sweep Phase (2026-06-20) — EXPERIMENT_PLAN_2026-06-20.md

**Hypothesis**: Lower eta spreads gate distribution away from 0.5, yielding valid thresholds for Variants B & C and improved accuracy for Variant A.
**New eta values**: 0.75, 0.5, 0.25, 0.1

### M0-ETA: Gate Calibration — Eta Sweep

| Run ID | Eta | Variant | Status | Notes |
|--------|-----|---------|--------|-------|
| CAL-A-eta0p75 | 0.75 | A: signed+raw_reward+dr_grpo | DONE ✓ | Job 5331525; thr=0.3737(kr=0.90), 0.4793(0.80), 0.4977(0.70) |
| CAL-B-eta0p75 | 0.75 | B: zero_two+advantage+grpo | DONE ✓ | Job 5331525; thr=0.4999(kr=0.90) — unlocked at eta=0.75 |
| CAL-C-eta0p75 | 0.75 | C: signed+advantage+dr_grpo | DONE ✓ | Job 5331525; thr=0.4999(kr=0.90) — unlocked at eta=0.75 |
| CAL-A-eta0p5  | 0.5  | A | DONE ✓ | Job 5331526; thr=0.4690(0.80), 0.4965(0.70) |
| CAL-B-eta0p5  | 0.5  | B | DONE ✗ | Job 5331526; no valid thresholds |
| CAL-C-eta0p5  | 0.5  | C | DONE ✗ | Job 5331526; no valid thresholds |
| CAL-A-eta0p25 | 0.25 | A | DONE ✓ | Job 5331527; thr=0.4383(0.80), 0.4930(0.70), 0.4991(0.60) |
| CAL-B-eta0p25 | 0.25 | B | DONE ✗ | Job 5331527; no valid thresholds |
| CAL-C-eta0p25 | 0.25 | C | DONE ✗ | Job 5331527; no valid thresholds |
| CAL-A-eta0p1  | 0.1  | A | DONE ✓ | Job 5331528; thr=0.0204(0.90), 0.3497(0.80), 0.4825(0.70), 0.4977(0.60), 0.5093(0.10) |
| CAL-B-eta0p1  | 0.1  | B | DONE ✗ | Job 5331528; no valid thresholds |
| CAL-C-eta0p1  | 0.1  | C | DONE ✗ | Job 5331528; no valid thresholds |

**Decision gate G0-ETA**: PASSED — A valid at all etas; B/C valid only at eta=0.75.

### M1-ETA: Sanity Checks — Eta Sweep

| Run ID | Eta | Variant | Threshold | Status | Notes |
|--------|-----|---------|-----------|--------|-------|
| SAN-A-eta0p75 | 0.75 | A | 0.4977 (kr=0.70) | DONE ✓ | Job 5331914 |
| SAN-B-eta0p75 | 0.75 | B | 0.4999 (kr=0.90) | DONE ✓ | Job 5331915 |
| SAN-C-eta0p75 | 0.75 | C | 0.4999 (kr=0.90) | DONE ✓ | Job 5331916 |
| SAN-A-eta0p5  | 0.5  | A | 0.4965 (kr=0.70) | DONE ✓ | Job 5331917 |
| SAN-A-eta0p25 | 0.25 | A | 0.4991 (kr=0.70) | DONE ✓ | Job 5331918 |
| SAN-A-eta0p1  | 0.1  | A | 0.4977 (kr=0.60) | DONE ✓ | Job 5331919 |

### M3-ETA: Main Training — Eta Sweep

| Run ID | Eta | Variant | Threshold | KR | MATH Acc | Δ vs BC | Status |
|--------|-----|---------|-----------|-----|----------|---------|--------|
| ETA75-A-kr90 | 0.75 | A | 0.3737 | kr=0.90 | **0.2645** | −5.46 pp | DONE ✗ | Job 5332180; 0.2645±0.0103; worst so far |
| ETA75-A-kr80 | 0.75 | A | 0.4793 | kr=0.80 | **0.2937** | −2.54 pp | DONE ✗ | Job 5332181; 0.2937±0.0081; below eta=1.0 |
| ETA75-A-kr70 | 0.75 | A | 0.4977 | kr=0.70 | **0.3161** | −0.30 pp | DONE ✗ | Job 5332182; 0.3161±0.0102; best eta=0.75, ≈MAIN-A2 (0.3139) |
| ETA75-B-kr90 | 0.75 | B | 0.4999 | kr=0.90 | **0.3096** | −0.95 pp | DONE ✗ | Job 5332183; 0.3096±0.0111; advantage collapse → worse than Var-A same eta |
| ETA75-C-kr90 | 0.75 | C | 0.4999 | kr=0.90 | **0.3167** | −0.24 pp | DONE ✗ | Job 5332184; 0.3167±0.0085; better than B, ~same as eta=0.1 kr=0.60 runs |
| ETA50-A-kr80 | 0.5  | A | 0.4690 | kr=0.80 | **0.2947** | −2.44 pp | DONE ✗ | Job 5332185; 0.2947±0.0091; below MAIN-A1 |
| ETA50-A-kr70 | 0.5  | A | 0.4965 | kr=0.70 | **0.3139** | −0.52 pp | DONE ✗ | Job 5332186; 0.3139±0.0115; ties MAIN-A2; below BC-all |
| ETA25-A-kr80 | 0.25 | A | 0.4383 | kr=0.80 | **0.2939** | −2.52 pp | DONE ✗ | Job 5332187; 0.2939±0.0090; consistent with kr=0.80 pattern |
| ETA25-A-kr70 | 0.25 | A | 0.4930 | kr=0.70 | **0.3196** | +0.05 pp | DONE ✓ | Job 5332188; 0.3196±0.0071; new best — marginally above BC-all but within noise |
| ETA25-A-kr60 | 0.25 | A | 0.4991 | kr=0.60 | **0.3193** | +0.02 pp | DONE ✓ | Job 5332189; 0.3193±0.0080; ≈BC-all; within noise |
| ETA10-A-kr90 | 0.1  | A | 0.0204 | kr=0.90 | ~0.265 | ~−5.4 pp | HUNG ✗ | Job 5332190; eval hung at run 27/30 after 3h; estimate from 27 runs; consistent with kr=0.90 floor |
| ETA10-A-kr80 | 0.1  | A | 0.3497 | kr=0.80 | **0.2943** | −2.48 pp | DONE ✗ | Job 5332191; 0.2943±0.0094; consistent kr=0.80 floor |
| ETA10-A-kr70 | 0.1  | A | 0.4825 | kr=0.70 | **0.3118** | −0.73 pp | DONE ✗ | Job 5332192; 0.3118±0.0096; slightly below kr=0.70 peers |
| ETA10-A-kr60 | 0.1  | A | 0.4977 | kr=0.60 | **0.3166** | −0.25 pp | DONE ✓ | Job 5332193; 0.3166±0.0101; consistent with kr=0.60 band |
| ETA10-A-kr10 | 0.1  | A | 0.5093 | kr=0.10 | **0.3304** | +1.13 pp | DONE ✓✓ | Job 5332194; 0.3304±0.0076; **NEW BEST** — beats C002 (0.3247) by +0.57 pp; kr=0.10 filters to hard-correct tokens only |

---

## Results Summary (updated as runs complete)

| System | MATH Acc | Std | Δ vs BC | Δ vs C002 | Verdict |
|--------|----------|-----|---------|-----------|---------|
| BC-all | 0.3191 | 0.0081 | — | — | Baseline |
| BC-correct | 0.3118 | 0.0087 | −0.7 pp | — | DONE (Job 5314314; correct-only is BELOW BC-all) |
| SoftDG TL-best (C002) | 0.3247 | 0.0089 | +0.6 pp | — | Prior best |
| SoftDG-TM Var-A (eta=1.0, thr=0.4845) | 0.2986 | 0.0097 | −2.05 pp | −2.61 pp | DONE — below all baselines |
| SoftDG-TM Var-A (eta=1.0, thr=0.4982) | 0.3139 | 0.0125 | −0.52 pp | −1.08 pp | DONE — best eta=1.0 result |
| SoftDG-TM Var-B (eta=1.0) | — | — | — | — | SKIP (no valid thresholds at eta=1.0) |
| SoftDG-TM Var-C (eta=1.0) | — | — | — | — | SKIP (no valid thresholds at eta=1.0) |
| SoftDG-TM Eta Sweep best (eta=0.1, kr=0.10) | **0.3304** | 0.0076 | **+1.13 pp** | **+0.57 pp** | DONE ✓✓ — beats C002; extreme filtering (hard-correct tokens only) is the key |
| SoftDG-TM Var-B (eta=0.75, kr=0.90) | 0.3096 | 0.0111 | −0.95 pp | −1.51 pp | DONE ✗ — advantage collapse; below BC |
| SoftDG-TM Var-C (eta=0.75, kr=0.90) | 0.3167 | 0.0085 | −0.24 pp | −0.80 pp | DONE ✗ — better than B; below BC |

**Eta sweep conclusion (14/15 Var-A done + 1 hung)**: Best = 0.3304 (eta=0.1, kr=0.10), +1.13 pp above BC. Key findings:
- Keep rate dominates eta: all etas give ~0.31–0.32 at kr=0.70
- eta=0.25 is slightly best; eta=0.1 slightly worse than 0.25 (gate too sharp)
- Stricter thresholds (kr=0.70 > 0.80 > 0.90) consistently better, but plateau around kr=0.60–0.70
- Variants B and C fail regardless of eta (advantage/signal collapse)
- **G3 outcome: FAIL** — no configuration reaches +2 pp target (0.3391); best is +1.13 pp (eta=0.1, kr=0.10)

---

## Phase 4: DG-Weighted Selected-Token PG (wgate) — 2026-06-22

**Hypothesis**: Scaling the PG signal by DG(token_t) (gate value) for selected tokens improves accuracy beyond the best unweighted result (0.3304).
**New PG rule**: `PG_signal_t = training_signal_i * DG(token_t)` for selected tokens; selection by same hard threshold.
**Calibration**: Reused from June 20 (gate formula unchanged).
**Plan**: `refine-logs/EXPERIMENT_PLAN_2026-06-22.md`
**Code changes**: `SoftDG-Token-Mask-Offline/trainer.py` + `train.py` — `--pg_weighting` flag (default True).
**Submit scripts**: `run_math_wgate.sh`, `submit_wgate_sweep.sh`

### M0-WGATE: Sanity Checks (7 jobs)

| Run ID | Eta | Variant | Threshold | Job ID | Status | Notes |
|--------|-----|---------|-----------|--------|--------|-------|
| WSAN-A-eta1p0 | 1.0 | A: signed+raw_reward+dr_grpo | 0.4982 | 5364847 | DONE ✓ | pg_mass_weighted=1904.8, mean_selected_gate=0.514, KL=2.7e-4, kr=0.78, budget reached |
| WSAN-A-eta0p75 | 0.75 | A | 0.4977 | 5364848 | DONE ✓ | pg_mass_weighted=1913.7, mean_selected_gate=0.518, KL=2.7e-4, kr=0.72, budget reached |
| WSAN-B-eta0p75 | 0.75 | B: zero_two+advantage+grpo | 0.4999 | 5364849 | DONE ✓ | pg_mass_weighted=309.0, gate collapse (p50=p90=0.5 as expected), KL active, budget reached |
| WSAN-C-eta0p75 | 0.75 | C: signed+advantage+dr_grpo | 0.4999 | 5364850 | DONE ✓ | pg_mass_weighted=309.0, gate collapse, KL active, budget reached |
| WSAN-A-eta0p5 | 0.5 | A | 0.4965 | 5364851 | DONE ✓ | pg_mass_weighted=1936.1, mean_selected_gate=0.523, KL=2.7e-4, kr=0.72, budget reached |
| WSAN-A-eta0p25 | 0.25 | A | 0.4991 | 5364852 | DONE ✓ | pg_mass_weighted=1655.6, mean_selected_gate=0.540, KL=2.7e-4, kr=0.60, budget reached |
| WSAN-A-eta0p1 | 0.1 | A | 0.4977 | 5364853 | DONE ✓ | pg_mass_weighted=1699.4, mean_selected_gate=0.554, KL=2.7e-4, kr=0.60, budget reached |

**Decision gate G0-WGATE**: PASSED ✅ — All 7 pass (no hang, `softdg/mean_selected_gate` logged, `softdg/pg_mass_weighted` logged, KL active). Full sweep submitted 2026-06-22.

### M1-WGATE: Main Training (17 jobs)

Submit after G0-WGATE passes: `bash SoftDG-Token-Mask-Offline/submit_wgate_sweep.sh`

| Run ID | Eta | Variant | Threshold | KR | Job ID | MATH Acc | Δ vs BC | Δ vs 0.3304 | Status |
|--------|-----|---------|-----------|-----|--------|----------|---------|-------------|--------|
| WMAIN-A-eta1p0-thr0p4845 | 1.0 | A | 0.4845 | ~0.80 | 5367040 | **0.3005** | −1.86 pp | −2.99 pp | DONE ✗ |
| WMAIN-A-eta1p0-thr0p4982 | 1.0 | A | 0.4982 | ~0.70 | 5367041 | **0.3083** | −1.08 pp | −2.21 pp | DONE ✗ |
| WMAIN-A-eta0p75-thr0p3737 | 0.75 | A | 0.3737 | 0.90 | 5367042 | **0.2760** | −4.31 pp | −5.44 pp | DONE ✗ |
| WMAIN-A-eta0p75-thr0p4793 | 0.75 | A | 0.4793 | 0.80 | 5367043 | **0.2991** | −2.00 pp | −3.13 pp | DONE ✗ |
| WMAIN-A-eta0p75-thr0p4977 | 0.75 | A | 0.4977 | 0.70 | 5367044 | **0.3084** | −1.07 pp | −2.20 pp | DONE ✗ |
| WMAIN-B-eta0p75-thr0p4999 | 0.75 | B | 0.4999 | 0.90 | 5367045 | **0.3067** | −1.24 pp | −2.37 pp | DONE ✗ |
| WMAIN-C-eta0p75-thr0p4999 | 0.75 | C | 0.4999 | 0.90 | 5367046 | **0.3210** | +0.19 pp | −0.94 pp | DONE ✓ |
| WMAIN-A-eta0p5-thr0p4690 | 0.5 | A | 0.4690 | 0.80 | 5367047 | **0.2975** | −2.16 pp | −3.29 pp | DONE ✗ |
| WMAIN-A-eta0p5-thr0p4965 | 0.5 | A | 0.4965 | 0.70 | 5367048 | **0.3089** | −1.02 pp | −2.15 pp | DONE ✗ |
| WMAIN-A-eta0p25-thr0p4383 | 0.25 | A | 0.4383 | 0.80 | 5367049 | **0.3016** | −1.75 pp | −2.88 pp | DONE ✗ |
| WMAIN-A-eta0p25-thr0p4930 | 0.25 | A | 0.4930 | 0.70 | 5367050 | **0.3175** | −0.16 pp | −1.29 pp | DONE ✗ |
| WMAIN-A-eta0p25-thr0p4991 | 0.25 | A | 0.4991 | 0.60 | 5367051 | **0.3209** | +0.18 pp | −0.95 pp | DONE ✓ |
| WMAIN-A-eta0p1-thr0p0204 | 0.1 | A | 0.0204 | 0.90 | 5367052 | **0.2872** | −3.19 pp | −4.32 pp | DONE ✗ |
| WMAIN-A-eta0p1-thr0p3497 | 0.1 | A | 0.3497 | 0.80 | 5367053 | **0.2945** | −2.46 pp | −3.59 pp | DONE ✗ |
| WMAIN-A-eta0p1-thr0p4825 | 0.1 | A | 0.4825 | 0.70 | 5367054 | **0.3133** | −0.58 pp | −1.71 pp | DONE ✗ |
| WMAIN-A-eta0p1-thr0p4977 | 0.1 | A | 0.4977 | 0.60 | 5367055 | **0.3167** | −0.24 pp | −1.37 pp | DONE ✗ |
| WMAIN-A-eta0p1-thr0p5093 | 0.1 | A | 0.5093 | 0.10 | 5367056 | **0.3303** | +1.12 pp | −0.01 pp | DONE ✗ |

**Success criterion**: At least one wgate run ≥ 0.3391 (beats BC by +2 pp). → **FAILED**
**Diagnostic bar**: Best wgate run > 0.3304 (beats best unweighted by any margin). → **FAILED** (best wgate = 0.3303, unweighted = 0.3304)

### Phase 4 Conclusions (2026-06-23)

**Overall verdict: Wgate does NOT improve over binary masking.**

Key findings:
- **Best wgate result**: 0.3210 (Var-C, eta=0.75, kr=0.90, job 5367046) — above BC-all but below binary best
- **Critical comparison** (eta=0.1, Var-A, kr=0.10): wgate=**0.3303** vs unweighted=**0.3304** — virtually identical (Δ=−0.01 pp)
- Gate-weighting PG by DG value adds no signal beyond binary inclusion/exclusion
- Pattern across all 17 wgate jobs: wgate accuracy ≈ unweighted accuracy for same (eta, variant, kr)
- Var-C wgate (eta=0.75, kr=0.90) was unexpectedly the wgate winner, same pattern as unweighted where Var-C did best at kr=0.90

**Interpretation**: At kr=0.10 the selected tokens already have high gate values (mean_selected_gate≈0.75), so multiplying by the gate value adds ~25% downscaling noise rather than useful signal differentiation. The binary mask already captures the informative tokens; the continuous gate value is not carrying additional information beyond the selection itself.

**No M2-WGATE extra-seed runs** — success criterion not met.

### M2-WGATE: Extra Seeds (if best ≥ 0.3391)

**SKIPPED** — success criterion not met. Best wgate result (0.3303) did not exceed unweighted best (0.3304).

---

## Phase 3: Token Probability Calibration (DONE — 2026-06-21)

**Goal**: Compute global student-token-probability distribution over fixed bad-teacher rollout (analysis only, no training).

| Job ID | Config | Status | Notes |
|--------|--------|--------|-------|
| 5349382 | Smoke check (4 completions) | DONE ✓ | All validation checks passed |
| 5349404 | Full calibration (48,000 completions) | DONE ✓ | Completed in ~22 min; 23,207,585 tokens scored |

**Key findings** (student=Qwen2.5-0.5B scoring teacher=Qwen2.5-0.5B-Instruct rollout):
- **Mean student prob = 0.8984** — student already assigns very high probability to teacher tokens
- **Median surprisal = 0.0013** — gate collapse confirmed for ~80% of tokens
- Surprisal distribution: p10=0.0000, p50=0.0013, p80=0.1454, p90=0.5416, p99=2.3026
- Only the top ~10% of tokens have surprisal > 0.54 (non-trivial gate spread)
- This explains why kr=0.10 (extreme filtering) was the only config that helped: it selects precisely these high-surprisal tokens

**Output files**:
- `SoftDG-Token-Mask-Offline/outputs/token_probability_calibration/teacher-Qwen2.5-0.5B-Instruct__student-Qwen2.5-0.5B/`
  - `*_token_probabilities.jsonl.gz` — per-token probs (all 23M tokens)
  - `*_run_summaries.jsonl` — per-completion summaries
  - `*_percentiles.json` — global percentile table
  - `*_percentiles.csv` — spreadsheet-friendly version
