# Experiment Tracker — SoftDG V2 (Dr.GRPO + Token Gating)

**Plan**: `refine-logs/EXPERIMENT_PLAN_2026-06-07.md`  
**Goal**: Beat BC baseline (0.3191) by ≥ +2 pp → target ≥ 0.3391

## Baseline
| System | MATH Acc | Notes |
|--------|----------|-------|
| BC-all (bad-teacher rollout) | 0.3191 ± 0.0081 | Existing, no rerun needed |

---

## M0: Gate Calibration — DONE (job 5165449, 2026-06-08)

| Run | Status | Config | Output |
|-----|--------|--------|--------|
| C0 | DONE | all 4 configs, eta=1.0 | outputs/gate_calibration_v2_thresholds.json |

**Decision: PROCEED**

### Calibration Findings

| Config | Valid Thresholds | Note |
|--------|-----------------|------|
| signed+raw_reward+token | 0.4687, 0.4777, **0.5000**, 0.5183, 0.5239 | thr=0.5 perfectly separates correct (kr=1.0) from wrong (kr=0.0) |
| zero_two+raw_reward+token | 0.5302, 0.5392 | Wrong completions gate exactly at 0.5, need thr>0.5 |
| zero_two+advantage+token | ✗ none | Both correct/wrong gate near 0.5 — skip this config |
| signed+raw_reward+completion | 0.4637, 0.4743, 0.5000, 0.5210, 0.5277 | Ablation reference; similar distribution to token mode |

**Key insight**: With `signed + raw_reward + token`, threshold=0.5 is the mathematically clean separator:
- Correct completions: signal=+1, delight_t=+surprisal_t>0 → gate_t>0.5 → mean_gate>0.5
- Wrong completions: signal=−1, delight_t=−surprisal_t<0 → gate_t<0.5 → mean_gate<0.5

---

## M1: Sanity Check

| Run | Job | Status | Config | threshold | eff_comp target | Notes |
|-----|-----|--------|--------|-----------|-----------------|-------|
| S0 | 5165620 | PENDING → RUNNING | signed+raw_reward+dr_grpo+token | 0.4687 | 32 | |

**Expected**: no DDP hang; gate logged; low-gate skips nonzero; effective=32; skipped rows zeroed

---

## M3: Main Method (submit after S0 sanity passes)

### signed+raw_reward+dr_grpo+token — 5 thresholds to sweep
| Run | Job | Status | threshold | kr_correct | kr_wrong | MATH Acc | Notes |
|-----|-----|--------|-----------|------------|----------|----------|-------|
| M001 | 5165653 | DONE | 0.4687 | 1.000 | 0.300 | **0.2995 ± 0.0121** | −2.0 pp vs BC; no collapse (len=715); gate active (35085 skipped, kr=0.578) |
| M002 | 5165654 | DONE | 0.4777 | 1.000 | 0.100 | **0.3170 ± 0.0102** | −0.2 pp vs BC; gate active (28031 skipped, 10% wrong) |
| M003 | 5165655 | DONE | 0.5000 | 1.000 | 0.000 | **0.3109 ± 0.0095** | −0.8 pp vs BC; train-only-on-correct; no collapse (len=~680) |
| M004 | 5165656 | DONE | 0.5183 | 0.901 | 0.000 | **0.3167 ± 0.0087** | −0.2 pp vs BC; within noise |
| M005 | 5165657 | DONE | 0.5239 | 0.702 | 0.000 | **0.3015 ± 0.0067** | −1.8 pp vs BC; dropping 30% correct completions hurts |

Success criterion: best variant MATH acc ≥ 0.3391

---

## M4: Conservative Comparison

### zero_two+raw_reward+dr_grpo+token — 2 thresholds
| Run | Job | Status | threshold | kr_correct | kr_wrong | MATH Acc | Notes |
|-----|-----|--------|-----------|------------|----------|----------|-------|
| C001 | 5165658 | DONE | 0.5302 | 0.901 | 0.000 | **0.3200 ± 0.0088** | ≈BC (+0.001 pp, within noise); zero_two signal |
| C002 | 5165659 | DONE | 0.5392 | 0.700 | 0.000 | **0.3247 ± 0.0089** | +0.6 pp vs BC (not sig.); best in sweep; hardest-correct selection |

---

## Results Summary (2026-06-08) — ALL M3/M4 COMPLETE

**Verdict: NEGATIVE — target ≥ 0.3391 not met. Best result (C002) is +0.6 pp above BC but not statistically significant.**

### Ranked results
| Run | Signal | thr | kr_correct | kr_wrong | MATH Acc | vs BC |
|-----|--------|-----|-----------|----------|----------|-------|
| C002 | zero_two | 0.5392 | 0.70 | 0.00 | **0.3247 ± 0.0089** | +0.6 pp (not sig) |
| C001 | zero_two | 0.5302 | 0.90 | 0.00 | 0.3200 ± 0.0088 | ≈BC |
| M002 | signed | 0.4777 | 1.00 | 0.10 | 0.3170 ± 0.0102 | −0.2 pp |
| M004 | signed | 0.5183 | 0.90 | 0.00 | 0.3167 ± 0.0087 | −0.2 pp |
| M003 | signed | 0.5000 | 1.00 | 0.00 | 0.3109 ± 0.0095 | −0.8 pp |
| M005 | signed | 0.5239 | 0.70 | 0.00 | 0.3015 ± 0.0067 | −1.8 pp |
| M001 | signed | 0.4687 | 1.00 | 0.30 | 0.2995 ± 0.0121 | −2.0 pp |

### Key observations
1. **zero_two > signed at equivalent keep rates**: C001/C002 outperform M004/M005 at similar kr_correct. The signed signal's negative gradient for wrong completions (when they slip through the gate) appears harmful.
2. **Inverted difficulty effect in zero_two**: C002 (kr_correct=0.70) > C001 (kr_correct=0.90) — harder/more surprising correct completions appear more informative. The opposite of what signed shows (M005 < M004).
3. **Negative signal is actively harmful**: Allowing even 10–30% wrong completions through the gate (M001, M002) hurts vs keeping zero wrong completions.
4. **Clean separator (thr=0.5) underperforms looser thresholds**: M003 (0.3109) < M002 (0.3170) despite M003 having perfect kr_wrong=0. The 10% extra wrong completions in M002 still pass through the signed-signal gradient.
5. **BC is a strong baseline**: Training on all demonstrations (correct + wrong) for behavior cloning matches or beats any GRPO-with-gating variant tried here.

### Decision on M5 ablations
**SKIP** — since no main variant meaningfully beats BC, isolating token vs completion gating or GRPO vs Dr.GRPO is not informative for the primary claim. M5 ablations were conditioned on ≥ 0.3391; none achieved it.

---

## M5: Ablations (after best main threshold identified)

| Run | Job | Status | Config | threshold | Notes |
|-----|-----|--------|--------|-----------|-------|
| A001 | — | PENDING | signed+raw_reward+dr_grpo+**completion** | 0.5000 | isolate token gating |
| A002 | — | PENDING | signed+raw_reward+**grpo**+token | 0.5000 | isolate Dr.GRPO |
| A003 | — | PENDING | zero_two+advantage+dr_grpo+completion | TBD | need completion calibration |

---

## Code Changes (2026-06-08)

- `SoftDG-Offline-to-the-bottom/trainer.py`: Added `dg_gating=token` + gate p10/p50/p90 metrics
- `SoftDG-Offline-to-the-bottom/train.py`: Added `--loss_type` and `--dg_gating` args
- `SoftDG-Offline-to-the-bottom/run_math.sh`: Added `LOSS_TYPE`, `DG_GATING` env vars; updated `RUN_TAG`
- `SoftDG-Offline-to-the-bottom/calibrate_gate.py`: New gate calibration script
- `SoftDG-Offline-to-the-bottom/run_calibrate.sh`: New calibration Slurm script
