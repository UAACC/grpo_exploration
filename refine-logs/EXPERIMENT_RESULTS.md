# Experiment Results

## Setup

- **Model**: Qwen2.5-0.5B (`/scratch/shuai14/models/Qwen2.5-0.5B`)
- **Rollout**: `rollouts_math_Qwen2.5-0.5B-Instruct_0.6_pick4.jsonl` (bad teacher, 48K completions, 35.1% correct)
- **Eval**: MATH test set, 500 problems, 30 runs, temperature=0.0
- **LoRA**: r=32, alpha=32, all-linear, LR=3e-6
- **Success criterion**: best SoftDG variant beats BC by ≥ **+2 absolute accuracy points** (≥ 0.3391)

---

## Milestone 0 — Sanity Check

| Item | Result |
|------|--------|
| Job ID | 5155798 |
| Config | zero_two + advantage, thr=0.2, target=32 |
| Status | **PASSED** |
| Effective / Scanned | 32 / 108 (keep_rate=0.296) |
| Gate mean | 0.499 — no low-gate skips at threshold=0.2 |
| Zero-signal skipped | 76 |

---

## Milestone 1 — BC Baseline (M1)

| Job | Config | MATH Accuracy | Std | Min / Max | Avg length |
|-----|--------|---------------|-----|-----------|------------|
| 5155939 | BC-all, 1 epoch, 48K completions | **0.3191** | 0.0081 | 0.300 / 0.334 | 661 tok |

**Target for SoftDG**: ≥ **0.3391**

---

## Milestone 2 — Main Sweep (M2) — COMPLETE

All runs: threshold=0.2, target=48K effective completions, max 20 epochs.

| Config | Job | MATH Accuracy | Std | Δ vs BC | Verdict |
|--------|-----|---------------|-----|---------|---------|
| zero_two + advantage | 5155940 | 0.2355 | 0.0091 | **−8.4 pp** | ✗ FAIL |
| zero_two + raw_reward | 5155941 | 0.3143 | 0.0081 | −0.5 pp | ✗ FAIL |
| signed + advantage | 5155942 | 0.2259 | 0.0082 | **−9.3 pp** | ✗ FAIL |
| signed + raw_reward | 5155943 | **0.0961** | 0.0051 | **−22.3 pp** | ✗ COLLAPSE |

### Training characteristics

| Config | Scanned | Keep rate | Low-gate skip | Zero-sig skip | Final KL | Avg len |
|--------|---------|-----------|---------------|---------------|----------|---------|
| zero_two + advantage | 136,728 | 0.351 | 0 | 88,720 | 0.0088 | — |
| zero_two + raw_reward | 136,224 | 0.352 | 0 | 88,208 | 0.0290 | — |
| signed + advantage | 136,728 | 0.351 | 0 | 88,720 | 0.0089 | — |
| signed + raw_reward | 48,024 | **1.000** | 0 | 0 | 0.00027 | **1602 tok** |

### Findings

1. **Gate inactive throughout**: `passes_gate_frac=1.0` in every run; `gate_mean ≈ 0.499` (sigmoid of near-zero delight). Threshold=0.2 never filtered a single completion — the gating mechanism did not engage.

2. **Advantage-based GRPO hurt badly (−8 to −9 pp)**: Group normalization zeros the signal for all-correct and all-wrong groups. The remaining signed gradients (correct=+, wrong=−) appear to destabilize the model against the BC trajectory, rather than helping it. Both zero_two and signed codings produced similarly poor results.

3. **zero_two + raw_reward closest to BC (−0.5 pp)**: Effectively selective BC — only correct completions (reward=2) carry non-zero signal (wrong completions have reward=0, so signal=gate×0=0). All 88K "wrong" completions are skipped as zero-signal. Nearest to BC but still below it.

4. **signed + raw_reward: model collapse (0.0961, avg_len 1602 tok)**: Every completion gets ±1 signal, no zero-signal filtering, keep_rate=1.0. The strong negative gradient on all 64.9% incorrect completions drove degenerate generation (long repetitive outputs). Final KL was tiny (0.00027) despite the collapse, suggesting the model diverged in output space without large log-prob shifts.

5. **Training stopped at epoch ≈ 0.71** for advantage/zero_two configs (48K effective reached mid-epoch-1), and at epoch ≈ 0.25 for signed+raw_reward (keep_rate=1.0 so target hit in one-third of a pass).

---

## Final Verdict

**NEGATIVE RESULT** — the success criterion is not met.

No SoftDG variant at threshold=0.2 beats the BC baseline by +2 pp. The best result (zero_two + raw_reward, 0.3143) is 0.5 pp *below* BC.

**Root cause**: The gating mechanism never activated. With the student model close to the teacher at initialization, `completion_surprisal` is nearly identical for all completions, so `delight ≈ 0` everywhere and `gate ≈ 0.5` throughout training. Threshold=0.2 is far below 0.5, meaning the gate provides no selection signal. SoftDG at this threshold reduces to plain GRPO (advantage mode) or selective-BC on correct completions (raw_reward + zero_two mode) — neither of which beats BC on a bad-teacher rollout.

---

## Milestone 3 — Threshold Ablation (M3)

**Status**: NOT RUN — halted pending analysis.

Per the plan, M3 sweeps thresholds {0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40} for the best M2 config. The best M2 config is `zero_two + raw_reward`.

However, since `gate_mean ≈ 0.499` with std ≈ 0.02 across all batches, thresholds below 0.48 will still pass essentially all completions. Thresholds ≥ 0.50 would begin cutting ~50%+ of completions — but those cuts would be nearly random with respect to quality (since delight variation is minimal). M3 is unlikely to rescue the result without first understanding why delight is near-zero (i.e., why surprisal is uninformative for this rollout/model pair).

---

## Code Changes

All new code in `SoftDG-Offline-to-the-bottom/` (DG-offline untouched):

| File | Description |
|------|-------------|
| `trainer.py` | `SoftDGOfflineTrainer`; gate computation, KL suppression via `effective_completion_mask`, DDP-safe counter, IS-ratio neutralization |
| `train.py` | Entry point; reward coding pipeline, `recompute_advantages`, `EffectiveCompletionCounter`, `EarlyStopOnTargetCallback` |
| `run_math.sh` | Slurm: train + MATH eval inline; env-var overrides for all hyperparams |
| `run_bc.sh` | Slurm: BC-all baseline |
| `configs/accelerate_ddp_4gpu.yaml` | 4×L40S DDP config |

**Codex-review fixes applied before deployment**:
1. **KL leakage** (CRITICAL): `effective_completion_mask = completion_mask * passes_gate` — zeroes KL for skipped rows, not just PG.
2. **Empty completion guard** (MAJOR): `passes_gate = (gate ≥ threshold) & has_tokens`.
3. **Float comparison** (MINOR): `gated_signal.abs() > 1e-12`.
