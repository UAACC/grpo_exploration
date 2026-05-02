# DG-offline: experiments and results

## Protocol for every number below

Each cell uses the audit protocol described in `MEMORY.md` and `feedback_always_audit.md`:
- **Greedy** at temp=0.0: pooled mean across multiple independent SLURM jobs × 30 seeds per job. The "(60-seed)" annotation means 2 jobs × 30 seeds. "(30-seed)" means 1 job × 30 seeds (within-node variance only).
- **pass@N** at temp=0.6, n=16: reported as mean ± std across 3 independent seeded jobs where that was audited; single-run otherwise.
- **"single-run"** means a 5-seed `run_eval.sh` with implicit ±3pp cross-node noise.

## Results per dataset (DG-offline rows only; compare with `../progress_reports/` for full tables)

### MATH (500 test problems) — NOT YET AUDITED

| η | Greedy | pass@1 | pass@16 | Source |
|---|---|---|---|---|
| 0.1 | 29.04% | 25.00% | 62.40% | `eval-dg-eta0.1-4549865` / `bon-dg-eta0.1-4549882` |
| **0.5** | **29.00%** | 26.60% | **64.20%** | `eval-dg-eta0.5-4545646` / `bon-dg-eta0.5-4546559` |
| 1.0 | 28.08% | 27.60% | 63.40% | `eval-dg-eta1.0-4546573` / `bon-dg-eta1.0-4548890` |
| 2.0 | 27.88% | 27.80% | 62.20% | `eval-dg-eta2.0-4546574` / `bon-dg-eta2.0-4548891` |

**Best η on MATH greedy: 0.5.** Gap over BC-all (27.40%) = +1.60pp. All numbers single-run; ±3pp noise floor applies. Needs audit.

### GSM8K (1319 test problems) — AUDITED

| η | Greedy (60-seed) | pass@1 (3-seed) | pass@16 (3-seed) |
|---|---|---|---|
| **0.1** | **49.47% ± 0.04** | **48.27 ± 0.88** | **85.49 ± 0.95** |
| 0.5 | 48.08% ± 0.99 | — | — |
| 1.0 | 48.50% ± 0.82 | 45.90 ± 1.53 | 84.94 ± 0.29 |
| 2.0 | 48.43% (pre-audit) | 45.41% | 84.69% |

**Best η on GSM8K greedy: 0.1.** Gap over BC-all (49.45% audited): +0.02pp (tied). On pass@16, DG-η=0.1 beats BC-correct (84.28 ± 0.23) by 1.21pp, outside one std on both sides — defensible.

Audit correction: the pre-audit report named η=1.0 as the pass@16 winner at 85.67% (single 5-seed). After 3-seed audit, η=1.0 dropped to 84.94 ± 0.29 and η=0.1 took the lead at 85.49 ± 0.95. The earlier winner was a lucky draw on one node.

### SVAMP (300 test problems) — AUDITED

| η | Greedy (30-seed) | std | pass@1 | pass@16 |
|---|---|---|---|---|
| **0.1** | **65.93%** | 0.37pp | 63.67% | 95.33% |
| 0.5 | 65.06% | 0.37pp | 60.33% | **97.00%** |
| 1.0 | 63.73% | 0.20pp | **64.67%** | 95.67% |
| 2.0 | 65.66% | 0.14pp | 61.33% | 95.00% |

**Best η on SVAMP greedy: 0.1.** Gap over BC-all (65.67% audited): +0.26pp (tied). No DG variant clearly separates from BC on SVAMP.

### ASDiv (461 test problems) — DG AUDITED, BC PENDING

| η | Greedy (30-seed) | std | pass@1 | pass@16 |
|---|---|---|---|---|
| 0.1 | 75.94% | 0.26pp | 73.75% | 94.36% |
| 0.5 | 75.18% (pre-audit) | n/a | 76.36% | 94.36% |
| 1.0 | 76.01% | 0.26pp | 75.49% | 94.58% |
| **2.0** | **76.23%** | 0.34pp | **75.70%** | **94.79%** |

**Best η on ASDiv greedy: 2.0.** Gap over BC-all (74.36%, single-run pre-audit): **+1.87pp**, about 5× the DG std. This is the strongest DG-over-BC result on any dataset. Needs a matching BC audit to confirm BC's row is tight and the gap is real.

Note on the ASDiv baseline: the pre-2026-04-17 "baseline 39.09%" was an eval artifact of `max_tokens=512` slicing the verbose 0.5B baseline mid-reasoning before it emitted `\boxed{}`. After bumping the budget to 1024 and 2048, the baseline is 73.58% and BC's apparent +34pp lift collapsed to +0.78pp (within noise). See `../progress_reports/2026_04_17.md` for the full story.

## Cross-dataset summary

| Dataset | Best η | DG greedy | BC-all greedy | Gap | Audited gap? |
|---|---|---|---|---|---|
| MATH | 0.5 | 29.00% | 27.40% | +1.60pp | no |
| GSM8K | 0.1 | 49.47% | 49.45% | +0.02pp | yes, tied |
| SVAMP | 0.1 | 65.93% | 65.67% | +0.26pp | yes, tied |
| ASDiv | 2.0 | 76.23% | 74.36% | +1.87pp | partial (DG yes, BC pending) |

**Observation: no single η dominates.** The best η varies across datasets: 0.1 on GSM8K and SVAMP, 0.5 on MATH, 2.0 on ASDiv. This is worth a sentence or two in any writeup.

Working hypothesis for why η differs by dataset:

- **Sharper gate (low η)** helps when reward is clean and most teacher rollouts agree — SVAMP and GSM8K have short, numeric answers with unambiguous rewards.
- **Softer gate (high η)** helps when the reward-gradient signal is noisier per rollout — ASDiv problems are slightly more varied in structure, and the 0.5B may benefit from softer filtering that admits more marginal gradients.
- **Middling η** helps on MATH where the student's base capability is low enough that aggressive filtering would discard most useful updates.

This is a hypothesis, not a finding. Would need a dedicated study (e.g., measure per-rollout delight distribution by dataset, correlate with winning η) to support.

## Two-stage experiment (DG-offline → Online GRPO)

Alex's idea: use a DG-offline checkpoint as the initialization for a fresh online GRPO run. Results so far:

| Dataset | Stage 1 (DG-offline) | Stage 2 (online on top) | Fresh-start online (reference) |
|---|---|---|---|
| GSM8K | 48.73% | **50.61%** (5-seed) | 55.82% (cross-task from MATH) |
| MATH | 29.00% | pending (audit suite in flight) | 32.10% |

GSM8K outcome: two-stage beats DG-offline alone by +1.88pp but loses to fresh-start online by -5.21pp. Possible mechanisms (all unverified): entropy collapse after stage 1, tighter KL anchor against the merged reference, LoRA search confined to a different low-rank subspace. Diagnosable by comparing entropy, `frac_reward_zero_std`, KL between the two-stage training log and the fresh-start log. Not yet done.

MATH outcome: pending. Training hit the full 15/15 epochs cleanly; eval audit suite (jobs 4722565 + 4722567-69) in flight.

## Open experimental threads

1. **MATH variance audit** (BC-all, BC-cc, DG-η=0.5). Without this, the +1.60pp DG-over-BC gap on MATH is noise-floor-limited.
2. **ASDiv BC-all 30-seed audit**. Needed to confirm the +1.87pp gap DG-η=2.0 vs BC-all.
3. **MATH two-stage eval audit**. In flight.
4. **Two-stage diagnosis on GSM8K**. Explain why the combined pipeline underperforms fresh-start online.
5. **η interaction hypothesis**. Does per-rollout delight distribution predict the winning η? Would need an offline analysis pass over saved training logs.

## Artifact list

| Artifact | Where |
|---|---|
| Training code | `DG-offline/trainer.py`, `DG-offline/train.py` |
| Unified launcher | `shared/run_train_offline.sh` (the `dg_offline` branch) |
| Merged checkpoints | `/scratch/mrli/merged/dg_offline_{math,gsm8k,svamp,asdiv}_eta{0.1,0.5,1.0,2.0}` |
| Training checkpoints | `/scratch/mrli/checkpoints/dg_offline_{dataset}_eta{η}` |
| Audit eval logs | `logs/audit-gsm8k-dg-*`, `logs/audit-dg-{svamp,asdiv}-eta*` |
| Theory doc | `docs/dg_offline/theory.md` (formerly `docs/recall/03_delightful_policy_gradient.md`) |
| Intuition | `docs/dg_offline/intuition.md` |
| Implementation | `docs/dg_offline/implementation.md` |
| Two-stage plan | `docs/dg_offline/plans/two_stage_experiment.md` |
| DG-Mixture design | `docs/dg_offline/plans/dg_mixture_design.md` |
