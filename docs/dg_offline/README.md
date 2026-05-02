# DG-offline: knowledge hub

DG-offline (Delightful Policy Gradient, offline variant) is the central algorithmic focus of this project. This folder is the single source of truth for everything DG-offline: intuition, theory, implementation, plans, and experiments. Anything not here is scattered historical context.

## Start here

| You want | Read |
|---|---|
| 30-second pitch | [The one-paragraph TL;DR](#one-paragraph-tldr) below |
| Plain-language explanation | [intuition.md](intuition.md) |
| Equations, comparison to IS/PPO, full derivation | [theory.md](theory.md) (516 lines, mentor-briefing depth) |
| Complete technical reference (design decisions + full implementation + experimental plan) | [technical_reference.md](technical_reference.md) |
| Quick code walkthrough with line references | [implementation.md](implementation.md) |
| All audited results and η sweep per dataset | [experiments.md](experiments.md) |
| Two-stage experiment plan and rationale | [plans/two_stage_experiment.md](plans/two_stage_experiment.md) |
| Multi-teacher experiment design (planned) | [plans/multi_teacher_experiment.md](plans/multi_teacher_experiment.md) |
| DG-Mixture prototype design | [plans/dg_mixture_design.md](plans/dg_mixture_design.md) |

## One-paragraph TL;DR

Standard offline GRPO corrects for off-policy data with an importance-sampling ratio `π_student / π_teacher`, which is numerically unstable over long sequences and needs PPO clipping bandages. DG-offline replaces the IS ratio with a sigmoid gate on `advantage × surprisal`, where surprisal is the negative log-probability under the student's *current* policy. The gate amplifies gradients from surprising successes (correct rollouts the student did not expect to emit), suppresses gradients from surprising failures (wrong rollouts the student was confident about, which probably reflect teacher-specific patterns the student cannot execute), and does not need teacher logprobs at all. Because surprisal is computed under the student's own policy on the student's own re-tokenization of the teacher's text, DG-offline is also teacher-tokenizer-agnostic: any teacher whose output the student can read works, including different-architecture and closed-API teachers — situations where IS-based offline GRPO is mathematically undefined. The temperature η controls how sharp the gate is: low η is nearly binary filtering, high η converges to unweighted REINFORCE.

### Why DG-offline over Offline-GRPO (capability differences)

| Requirement on the teacher | Offline-GRPO (IS) | DG-offline |
|---|---|---|
| Per-token logprobs available | **Required** | Not used |
| Same tokenizer as student | **Required** (token IDs must align) | Not required |
| Same vocab as student | **Required** (otherwise IS is undefined) | Not required |
| Callable with logprob endpoint | **Required** | Not required (text-only API works) |

DG-offline therefore extends to teachers Offline-GRPO cannot run on. The full comparison and the cross-tokenizer recipe (decode → re-tokenize under student) are in [theory.md](theory.md) §6. The planned experiment that exercises this capability is [plans/multi_teacher_experiment.md](plans/multi_teacher_experiment.md). Note: the current shared rollout loader (`offline_grpo/data.py`) carries pre-tokenized `completion_ids` through to training, so the *current code path* still inherits a same-tokenizer constraint — that's an implementation artifact of code-sharing, not a property of DG.

## Headline results (audited where marked)

| Dataset | Best η | DG greedy | BC-all greedy | Gap | Audit |
|---|---|---|---|---|---|
| MATH | 0.5 | 29.00% | 27.40% | +1.60pp | neither audited (±3pp noise) |
| GSM8K | 0.1 | 49.47% | 49.45% | +0.02pp (tied) | ✓ 60-seed |
| SVAMP | 0.1 | 65.93% | 65.67% | +0.26pp (tied) | ✓ 30-seed |
| ASDiv | 2.0 | 76.23% | 74.36% | +1.87pp | DG ✓, BC pending |

No single η dominates: 0.1 wins on GSM8K and SVAMP, 0.5 on MATH, 2.0 on ASDiv. The paper will need a sentence or two on this dataset-dependence; mechanistic explanation is an open question. Full per-η sweep and pass@N numbers in [experiments.md](experiments.md).

## File layout of this hub

```
docs/dg_offline/
├── README.md              # this file (hub)
├── intuition.md           # plain-language explanation
├── theory.md              # equations + comparison to IS/PPO + numerical walkthrough (516 lines, moved from docs/recall/03_*)
├── technical_reference.md # full technical doc: background, design, implementation, experimental plan (285 lines, moved from repo root)
├── implementation.md      # quick code walkthrough of DG-offline/trainer.py with line refs
├── experiments.md         # audited results per dataset with η sweep
└── plans/
    ├── two_stage_experiment.md       # Alex's DG-offline → Online idea
    ├── multi_teacher_experiment.md   # Multi-teacher / cross-tokenizer experiment (planned)
    └── dg_mixture_design.md          # DG-Mixture prototype (parked)
```

The two implementation docs serve different purposes: `technical_reference.md` is the canonical written document (background reasoning, design decisions, experimental plan, known limitations); `implementation.md` is a focused code-walkthrough with line numbers to the current `trainer.py`.

## Code layout (not in this folder; the code lives with the other methods)

| File | Purpose |
|---|---|
| `DG-offline/trainer.py` | `DGOfflineTrainer` subclass of TRL's `GRPOTrainer`; gate + IS neutralization |
| `DG-offline/train.py` | CLI and accelerate launch glue |
| `DG-offline/run_math.sh`, `run_gsm8k.sh` | Single-η SLURM wrappers |
| `shared/run_train_offline.sh` (`dg_offline` branch) | Unified launcher: `METHOD=dg_offline DATASET=svamp DG_ETA=0.1 sbatch shared/run_train_offline.sh` |
| `shared/run_dg_then_online.sh` | Two-stage wrapper: DG-offline → online GRPO |

## External references in the repo (kept where they are, not moved)

| File | Why it matters for DG-offline |
|---|---|
| `docs/experiments/exp08_capacity_vs_data.md` | Earlier "capacity vs. data" hypothesis with a DG-offline subsection; broader scope than DG, so left in place |
| `docs/progress_reports/2026_03_28_30.md` | First comprehensive DG-offline η sweep results on MATH and GSM8K, pre-audit |
| `docs/progress_reports/2026_04_17.md` | Audit results and the generation-cap bug fix for SVAMP / ASDiv |

## Open threads (live)

1. **MATH variance audit** for {BC-all, BC-cc, DG-η=0.5}. Without it the +1.60pp DG advantage on MATH is noise-floor-limited.
2. **ASDiv BC-all 30-seed audit**. Needed to lock in the +1.87pp DG-over-BC gap on ASDiv.
3. **MATH two-stage eval audit** (jobs 4722565 + 4722567-69 in flight). Answers whether Alex's two-stage beats fresh-start online on MATH.
4. **Two-stage underperformance on GSM8K** (two-stage = 50.61% vs. fresh-start online = 55.82%). Diagnose by comparing entropy / `frac_reward_zero_std` / KL between the two training logs.
5. **Mechanistic explanation for η dataset-dependence**. Hypothesis in [experiments.md](experiments.md); needs a delight-distribution analysis per dataset to test.
6. **Multi-teacher experiment** ([plans/multi_teacher_experiment.md](plans/multi_teacher_experiment.md)). Design locked 2026-05-01: replace teacher with DeepSeek-R1-Distill-Qwen-7B on MATH, train 7 configs (BC-all, BC-cc, Offline-GRPO, DG×4η), two-pass audit funnel (cheap eval all → audit per-method winners). Loader (`DG-offline/teacher_agnostic_loader.py`) implemented and smoke-tested. Awaiting Stage 0 pre-check + execution.
7. **DG-Mixture prototype**. Lower priority until DG-offline audits are closed.

## External reference (original paper)

Osband (2026), "Delightful Distributed Policy Gradient", arXiv:2603.20521. Our DG-offline is an adaptation of the DG gate to the teacher-student offline setting: we removed the need for behavior (teacher) logprobs, aggregated surprisal at the completion level for TRL compatibility, and added an η hyperparameter for gate sharpness. See [theory.md](theory.md) § 2 and § 3 for the side-by-side comparison.
