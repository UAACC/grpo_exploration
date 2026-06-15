# AutoAI Branch Handover

Date: 2026-06-15

Comparison basis:
- `main`: `989cf1b`
- `AutoAI` before this handover note: `cbd9cd2`

## Branch Status

`AutoAI` is a strict superset of `main` in the local repository:

- 13 commits are unique to `AutoAI`.
- 0 commits are unique to `main`.
- Net diff from `main` to `AutoAI`: 52 files changed, 7,395 insertions, 298 deletions.

Unique commits on `AutoAI`:

1. `4ae0017` `shuai`
2. `825cb13` `merge`
3. `7cf4563` `shuai`
4. `b32f6f7` `Merge remote-tracking branch 'origin/main' into shuai mixture done right`
5. `c55f359` `shuai`
6. `926e41f` `AWR-offline`
7. `4e5dea5` `AWR`
8. `b6dfb80` `multiple teachers`
9. `6316477` `offline experiments`
10. `333acd1` `before auto AI`
11. `9992261` `handover`
12. `8568618` `Add SoftDG offline training scripts`
13. `cbd9cd2` `Add refine logs markdown reports`

## Main Changes

### Offline Training Variants

`AutoAI` adds new offline training method directories:

- `AWR-offline/`: reward-weighted offline GRPO trainer and MATH launch script.
- `RWR-offline/`: RWR-style offline GRPO trainer, GSM8K/MATH launch scripts, evaluation sweep script, and accelerate config.
- `SoftDG-Offline-to-the-bottom/`: SoftDG offline trainer, BC/MATH/calibration scripts, and gate calibration tooling.

SoftDG adds:

- Gate-threshold skipping for low-gate completions.
- Reward coding options: `zero_two` and `signed`.
- Training signal options: `advantage` and `raw_reward`.
- `grpo` vs `dr_grpo` loss selection.
- Completion-level or token-level DG gating.
- Early stopping by target effective completions.
- Gate calibration output for choosing thresholds before full runs.

### DG Offline Updates

`DG-offline/` is extended for multi-teacher and multi-file rollouts:

- `--rollout_path` accepts multiple JSONL files.
- `num_generations` can be inferred or validated from rollout counts.
- Run IDs are remapped per question across multiple rollout files.
- Training regimes include the existing mode and signed-reward variants.
- Loss type can be selected between GRPO and Dr.GRPO modes.
- New scripts cover MATH, GSM8K, smoke runs, and multi-teacher runs.

### Rollout and Dataset Tooling

Shared rollout tooling is expanded:

- `shared/datasets_registry.py` adds AceReason-Math support and a cached Arrow fallback path.
- `shared/generate_rollouts.py` adds `top_k` sampling and flushes output after each written example.
- `shared/generate_rollouts_by_batch.py` adds batched rollout generation.
- `shared/merge_rollouts.py` adds rollout merging.
- `shared/repair_pick4_rollouts.py` adds repair logic for pick-4 rollouts.
- `shared/select_rollout_runs.py` adds rollout subset selection.
- `shared/run_generate_acereason_rollouts.sh` adds AceReason rollout generation.

### Baselines and Mixture Methods

Behavior cloning and mixture GRPO scripts are updated:

- `bc/train_bc.py` supports multiple rollout files through dataset concatenation.
- `bc/run_bc_math.sh` and `bc/run_bc_correct_only_math.sh` are expanded for current experiment settings.
- `dr_mixture_grpo/` adds configurable `loss_type`, colocated vLLM baseline generation, and associated logging.
- `dr_mixture_grpo/dr_dg_mixture/` receives the same baseline-generation and loss-type controls.
- `mixture_grpo/evaluate.py` re-execs evaluation after LoRA merge so it evaluates the merged model path cleanly.

### Experiment Documentation

New documentation and logs were added:

- `docs/progress_reports/2026_05_27.md`
- `docs/progress_reports/2026_06_01_multi_teacher.md`
- `refine-logs/EXPERIMENT_PLAN.md`
- `refine-logs/EXPERIMENT_PLAN_2026-06-07.md`
- `refine-logs/EXPERIMENT_RESULTS.md`
- `refine-logs/EXPERIMENT_TRACKER.md`

The refine-log results report that the SoftDG V2 sweep did not beat the BC baseline by the target margin. The best reported SoftDG result was `C002` at `0.3247 +/- 0.0089`, about +0.6 percentage points over the BC baseline and not statistically significant.

## Working Tree Note

At the time this handover note was written, two non-handover changes were still staged locally and intentionally left out of this commit:

- `.gitignore`
- `SoftDG-Offline-to-the-bottom/configs/accelerate_ddp_4gpu.yaml`
