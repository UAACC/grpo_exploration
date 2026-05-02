# DG-Mixture: Online student rollouts + DG-gated teacher rollouts

Complete design document preserved from the approved ultraplan. This is the full
rationale and blueprint that drove the implementation in `mixture_grpo/dg_mixture/`.

## Context

The progress report (`docs/progress_reports/2026_03_28_30.md`) establishes two
results that motivate this change:

1. **DG-offline (η=0.5) is the only offline method that raises pass@16 on MATH**
   (61.4% → 64.2%). Its asymmetric sigmoid gate (`σ(advantage × surprisal / η)`)
   amplifies surprising successes and suppresses surprising failures, so it
   actually expands the set of problems the student can solve.
2. **Online GRPO is the only method that matches or beats DG on both pass@1 and
   pass@16**, because it learns from the student's own successful rollouts —
   the on-policy signal directly reinforces strategies the student is
   discovering itself.

Neither signal alone is ideal: DG-offline caps out below online GRPO on pass@1
(29.0 vs 31.8), and online GRPO wastes its compute budget re-sampling problems
the student can already solve while never seeing the teacher's correct
strategies on hard problems. The goal of this change is a single trainer that
combines **online student rollouts (standard GRPO loss)** with **offline
teacher rollouts (DG-gated loss)**, so each step learns from both sources.

Existing mixture trainers (`mixture_grpo/method_A_unified`,
`mixture_grpo/method_B_weighted`) mix student+teacher but use plain IS-ratio
weighting on teacher completions — which is exactly the weighting DG-offline
outperformed. This plan adds a third mixture (DG-Mixture) that replaces the
teacher-side loss with the DG-offline gated loss.

## Design

```
                  prompts (qid, answer)
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
  student rollouts                 teacher rollouts
  (HF model.generate)         (lookup by qid in teacher_data)
          │                               │
  student_rewards                  teacher_rewards
          │                               │
  online advantages               teacher-only group advantages
  (student-only group stats)      (pre-computed at load time)
          │                               │
          │                         DG gate: σ(adv·surprisal/η)
          │                         × advantage   (pre-multiplied)
          │                               │
          ▼                               ▼
   L_online = TRL GRPO         L_dg_teacher = gated REINFORCE
   (std PPO clip, KL vs ref)   (old_logps = current ⇒ ratio≈1,
          │                     no KL term for teacher)
          │                               │
          └──────────────┬────────────────┘
                         ▼
          L_total = L_online + λ · L_dg_teacher
                   (+ β · KL on student only)
```

Key design choices (grounded in what already exists):

- **Skeleton**: fork `mixture_grpo/method_B_weighted/` — it already handles
  separate online/offline loss terms and the `teacher_*`-prefixed output dict.
  Method A (unified advantages) does not fit because the DG gate must apply
  only to teacher completions.
- **Teacher advantage baseline**: teacher-only group stats (same as
  DG-offline), *not* student-stats (Method B's choice). Rationale: the DG
  gate's calibration was developed on teacher-only group normalization, and
  mixing baselines would distort the `delight = advantage × surprisal` scale
  that η is tuned against.
- **IS-ratio neutralization on teacher**: follow DG-offline exactly — set
  `teacher_old_per_token_logps = current_logps.detach()` at generation time
  so the PPO ratio ≈ 1 at the step's first microbatch, and the effective
  teacher loss becomes gated REINFORCE.
- **KL penalty**: student-only (as in Method B). Teacher samples are
  off-policy and would bias the k3 KL estimator. The `kl_mask` mechanism
  from Method A is not needed because student and teacher flow through
  *separate* loss calls in Method B's layout.
- **Gate granularity**: start with `dg_gating="completion"` (mean per-token
  surprisal), matching DG-offline's best result.
- **Data loading**: reuse `offline_grpo/data.py::load_rollouts` +
  `compute_rewards_and_advantages` (produces teacher-only group advantages),
  then reshape into a `{qid: {"runs": [...]}}` dict the trainer can look up
  by question_id. This avoids re-implementing advantage normalization.

## Files

### New

- **`mixture_grpo/dg_mixture/__init__.py`** — empty.
- **`mixture_grpo/dg_mixture/trainer.py`** — `DGMixtureGRPOTrainer(GRPOTrainer)`.
  Fork of `method_B_weighted/trainer.py` with two changes:
    1. `_generate_and_score_completions`:
       - Student half: unchanged (student rewards → group-normalized online
         advantages → `student_old_logps` from one forward on
         `self.model`).
       - Teacher half: look up runs by qid from `self._teacher_data`; each
         run already carries a pre-computed `advantage` (teacher-only group
         stats). Compute current-policy logprobs on the teacher completions
         under `torch.no_grad()` → per-completion mean surprisal →
         `gate = sigmoid(advantage × mean_surprisal / self._dg_temperature)`
         → `gated_advantage = gate × advantage`. Set
         `teacher_old_per_token_logps = current_teacher_logps.detach()`.
       - Output dict: same `teacher_*` keys as Method B but carrying gated
         advantages and neutralized old logps.
    2. `_compute_loss`:
       - Call `super()._compute_loss(model, inputs)` for the online student
         loss (unchanged — this already computes KL vs ref on the student).
       - If `teacher_completion_ids` is in `inputs`, call a new
         `_compute_dg_teacher_loss(...)` that mirrors
         `_compute_offline_loss` but: (a) uses the pre-gated advantages
         passed in; (b) since `old ≈ current`, the clipped surrogate is a
         near-identity on the gradient and reduces to `-gated_adv ·
         log π_current`; (c) logs `dg/gate_mean`, `dg/gate_min`,
         `dg/gate_max`, `dg/surprisal_mean`, `dg/delight_mean` — the same
         names `DG-offline/trainer.py` uses, so existing dashboards work.
       - Final: `loss = online_loss + self._dg_offline_weight * dg_teacher_loss`.
  Constructor args mirror Method B plus `dg_temperature: float = 0.5`,
  `dg_gating: str = "completion"`. Keep `_sync_ref_adapter` /
  `_get_ref_logprobs` verbatim from Method B.

- **`mixture_grpo/dg_mixture/train.py`** — fork of
  `method_B_weighted/train.py` with:
    - Import `DGMixtureGRPOTrainer` instead of `WeightedMixtureGRPOTrainer`.
    - Add `--dg_temperature` (default 0.5), `--dg_gating`
      (choices `completion|token`, default `completion`), rename
      `--offline_weight` to `--dg_offline_weight` for clarity.
    - Replace the `load_teacher_rollouts` call with a helper that loads
      via `offline_grpo.data.load_rollouts` + `compute_rewards_and_advantages`
      and reshapes to `{qid: {"runs": [{"completion_ids", "reward",
      "advantage", "response"}, ...]}}`. Put the helper in
      `mixture_grpo/dg_mixture/train.py` (it's ~15 lines, no need to
      pollute `mixture_grpo/data.py`). Import path trick matches
      `DG-offline/train.py:28-33`: add `offline_grpo/` to `sys.path` after
      importing the local trainer.
    - Default hyperparameters: `learning_rate=5e-6`, `beta=0.01`,
      `num_generations=4`, `num_teacher_per_prompt=4`,
      `dg_offline_weight=0.3`, `dg_temperature=0.5`, LoRA r=32/α=32
      (matching `run_method_A_math.sh`).

- **`mixture_grpo/run_dg_mixture_math.sh`** — fork of
  `run_method_A_math.sh`. Point at `dg_mixture/train.py`, set
  `WANDB_PROJECT="dg-mixture-math"`, plumb through `DG_ETA` and
  `DG_LAMBDA` env vars so sweeps work the same way `DG-offline/run_math.sh`
  exposes `DG_ETA`. Output / merged dirs: `dg_mixture_math` /
  `dg_mixture_math_merged`.

### Modified

- **`README.md`** — add one row to the Methods table for
  `mixture_grpo/dg_mixture/` ("DG-Mixture: online GRPO + DG-gated teacher
  loss, combining Method B's mixture structure with DG-offline's gate"),
  and add a `### DG-Mixture` subsection under "Mixture Methods"
  with a minimal usage snippet.

### Untouched (reused as-is)

- `offline_grpo/data.py::load_rollouts, compute_rewards_and_advantages` — reused for teacher-side advantage normalization.
- `mixture_grpo/configs.py` — prompts and answer extractors are unchanged.
- `mixture_grpo/data.py::compute_math_correctness, compute_gsm8k_correctness` — used by the trainer to reward student completions (same as Method B).
- `mixture_grpo/evaluate.py` — same eval path as all existing methods.
- `bc/eval_best_of_n.py` — used for the pass@16 comparison.

## Implementation order

1. Create `mixture_grpo/dg_mixture/__init__.py`, `trainer.py`,
   `train.py`.
2. Start `trainer.py` as a literal copy of
   `method_B_weighted/trainer.py`, then:
   - Rename the class, update the docstring, add `dg_temperature` /
     `dg_gating` to `__init__`.
   - In `_generate_and_score_completions`, after the teacher advantages are
     gathered but before the output dict is built, insert the DG gate block:
     forward-pass `self.model` on `teacher_prompt_ids + teacher_completion_ids`
     under `no_grad`, compute mean surprisal over
     `teacher_completion_mask`, multiply into advantages via
     `sigmoid(delight / η)`, and overwrite `teacher_old_per_token_logps`
     with the freshly computed current logprobs (`.detach()`).
   - Replace `_compute_offline_loss` with `_compute_dg_teacher_loss`.
     Structure is identical up to the per-token surrogate, but: drop the
     IS-ratio diagnostics block (mostly irrelevant when ratio≈1) and add
     the `dg/*` metrics.
3. Copy `method_B_weighted/train.py` → `dg_mixture/train.py`, update
   imports, add the new CLI flags, and swap the teacher-data loader for
   the `load_rollouts + compute_rewards_and_advantages` reshape helper.
4. Copy `run_method_A_math.sh` → `run_dg_mixture_math.sh`, flip paths /
   project name, add `DG_ETA` + `DG_LAMBDA` env-var plumbing.
5. README update (table row + usage subsection).

## Verification

Run in order:

1. **Static smoke** (login node, no GPUs needed):
   ```
   cd mixture_grpo && python -c "from dg_mixture.trainer import DGMixtureGRPOTrainer"
   ```
   Confirms imports and sys.path juggling work.

2. **1-step training smoke** (single GPU, tiny config) — run the train
   script with `--num_train_epochs 1 --per_device_train_batch_size 2
   --gradient_accumulation_steps 1 --save_steps 999999 --logging_steps 1`
   against a ~50-prompt subset of the MATH rollouts. Confirm in stdout /
   wandb that:
   - `loss` is finite and non-zero,
   - `offline_loss` (or `dg_teacher_loss`) is present,
   - `dg/gate_mean` is in (0, 1) and not pinned to 0 or 1,
   - `reward` for student matches baseline ~0.27 on MATH at step 1,
   - teacher completions do not crash on OOV tokens (the 7B teacher has
     128 extra tokens — `load_rollouts(vocab_size=...)` handles this).

3. **Full MATH training** via `sbatch run_dg_mixture_math.sh train`
   with defaults (η=0.5, λ=0.3), mirroring the DG-offline MATH budget
   (1 epoch, 4 student + 4 teacher per prompt).

4. **Evaluation** — two numbers to report, both relative to the table in
   the progress report §1.3:
   - **pass@1 (MATH, greedy)** via `sbatch run_dg_mixture_math.sh eval`
     (uses `mixture_grpo/evaluate.py` at `temperature=0.0`, 5 runs).
     Target: ≥ 29.0% (DG-offline η=0.5). Stretch: ≥ 31.8% (online GRPO).
   - **pass@16 (MATH, temp=0.6)** via `python bc/eval_best_of_n.py
     --model_path <merged> --n_samples 16 --temperature 0.6
     --dataset_type math`. Target: ≥ 64.2% (DG-offline η=0.5).

5. **η/λ sweep (optional, if first run works)**: η ∈ {0.1, 0.5, 1.0}, λ ∈
   {0.1, 0.3, 1.0} using `DG_ETA=... DG_LAMBDA=...
   CHECKPOINT_DIR=/scratch/$USER/checkpoints/dg_mixture_math_eta${η}_lam${λ}
   sbatch --job-name=dg-mix-eta${η}-lam${λ} run_dg_mixture_math.sh`. The
   env-var indirection is how `DG-offline/run_math.sh` already supports
   DG_ETA sweeps — reuse that exact pattern.
