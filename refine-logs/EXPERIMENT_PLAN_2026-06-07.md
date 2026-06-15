# Experiment Plan

## Goal
Test whether **SoftDG with Dr.GRPO + token-level gating** improves accuracy on bad-teacher MATH without the collapse observed in `signed + raw_reward + GRPO`.

The goal is to beat behavior cloning by at least **+2 absolute MATH accuracy points**. Higher is better.

Secondary goal:
- prevent signed-reward collapse, measured by MATH accuracy staying above `0.25` and average eval length staying below `900` tokens.

## Existing Code
- Main training file: `SoftDG-Offline-to-the-bottom/train.py`
- Main trainer file: `SoftDG-Offline-to-the-bottom/trainer.py`
- Evaluation file: `mixture_grpo/evaluate.py`
- Reference implementation for token-level gating:
  - `DG-offline/trainer.py`
- Reference implementation for Dr.GRPO config:
  - `DG-offline/train.py`
  - `DG-offline/run_math.sh`
- Config files:
  - `SoftDG-Offline-to-the-bottom/configs/accelerate_ddp_4gpu.yaml`
  - `SoftDG-Offline-to-the-bottom/run_math.sh`

## Implementation Idea
Continue using `SoftDG-Offline-to-the-bottom/*`.

Do not directly modify `DG-offline/*`.

Add selectable loss type:
- `loss_type=grpo`
- `loss_type=dr_grpo`

Default new V2 experiments to:
- `loss_type=dr_grpo`

Add selectable DG gating:
- `dg_gating=completion`: current SoftDG behavior, using mean per-token surprisal.
- `dg_gating=token`: token-level gate, matching `DG-offline/trainer.py`.

Token-level SoftDG rule:
- Compute per-token surprisal:
  - `surprisal_t = -log p(token_t)`
- Compute per-token delight:
  - `delight_t = training_signal * surprisal_t`
- Compute per-token gate:
  - `gate_t = sigmoid(delight_t / eta)`
- Average over non-padding completion tokens:
  - `gate = masked_mean(gate_t, completion_mask)`
- Skip completion-level backprop when:
  - `gate < softdg_gate_threshold`
- For selected rows, use:
  - `gate * training_signal`

Important edge cases:
- Padding tokens must not affect the averaged token gate.
- Empty completions must be skipped.
- With `reward_coding=zero_two` and `training_signal=raw_reward`, wrong completions have raw reward `0`, so they must not count as effective updates.
- Skipped rows must zero both policy-gradient signal and KL mask.
- Gibberish/truncated completions should be diagnostic canaries: under nonzero negative signal, if they are wrong and surprising, their gate should be low. If they are not low-gate, the gate is not identifying bad data and threshold tuning alone is unlikely to help.

## Experiments

### 0. Gate Calibration
- Dataset: bad-teacher MATH rollout only.
- Model: `/scratch/shuai14/models/Qwen2.5-0.5B`
- No training.
- Purpose:
  - measure actual gate and surprisal distributions before choosing thresholds.
  - avoid guessing thresholds before we know what token-level gates look like.
- Configs to calibrate:
  - `signed + raw_reward + dr_grpo + token`
  - `zero_two + raw_reward + dr_grpo + token`
  - `zero_two + advantage + dr_grpo + token`
  - optional comparison: `signed + raw_reward + dr_grpo + completion`
- Calibration outputs:
  - gate mean/min/max
  - gate p01/p05/p10/p25/p50/p75/p90/p95/p99
  - gate distribution split by correct vs wrong
  - gate distribution split by heuristic data-quality flags
  - keep-rate curve over candidate thresholds
  - estimated effective completions per raw epoch
- Data-quality flags:
  - no `\boxed{}` or missing extracted answer
  - max-length or truncation-looking completion
  - high repeated-line / repeated-tail heuristic
  - very short completion
  - exact duplicate within a question
- Threshold selection rule:
  - choose an open candidate set from calibration quantiles, not a fixed hand-written sweep.
  - include thresholds around the low-quality or wrong-completion gate distribution:
    - p50, p60, p70, p80, p90
  - include thresholds around the positive/effective-completion gate distribution:
    - p10, p20, p30
  - always include `0.50` as the neutral sigmoid boundary.
  - remove duplicate thresholds within `0.005`.
  - keep only thresholds whose estimated keep rate is between `0.10` and `0.80`.
- Decision gate:
  - proceed only if calibration shows at least one threshold with:
    - nonzero low-gate skips
    - estimated correct/effective keep rate above wrong/zero-effective keep rate
    - gibberish/truncation-flagged wrong completions mostly below threshold

### 1. Sanity Check
- Dataset: bad-teacher MATH rollout only.
- Model: `/scratch/shuai14/models/Qwen2.5-0.5B`
- Tiny run size: `target_effective_completions=32`
- Config:
  - `reward_coding=signed`
  - `training_signal=raw_reward`
  - `loss_type=dr_grpo`
  - `dg_gating=token`
  - `softdg_gate_threshold`: first threshold selected by calibration
- Expected outcome:
  - no DDP hang
  - token-level gate is logged
  - low-gate rows are skipped
  - effective update count reaches exactly 32
  - skipped rows zero both PG and KL
- Command template:
```bash
REWARD_CODING=signed TRAINING_SIGNAL=raw_reward \
LOSS_TYPE=dr_grpo DG_GATING=token \
SOFTDG_GATE_THRESHOLD=<calibrated_threshold> TARGET_EFFECTIVE_COMPLETIONS=32 \
sbatch SoftDG-Offline-to-the-bottom/run_math.sh train
```

### 2. Baseline
Reuse existing BC-all baseline on the same bad-teacher rollout.

Existing baseline:
- MATH accuracy: `0.3191`
- Success target: `0.3391`

No need to rerun unless code, dataset, tokenizer, or eval protocol changes.

### 3. Main Method
- Method: SoftDG threshold skipping with Dr.GRPO and token-level gating.
- Main config:
  - `loss_type=dr_grpo`
  - `dg_gating=token`
  - `target_effective_completions=48000`
  - max epochs: `20`

Primary sweeps:
- `signed + raw_reward`
- `zero_two + advantage`
- thresholds:
  - all calibrated thresholds for that config satisfying the keep-rate and selectivity rule

Why:
- `signed + raw_reward` directly tests whether Dr.GRPO plus token gating prevents the previous signed-reward collapse.
- `zero_two + advantage` is the required advantage-mode test under the new Dr.GRPO/token-gate setup.
- Thresholds must be chosen from measured gate distributions, because we do not know the surprisal scale in advance.

Metrics:
- MATH accuracy
- average response length
- improvement over BC
- effective updates
- raw completions scanned
- raw epochs used
- keep rate
- skipped low-gate count
- skipped zero-signal count
- gate mean/min/max
- gate p10/p50/p90
- final KL
- collapse indicators:
  - average response length
  - repeated/degenerate eval outputs

### 4. Conservative Comparison
Run positive-only raw-reward variants to check whether the new setup is stable even without negative rewards.

Configs:
- `zero_two + raw_reward + dr_grpo + token`
- thresholds:
  - calibrated thresholds from Milestone 0, filtered to keep-rate between `0.10` and `0.80`

Why:
- This tests whether token-level gating plus Dr.GRPO can improve over the prior best SoftDG result without introducing negative raw-reward gradients.
- Wrong completions still have zero raw reward, so collapse should not occur.

### 5. Ablation
Run only after the best main-method threshold is identified.

Ablations:
- `signed + raw_reward + dr_grpo + completion`
  - thresholds: calibrated completion-gate thresholds, not copied from token mode
  - isolates token-level gating
- `signed + raw_reward + grpo + token`
  - threshold: best calibrated token threshold
  - isolates Dr.GRPO
- `zero_two + advantage + dr_grpo + completion`
  - threshold: calibrated completion-gate threshold for this config
  - isolates token-level gating for the required advantage-mode test
- Optional:
  - `signed + advantage + dr_grpo + token`

Metrics:
- accuracy first
- collapse indicators second:
  - average eval length
  - repeated/degenerate outputs
  - final KL
- gate selectivity:
  - skipped low-gate count
  - keep rate
  - correct keep rate vs wrong keep rate

## Success Criterion
The method is successful if the best V2 SoftDG variant beats BC by at least **+2 absolute MATH accuracy points**.

Main success threshold:
- MATH accuracy `>= 0.3391`

Minimum useful result:
- signed raw-reward no longer collapses:
  - MATH accuracy `> 0.25`
  - average eval length `< 900`
  - low-gate skips are nonzero
  - `passes_gate_frac < 1.0`
  - correct keep rate is higher than wrong keep rate

Main comparison:
- best `signed + raw_reward + dr_grpo + token`
- best `zero_two + advantage + dr_grpo + token`
- best `zero_two + raw_reward + dr_grpo + token`
- best completion-gate ablation
- BC bad-teacher baseline

## Compute Budget
- GPU type: 4x L40S.
- Calibration first; do not launch full training before calibration.
- First pass: 1 seed per calibrated threshold.
- Final pass: 3 seeds only for the best config if it reaches at least BC-level accuracy or clearly avoids collapse.
- Max parallel runs: launch only after sanity check passes.
- Do not launch the ablation stage until the main threshold sweep identifies a best threshold.

## Output Format
Save results to:
- `SoftDG-Offline-to-the-bottom/outputs/gate_calibration_v2.jsonl`
- `SoftDG-Offline-to-the-bottom/outputs/gate_calibration_v2.csv`
- `SoftDG-Offline-to-the-bottom/outputs/results_v2.jsonl`
- `SoftDG-Offline-to-the-bottom/outputs/summary_v2.csv`

Each training run should record:
- `reward_coding`
- `training_signal`
- `loss_type`
- `dg_gating`
- `softdg_gate_threshold`
- `dg_temperature`
- `target_effective_completions`
- MATH accuracy mean/std/min/max
- average response length
- effective updates
- scanned completions
- keep rate
- skipped low-gate count
- skipped zero-signal count
- final KL

Each calibration row should record:
- `question_id`
- `run_id`
- reward and training signal
- gate
- surprisal mean
- token-gate mean/min/max
- completion length
- data-quality flags
