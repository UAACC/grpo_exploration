# Experiment Plan

## Goal
Test whether **SoftDG gate-threshold skipping** improves accuracy on bad-teacher MATH.

The goal is to beat behavior cloning by at least **+2 absolute MATH accuracy points**. Higher is better.

## Existing Code
- Main training file: `DG-offline/train.py`
- Evaluation file: `mixture_grpo/evaluate.py`
- Important modules:
  - `DG-offline/trainer.py`
  - `DG-offline/teacher_agnostic_loader.py`
  - `bc/train_bc.py`
- Config files:
  - `DG-offline/configs/accelerate_ddp_4gpu.yaml`
  - `DG-offline/run_math.sh`

## Implementation Idea
Create a new folder: `SoftDG-Offline-to-the-bottom/*`.

Do not directly modify `DG-offline/*`.

Add selectable reward coding:
- `reward_coding=zero_two`: wrong = `0`, correct = `2`
- `reward_coding=signed`: wrong = `-1`, correct = `+1`

Add selectable training signal:
- `training_signal=advantage`: use group-normalized advantage computed from the chosen reward coding.
- `training_signal=raw_reward`: use the chosen reward value directly.

SoftDG threshold rule:
- Compute `delight = training_signal * completion_surprisal`.
- Compute `gate = sigmoid(delight / eta)`.
- Skip completion-level backprop when `gate < softdg_gate_threshold`.
- For selected rows, keep current SoftDG signal: `gate * training_signal`.
- Count one effective update only when the selected completion has nonzero effective signal.
- Continue until `target_effective_completions=48000`.

Important edge case:
- With `reward_coding=zero_two` and `training_signal=raw_reward`, wrong completions have raw reward `0`, so they must not count as effective updates.

## Experiments

### 0. Sanity Check
- Dataset: bad-teacher MATH rollout only.
- Model: `/scratch/shuai14/models/Qwen2.5-0.5B`
- Tiny run size: `target_effective_completions=32`
- Expected outcome:
  - no DDP hang
  - low-gate rows skipped
  - zero-signal rows not counted
  - effective update count reaches exactly 32
- Command to run, if known:
```bash
REWARD_CODING=zero_two TRAINING_SIGNAL=advantage \
SOFTDG_GATE_THRESHOLD=0.2 TARGET_EFFECTIVE_COMPLETIONS=32 \
sbatch SoftDG-Offline-to-the-bottom/run_math.sh train
```

### 1. Baseline
- Method: BC-all on the same bad-teacher rollout.
- Hyperparameters:
  - 1 epoch
  - 48,000 completions
  - LR `3e-6`
  - LoRA r=32, alpha=32
- Metrics:
  - MATH accuracy over 30 eval runs
  - average response length
- Seeds: 1 initial, 3 final.

### 2. Main Method
- Method: SoftDG threshold skipping.
- Hyperparameters:
  - reward coding: compare `zero_two` vs `signed`
  - training signal: compare `advantage` vs `raw_reward`
  - threshold: open sweep
  - target effective completions: `48000`
- Metrics:
  - MATH accuracy
  - improvement over BC
  - effective updates
  - raw completions scanned
  - raw epochs used
  - keep rate
  - skipped low-gate count

### 3. Ablation
- Change:
  - Sweep thresholds for each reward/signal pair:
    - `0.00`, `0.05`, `0.10`, `0.15`, `0.20`, `0.25`, `0.30`, `0.40`
- Why:
  - We do not know the best gate cutoff.
  - Reward coding and training signal may interact strongly with the threshold.
- Metrics:
  - accuracy first
  - keep rate, raw epochs, and wall-clock second

## Success Criterion
The method is successful if the best SoftDG-threshold variant beats BC on the same bad-teacher rollout by at least **+2 absolute MATH accuracy points**.

Main comparison:
- best `zero_two + advantage`
- best `zero_two + raw_reward`
- best `signed + advantage`
- best `signed + raw_reward`
- BC bad-teacher baseline

## Compute Budget
- GPU type: 4x L40S.
- Max GPU hours: keep first sweep to 1 seed per config.
- Max parallel runs: launch only after sanity check passes.
- Do not launch full runs until sanity check passes.

## Output Format
Save results to:
- `SoftDG-Offline-to-the-bottom/outputs/results.jsonl`
- `SoftDG-Offline-to-the-bottom/outputs/summary.csv`
