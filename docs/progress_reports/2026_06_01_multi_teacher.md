# multi-teacher progress report: june 1, 2026

## summary

This document tracks MATH-500 performance for Qwen2.5-0.5B under different teacher choices. The goal is to compare BC, DG, GRPO-style methods, and related offline algorithms as the teacher pool changes.

Current completed multi-teacher result: DG with Qwen2.5-Math-7B-Instruct + Qwen2.5-Math-1.5B-Instruct reaches 33.15 ± 0.87% on MATH-500 over 30 greedy evaluation runs.

For reference, the best matching single-teacher base-student DG result from the May 27 report was 31.09 ± 0.97% with Qwen2.5-Math-7B-Instruct only. The multi-teacher DG run is therefore +2.06pp higher, and is also above the 0/2 reward Dr.GRPO run at 31.69 ± 0.75%.

---

## shared setup

| param | value |
|---|---|
| student | Qwen2.5-0.5B |
| eval | MATH-500, 30 runs, greedy (T=0.0) |
| primary teacher | Qwen2.5-Math-7B-Instruct |
| additional teacher for multi-teacher run | Qwen2.5-Math-1.5B-Instruct |
| 7B rollout file | `/scratch/shuai14/rollouts/math_teacher/rollouts_full.jsonl` |
| 1.5B rollout file | `/scratch/shuai14/rollouts/math_teacher/rollouts_math_Qwen2.5-Math-1.5B-Instruct_0.6.jsonl` |
| LoRA | r=32, alpha=32, targets=all-linear |
| beta | 0.001 |
| output checkpoint | `/scratch/shuai14/checkpoints/DG_offline_math_multi_teacher` |

The multi-teacher DG run loaded 108,000 completions from the two rollout files, inferred `num_generations=9`, and logged 84,204/108,000 correct rewards (78.0%).

Teacher names are inferred from the rollout file when needed. In particular, `rollouts_full.jsonl` maps to Qwen2.5-Math-7B-Instruct; model-specific rollout filenames use the model name embedded in the filename.

---

## rollout correctness by verifier

These are rollout-level correctness rates computed by the loader/verifier path used by each run. They are not the trained-student MATH-500 results below, and duplicate rows for the same rollout file mean the verifier changed, not the file. Counts are taken from the corresponding log headers; repeated DDP prints are counted once.

| teacher / rollout source | rollout file | completions | correct | verifier accuracy | source / verifier path |
|---|---|---:|---:|---:|---|
| Qwen2.5-Math-7B-Instruct | `rollouts_full.jsonl` | 48,000 | 37,717 | 78.6% | `logs/dg-offline-math-5077220.out` |
| Qwen2.5-Math-1.5B-Instruct | `rollouts_math_Qwen2.5-Math-1.5B-Instruct_0.6_pick4.jsonl` | 48,000 | 37,342 | 77.8% | `logs/dg-offline-math-5096479.out` |
| Qwen2.5-Math-1.5B-Instruct | `rollouts_math_Qwen2.5-Math-1.5B-Instruct_0.6_pick4.jsonl` | 48,000 | 37,440 | 78.0% | `logs/bc_math/run_bc_math.sh-5108858.out` |
| Phi-4-mini-instruct | `rollouts_math_Phi-4-mini-instruct_0.6_pick4.jsonl` | 48,000 | 31,180 | 65.0% | `logs/dg-offline-math-5108193.out` |
| Phi-4-mini-instruct | `rollouts_math_Phi-4-mini-instruct_0.6_pick4.jsonl` | 48,000 | 30,082 | 62.7% | `logs/bc_math/run_bc_math.sh-5102113.out` |
| DeepSeekMath-7B-RL | `rollouts_math_DeepSeekMath-7B-RL_0.7_pick4.jsonl` | 48,000 | 27,285 | 56.8% | `logs/dg-offline-math-5109977.out` |
| DeepSeekMath-7B-RL | `rollouts_math_DeepSeekMath-7B-RL_0.7_pick4.jsonl` | 48,000 | 4,524 | 9.4% | `logs/bc_math/run_bc_math.sh-5108019.out` |
| Qwen2.5-0.5B-Instruct | `rollouts_math_Qwen2.5-0.5B-Instruct_0.6_pick4.jsonl` | 48,000 | 16,844 | 35.1% | `logs/dg-offline-math-5100942.out` |
| Qwen2.5-0.5B-Instruct | `rollouts_math_Qwen2.5-0.5B-Instruct_0.6_pick4.jsonl` | 48,000 | 16,834 | 35.1% | `logs/bc_math/run_bc_math.sh-5100940.out` |
| Qwen2.5-Math-7B-Instruct + Qwen2.5-Math-1.5B-Instruct | `rollouts_full.jsonl` + `rollouts_math_Qwen2.5-Math-1.5B-Instruct_0.6.jsonl` | 108,000 | 84,204 | 78.0% | `logs/dg-offline-math-mt-5091241.out` |
| Qwen2.5-Math-7B-Instruct + Qwen2.5-Math-1.5B-Instruct | `rollouts_full.jsonl` + `rollouts_math_Qwen2.5-Math-1.5B-Instruct_0.6.jsonl` | 60,000 | 46,603 | 77.7% | `logs/bc_math/run_bc_math.sh-5097722.out` |

For MATH, DG recomputes correctness with the question-aware multi-candidate verifier, while BC uses the older single extracted-answer verifier. The source column is part of the number.

---

## results by teacher set

| teacher set | algorithm | job | reward regime | eta | loss | greedy ± std | notes |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Math-7B-Instruct | DG-offline | 5077220 | signed reward (-1/+1) | 1.0 | dr_grpo | 31.09 ± 0.97% | raw reward |
| Qwen2.5-Math-7B-Instruct | DG-offline | 5087846 | unsigned reward (0/2) | 1.0 | grpo | 29.37 ± 0.91% | advantage function |
| Qwen2.5-Math-7B-Instruct | AWR-adv | 5114570 | unsigned reward (0/2) | 1.0 | grpo | 29.46 ± 0.78% | advantage function |
| Qwen2.5-Math-7B-Instruct | AWR-rwd | 5115174 | signed reward (-1/1) | 1.0 | dr_grpo | 30.05 ± 0.91% | advantage function |
| Qwen2.5-Math-7B-Instruct | AWR-adv | 5115194 | unsigned reward (0/2) | 1.0 | dr_grpo | 29.44 ± 0.84% | advantage function |
| Qwen2.5-Math-7B-Instruct | RWR-adv | 5116088 | unsigned reward (0/2) | 1.0 | dr_grpo | 11.17 ± 0.75% | advantage function |
| Qwen2.5-Math-7B-Instruct | RWR-rwd | 5116121 | signed reward (-1/1) | 1.0 | dr_grpo | 28.82 ± 0.85% | advantage function |
| Qwen2.5-Math-7B-Instruct | BC-all | 5077583 | n/a | n/a | cross-entropy | 29.79 ± 1.03% | all teacher completions |
| Qwen2.5-Math-7B-Instruct | BC-correct-only | 5077587 | n/a | n/a | cross-entropy | 28.63 ± 0.92% | correct teacher completions only |
| Phi-4-mini-instruct | DG-offline | 5102114 | signed_reward | 1.0 | dr_grpo | 29.13 ± 0.87% | teacher inferred from `rollouts_math_Phi-4-mini-instruct_0.6_pick4.jsonl` |
| Phi-4-mini-instruct | DG-offline | 5108193 | current | 1.0 | dr_grpo | 30.49 ± 1.03% | log-confirmed current-regime rerun of 5102114; teacher inferred from `rollouts_math_Phi-4-mini-instruct_0.6_pick4.jsonl`; 48k rows, 65.0% correct rewards |
| Phi-4-mini-instruct | BC-all | 5102113 | n/a | 1.0 | dr_grpo | 30.07 ± 0.87% | teacher inferred from `rollouts_math_Phi-4-mini-instruct_0.6_pick4.jsonl` |
| DeepSeekMath-7B-RL | DG-offline | 5107885 | signed_reward | 1.0 | dr_grpo | 23.06 ± 0.72% | teacher inferred from `rollouts_math_DeepSeekMath-7B-RL_0.7_pick4.jsonl`; 48k rows, 56.8% correct rewards |
| DeepSeekMath-7B-RL | DG-offline | 5109977 | current | 1.0 | dr_grpo | 23.93 ± 0.68% | log-confirmed current-regime rerun of 5107885; teacher inferred from `rollouts_math_DeepSeekMath-7B-RL_0.7_pick4.jsonl`; 48k rows, 56.8% correct rewards |
| DeepSeekMath-7B-RL | BC-all | 5108019 | n/a | n/a | cross-entropy | 26.12 ± 0.81% | teacher inferred from `rollouts_math_DeepSeekMath-7B-RL_0.7_pick4.jsonl`; 48k completions, 9.4% correct |
| Qwen2.5-0.5B-Instruct | DG-offline | 5100942 | signed_reward | 1.0 | dr_grpo | 19.63 ± 0.98% | teacher inferred from `rollouts_math_Qwen2.5-0.5B-Instruct_0.6_pick4.jsonl`; 48k rows, 35.1% correct rewards |
| Qwen2.5-0.5B-Instruct | BC-all | 5100940 | n/a | n/a | cross-entropy | 32.04 ± 0.84% | teacher inferred from `rollouts_math_Qwen2.5-0.5B-Instruct_0.6_pick4.jsonl`; 48k completions, 35.1% correct |
| Qwen2.5-0.5B-Instruct | AWR | 5112417 | unsigned_reward | 1.0 | grpo | 31.65 ± 0.98% | teacher inferred from `rollouts_math_Qwen2.5-0.5B-Instruct_0.6_pick4.jsonl`; 48k completions, 35.1% correct |
| Qwen2.5-Math-1.5B-Instruct | DG-offline | 5096479 | signed_reward | 1.0 | dr_grpo | 31.53 ± 1.04% | teacher inferred from `rollouts_math_Qwen2.5-Math-1.5B-Instruct_0.6_pick4.jsonl`; 48k rows, 77.8% correct rewards |
| Qwen2.5-Math-7B-Instruct + Qwen2.5-Math-1.5B-Instruct | DG-offline | 5091241 | signed reward (-2/+2) | 1.0 | dr_grpo | 33.15 ± 0.87% | multi-teacher rollouts, completion-level DG gating |
| Qwen2.5-Math-7B-Instruct + Qwen2.5-Math-1.5B-Instruct | BC-all | 5097722 | n/a | n/a | cross-entropy | 31.05 ± 0.84% | multi-teacher BC-all; 60k completions, 77.7% correct |
| Qwen2.5-Math-1.5B-Instruct | BC-all | 5108858 | n/a | n/a | cross-entropy | 31.33 ± 0.87% | teacher inferred from `rollouts_math_Qwen2.5-Math-1.5B-Instruct_0.6_pick4.jsonl`; BC loader: 37,440/48,000 correct (78.0%); DG-canonical: 37,342/48,000 (77.8%) |

## DG Phi-4-mini teacher details

### 5108193: current reward regime

**method**: DG-offline on the same Phi-4-mini-instruct rollout file as job 5102114, but with `Training regime: current` rather than `signed_reward`. The log header confirms `Training regime: current` and `Loss type: dr_grpo`. The teacher is inferred from `rollouts_math_Phi-4-mini-instruct_0.6_pick4.jsonl`; only `rollouts_full.jsonl` maps to Qwen2.5-Math-7B-Instruct. The run uses `DG_ETA=1.0` and `dr_grpo`.

| param | value |
|---|---|
| job | 5108193 |
| status | complete |
| student | Qwen2.5-0.5B |
| teacher | Phi-4-mini-instruct |
| rollout file | `/scratch/shuai14/rollouts/math_teacher/rollouts_math_Phi-4-mini-instruct_0.6_pick4.jsonl` |
| DG_ETA | 1.0 |
| reward regime | current |
| loss type | dr_grpo |
| dataset rows | 48,000 |
| correct rewards | 31,180/48,000 (65.0%) |
| output checkpoint | `/scratch/shuai14/checkpoints/DG_offline_math` |
| eval dataset | MATH-500 test split |

| job | greedy ± std | min | max | avg length |
|---|---|---|---|---|
| 5108193 | 30.49 ± 1.03% | 28.00% | 32.60% | 699.7 |

---

**method**: DG-offline on Phi-4-mini-instruct rollouts. The teacher is inferred from the rollout file `rollouts_math_Phi-4-mini-instruct_0.6_pick4.jsonl`; by the current naming rule, only `rollouts_full.jsonl` corresponds to Qwen2.5-Math-7B-Instruct. The run uses `DG_ETA=1.0`, `signed_reward`, and `dr_grpo`.

| param | value |
|---|---|
| job | 5102114 |
| student | Qwen2.5-0.5B |
| teacher | Phi-4-mini-instruct |
| rollout file | `/scratch/shuai14/rollouts/math_teacher/rollouts_math_Phi-4-mini-instruct_0.6_pick4.jsonl` |
| DG_ETA | 1.0 |
| reward regime | signed_reward |
| loss type | dr_grpo |
| dataset rows | 48,000 |
| eval dataset | MATH-500 test split |

| job | greedy ± std | min | max | avg length |
|---|---|---|---|---|
| 5102114 | 29.13 ± 0.87% | 27.40% | 31.20% | 489.3 |

This Phi-4-mini teacher run lands below the 7B-teacher DG references in this document: 29.13% vs 31.09% for signed-reward DG and 31.69% for the 0/2 Dr.GRPO run.

## DeepSeekMath-7B-RL teacher details

### 5107885: DG signed reward

**method**: DG-offline on DeepSeekMath-7B-RL rollouts. The teacher is inferred from `rollouts_math_DeepSeekMath-7B-RL_0.7_pick4.jsonl`; only `rollouts_full.jsonl` maps to Qwen2.5-Math-7B-Instruct. The run uses `DG_ETA=1.0`, `signed_reward`, and `dr_grpo`.

| param | value |
|---|---|
| job | 5107885 |
| status | complete |
| student | Qwen2.5-0.5B |
| teacher | DeepSeekMath-7B-RL |
| rollout file | `/scratch/shuai14/rollouts/math_teacher/rollouts_math_DeepSeekMath-7B-RL_0.7_pick4.jsonl` |
| DG_ETA | 1.0 |
| reward regime | signed_reward |
| loss type | dr_grpo |
| dataset rows | 48,000 |
| correct rewards | 27,285/48,000 (56.8%) |
| output checkpoint | `/scratch/shuai14/checkpoints/DG_offline_math` |
| eval dataset | MATH-500 test split |

| job | greedy ± std | min | max | avg length |
|---|---|---|---|---|
| 5107885 | 23.06 ± 0.72% | 21.40% | 24.40% | 261.5 |

### 5109977: DG current reward regime

**method**: DG-offline on the same DeepSeekMath-7B-RL pick4 rollout file as job 5107885, but with `Training regime: current` rather than `signed_reward`. The log header confirms `Training regime: current` and `Loss type: dr_grpo`. The submitted command was `TRAINING_REGIME=current LOSS_TYPE=dr_grpo DG_ETA=1.0 ROLLOUT_PATH=/scratch/shuai14/rollouts/math_teacher/rollouts_math_DeepSeekMath-7B-RL_0.7_pick4.jsonl sbatch DG-offline/run_math.sh train`.

| param | value |
|---|---|
| job | 5109977 |
| status | complete |
| student | Qwen2.5-0.5B |
| teacher | DeepSeekMath-7B-RL |
| rollout file | `/scratch/shuai14/rollouts/math_teacher/rollouts_math_DeepSeekMath-7B-RL_0.7_pick4.jsonl` |
| DG_ETA | 1.0 |
| reward regime | current |
| loss type | dr_grpo |
| dataset rows | 48,000 |
| correct rewards | 27,285/48,000 (56.8%) |
| log path | `logs/dg-offline-math-5109977.out` |
| eval result | 23.93 ± 0.68% |

| job | greedy ± std | min | max | avg length |
|---|---|---|---|---|
| 5109977 | 23.93 ± 0.68% | 22.00% | 25.40% | 523.0 |

### 5108019: BC-all

**method**: offline behavior cloning on all DeepSeekMath-7B-RL teacher completions. The teacher is inferred from `rollouts_math_DeepSeekMath-7B-RL_0.7_pick4.jsonl`.

| param | value |
|---|---|
| job | 5108019 |
| status | complete |
| student | Qwen2.5-0.5B |
| teacher | DeepSeekMath-7B-RL |
| rollout file | `/scratch/shuai14/rollouts/math_teacher/rollouts_math_DeepSeekMath-7B-RL_0.7_pick4.jsonl` |
| algorithm | BC-all |
| loaded completions | 48,000 |
| correct completions | 4,524/48,000 (9.4%) |
| max_length | 2304 |
| output checkpoint | `/scratch/shuai14/checkpoints/bc_math` |
| eval dataset | MATH-500 test split |

| job | greedy ± std | min | max | avg length |
|---|---|---|---|---|
| 5108019 | 26.12 ± 0.81% | 24.20% | 27.60% | 583.6 |

## Qwen2.5-0.5B-Instruct teacher details

### 5100942: DG signed reward

**method**: DG-offline on Qwen2.5-0.5B-Instruct rollouts. The teacher is inferred from `rollouts_math_Qwen2.5-0.5B-Instruct_0.6_pick4.jsonl`; only `rollouts_full.jsonl` maps to Qwen2.5-Math-7B-Instruct. The run uses `DG_ETA=1.0`, `signed_reward`, and `dr_grpo`.

| param | value |
|---|---|
| job | 5100942 |
| student | Qwen2.5-0.5B |
| teacher | Qwen2.5-0.5B-Instruct |
| rollout file | `/scratch/shuai14/rollouts/math_teacher/rollouts_math_Qwen2.5-0.5B-Instruct_0.6_pick4.jsonl` |
| DG_ETA | 1.0 |
| reward regime | signed_reward |
| loss type | dr_grpo |
| dataset rows | 48,000 |
| correct rewards | 16,844/48,000 (35.1%) |
| output checkpoint | `/scratch/shuai14/checkpoints/DG_offline_math` |
| eval dataset | MATH-500 test split |

| job | greedy ± std | min | max | avg length |
|---|---|---|---|---|
| 5100942 | 19.63 ± 0.98% | 18.20% | 21.60% | 1382.1 |

### 5100940: BC-all

**method**: offline behavior cloning on all Qwen2.5-0.5B-Instruct teacher completions. The teacher is inferred from `rollouts_math_Qwen2.5-0.5B-Instruct_0.6_pick4.jsonl`.

| param | value |
|---|---|
| job | 5100940 |
| student | Qwen2.5-0.5B |
| teacher | Qwen2.5-0.5B-Instruct |
| rollout file | `/scratch/shuai14/rollouts/math_teacher/rollouts_math_Qwen2.5-0.5B-Instruct_0.6_pick4.jsonl` |
| algorithm | BC-all |
| loaded completions | 48,000 |
| correct completions | 16,834/48,000 (35.1%) |
| max_length | 2304 |
| output checkpoint | `/scratch/shuai14/checkpoints/bc_math` |
| eval dataset | MATH-500 test split |

| job | greedy ± std | min | max | avg length |
|---|---|---|---|---|
| 5100940 | 32.04 ± 0.84% | 30.20% | 33.60% | 659.9 |

### 5112417: AWR

**method**: AWR-offline on all Qwen2.5-0.5B-Instruct teacher completions. The teacher is inferred from `rollouts_math_Qwen2.5-0.5B-Instruct_0.6_pick4.jsonl`; the log header still prints the default teacher path, but the rollout file determines the documented teacher.

| param | value |
|---|---|
| job | 5112417 |
| student | Qwen2.5-0.5B |
| teacher | Qwen2.5-0.5B-Instruct |
| rollout file | `/scratch/shuai14/rollouts/math_teacher/rollouts_math_Qwen2.5-0.5B-Instruct_0.6_pick4.jsonl` |
| algorithm | AWR |
| AWR eta | 1.0 |
| AWR gating | completion |
| loaded completions | 48,000 |
| correct completions | 16,834/48,000 (35.1%) |
| output checkpoint | `/scratch/shuai14/checkpoints/AWR_offline_math` |
| eval dataset | MATH-500 test split |

| job | greedy ± std | min | max | avg length |
|---|---|---|---|---|
| 5112417 | 31.65 ± 0.98% | 29.80% | 33.80% | 977.0 |

For this teacher, BC-all and AWR are close: 32.04% vs 31.65%. Both substantially outperform DG signed reward at 19.63%. The DG model also generates much longer completions on average (1382.1 tokens) than AWR (977.0) and BC-all (659.9), which is worth checking when comparing failure modes.

## Qwen2.5-Math-1.5B-Instruct teacher details

### 5096479: DG signed reward

**method**: DG-offline on Qwen2.5-Math-1.5B-Instruct rollouts. The teacher is inferred from `rollouts_math_Qwen2.5-Math-1.5B-Instruct_0.6_pick4.jsonl`; only `rollouts_full.jsonl` maps to Qwen2.5-Math-7B-Instruct. The run uses `DG_ETA=1.0`, `signed_reward`, and `dr_grpo`.

| param | value |
|---|---|
| job | 5096479 |
| student | Qwen2.5-0.5B |
| teacher | Qwen2.5-Math-1.5B-Instruct |
| rollout file | `/scratch/shuai14/rollouts/math_teacher/rollouts_math_Qwen2.5-Math-1.5B-Instruct_0.6_pick4.jsonl` |
| DG_ETA | 1.0 |
| reward regime | signed_reward |
| loss type | dr_grpo |
| dataset rows | 48,000 |
| correct rewards | 37,342/48,000 (77.8%) |
| output checkpoint | `/scratch/shuai14/checkpoints/DG_offline_math` |
| eval dataset | MATH-500 test split |

| job | greedy ± std | min | max | avg length |
|---|---|---|---|---|
| 5096479 | 31.53 ± 1.04% | 29.40% | 33.60% | 653.4 |

### 5108858: BC-all

**method**: offline behavior cloning on all Qwen2.5-Math-1.5B-Instruct teacher completions. The teacher is inferred from `rollouts_math_Qwen2.5-Math-1.5B-Instruct_0.6_pick4.jsonl`.

| param | value |
|---|---|
| job | 5108858 |
| student | Qwen2.5-0.5B |
| teacher | Qwen2.5-Math-1.5B-Instruct |
| rollout file | `/scratch/shuai14/rollouts/math_teacher/rollouts_math_Qwen2.5-Math-1.5B-Instruct_0.6_pick4.jsonl` |
| algorithm | BC-all |
| loaded completions | 48,000 |
| BC-loader correct completions | 37,440/48,000 (78.0%) |
| DG-canonical rollout correctness | 37,342/48,000 (77.8%) |
| max_length | 2304 |
| output checkpoint | `/scratch/shuai14/checkpoints/bc_math` |
| eval dataset | MATH-500 test split |

| job | greedy ± std | min | max | avg length |
|---|---|---|---|---|
| 5108858 | 31.33 ± 0.87% | 29.40% | 32.80% | 682.6 |

---

## multi-teacher BC-all details

**method**: offline behavior cloning on all completions from the combined Qwen2.5-Math-7B-Instruct and Qwen2.5-Math-1.5B-Instruct rollout files. `rollouts_full.jsonl` maps to Qwen2.5-Math-7B-Instruct; the second rollout file identifies Qwen2.5-Math-1.5B-Instruct.

| param | value |
|---|---|
| job | 5097722 |
| student | Qwen2.5-0.5B |
| teachers | Qwen2.5-Math-7B-Instruct + Qwen2.5-Math-1.5B-Instruct |
| rollout files | `/scratch/shuai14/rollouts/math_teacher/rollouts_full.jsonl`, `/scratch/shuai14/rollouts/math_teacher/rollouts_math_Qwen2.5-Math-1.5B-Instruct_0.6.jsonl` |
| algorithm | BC-all |
| loaded completions | 60,000 |
| correct completions | 46,603/60,000 (77.7%) |
| max_length | 2304 |
| output checkpoint | `/scratch/shuai14/checkpoints/bc_math` |
| eval dataset | MATH-500 test split |

| job | greedy ± std | min | max | avg length |
|---|---|---|---|---|
| 5097722 | 31.05 ± 0.84% | 29.60% | 32.60% | 748.4 |

---

## DG multi-teacher details

**method**: DG-offline on multi-teacher rollouts. Teacher trajectories come from both Qwen2.5-Math-7B-Instruct and Qwen2.5-Math-1.5B-Instruct. The reward regime is signed reward with r in {-2, +2}, DG eta is 1.0, and the loss is `dr_grpo`.

| param | value |
|---|---|
| job | 5091241 |
| student | Qwen2.5-0.5B |
| teachers | Qwen2.5-Math-7B-Instruct + Qwen2.5-Math-1.5B-Instruct |
| DG_ETA | 1.0 |
| DG gating | completion |
| reward regime | signed reward (-2/+2) |
| loss type | dr_grpo |
| num_generations | 9 inferred from rollout files |
| completions | 108,000 |
| correct rewards | 84,204/108,000 (78.0%) |
| eval dataset | MATH-500 test split |

| job | greedy ± std | min | max | avg length |
|---|---|---|---|---|
| 5091241 | 33.15 ± 0.87% | 31.40% | 35.00% | 648.2 |

The multi-teacher DG run is the first row in this document that directly tests whether adding the smaller math teacher improves the base student relative to the 7B-only teacher. It clears the 7B-only signed-reward DG reference by about 2pp, though the comparison is not isolated to teacher set alone because the multi-teacher run also uses the combined rollout pool and the signed reward scale noted above.


---

## deployed jobs snapshot: june 1

Slurm accounting gives the following submit commands for the currently deployed jobs. For jobs without logs yet, the configuration below is inferred from the script defaults at documentation time. Environment overrides are recorded below when available; for jobs without logs yet, these should be checked against the first log header when the job starts.

| job | state | submit command | script/log path | inferred config | report status |
|---|---|---|---|---|---|
| 5107885 | complete | `sbatch DG-offline/run_math.sh train` | `logs/dg-offline-math-5107885.out` | DG-offline, DeepSeekMath-7B-RL rollouts, `signed_reward`, `DG_ETA=1.0`, `dr_grpo`; 27,285/48,000 correct rewards (56.8%) | 23.06 ± 0.72% |
| 5108193 | complete | `sbatch DG-offline/run_math.sh train` | `logs/dg-offline-math-5108193.out` | DG-offline, Phi-4-mini-instruct rollouts, `current`, `DG_ETA=1.0`, `dr_grpo`; 31,180/48,000 correct rewards (65.0%) | 30.49 ± 1.03% |
| 5109977 | complete | `TRAINING_REGIME=current LOSS_TYPE=dr_grpo DG_ETA=1.0 ROLLOUT_PATH=/scratch/shuai14/rollouts/math_teacher/rollouts_math_DeepSeekMath-7B-RL_0.7_pick4.jsonl sbatch DG-offline/run_math.sh train` | `logs/dg-offline-math-5109977.out` | DG-offline, DeepSeekMath-7B-RL pick4 rollouts, Qwen2.5-0.5B student, `current`, `DG_ETA=1.0`, `dr_grpo`; 27,285/48,000 correct rewards (56.8%) | 23.93 ± 0.68% |
| 5108858 | complete | `sbatch bc/run_bc_math.sh train` | `logs/bc_math/run_bc_math.sh-5108858.out` | BC-all, Qwen2.5-Math-1.5B-Instruct pick4 rollouts, Qwen2.5-0.5B student; BC loader 37,440/48,000 correct (78.0%); DG-canonical 37,342/48,000 (77.8%) | 31.33 ± 0.87% |
| 5108994 | pending | `TRAINING_REGIME=current LOSS_TYPE=dr_grpo DG_ETA=1.0 ROLLOUT_PATH=/scratch/shuai14/rollouts/math_teacher/rollouts_math_Qwen2.5-0.5B-Instruct_0.6.jsonl sbatch DG-offline/run_math.sh train` | `logs/dg-offline-math-5108994.out` | DG-offline, Qwen2.5-0.5B student, Qwen2.5-0.5B-Instruct rollouts, `current`, `DG_ETA=1.0`, `dr_grpo` | awaiting log |
| 5109000 | pending | `ROLLOUT_PATH=/scratch/shuai14/rollouts/math_teacher/rollouts_math_Qwen2.5-0.5B-Instruct_0.6.jsonl sbatch RWR-offline/run_math.sh train` | `logs/rwr-offline-math-5109000.out` | RWR, Qwen2.5-0.5B-Instruct student, Qwen2.5-0.5B-Instruct rollouts, eta=1.0, lr=3e-6 | awaiting log |
| 5109002 | pending | `DG_ETA=1.0 ROLLOUT_PATH=/scratch/shuai14/rollouts/math_teacher/rollouts_math_Qwen2.5-0.5B-Instruct_0.6.jsonl sbatch AWR-offline/run_math.sh train` | `logs/AWR-offline-math-5109002.out` | AWR, Qwen2.5-0.5B-Instruct student, Qwen2.5-0.5B-Instruct rollouts, eta=1.0, lr=6e-7 | awaiting log |
| 5112417 | complete | `AWR-offline/run_math.sh train` inferred from log name/header | `logs/AWR-offline-math-5112417.out` | AWR, Qwen2.5-0.5B student, Qwen2.5-0.5B-Instruct pick4 rollouts, eta=1.0, completion gating; 16,834/48,000 correct (35.1%) | 31.65 ± 0.98% |
