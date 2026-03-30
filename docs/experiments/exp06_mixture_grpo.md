# GSM8K GRPO Experiment Analysis

## Models
- **Student**: Qwen2.5-0.5B-Instruct
- **Teacher**: Qwen2.5-Math-7B-Instruct (95.1% accuracy on GSM8K train)
- **Baseline**: Student without training

## Experiments Overview

| | Online GRPO | Offline GRPO | Method A (Unified) | Method B (Weighted) |
|---|---|---|---|---|
| **Job ID** | (completed) | 4301147 | 4301405 | 4301406 |
| **Code** | `online_grpo/` | `offline_grpo/` | `mixture_grpo/method_A_unified/` | `mixture_grpo/method_B_weighted/` |
| **Output** | `online_grpo_gsm8k` | `offline_grpo_gsm8k` | `mixture_A_unified` | `mixture_B_weighted` |

### Key Hyperparameter Differences

| Parameter | Online GRPO | Offline GRPO | Method A | Method B |
|---|---|---|---|---|
| **Student online rollouts** | 5/prompt | 0 | 5/prompt | 5/prompt |
| **Teacher offline rollouts** | 0 | 5/prompt | 5/prompt | 5/prompt |
| **beta (KL)** | **0.001** | **0.1** | **0.01** | **0.01** |
| **learning_rate** | 3e-6 | **5e-6** | 3e-6 | 3e-6 |
| **num_train_epochs** | 15 | **1** | 15 | 15 |
| **offline_weight (lambda)** | N/A | N/A | N/A | **0.3** |
| per_device_batch_size | 5 | 5 | 5 | 5 |
| gradient_accumulation | 2 | 2 | 2 | 2 |
| max_completion_length | 1024 | 1024 | 1024 | 1024 |
| max_grad_norm | 1.0 | 1.0 | 1.0 | 1.0 |
| LoRA r/alpha | 32/32 | 32/32 | 32/32 | 32/32 |
| temperature (generation) | 0.7 | 0.7 | 0.7 | 0.7 |

### Algorithmic Differences

**Online GRPO**: Standard GRPO. Student generates G completions per prompt online. Advantage = group-normalized reward within G student completions. Single loss.

**Offline GRPO**: Pure offline. Uses pre-generated teacher rollouts only. Advantage = group-normalized reward within K teacher completions. Importance ratio = pi_student / pi_teacher.

**Method A (Unified Mixture)**: Pools G student + K teacher completions into one group of G+K. Unified advantage baseline (mixed mean/std across both). Single loss over all G+K completions. Each uses its own behavior logprobs for importance ratio.

**Method B (Weighted Mixture)**: Separate online and offline losses. L = L_online + lambda * L_offline. Student advantage uses student-only group stats. Teacher advantage uses student's online mean as baseline (linking teacher signal to student ability). Clipped PPO surrogate for teacher completions.

## Notable Differences to Investigate

1. **beta差异很大**: Online=0.001, Offline=0.1, Mixture=0.01. KL penalty强度差100倍。这会显著影响student偏离reference model的程度。
2. **lr差异**: Offline用5e-6 (更大), 其他用3e-6。
3. **epochs差异**: Offline只训1个epoch, 其他训15个。Offline数据量固定不变，多epoch可能过拟合。
4. **Method A vs B的核心区别**: advantage的计算方式不同。A把teacher和student混在一起算baseline; B分开算，teacher的advantage用student的online统计量做baseline。

## Results

### Existing Results
- **Baseline (no training)**: ~35% (TODO: confirm)
- **Online GRPO**: 55.82% (best checkpoint)
- **VERL benchmark**: 54.3%

### Pending Results
- **Offline GRPO**: Job 4301147 (running)
- **Method A**: Job 4301405 (running)
- **Method B**: Job 4301406 (pending)

### Evaluation Commands
```bash
cd /home/shuai14/projects/aip-szepesva/shuai14/backup_dongheng/mixture_grpo

# Method A
sbatch --job-name=mixture-A-eval run_method_A.sh eval

# Method B
sbatch --job-name=mixture-B-eval run_method_B.sh eval

# Offline GRPO
cd /home/shuai14/projects/aip-szepesva/shuai14/backup_dongheng/offline_grpo
sbatch --job-name=gsm8k-eval run_gsm8k_offline.sh eval
```

### Results Table (fill in after evaluation)

| Method | GSM8K Test Accuracy | Std | Avg Length | Notes |
|---|---|---|---|---|
| Baseline (no training) | | | | |
| Online GRPO | 55.82% | | | beta=0.001 |
| Offline GRPO | | | | beta=0.1, lr=5e-6, 1 epoch |
| Method A (Unified) | | | | beta=0.01 |
| Method B (Weighted, λ=0.3) | | | | beta=0.01 |

## Analysis Questions

1. **Does mixing teacher rollouts help vs pure online?** Compare Method A/B vs Online GRPO.
2. **Does mixing help vs pure offline?** Compare Method A/B vs Offline GRPO.
3. **Unified vs Weighted mixing?** Compare Method A vs Method B.
4. **How much does beta matter?** The three setups use very different beta values.
5. **Is the teacher signal useful?** Teacher has 95.1% accuracy vs student's ~35% baseline. The gap is large — does this help or hurt (distribution mismatch)?

## WandB Projects
- Online GRPO: `online-grpo-gsm8k`
- Offline GRPO: `offline-grpo-gsm8k`
- Method A: `mixture-grpo-A`
- Method B: `mixture-grpo-B`
