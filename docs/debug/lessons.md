# Mixture GRPO Lessons Learned

## Lesson 1: TRL generation batch vs training batch — OOM的真正来源

**问题**: Method A MATH 反复 OOM，尝试把 `per_device_train_batch_size` 从 2 减到 1（同时 `gradient_accumulation_steps` 从 8 加到 16），但仍然 OOM，报完全相同的错误：`Tried to allocate 14.24 GiB`。

**错误位置**: `trl/trainer/grpo_trainer.py` → `_get_per_token_logps_and_entropies` → `logits = logits / self.temperature`

**为什么减 batch 没用**:

减小 `per_device_train_batch_size` 只影响 TRL **split 之后**每个 accumulation step 的大小。但 OOM 发生在 **split 之前**的 `_generate_and_score_completions` 阶段，这时处理的是整个 **generation batch**。

Generation batch 大小 = `per_device_train_batch_size × gradient_accumulation_steps`

所以 batch=2, accum=8 和 batch=1, accum=16 的 generation batch 都是 16 unique prompts。

Method A 每个 prompt 有 8 个 completion（4 student + 4 teacher），所以 generation batch = 16 × 8 = **128 sequences**。

`_get_per_token_logps_and_entropies` 默认对这 128 个 sequence 做**一次** forward pass。logits tensor 大小：
```
128 sequences × 786 tokens × 151936 vocab × 2 bytes (bf16) = 14.24 GiB
```

这就是 OOM 分配的精确数字。

**解决**: TRL 的 `_get_per_token_logps_and_entropies` 有一个 `batch_size` 参数，可以把 forward pass 分成多个 chunk：

```python
# 旧（默认处理全部 128 sequences）:
ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
    model, prompt_completion_ids, attention_mask, logits_to_keep,
)

# 新（每次处理 16 sequences，做 8 次 forward）:
chunk_bs = self.args.per_device_train_batch_size * (self._num_teacher + self.num_generations)
ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
    model, prompt_completion_ids, attention_mask, logits_to_keep,
    batch_size=chunk_bs,
)
```

每个 chunk 的 logits = 16 × 786 × 151936 × 2 ≈ **1.8 GiB**，轻松放进 L40s。

**教训**:

1. **OOM 要看分配大小，不要盲目减 batch**。`14.24 GiB` 这个数字不管 batch=1 还是 batch=2 都一样 → 说明瓶颈不在 training batch size
2. **TRL 有两个 batch 概念**：
   - Generation batch（split 前）= `per_device_batch × grad_accum × completions_per_prompt`
   - Training batch（split 后）= `per_device_batch × completions_per_prompt`
   - OOM 在 split 前 → 调 per_device_batch 无效，因为 grad_accum 会同比增大来保持 effective batch
3. **先读框架源码再 debug**。TRL 的 `_get_per_token_logps_and_entropies` 已经提供了 `batch_size` 参数用于分 chunk，只是默认没启用
4. **Method B 不受影响**是因为它只对 student completions（4 per prompt）算 ref logprobs，不包含 teacher completions。128 vs 64 sequences 的区别

**涉及文件**: `method_A_unified/trainer.py` — `_get_ref_logprobs` 和 `_generate_and_score_completions` 中对 ref logprobs 的调用

---

## Lesson 2: KL penalty 只能用 on-policy 样本估计

**问题**: Method A 把 student + teacher completions 合并成一个 unified batch，TRL 的 `_compute_loss` 对**每一行**都算 KL penalty。但 teacher completions 来自 7B teacher，不是 student 策略的样本。

**为什么这不对**:

KL(π_θ || π_ref) = E_{y ~ π_θ}[log π_θ(y) - log π_ref(y)]

关键：期望下面的 y 必须来自 π_θ。Teacher completions 来自 π_teacher，在 student 不太可能生成的区域有高概率。用这些样本估计 KL 会**偏向高估**这些区域的 KL，等于惩罚 student 学习 teacher 的风格。

**解决**: 加 `kl_mask`（student=1.0, teacher=0.0），在 `_compute_loss` 里 `per_token_kl *= kl_mask`。

**教训**: 混合不同来源的数据进同一个 batch 时，检查所有 batch-level 操作是否对每种数据都合理。

---

## Lesson 3: 零方差保护可能误杀有效信号

**问题**: Method B 计算 teacher advantage 时用了 student 的 (mean, std) 做归一化。当所有 student reward 相同（std=0）时，代码把 teacher advantage 也设为 0：

```python
tea_adv = (tea_reward - mean_r) / (std_r + eps) if std_r > eps else 0.0
```

**为什么这有害**: 在 MATH 上 student baseline 只有 ~5-10%，4 个 student 全错的概率 ≈ 75%。这意味着 75% 的 prompt 上 teacher signal 被完全杀死——而这恰恰是 teacher 最有价值的地方。

**解决**: `else 0.0` → `else (tea_reward - mean_r)`，用未归一化的差值替代。

**教训**: 零方差保护对 group **内部**是合理的（无法区分），但对 group **外部**的 baseline 比较不应该适用。

---

## Lesson 4: Importance ratio 在大分布差异下失效

**问题**: Method B 的 offline loss 用 importance ratio `π_θ/π_teacher`。Student 0.5B 和 teacher 7B 在 teacher 生成的 token 上概率差异巨大：

```
单 token: π_student/π_teacher ≈ exp(-2.5) ≈ 0.08
PPO clip: min(0.08 * A, 0.8 * A) = 0.08 * A  ← clip 不起作用
梯度系数只有正常的 8%
```

**为什么 PPO clip 救不了**: Clip [0.8, 1.2] 是**上限保护**（防 ratio 太大），不是下限保护。ratio=0.08 远小于 0.8 时，`min(ratio*A, 0.8*A) = ratio*A`，clip 形同虚设。

**教训**: Importance sampling 要求两个分布有足够的重叠。0.5B vs 7B 重叠极小 → ratio → 0 → 梯度消失。考虑用 advantage-weighted SFT 替代。

---

## Lesson 5: Sequence-level importance ratio → 数值下溢

**问题**: Method B offline loss 早期用 sequence-level ratio：把每个 token 的 log ratio 加起来再 exp。100 个 token 累加后 `exp(-250) ≈ 0`。

**为什么 per-token 能 work**: 单 token `exp(-2.5) = 0.08`，clip 到 0.8 后仍有梯度。每个 token 独立处理，不受序列长度影响。

**教训**: RL 中的 importance ratio 必须 per-token 处理。Sequence-level 的乘积在长序列上必然归零。

---

## Lesson 6: TRL split 机制只处理 output dict 里的数据

**问题**: Method B 把 teacher 数据存在 `self._current_teacher_batch`（实例变量），不在 `_generate_and_score_completions` 的返回 dict 里。TRL 的 gradient accumulation split 只切分 output dict → teacher 数据没被 split → 每个 accumulation step 都对全量 teacher 做 forward → OOM。

**解决**: 把 teacher 数据放进 output dict（加 `teacher_` 前缀），让 TRL 自动 split。

**教训**: 用 side-channel（实例变量）传递数据会绕过 TRL 的 split/accumulation 机制。所有需要跟 batch 同步切分的数据都应该放进 output dict。

---

## Lesson 7: TRL batch 去重错误

**问题**: 误以为需要对 TRL dataloader 的重复 prompt 做去重，然后用 `num_return_sequences` 生成多个 completion。实际上 TRL 用 `RepeatSampler` 重复 prompt，每行生成 1 个 completion（通过随机采样产生多样性）。

**教训**: 两种生成多样 completion 的方式效果相同，但必须跟框架一致：
1. TRL 方式：重复 prompt N 次，每次独立采样 1 个
2. HF generate 方式：1 个 prompt，`num_return_sequences=N`

---

## Lesson 8: 离线环境下 HF 缓存路径必须精确

**问题**: `HF_DATASETS_CACHE` 写了 `/home/.../datasets` 而不是 `/home/.../datasets/GSM8K`，在 `HF_DATASETS_OFFLINE=1` 下找不到缓存，任务 ~25 秒就挂。

**教训**: 快速失败（<30s）几乎总是环境问题。对比已知可用的参考脚本来排查。
