# Mixture GRPO Debug Log

## Bug 1: HF_DATASETS_CACHE 路径错误

**现象**: 两个 mixture 训练任务提交后 ~25 秒立即失败 (Jobs 4300733, 4300735)

**错误信息**: `ConnectionError: Couldn't reach 'openai/gsm8k' on the Hub (OfflineModeIsEnabled)`

**排查思路**:
1. 任务 ~25 秒就挂 → 不是训练逻辑的问题，是启动阶段的环境/配置问题
2. 错误信息明确指向 HuggingFace Hub 连接 → 离线模式下试图联网 → 本地缓存没找到
3. 对比已经能正常跑的 `offline_grpo/run_gsm8k_offline.sh` 的环境变量设置 → 发现 `HF_DATASETS_CACHE` 路径不同
4. `ls` 验证两个路径 → mixture 脚本的路径下没有数据集缓存文件

**原因**: 脚本设了 `HF_DATASETS_OFFLINE=1`（集群无外网），GSM8K 数据集缓存在 `/home/shuai14/scratch/datasets/GSM8K`，但 mixture 脚本写的是 `HF_DATASETS_CACHE="/home/shuai14/scratch/datasets"`（少了 `/GSM8K` 子目录），找不到缓存。

**解决**: `export HF_DATASETS_CACHE="/home/shuai14/scratch/datasets/GSM8K"`，对齐 offline GRPO 脚本的写法。

**教训**: 在离线环境跑实验时，数据集缓存路径必须精确到 HF datasets 的缓存子目录。可以用 `ls $HF_DATASETS_CACHE` 提前验证。快速失败（<30s）几乎总是环境问题，优先对比已知可用的参考脚本。

---

## Bug 2: TRL 的 batch 重复机制 vs num_return_sequences

**现象**: 修复 Bug 1 后两个任务都在第一步就报 `IndexError: list index out of range` (Jobs 4301292, 4301293)

**错误位置**: `student_rewards[stu_idx + g]` 越界

**排查思路**:
1. 第一步就崩 → 不是显存问题，是逻辑/shape 问题
2. `IndexError: list index out of range` → 某个 list 比预期短
3. 看 traceback：`student_rewards[stu_idx + g]` 越界 → `student_rewards` 的长度不对
4. 向上追溯：`student_rewards` 的长度 = `student_outputs` 的行数 = `generate()` 的输入行数
5. 发现代码做了去重：`unique_prompt_ids = prompt_ids[::num_gen]`，batch_size=10 去重后只有 2 行
6. `generate()` 输入 2 行，输出 2 行，但后续代码期望 10 个 reward（因为 `stu_idx` 按 batch_size=10 遍历）
7. 关键问题：为什么要去重？→ 我们假设需要自己用 `num_return_sequences` 生成多个 completion
8. 阅读 TRL 源码 `_get_train_sampler` → 发现 `RepeatSampler` 已经重复了 prompt，框架设计就是每行生成 1 个
9. 验证：TRL 的 `generation_config` 里没有设 `num_return_sequences`，默认=1

**原因**: 对 TRL GRPOTrainer 内部 batch 机制理解有误。

TRL 的做法是：dataloader 通过 `RepeatSampler` 把每个 prompt **重复 `num_generations` 次**放进 batch，然后对每行生成 **1 个** completion（`do_sample=True`，随机采样产生多样性）。

我们的代码误以为需要自己去重（`unique_prompt_ids = prompt_ids[::num_gen]`），然后靠 `generation_config` 里的 `num_return_sequences` 一次返回多个结果。但 TRL 的 `generation_config` **根本没设 `num_return_sequences`**（默认=1）。

结果：`batch_size=10`（5 个 prompt × 2 重复），去重后只有 2 个 prompt，generate 只返回 2 个结果，但后续代码期望 10 个 reward → 越界。

**解决**: 不去重，直接用重复的 prompt batch 生成。每行独立采样，自然得到不同结果：
```python
# 删除:
# unique_prompt_ids = prompt_ids[::num_gen]
# 直接用:
student_outputs = self.model.generate(input_ids=prompt_ids, ...)
```

**教训**: 两种生成多样 completion 的方式：
1. **TRL 方式**: 重复 prompt N 次，每次独立采样 1 个（通过 dataloader 重复）
2. **HF generate 方式**: 1 个 prompt，设 `num_return_sequences=N`

效果相同，但必须跟框架一致。阅读框架源码（特别是 `_get_train_sampler` 和 `_generate_and_score_completions`）是理解 batch 结构的关键。

---

## Bug 3: TRL `_compute_loss` 返回值格式

**现象**: Method B 第一步报 `TypeError: iteration over a 0-d tensor` (Job 4301406)

**错误位置**: `loss, metrics = super()._compute_loss(model, inputs)` — 试图 unpack 标量 tensor

**排查思路**:
1. `TypeError: iteration over a 0-d tensor` → 试图对标量 tensor 做 iteration（unpack）
2. 看 traceback 指向 `loss, metrics = super()._compute_loss(...)` → 返回值只有一个标量，不是元组
3. 为什么写成 tuple unpack？→ 参考了 HF Trainer 的 `compute_loss`（返回 `(loss, outputs)`），误以为 TRL 也这样
4. 阅读 TRL 的 `_compute_loss` 源码 → 确认只返回 `loss` 标量
5. 那 metrics 怎么记录？→ 源码里直接用 `self._metrics[mode]["xxx"].append(...)` 写入，不通过返回值
6. 注意：此 bug 只影响 Method B（因为 Method B override 了 `_compute_loss`），Method A 没有 override 所以不受影响

**原因**: TRL 的 `_compute_loss` 只返回 `loss`（一个标量 tensor），不是 `(loss, metrics)` 元组。metrics 是通过 `self._metrics[mode]` 直接 append 的，不通过返回值传递。

**解决**:
```python
# 旧: loss, metrics = super()._compute_loss(model, inputs)
# 新: loss = super()._compute_loss(model, inputs)
# 用 self._metrics[mode]["offline_loss"].append(...) 记录 teacher loss
```

**教训**: override 父类方法时，先看父类的实际返回值签名，不要假设。TRL 的 `_compute_loss` 和 HF Trainer 的 `compute_loss` 返回值不同。

---

## Bug 4: Method A — CUDA OOM（显存不足）

**现象**: Method A 跑了 7 步后 OOM (Job 4301405)

**错误信息**: `torch.OutOfMemoryError: Tried to allocate 5.80 GiB. GPU has 44.40 GiB total, 5.42 GiB free.`

**排查思路**:
1. 跑了 7 步才 OOM → 不是 shape 错误，是显存逐步积累或碰到较长序列
2. 看 OOM 信息：需要 5.80 GiB，只有 5.42 GiB free → 差距不大（0.38 GiB），说明接近极限
3. 为什么比标准 online GRPO 占更多显存？→ 检查每步实际处理了多少数据
4. 核心发现：`_generate_and_score_completions` 返回 G+K=10 条，而 online GRPO 只返回 G=5 条
5. 但有 gradient accumulation 的 split 机制啊？→ 计算实际每个 split 的大小
6. 逐步推算 batch 流转（见下面的分析过程），确认每步 10 条 vs 原来 5 条 → 翻倍
7. 为什么第 7 步才挂？→ 前几步序列较短，显存刚好够用。第 7 步碰到较长的 completion → 激活值更大 → OOM
8. 解决方向：减少每步处理量。两种方案 — gradient checkpointing（用计算换显存）vs 减小配置（4+4 代替 5+5）

**原因**: Method A 的 `_generate_and_score_completions` 返回 G+K 个 completion（student 5 + teacher 5 = 10），而 online GRPO 只返回 G=5 个。每步处理的 sequence 数量翻倍，显存不够。

**分析过程**:
- `per_device_train_batch_size=5, num_generations=5, gradient_accumulation_steps=2`
- `steps_per_generation = grad_accum = 2`
- 每 GPU generation batch = 5 × 2 = 10 items → 2 unique prompts
- Method A output: 2 × (5+5) = 20 rows → split 成 2 slices → 每步 10 条
- Online GRPO: 2 × 5 = 10 rows → split 成 2 slices → 每步 5 条
- 每步多了 1 倍，且只差 0.4GB，所以前几步侥幸通过，第 7 步序列较长时 OOM

**解决**: 改为 4+4 配置：
- `num_generations=4, num_teacher_per_prompt=4`
- `per_device_train_batch_size=4`（必须能被 `num_generations` 整除）
- `gradient_accumulation_steps=3`（保持 effective batch ≈ 48，接近原来的 40）
- 每步 completion 数: 2 × (4+4) / 3 ≈ 5.3 → 实际是 8，比之前 10 少 20%

**备选方案**（未采用）:
- `gradient_checkpointing=True`: 用时间换空间，省 ~30-50% 显存，但训练慢 ~30%
- 减少 `max_completion_length`
- 减少 `num_teacher_per_prompt`

**教训**: 修改 `_generate_and_score_completions` 的输出大小时，要考虑 TRL 的 split/accumulation 机制对每步实际处理量的影响。输出行数 × 序列长度 = 显存占用的主要因素。

---

## Bug 5: Method B — CUDA OOM（teacher batch 未 split）

**现象**: Method B 4+4 配置下跑了 3 步后 OOM (Job 4301501)

**错误信息**: `Tried to allocate 5.68 GiB, 3.89 GiB free`

**排查思路**:
1. 又是 OOM，但 Method B 已经改成 4+4 了（Bug 4 后），而且 Method B 的 output dict 只包含 student 数据 → 按理每步只有 4 条，应该没问题
2. 关键疑问：output dict 大小正确，为什么还 OOM？→ 一定有额外的显存消耗来源
3. 检查 `_compute_loss` 里做了什么 → 发现 `self._current_teacher_batch` 包含 teacher 数据，每次都对其做 forward
4. 关键发现：`self._current_teacher_batch` 是在 `_generate_and_score_completions` 里存的实例变量，**不在 output dict 里**
5. 推理：TRL 的 split 机制只处理 output dict → teacher 数据没有被 split → 每次 `_compute_loss` 都用全量 teacher 数据
6. 验证计算：3 unique prompts × 4 teacher = 12 条，每次 `_compute_loss` 都 forward 全部 12 条 + 4 student = 16 条 → 远超 8 条预期
7. 解决思路：让 teacher 数据也被 split → 放进 output dict。但前提是行数要对齐 → 检查：student 和 teacher 都是 num_unique × 4 = 相同行数 → 可以放一起

**原因**: 跟 Method A 的 OOM 不同。Method B 的 `_generate_and_score_completions` 只返回 student 数据（大小正确），但 teacher 数据存在 `self._current_teacher_batch` 里，**没有被 TRL 的 split 机制处理**。

TRL 的工作流：
```
_generate_and_score_completions → 返回 output dict (12 rows)
  → shuffle_sequence_dict → split_tensor_dict(output, steps_per_gen=3)
  → 3 slices, 每个 4 rows
  → _compute_loss 被调用 3 次，每次处理 4 student rows
```

但 `self._current_teacher_batch` 有 12 条（3 unique × 4 teacher），每次 `_compute_loss` 都对全部 12 条做 forward。所以每步实际是 4 student + 12 teacher = **16 条**，远超预期的 8 条。

**解决**: 把 teacher 数据放进 output dict，加 `teacher_` 前缀，让 TRL 自动 split：
```python
# 旧: self._current_teacher_batch = {"prompt_ids": ..., "completion_ids": ..., ...}
# 新: output["teacher_prompt_ids"] = ...
#     output["teacher_completion_ids"] = ...
#     ...
# 在 _compute_loss 里从 inputs 取: inputs["teacher_completion_ids"]
```

这样 teacher 数据跟 student 数据一起被 split，每步 4 student + 4 teacher = 8 条。

**前提条件**: `num_generations == num_teacher_per_prompt`（都是 4），所以 student 和 teacher 的行数相同（都是 `num_unique × 4`），可以放在同一个 dict 里被均匀 split。

**教训**:
1. TRL 的 gradient accumulation 通过 split output dict 实现，**只有在 output dict 里的数据才会被 split**
2. 用 side-channel（实例变量）传递数据会绕过 split 机制
3. 设计时要考虑：哪些数据需要跟 batch 同步 split，哪些是全局共享的

---

## 架构经验总结

### TRL GRPOTrainer 的关键机制

1. **Batch 构建**: `RepeatSampler` 把每个 prompt 重复 `num_generations` 次。`per_device_train_batch_size` 是包含重复后的总数，必须能被 `num_generations` 整除。

2. **Generation batch**: 大小 = `per_device_train_batch_size × steps_per_generation`（per GPU）。`steps_per_generation` 默认等于 `gradient_accumulation_steps`。

3. **Split 机制**: `_generate_and_score_completions` 的输出被 split 成 `steps_per_generation` 份，每份在一个 accumulation step 中用于 `_compute_loss`。

4. **`generation_config`**: 不含 `num_return_sequences`，每行生成 1 个 completion。

5. **`_compute_loss` 返回值**: 只返回 `loss` 标量，metrics 通过 `self._metrics` 记录。

### Debug 方法论

1. **快速失败**（<30s）→ 检查环境配置（路径、离线模式、import 错误）
2. **第一步失败** → 检查 tensor shape、batch 大小、返回值格式
3. **跑几步后失败** → 检查显存（OOM 通常在序列较长时触发）
4. **读 `.err` 文件**比 `.out` 更有用，错误信息和 traceback 都在 stderr
5. **DDP 多卡**错误信息会交织，找 `[rank0]` 或 `Root Cause` 部分

### 显存估算

- 0.5B 模型 + LoRA: 基础 ~2GB
- 每条 completion（1024 tokens）的激活值: forward ~0.5GB, backward ~0.5GB
- 估算公式: `每步 completion 数 × 单条显存 + 模型 + 优化器 < GPU 总显存`
- L40s: 44.4GB，安全上限约 8-10 条 × 1024 tokens

---

## Bug 6: Method B — offline_loss ≈ 0（Sequence-level vs Per-token Clipping）

**现象**: Method B 训练正常但 `offline_loss` 始终接近 0（约 1e-6 ~ 1e-7），teacher 信号完全没有发挥作用。Method B 实际上退化成了纯 online GRPO。

**排查思路**:
1. 训练没有崩溃，loss 在下降 → 不是 crash 类 bug，是逻辑/数值问题
2. 查 wandb/日志发现 `offline_loss` ≈ 1e-6 ~ 1e-7 → 几乎为零，teacher 信号形同虚设
3. Method B 的 total loss = L_online + λ × L_offline → 如果 L_offline ≈ 0，就退化成纯 online GRPO
4. 进入 `_compute_offline_loss`，逐行检查数值范围：
   - `log_ratio = per_token_logps - old_per_token_logps` → teacher 7B 和 student 0.5B 分布差异大，单 token log_ratio ≈ -2.5
   - 旧代码：`log_ratio_sum = (log_ratio * mask).sum(dim=1)` → 100 tokens 累加 ≈ -250
   - `ratio = torch.exp(-250)` ≈ 0 → 就是这里归零了
5. 对比 TRL 内置的 GRPO loss → 发现它是 per-token 的，每个 token 独立 exp，不做 sum
6. 为什么 per-token 能 work？→ 单 token exp(-2.5) = 0.082，虽然小但 clamp 到 0.8 后仍有梯度
7. 为什么 sequence-level 不 work？→ exp(sum) = 所有 token 概率比值的**乘积**，100 个 0.08 相乘 ≈ 0
8. 核心区别：clipping 发生在 exp 之后。per-token 是先 exp 再 clip 每个 token → 有效。sequence-level 是 sum 完 exp 出 0 再 clip → min(0, 0.8) = 0，clip 救不回来

**原因**: `_compute_offline_loss` 中 importance ratio 是 sequence-level 的：先把每个 token 的 log ratio 加起来（sum over 100+ tokens），再 exp。

由于 teacher (7B) 和 student (0.5B) 的分布差异巨大，单 token log ratio ≈ -2.5，100 个 token 累加后 log_ratio_sum ≈ -250，`exp(-250) ≈ 0`。Clipping 到 0.8 后，`min(0 × adv, 0.8 × adv) = 0`。

而 TRL 内置的 GRPO loss 是 per-token 的：每个 token 独立 exp、独立 clip。单 token `exp(-2.5) ≈ 0.08`，clip 到 0.8，仍能产生有效梯度。

**解决**: 将 `_compute_offline_loss` 改为 per-token clipping，与 TRL 的标准 GRPO loss 一致：
```python
# 旧 (sequence-level):
log_ratio_sum = (log_ratio * mask).sum(dim=1)  # sum over tokens
ratio = torch.exp(log_ratio_sum)                # 一个 sequence 一个 ratio

# 新 (per-token):
ratio = torch.exp(log_ratio)                    # 每个 token 独立 exp
clipped = torch.clamp(ratio, 0.8, 1.2)          # 每个 token 独立 clip
per_token_loss = -min(ratio * adv, clipped * adv)  # 每个 token 独立算 loss
loss = (per_token_loss * mask).sum(-1) / mask.sum(-1)  # 先 token 平均，再 batch 平均
```

**教训**: Importance ratio 在 RL 中必须注意数值稳定性。sequence-level 的乘积（对应 log 的 sum）在长序列上极易爆炸或归零。Per-token clipping 是 PPO/GRPO 处理这个问题的标准做法。设计 loss 时要参考框架内置实现的数值处理方式。

### 数值对比示例：Sequence-level vs Per-token Clipping

假设每个 token 的 log ratio（log π_student − log π_teacher）= −2.5：

**Sequence-level（错误做法）**:
```
4 tokens:   sum = −10.0  → exp(−10) = 0.0000454  → clamp(0.8, 1.2) = 0.8
                          → min(0.0000454 × adv, 0.8 × adv) = 0.0000454 × adv  ← 几乎为零
100 tokens: sum = −250.0 → exp(−250) ≈ 0          → clamp = 0.8
                          → min(0 × adv, 0.8 × adv) = 0                        ← 完全为零
```
问题：ratio 是所有 token 概率的**乘积**（log 的 sum → exp），token 越多衰减越严重。

**Per-token（正确做法）**:
```
每个 token: exp(−2.5) = 0.082 → clamp(0.8, 1.2) = 0.8
           → min(0.082 × adv, 0.8 × adv) = 0.082 × adv  ← 有效梯度 ✓
无论 4 tokens 还是 100 tokens，每个 token 独立处理，不受序列长度影响。
```

### Method A vs Method B 中 per-token clipping 的区别

| | Method A (Unified) | Method B (Weighted) |
|---|---|---|
| **谁算 clipping** | TRL 框架统一算 | online 用 TRL，offline 自己写 `_compute_offline_loss` |
| **π_old 来源** | student 和 teacher 都用当前 student policy forward 出的 logprobs | online 用 student 的，offline 用 **teacher 7B** 生成时的 logprobs |
| **分布差异** | 小（student vs student 几步前） | offline 很大（student 0.5B vs teacher 7B） |
| **clipping 压力** | 低，ratio 接近 1.0 | 高，ratio ≈ 0.08，经常被 clamp 到 0.8 |

Method A 没有分布差异问题，因为 teacher completions 的 π_old 是用当前 student forward 出来的。Method B 的 offline π_old 来自 teacher 7B，跟 student 0.5B 差异大，所以 per-token clipping 对 Method B 至关重要。

---

## Bug 7: Method A — KL penalty 错误地作用于 teacher completions

**现象**: 代码审查发现的理论错误，尚未通过实验验证影响。但从 GRPO 理论角度，这是一个必须修正的实现 bug。

**发现过程**:
1. 在分析 mixture GRPO 中 KL penalty 的实现时，追踪数据流：`_generate_and_score_completions` 构建 ref logprobs → output dict → TRL `_compute_loss` 消费
2. Method A 把 student + teacher completions 合并成一个 unified batch（line 308: `torch.stack(all_completion_ids)` 交织排列），然后对**整个 batch** 计算 ref logprobs（line 317-330）
3. TRL 的 `_compute_loss`（`grpo_trainer.py:1844-1886`）对 batch 中**每一行**都计算 KL：
   ```python
   per_token_kl = exp(ref - current) - (ref - current) - 1
   per_token_loss = per_token_loss + β * per_token_kl
   ```
4. 这意味着 teacher completions 也被施加了 KL penalty — 但这理论上不对

**排查思路**:
1. KL(π_θ || π_ref) 的定义是 `E_{y ~ π_θ}[log π_θ(y) - log π_ref(y)]`
2. 关键：期望下面的 `y ~ π_θ` — **样本必须来自当前策略**，这样才是 KL 的无偏估计
3. Student 生成的 completion：y ~ π_θ ✓ → KL 估计正确
4. Teacher 生成的 completion：y ~ π_teacher_7B ✗ → 用错了分布，KL 估计**有偏**
5. 具体影响：teacher 7B 生成的文本（如复杂证明）是 student 不太可能生成的区域。在真正的 KL(π_θ || π_ref) 中，这些区域因为 π_θ 概率低，贡献很小。但 Method A 因为 teacher completion 的存在，**人为放大了这些区域在 KL 估计中的权重**
6. 实际效果：KL penalty 惩罚 student 在 teacher 文本上偏离 base weights → **限制了 student 学习 teacher 信号的速度**

**数值示例**:
```
Teacher completion: "We apply the quadratic formula... therefore x = ..."

log π_ref(each token)  ≈ -5.0  (base student 在 teacher 文本上概率低)
log π_θ(each token)    ≈ -4.5  (LoRA 训练后稍微升高)

per_token_kl = exp(-5.0 - (-4.5)) - (-5.0 - (-4.5)) - 1
             = exp(-0.5) - (-0.5) - 1
             = 0.607 + 0.5 - 1 = 0.107

β × 0.107 被加到 teacher completion 的 loss 里 → 惩罚 student 学习 teacher 的内容
```

而这段 teacher 文本在真正的 KL 中，因为 π_θ 的采样概率低，贡献应该很小。Method A 的实现等于给了它一个不应有的、额外的权重。

**对比 Method B**: Method B 只对 student completions 算 KL（line 280: `torch.cat([prompt_ids, student_completion_ids])`），teacher 的 offline loss（`_compute_offline_loss`）是纯 clipped surrogate，没有 KL term。这恰好是理论正确的做法。

**解决方案**: 在 Method A 中加入 `kl_mask` 机制，override `_compute_loss` 让 KL 只作用于 student completions。

1. 在 `_generate_and_score_completions` 的 interleave 阶段构建 `kl_mask`：
   ```python
   all_kl_mask = []
   # Student completions:
   all_kl_mask.append(1.0)   # KL 生效
   # Teacher completions:
   all_kl_mask.append(0.0)   # KL 不生效
   ```

2. 把 `kl_mask` 放进 output dict → 自动被 TRL split 机制处理

3. Override `_compute_loss`，在计算 KL 后乘以 mask：
   ```python
   per_token_kl = (exp(ref - current) - (ref - current) - 1)
   per_token_kl = per_token_kl * kl_mask.unsqueeze(1)  # teacher 行的 KL 归零
   per_token_loss = per_token_loss + β * per_token_kl
   ```

**为什么不直接跳过 teacher 的 ref logprobs 计算**: ref logprobs 在 `_generate_and_score_completions` 中一次性计算（line 430-445），此时 student 和 teacher 已经合并成一个 batch。拆开来分别计算虽然能节省一点计算量，但需要拆分/重组 batch，增加复杂度且容易出错。选择在 loss 端 mask 掉 KL 是更干净的方案，代价仅是多算了一些不会被使用的 ref logprobs。

**影响文件**:
- `method_A_unified/trainer.py`:
  - `_generate_and_score_completions`: 新增 `all_kl_mask` 列表，student=1.0, teacher=0.0，打包进 output dict
  - 新增 `_compute_loss` override：复制 TRL 的实现，在 `per_token_kl` 后乘以 `kl_mask`
- Method B **不受影响**：teacher offline loss 本来就没有 KL

**教训**:
1. 当把不同来源的数据混合进同一个 batch 时，要检查 batch 级别的操作（如 KL penalty）是否对所有行都适用
2. KL(π_θ || π_ref) 的估计必须用 on-policy 样本。off-policy 样本（teacher completions）在 student 不常访问的区域有高权重，导致 KL 估计偏向这些不重要的区域
3. 代码审查（code review）比实验更高效地发现理论性 bug — 这类 bug 不会导致 crash 或明显的数值异常，但会默默地损害训练效果

---

## Bug 8: Method B — 零方差时 teacher advantage 被完全杀死

**现象**: 代码审查发现。当一个 prompt 的所有 student completions reward 相同时（尤其是全部答错，reward 全为 0），teacher 的 advantage 被强制设为 0，teacher 梯度信号完全消失。

**发现过程**:
1. 追踪 Bug 7 时顺便审查 Method B 的 advantage 计算
2. 在 `method_B_weighted/trainer.py:222-234` 发现 student advantage 的零方差保护：
   ```python
   adv = (reward - mean_r) / (std_r + eps) if std_r > eps else 0.0
   ```
3. 这个保护对 student advantage 是合理的（标准 GRPO 行为：group 内所有 reward 一样 → 无法区分好坏 → advantage = 0）
4. 但在 line 263，**teacher advantage 也用了同样的保护**：
   ```python
   mean_r, std_r = online_stats[q]   # 来自 student 的 (mean, std)
   tea_adv = (tea_reward - mean_r) / (std_r + eps) if std_r > eps else 0.0
   ```
5. 当 std_r = 0 时，teacher advantage 也被强制归零 → teacher 的梯度信号完全消失
6. 推理影响范围：什么时候 std_r = 0？→ 所有 student reward 相同 → 最常见的场景是 **4 个 student 全部答错（reward 全为 0）**

**排查思路 — 影响范围估算**:

GSM8K（student baseline ~50%）:
```
4 个 student 全错的概率 ≈ (1 - 0.5)^4 = 0.0625 → ~6% 的 prompt 受影响
```

MATH（student baseline ~5-10%）:
```
4 个 student 全错的概率 ≈ (1 - 0.07)^4 ≈ 0.748 → ~75% 的 prompt 受影响
```

**在 MATH 上，约 75% 的 prompt 的 teacher 信号被完全杀死。**

而这恰恰是 mixture 最有价值的场景——student 完全不会做的难题，正是最需要 teacher 指导的地方。

**数值示例**:

场景：一道 MATH 难题，4 个 student 全错，teacher 有 3 个对 1 个错。

```
Student rewards: [0, 0, 0, 0]
  mean_r = 0.0
  std_r  = 0.0
  std_r > eps(1e-4)?  → False

Student advantages: [0.0, 0.0, 0.0, 0.0]    ← 正常（标准 GRPO，无法区分）

Teacher rewards: [2.0, 2.0, 0.0, 2.0]
  tea_adv = (tea_reward - 0.0) / (0.0 + 1e-4) if 0.0 > 1e-4 else 0.0
          = 0.0 for ALL teachers                ← 问题！

→ teacher_advantages = [0.0, 0.0, 0.0, 0.0]
→ offline_loss 中: per_token_loss = -min(ratio * 0, clip * 0) = 0
→ ∂(offline_loss)/∂θ = 0
→ Student 从 teacher 学到了：什么都没有
```

如果 teacher advantage 正常工作（比如用 raw difference）：
```
tea_adv = tea_reward - mean_r = 2.0 - 0.0 = 2.0

→ offline_loss 中: per_token_loss = -min(ratio * 2.0, clip * 2.0)
→ 有效梯度，student 朝 teacher 正确解法学习
```

**对比 Method A**:

Method A 把 student + teacher rewards 放在同一个 group：
```
group_rewards = [0, 0, 0, 0, 2, 2, 0, 2]    ← 4 student + 4 teacher
mean_r = 0.75
std_r  = sqrt(mean([0.5625]*5 + [1.5625]*3)) = sqrt(0.9375) = 0.968
std_r > eps ✓ → advantage 正常计算

Teacher correct: adv = (2.0 - 0.75) / 0.968 = +1.29   ← 有效正信号 ✓
Student wrong:   adv = (0.0 - 0.75) / 0.968 = -0.77   ← 有效负信号 ✓
```

Method A **天然不存在这个问题**，因为 teacher rewards 的多样性为 group 提供了方差。

**根本原因**: Method B 的设计是把 student 和 teacher 的 advantage 分开计算，teacher advantage 用 student 的 (mean, std) 归一化。但当 student 的 std = 0 时（所有 student reward 相同），归一化未定义。代码选择了最差的 fallback：直接设为 0。

**解决方案**: 当 std_r = 0 时，teacher advantage 不应该归零，应该用未归一化的差值：

```python
# 旧:
tea_adv = (tea_reward - mean_r) / (std_r + eps) if std_r > eps else 0.0

# 新:
tea_adv = (tea_reward - mean_r) / (std_r + eps) if std_r > eps else (tea_reward - mean_r)
```

这样：
- Student 全错 (mean=0, std=0)，teacher 对 (reward=2.0) → tea_adv = 2.0 - 0.0 = 2.0 ✓
- Student 全错，teacher 也错 (reward=0.0) → tea_adv = 0.0 - 0.0 = 0.0 ✓
- Student 全对 (mean=1, std=0)，teacher 对 (reward=1.0) → tea_adv = 1.0 - 1.0 = 0.0 ✓（不需要学）
- Student 全对，teacher 错 (reward=0.0) → tea_adv = 0.0 - 1.0 = -1.0 ✓（抑制错误解法）

注意：这会导致 advantage 在 std > 0 时是归一化的（量级 ~±2），在 std = 0 时是 raw（量级取决于 reward scale，GSM8K ~±1，MATH ~±2）。两者量级相近，不会导致梯度尺度突变。

**影响文件**: `method_B_weighted/trainer.py` line 263

**教训**:
1. 零方差保护（`if std > eps else 0.0`）对 group 内部的 advantage 是正确的（无法区分好坏），但对 group 外部的 baseline 比较（teacher vs student 平均水平）是有害的
2. Mixture 方法的核心价值在于 "student 不会做但 teacher 会做" 的场景。如果代码恰恰在这个场景杀死 teacher 信号，整个方法的理论优势就被实现 bug 抵消了
3. 在 hard dataset (MATH) 上这个 bug 影响 ~75% 的 prompt，几乎完全瘫痪 teacher signal，可能是 mixture 方法在 MATH 上表现不佳的主要原因之一

---

## Bug 9: Method B — Importance ratio 导致 teacher 梯度信号极弱（设计缺陷）

**现象**: 代码审查 + 理论分析发现。即使 Bug 8 修复后 teacher advantage 非零，`_compute_offline_loss` 中的 importance ratio `π_θ/π_teacher` 也会把 teacher 梯度信号压缩到正常强度的 ~8%。

**发现过程**:
1. 修复 Bug 8 后继续追踪 teacher 信号的完整链路：advantage → offline_loss → 梯度
2. 在 `_compute_offline_loss` (line 114-140) 中发现 per-token importance ratio:
   ```python
   log_ratio = per_token_logps - old_per_token_logps    # log(π_θ / π_teacher)
   ratio = torch.exp(log_ratio)                          # π_θ / π_teacher
   ```
3. π_θ 是 student 0.5B，π_teacher 是 teacher 7B → 分布差异巨大
4. 逐 token 计算：student 在 teacher 生成的 token 上概率远低于 teacher → ratio ≈ 0.08

**排查思路**:

对一个 token 的 loss 做完整展开：

```
Teacher 7B 生成 token "quadratic":
  log π_teacher("quadratic" | context) = -2.8   → P = 0.06
  log π_student("quadratic" | context) = -5.3   → P = 0.005

  log_ratio = -5.3 - (-2.8) = -2.5
  ratio = exp(-2.5) = 0.082

  clipped_ratio = clamp(0.082, 0.8, 1.2) = 0.8

  假设 advantage A = 1.5:
  loss1 = ratio * A       = 0.082 * 1.5 = 0.123
  loss2 = clipped * A     = 0.8   * 1.5 = 1.200
  per_token_loss = -min(0.123, 1.200) = -0.123

  对比 on-policy (ratio=1.0):
  per_token_loss = -min(1.5, 1.5) = -1.5

  teacher 信号强度: 0.123 / 1.5 = 8.2%
```

**PPO clip 为什么救不了**:

Clip 的机制：`min(ratio * A, clip(ratio) * A)`

- 当 ratio > 1.2 且 A > 0 时：clip 生效，用 1.2 代替过大的 ratio → **上限保护** ✓
- 当 ratio < 0.8 且 A > 0 时：min 选 ratio*A（更小的那个） → **clip 完全不生效** ✗

Clip 是上限机制，不是下限机制。它防止过大更新，但不会把太小的 ratio 放大。当 ratio = 0.08 远小于 0.8 时，clip 形同虚设。

**梯度视角**:

```
∂(offline_loss)/∂θ = -ratio * A * ∂(log π_θ)/∂θ
                   = -0.08 * A * ∂(log π_θ)/∂θ
                        ^^^^
                  梯度系数只有正常的 8%
```

这意味着在 Method B 的 total loss 中：
```
total_loss = online_loss + λ * offline_loss

∂(total)/∂θ = ∂(online)/∂θ + 0.3 * ∂(offline)/∂θ
              ^^^^^^^^^^^^         ^^^^^^^^^^^^^^^^
              ratio ≈ 1.0          ratio ≈ 0.08
              正常强度               只有 0.3 * 8% = 2.4% 的强度
```

**Teacher 的实际梯度贡献不到 total 的 3%。**

**根本原因**: Importance-weighted policy gradient 的理论要求 `∇J = E_{y~π_old}[(π_θ/π_old) * A * ∇log π_θ]`。当 π_old = π_teacher 且 distribution gap 巨大时，importance ratio π_θ/π_teacher → 0，梯度信号消失。这是 importance sampling 在大 distribution gap 下的根本局限，不是实现 bug，而是**算法设计层面的缺陷**。

PPO clipping 的设计假设是 ratio ≈ 1（near on-policy），clip 区间 [0.8, 1.2] 处理的是小偏移。当 ratio = 0.08 时已完全超出 PPO 的设计范围。

**可能的改进方向**（未实施，需要进一步研究）:

| 方案 | Teacher offline loss 公式 | 信号强度 | 风险 |
|---|---|---|---|
| 当前 (importance-weighted) | `-min(π_θ/π_teacher * A, clip * A)` | ~8% | 太弱，teacher 信号被淹没 |
| Direct PG (去掉 ratio) | `-A * log π_θ(teacher_token)` | 100% | 无 off-policy 修正，可能不稳定 |
| Advantage-weighted SFT | `-max(A, 0) * log π_θ(teacher_token)` | 100%（仅正 A） | 忽略负 advantage（差的 teacher completion）|
| KL-regularized SFT | `-log π_θ + β * KL(π_θ \|\| π_ref)` | 100% | 需要另外平衡 KL |

**与 Bug 8 的关系**: Bug 8（零方差杀死 advantage）和 Bug 9（ratio 压缩梯度）是**叠加**的。即使修了 Bug 8 让 advantage 非零，Bug 9 仍然会把 teacher 梯度压到 ~8%。两者共同导致 teacher signal 在 Method B 中几乎无效。

**教训**:
1. Importance sampling 的有效性取决于两个分布的重叠程度。0.5B student 和 7B teacher 的分布重叠极小 → importance ratio 接近 0 → 梯度消失
2. PPO clipping 是为 near-on-policy 设计的，强行套用到 extreme off-policy 场景不会 work
3. 这解释了为什么 GSM8K 实验中 Method B (51.58%) 仅略优于 online GRPO baseline (~50%) — teacher 信号贡献极弱，Method B 实质上退化为 online GRPO + 极微弱的 teacher 噪声
