# Offline BC 实验设计文档

## 1. 实验目的

本实验的目标不是提出一个新方法，而是建立一个**最干净、最可解释的 offline imitation baseline**，用于回答下面这个核心问题：

> 在**同样的 teacher rollout 数据**上，student 仅靠模仿 teacher，能不能稳定学到东西？

这个实验的作用是把你当前的研究问题拆开：

- 如果 **BC 有效，但 offline GRPO 退化**，说明问题主要出在 **offline RL / off-policy policy improvement**。
- 如果 **BC 也无效**，说明问题更底层，可能出在：
  - offline 数据构造
  - teacher / student gap
  - token 对齐 / mask / padding
  - 训练配置

---

## 2. 为什么 BC 实验必须做

你已经观察到：

- **online GRPO 可以超过 baseline**
- **offline GRPO 反而比不训练更差**

这说明：

- 你的 **核心 GRPO 实现大概率不是根本性错误**
- 问题更可能集中在 **offline setting 本身**

但是，当前还缺少一个关键对照：

> **同样用 offline teacher 数据，但不做 RL-style policy improvement，只做 imitation，会怎样？**

BC 正是这个对照。

它可以帮助你判断：

1. **offline teacher data 本身是否可学**
2. **student 是否有能力在 teacher prefix 上模仿 teacher token**
3. **offline GRPO 的退化，是否来自 ratio / advantage / clipping / off-policy bias**

---

## 3. 核心假设

本实验验证以下假设：

### 假设 H1
使用 teacher rollout 构造的 offline 数据，student 通过标准 BC 可以提升 teacher token 的拟合能力。

### 假设 H2
如果 BC 有效而 offline GRPO 无效，则说明：

> teacher data 中存在 supervised signal，  
> 但 strict offline GRPO 的 off-policy policy improvement 部分破坏了这种信号。

### 假设 H3
如果 BC 也无效，则说明问题更底层，不应优先继续调 offline GRPO。

---

## 4. 数据设置

### 4.1 数据来源
使用**与 offline GRPO 完全相同**的 teacher rollout 数据。

每条样本至少包含：

- `prompt`
- `teacher completion`
- `reward`（BC 主实验中可不用）
- `teacher logprob`（BC 主实验中不用）

### 4.2 数据组织形式
每条训练样本为：

- 输入：`[prompt] + [teacher completion]`
- 监督目标：只在 `teacher completion` 部分计算 next-token prediction loss

### 4.3 数据一致性要求
为了与 offline GRPO 可比，以下内容必须尽量保持一致：

- 相同 student 初始化
- 相同训练 / 验证 / 测试划分
- 相同 max length / truncation 策略
- 相同 tokenizer
- 相同 EOS 处理
- 相同 batch size（如可行）
- 相同 optimizer / lr scheduler（尽量）
- 相近训练 step 或总 token 数

---

## 5. 模型与训练目标

## 5.1 模型
使用与 offline GRPO 完全相同的 student 模型：

- 同一个 base checkpoint
- 同样的 LoRA / QLoRA 配置（如果你现在用的是 LoRA）
- 同样的 precision / distributed config

### 5.2 BC 目标函数

标准 behavior cloning 目标为：

\[
L_{BC}
=
-\frac{1}{N}
\sum_{t \in \text{completion}}
\log \pi_\theta(y_t^{teacher} \mid x, y_{<t}^{teacher})
\]

其中：

- \(x\)：prompt
- \(y_t^{teacher}\)：teacher completion 的第 \(t\) 个 token
- \(N\)：参与监督的 completion token 数

直觉上，这就是：

> 在 teacher 的 prefix 下，让 student 预测 teacher 的下一个 token。

---

## 6. 输入与标签构造

设一条样本为：

- prompt: `What is 2+3?`
- teacher completion: `We compute 2+3 = 5.`

### 6.1 输入序列
将其拼接为：

```text
[prompt tokens] + [completion tokens]
```

### 6.2 标签构造
`labels = input_ids.clone()`

然后：

- prompt 对应位置：设为 `-100`
- completion 对应位置：保留原 token id
- padding 对应位置：设为 `-100`

### 6.3 Loss 计算规则
只在 **completion token** 上计算标准 causal LM cross entropy。

---

## 7. 最关键的实现要求

### 7.1 只监督 completion，不监督 prompt
这是最重要的一条。

prompt 部分必须 mask 掉，否则 loss 会被 prompt 污染。

### 7.2 保证 causal shift 正确
标准 causal LM 会自动做 next-token shift，但你仍然要确认：

- label 没有错位
- EOS 没有对齐错
- prompt / completion 边界没有偏移

### 7.3 padding 必须 mask 掉
否则会引入伪梯度。

### 7.4 先做纯 BC，不引入其他项
主实验不要加入：

- importance ratio
- clipping
- KL 项
- reward weighting
- group normalization
- advantage
- teacher logprob correction

先做最纯的 BC baseline。

---

## 8. 训练配置建议

下面给的是**推荐原则**，不是硬性值。

### 8.1 与 offline GRPO 尽量对齐
建议首先保持：

- optimizer 一致
- learning rate 一致或更小
- warmup 一致
- batch size 一致
- gradient accumulation 一致

这样结果更好解释。

### 8.2 若 BC 不稳定
优先尝试：

- 降低学习率
- 减少训练步数
- 增强 gradient clipping
- 确认 LoRA target modules 正确

### 8.3 对齐策略
建议至少采用以下其一：

#### 方案 A：按相同训练 step 对齐
例如：

- offline GRPO 训练 2k step
- BC 也训练 2k step

#### 方案 B：按相同监督 token 数对齐
例如：

- 两者训练时总共看过的 completion token 数近似一致

如果实验资源有限，先用 **方案 A** 即可。

---

## 9. 必须记录的指标

这个实验不应该只看最终 benchmark。  
你需要记录“模仿是否真的发生”。

### 9.1 训练 loss
记录：

- token-level cross entropy
- completion-only NLL

预期：应持续下降。

### 9.2 teacher token 平均 logprob
记录：

\[
\mathbb{E}[\log \pi_\theta(a_{teacher}\mid s)]
\]

预期：应持续上升。

这是最重要的诊断指标之一。

### 9.3 teacher token top-1 / top-k 命中率
例如：

- top-1 accuracy
- top-5 hit rate

在 teacher prefix 下，检查 teacher token 是否更容易被 student 选中。

### 9.4 completion-level imitation 指标
可选：

- exact match
- token overlap
- sequence logprob

### 9.5 下游 evaluation benchmark
例如你当前用来比较 online/offline GRPO 的 benchmark。

目的不是只看它是否暴涨，而是结合 imitation 指标一起解释。

### 9.6 KL 与 entropy（可选但建议）
如果现有 logging 已支持，继续记录：

- KL(base -> current)
- entropy

BC 下通常会：

- KL 平稳上升
- entropy 平稳下降

如果出现快速崩坏，说明训练配置可能有问题。

---

## 10. 推荐的实验对照矩阵

最低限度建议跑以下 3 个：

### Run A: No training
- 当前 baseline
- 用于提供零训练参考

### Run B: Pure BC
- 只用 teacher completion 做 teacher forcing
- 不加任何 RL 项

### Run C: Current offline GRPO
- 使用你当前的 strict offline GRPO 设置

如果还有余力，补一个：

### Run D: Weighted BC
例如：
- 只用高 reward completion
- 或按 reward / positive advantage 加权

---

## 11. 结果解释逻辑

### 情况 1：BC 明显优于 no-training
说明：

- offline teacher data 是可学的
- student 有能力模仿 teacher
- offline GRPO 的退化主要不是因为“teacher data 没用”

这时若 offline GRPO 仍更差，可以较强地支持以下结论：

> strict offline GRPO 的 off-policy optimization 部分在伤害训练。

### 情况 2：BC 和 no-training 差不多
说明：

- teacher imitation 信号很弱
- 或 teacher / student gap 太大
- 或数据质量有限
- 或训练设置不适合 offline imitation

### 情况 3：BC 也比 no-training 更差
优先排查：

- label shift
- mask
- padding
- EOS
- truncation
- LoRA target modules
- learning rate
- 数据对齐

这时不建议先继续深挖 offline GRPO。

### 情况 4：BC > offline GRPO
这是你当前最值得关注的可能结果。

它意味着：

> 同样的 offline teacher data，  
> imitation 能学到一些东西，  
> 但 RL-style offline improvement 反而把模型学坏了。

这会非常有研究解释力。

---

## 12. 与 offline GRPO 的关系

本实验并不是要证明 BC 比 GRPO“更高级”，而是要建立以下判断链：

1. **online GRPO works**
2. **offline BC works or not**
3. **offline GRPO fails**

这三者组合起来，才能清楚说明问题发生在哪一层。

如果最终观察到：

\[
\text{online GRPO} > \text{offline BC} > \text{offline GRPO}
\]

那么一个合理结论是：

> teacher-generated fixed dataset contains useful supervised signal,  
> but strict offline policy improvement suffers from off-policy bias and policy drift.

---

## 13. 可选扩展：Weighted BC

在 pure BC 之后，最自然的扩展是 Weighted BC。

目标函数：

\[
L = - \sum_i w_i \sum_{t \in i} \log \pi_\theta(y_t^{teacher}\mid s_t)
\]

其中 \(w_i\) 是第 \(i\) 条 completion 的权重。

### 可选权重形式

#### 方案 1：按 reward 加权
\[
w_i = reward_i
\]

#### 方案 2：只保留正 advantage
\[
w_i = \mathbf{1}[A_i > 0]
\]

#### 方案 3：截断后的正权重
\[
w_i = \max(A_i, 0)
\]

Weighted BC 的意义在于：

> 保留 teacher trajectories 的质量排序信息，  
> 但不引入 off-policy importance ratio。

如果 Weighted BC 明显优于 offline GRPO，则进一步说明：

> 问题主要不是 reward ranking，而是 ratio / off-policy correction。

---

## 14. 实现伪代码

下面给出一个最简伪代码示意：

```python
input_ids = concat(prompt_ids, completion_ids)

labels = input_ids.clone()
labels[:prompt_len] = -100
labels[padding_positions] = -100

outputs = model(input_ids=input_ids, labels=labels)
loss = outputs.loss

loss.backward()
optimizer.step()
optimizer.zero_grad()
```

如果是 batch 版本，每条样本都按各自的 prompt 长度构造 mask。

---

## 15. BC 实验的诊断 checklist

在开始正式训练前，建议做如下检查：

### 数据检查
- prompt / completion 是否正确拼接
- truncation 是否截断了关键 completion token
- EOS 是否重复或缺失
- padding 是否正确

### label 检查
- prompt label 是否全部为 `-100`
- completion label 是否保留正确 token id
- padding label 是否为 `-100`

### 单 batch 检查
- 随机打印一条样本的：
  - input tokens
  - labels
  - loss mask
- 手动确认 teacher completion 位置正确参与 loss

### 训练前 sanity check
- 初始 teacher-token logprob
- 初始 token accuracy
- 初始 eval benchmark

### 训练后对比
- BC 是否提高 teacher-token logprob
- BC 是否降低 completion NLL
- BC 是否优于 no-training

---

## 16. 预期结果与研究意义

基于你当前现象，一个合理预期是：

- online GRPO 最好
- pure BC 稳定且优于 strict offline GRPO
- strict offline GRPO 最容易发生 drift

如果结果符合这一趋势，那么这个 BC 实验将帮助你明确写出如下结论：

> 在 teacher-generated offline dataset 上，  
> imitation-based objectives are substantially more robust than strict off-policy GRPO.  
> This suggests that the degradation in offline GRPO is not due to the absence of useful teacher signal,  
> but due to off-policy bias, ratio instability, and policy drift.

---

## 17. 最终建议

本实验建议按以下顺序执行：

1. 跑 **Pure BC**
2. 记录 imitation 指标 + benchmark
3. 与 **offline GRPO** 对比
4. 若 BC 有效，再做 **Weighted BC**
5. 再决定是否转向：
   - GRPO + BC anchor
   - hybrid offline/online
   - offline warm start + online RL

---

## 18. 一句话总结

这个 BC 实验的核心不是“换个方法试试”，而是：

> 用一个最干净的 offline imitation baseline，  
> 判断 teacher data 到底是**不可学**，还是被 strict offline GRPO **学坏了**。
