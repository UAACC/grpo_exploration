# Deep Learning Wiki — From Basics to Our Pipeline

Everything explained from the ground up, with connections to how each concept appears in our offline GRPO project.

---

## Table of Contents

1. [How a Neural Network Learns](#1-how-a-neural-network-learns)
2. [Forward Pass and Backward Pass](#2-forward-pass-and-backward-pass)
3. [Loss Function](#3-loss-function)
4. [Gradient and Gradient Descent](#4-gradient-and-gradient-descent)
5. [Learning Rate](#5-learning-rate)
6. [The Adam Optimizer](#6-the-adam-optimizer)
7. [Batch, Step, and Epoch](#7-batch-step-and-epoch)
8. [Gradient Accumulation](#8-gradient-accumulation)
9. [Gradient Clipping](#9-gradient-clipping)
10. [Weight Decay](#10-weight-decay)
11. [Learning Rate Schedules](#11-learning-rate-schedules)
12. [Numerical Precision (bf16, fp16, fp32)](#12-numerical-precision-bf16-fp16-fp32)
13. [Overfitting and Regularization](#13-overfitting-and-regularization)
14. [How Language Models Work](#14-how-language-models-work)
15. [Autoregressive Generation vs Single-Pass Evaluation](#15-autoregressive-generation-vs-single-pass-evaluation)
16. [Tokens and Tokenization](#16-tokens-and-tokenization)
17. [Log-Probabilities (Logprobs)](#17-log-probabilities-logprobs)
18. [KL Divergence](#18-kl-divergence)
19. [Importance Sampling](#19-importance-sampling)
20. [PPO-Style Clipping](#20-ppo-style-clipping)
21. [LoRA — Low-Rank Adaptation](#21-lora--low-rank-adaptation)
22. [Multi-GPU Training](#22-multi-gpu-training)
23. [Gradient Checkpointing](#23-gradient-checkpointing)
24. [Checkpointing and Resuming](#24-checkpointing-and-resuming)
25. [Putting It All Together — Our Pipeline](#25-putting-it-all-together--our-pipeline)

---

## 1. How a Neural Network Learns

A neural network is a giant math function with millions of adjustable numbers called **weights** (or **parameters**). Learning means finding the right values for these weights.

The process:
1. Show the network some input (a math problem)
2. The network produces an output (a solution attempt)
3. Compare the output to what we wanted — measure how wrong it is (the **loss**)
4. Figure out which direction to nudge each weight to make the loss smaller (the **gradient**)
5. Nudge the weights a tiny bit in that direction
6. Repeat millions of times

That's it. Everything else in deep learning is details about how to do these steps more efficiently, more stably, or with less memory.

---

## 2. Forward Pass and Backward Pass

### Forward Pass
Data flows **forward** through the network: input → layer 1 → layer 2 → ... → output.

```
Input: "What is 2+3?"
         ↓
   [ Layer 1: multiply by weights, add bias ]
         ↓
   [ Layer 2: multiply by weights, add bias ]
         ↓
   [ ... 24 more layers ... ]
         ↓
Output: probability distribution over next token
```

The forward pass is just matrix multiplications and simple math operations, applied in sequence. It produces the network's prediction.

### Backward Pass (Backpropagation)
After we compute the loss, we need to know: "how should each weight change to reduce the loss?" This is computed by running the chain rule of calculus **backwards** through the network.

```
Loss = 3.7
  ↓ how much did the last layer contribute?
Layer 26 gradients computed
  ↓ how much did the second-to-last layer contribute?
Layer 25 gradients computed
  ↓ ...
Layer 1 gradients computed
```

The backward pass computes a **gradient** for every single weight in the network. This tells us: "if I increase this weight by a tiny amount, how much does the loss change?"

### In our pipeline
- **Training** (student): does both forward AND backward passes. Forward to get logprobs, backward to compute gradients, then update LoRA weights.
- **Rollout generation** (teacher): forward pass only. No learning, just generating text.
- **Evaluation**: forward pass only. Just generating and checking answers.

---

## 3. Loss Function

The loss is a single number that measures **how bad the model's prediction is**. Lower = better. The entire goal of training is to make this number go down.

Different tasks use different loss functions:

| Task | Loss function | What it measures |
|---|---|---|
| Classification | Cross-entropy | How wrong the predicted class probabilities are |
| Regression | Mean squared error | How far the predicted number is from the true number |
| Language modeling | Cross-entropy on tokens | How surprised the model is by the actual next token |
| **Our GRPO** | Policy gradient loss | How well the model assigns probability to good solutions vs bad ones |

### Our loss (simplified)

```
loss = -advantage × ratio + β × KL_penalty
```

- When advantage is positive (good solution) and ratio is high (student likes it): loss is very negative → good → gradient descent keeps going this direction
- When advantage is negative (bad solution): loss pushes student away from those tokens
- The KL penalty prevents the student from changing too much

The loss in our exp02 training logs hovers around 0.007–0.01. This is small because most of the work happens in the ratio and advantage — the actual loss magnitude doesn't tell you much by itself.

---

## 4. Gradient and Gradient Descent

### What is a gradient?

A gradient is the **slope** of the loss with respect to a weight. It answers: "if I increase this weight by 0.001, does the loss go up or down, and by how much?"

```
weight = 0.5, loss = 3.7
gradient = +0.02

Interpretation: increasing this weight slightly would increase the loss by 0.02
Action: DECREASE this weight to reduce the loss
```

For a model with 500 million weights, the backward pass computes 500 million gradients — one for each weight.

### Gradient descent

The simplest learning algorithm:

```
new_weight = old_weight - learning_rate × gradient
```

If the gradient is positive (increasing weight increases loss), we subtract → weight goes down → loss goes down.
If the gradient is negative (increasing weight decreases loss), we subtract a negative → weight goes up → loss goes down.

Every optimization algorithm (Adam, SGD, etc.) is a variation of this basic idea.

---

## 5. Learning Rate

The learning rate controls **how big each step is**.

```
new_weight = old_weight - learning_rate × gradient
                          ↑ this multiplier
```

| Learning rate | Effect |
|---|---|
| Too high (e.g., 0.01) | Steps are too big. The model overshoots, loss oscillates wildly or explodes to infinity. Training diverges. |
| Too low (e.g., 1e-8) | Steps are too tiny. The model barely changes. Training is extremely slow or gets stuck. |
| Just right (e.g., 5e-6 for us) | Model improves steadily without instability. |

### Why ours is 5e-6 (0.000005)

This is lower than the typical default of 5e-5. Two reasons:
1. **Importance sampling ratios** can amplify gradients — a ratio of 2.0 doubles the effective gradient. Lower LR compensates.
2. **LoRA** focuses updates on a small number of parameters — each update has outsized impact, so smaller steps are safer.

---

## 6. The Adam Optimizer

Plain gradient descent has problems: it uses only the current gradient, which can be noisy and misleading. **Adam** (Adaptive Moment Estimation) is smarter — it keeps a running history.

### What Adam tracks for each weight

```
m = running average of gradients         (momentum, "which direction are we generally going?")
v = running average of squared gradients (variance, "how much does the gradient fluctuate?")
```

### How Adam updates a weight

```
m = β1 × m_old + (1 - β1) × gradient          ← smooth the direction
v = β2 × v_old + (1 - β2) × gradient²         ← smooth the magnitude

new_weight = old_weight - learning_rate × m / (√v + ε)
```

### What the hyperparameters do

| Parameter | Our value | Default | What it controls |
|---|---|---|---|
| **β1 (beta1)** | 0.9 | 0.9 | How much to trust past gradient direction vs current. 0.9 = "90% memory, 10% new." Higher = smoother but slower to react. |
| **β2 (beta2)** | 0.99 | 0.999 | How much to trust past gradient magnitude. We use 0.99 instead of 0.999 — forgets old magnitudes faster, adapts quicker to changing gradient scale. |
| **ε (epsilon)** | 1e-8 | 1e-8 | Tiny number added to prevent dividing by zero. Almost never needs changing. |

### Why Adam matters

Without Adam, training is like walking downhill blindfolded, feeling only the slope under your feet right now. With Adam, you remember which direction you've been going (momentum) and how steep the terrain generally is (variance), so you take smarter steps.

### Memory cost

Adam stores **two extra numbers per weight** (m and v). For a 7B model:
```
Weights:    7B × 2 bytes (bf16) = 14 GB
Adam m:     7B × 4 bytes (fp32) = 28 GB    ← stored in full precision
Adam v:     7B × 4 bytes (fp32) = 28 GB
                                   ------
Total:                             70 GB    ← doesn't fit on one 48 GB GPU!
```

This is why LoRA matters — with only 13M trainable parameters, Adam needs only ~100 MB instead of 56 GB.

---

## 7. Batch, Step, and Epoch

### Batch
Instead of updating weights after every single example, we process a **batch** of examples and average their gradients. This is more stable and efficient.

```
batch_size = 2 means: process 2 examples, average their gradients, then update
```

Why not batch_size = 1?
- Single examples give noisy gradients (one easy problem vs one hard problem gives very different signals)
- GPUs are designed for parallel computation — processing 2 examples takes almost the same time as 1

Why not batch_size = 1000?
- GPU memory. Each example in the batch needs to be stored simultaneously.
- For us, each example is ~1042 tokens long. With a 500M parameter model, 2 examples already use several GB.

### Step
One **step** = one weight update. With `gradient_accumulation_steps=8` (explained below), one step involves 8 forward+backward passes but only 1 weight update.

### Epoch
One **epoch** = one complete pass through all training data.

```
48,000 samples ÷ 16 effective batch size = 3,000 optimizer steps per epoch
(but TRL/GRPO groups 4 completions per prompt, so it's ~12,000 steps in practice)

We train for 1 epoch = see every sample exactly once
```

| Concept | In our setup |
|---|---|
| Batch size (per device) | 2 samples |
| Gradient accumulation | 8 |
| Effective batch size | 2 × 8 = 16 |
| Samples per epoch | 48,000 |
| Steps per epoch | ~12,000 |
| Epochs | 1 |

---

## 8. Gradient Accumulation

**Problem**: We want an effective batch size of 16, but we can only fit 2 samples in GPU memory at once.

**Solution**: Do 8 forward+backward passes, **accumulate** (add up) the gradients, then do one weight update.

```
Step 1 (no update yet):
  Forward+backward on samples 1-2 → gradients g1

Step 2 (no update yet):
  Forward+backward on samples 3-4 → gradients g2
  Accumulated: g1 + g2

Step 3 (no update yet):
  Forward+backward on samples 5-6 → gradients g3
  Accumulated: g1 + g2 + g3

... 5 more times ...

Step 8 (UPDATE!):
  Forward+backward on samples 15-16 → gradients g8
  Accumulated: g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8
  Average them: (g1+g2+...+g8) / 8
  Update weights using this averaged gradient
```

The result is **mathematically identical** to processing 16 samples at once — but uses only enough memory for 2 samples.

**Trade-off**: Takes 8× longer per update (8 forward+backward passes instead of 1), but uses 8× less memory.

### In our setup
```
per_device_train_batch_size = 2    ← fits in GPU memory
gradient_accumulation_steps = 8    ← accumulate 8 mini-batches
effective_batch_size = 2 × 8 = 16  ← as if we processed 16 at once
```

---

## 9. Gradient Clipping

Sometimes a single example produces an **enormous gradient** — maybe a weird outlier in the data, or a numerical instability. If we apply this giant gradient, the weights jump wildly and training can blow up.

**Gradient clipping** puts a speed limit on gradients:

```
If ||gradient|| > max_norm:
    gradient = gradient × (max_norm / ||gradient||)
```

This shrinks the gradient vector to have maximum length `max_norm`, while keeping its direction the same. It's like saying "you can point any direction, but you can't step further than this."

### Our setting: `max_grad_norm = 0.1`

This is aggressive — the default is 1.0. We use 0.1 because importance sampling ratios can cause gradient spikes. If the student and teacher disagree strongly on a token (ratio >> 1 or << 1), the gradient can be large. Clipping at 0.1 prevents these spikes from destabilizing training.

### What `grad_norm` means in our logs

The training logs show `grad_norm` — this is the magnitude of the gradient **before** clipping. In exp02:
- Average grad_norm: ~0.1 (right at the clip threshold)
- Max grad_norm: 6.17 (one spike, clipped down to 0.1)
- grad_norm = 0.0: means the loss was zero (all-same rewards → zero advantage → zero gradient)

---

## 10. Weight Decay

Weight decay is a form of **regularization** — it gently pushes all weights toward zero. Without it, weights can grow arbitrarily large, which usually means the model is memorizing noise rather than learning patterns.

```
new_weight = old_weight - learning_rate × gradient - weight_decay × old_weight
                                                     ↑ this shrinks the weight
```

Our setting: `weight_decay = 0.1` — applied to all weights except bias terms and layer normalization. This is a fairly standard value.

Think of it as a tax on large weights: the bigger a weight gets, the more it costs to keep it that big. The model has to "justify" large weights by them actually helping reduce the loss.

---

## 11. Learning Rate Schedules

Instead of keeping the learning rate constant, we change it during training. The idea: take big steps at the beginning (explore), smaller steps later (fine-tune).

### Warmup

For the first N steps, the learning rate gradually increases from 0 to the target value.

```
Steps 1-1200 (10% of 12000 total):
  LR: 0 → 5e-6 (linear increase)
```

Why? At the start, the model's gradients are based on random-ish predictions. Taking a big step based on garbage gradients is dangerous. Warmup lets the model "warm up" — get reasonable gradients before taking full-sized steps.

### Cosine Schedule

After warmup, the learning rate follows a cosine curve from the peak down to ~0:

```
Step:    0        1200       6000       12000
LR:     0 → 5e-6 → ~3.5e-6 → ~0
        |  warmup |  cosine decay      |
```

```
LR
5e-6 |    /\
     |   /  \
     |  /    \
     | /      --------___
0    |/                   \___
     +--------------------------→ steps
     0   1200              12000
```

### Why cosine?

- **Early training**: high LR, big steps, model learns the main patterns quickly
- **Late training**: low LR, small steps, model polishes details without overshooting

### The problem in our exp02

KL divergence plateaued at epoch 0.3 (~step 3600). At that point, cosine had already reduced the LR to ~60% of peak. By epoch 0.5, LR was down to ~25%. The model spent 70% of training making near-zero updates.

**Alternative schedules**:
- **Constant**: LR stays at 5e-6 the whole time. Simple, keeps learning throughout.
- **Linear decay**: LR decreases linearly from 5e-6 to 0. Slower decay than cosine.
- **Cosine with more epochs**: Train for 3 epochs. The cosine still decays within each epoch but the model sees the data multiple times.

### `warmup_ratio` vs `warmup_steps`

```
warmup_ratio = 0.1 means: spend 10% of total steps warming up
warmup_steps = 1200 means: spend exactly 1200 steps warming up

Both achieve the same thing. We use warmup_ratio so it auto-scales with total steps.
```

---

## 12. Numerical Precision (bf16, fp16, fp32)

Every number in the computer is stored with limited precision. More bits = more precise = more memory.

| Format | Bits | Memory per number | Range | Precision | Use |
|---|---|---|---|---|---|
| **fp32** (float32) | 32 | 4 bytes | ±3.4×10³⁸ | ~7 decimal digits | Optimizer states (Adam m, v) |
| **fp16** (float16) | 16 | 2 bytes | ±65,504 | ~3 decimal digits | Older mixed precision |
| **bf16** (bfloat16) | 16 | 2 bytes | ±3.4×10³⁸ | ~3 decimal digits | **What we use** for model weights and forward/backward |

### Why bf16?

fp16 can only represent numbers up to 65,504. In deep learning, loss values and gradients sometimes spike beyond this → **inf** → training crashes. bf16 has the same range as fp32 (huge numbers are fine) but only 2 bytes per number.

```
fp32: 4 bytes × 500M weights = 2.0 GB
bf16: 2 bytes × 500M weights = 1.0 GB  ← half the memory, same range
```

### Mixed precision

"Mixed precision" means: do the forward and backward pass in bf16 (fast, low memory), but keep the optimizer states and weight updates in fp32 (accurate). This gives you the speed of bf16 with the accuracy of fp32.

Our setting: `bf16=True` — the model weights and computations use bfloat16, Adam states use fp32.

---

## 13. Overfitting and Regularization

### Overfitting

The model memorizes the training data instead of learning general patterns. It performs great on training data but poorly on new data.

```
Analogy: A student who memorizes every answer in the textbook word-for-word
         but can't solve a new problem that's slightly different.
```

Signs of overfitting:
- Training loss keeps going down
- Validation/test accuracy stops improving or gets worse
- Model outputs become very confident (low entropy)

### Regularization

Techniques to prevent overfitting — they make it harder for the model to "cheat" by memorizing:

| Technique | How it helps | Our setting |
|---|---|---|
| **Weight decay** | Penalizes large weights, keeps model simple | 0.1 |
| **Dropout** | Randomly zeros out neurons during training, forces redundancy | 0.05 (LoRA only) |
| **KL penalty (beta)** | Prevents student from deviating too far from base model | 0.1 |
| **Gradient clipping** | Prevents wild weight updates | max_grad_norm=0.1 |
| **LoRA** (low rank) | Limits the expressiveness of updates to low-rank changes | r=16 |
| **Early stopping** | Stop training when validation accuracy plateaus | We don't use this (we should) |

### Entropy as a health check

**Entropy** measures how spread out the model's predictions are. High entropy = uncertain, predicts many tokens roughly equally. Low entropy = confident, strongly predicts specific tokens.

In our exp02, entropy stayed stable at ~0.24 throughout training. If it dropped toward 0, that would mean the model collapsed to always predicting the same thing — a sign of overfitting or mode collapse.

---

## 14. How Language Models Work

A language model predicts the **next token** given all previous tokens. That's it. Everything else (conversations, math, code) emerges from this simple task.

### The core operation

```
Input:  "The capital of France is"
Output: probability distribution over ~150,000 possible next tokens

  "Paris"  → 0.85
  "Lyon"   → 0.02
  "the"    → 0.01
  "Berlin" → 0.001
  ...
```

The model doesn't "know" facts — it has learned statistical patterns from training data. "Paris" follows "The capital of France is" with high probability because it saw similar patterns billions of times.

### Transformer architecture (simplified)

```
Input tokens → Embedding (look up a vector for each token)
            → Attention Layer 1 (tokens look at each other)
            → MLP Layer 1 (process each token independently)
            → Attention Layer 2
            → MLP Layer 2
            → ... (24 layers for 0.5B, 32 layers for 7B)
            → Output projection (convert back to vocabulary probabilities)
```

**Attention** is the key innovation: each token can "attend to" (look at) all previous tokens to decide what comes next. "France" attends to "capital" to know that "Paris" is likely.

**MLP** (Multi-Layer Perceptron, also called Feed-Forward Network) processes each token position independently. It's where a lot of the "knowledge" is stored.

### Decoder-only

Qwen (and GPT, LLaMA, etc.) are "decoder-only" — they only predict forward, never backward. Each position can only attend to positions before it (causal attention mask). This is what makes autoregressive generation work.

---

## 15. Autoregressive Generation vs Single-Pass Evaluation

### Autoregressive generation (slow, for creating text)

The model generates text one token at a time:

```
Step 1: Input "What is 2+3?" → predict "The" → append
Step 2: Input "What is 2+3? The" → predict "answer" → append
Step 3: Input "What is 2+3? The answer" → predict "is" → append
Step 4: Input "What is 2+3? The answer is" → predict "5" → append
...
```

Each step requires a full forward pass. 100 tokens = 100 forward passes. This is what the teacher does during rollout generation, and what the student does during evaluation. **vLLM** speeds this up with KV caching (remembering previous computations instead of recomputing from scratch).

### Single-pass evaluation (fast, for computing logprobs)

When we already know the full token sequence, we can compute all logprobs in **one** forward pass:

```
Input: ["What", "is", "2+3?", "The", "answer", "is", "5"]  ← all at once
Output: P(is|What), P(2+3?|What is), P(The|...), P(answer|...), P(is|...), P(5|...)
```

The causal attention mask ensures each position only sees tokens before it, so the results are identical to autoregressive generation. But instead of 6 forward passes, it's just 1.

### Where each is used in our pipeline

| Stage | Method | Why |
|---|---|---|
| Rollout generation (teacher) | Autoregressive (vLLM) | Don't know the text yet — generating it |
| Training (student logprobs) | Single-pass (PyTorch) | Text is known (teacher's solution) — just need logprobs |
| Training (ref logprobs) | Single-pass (PyTorch) | Same text, disable_adapter() |
| Evaluation (student) | Autoregressive (vLLM) | Generating new solutions to check accuracy |

---

## 16. Tokens and Tokenization

Models don't see text — they see **token IDs** (integers). A **tokenizer** converts between text and IDs.

### How tokenization works

```
Text:   "The answer is 5."
Tokens: ["The", " answer", " is", " 5", "."]
IDs:    [464, 3280, 374, 220, 13]
```

Tokens are NOT words. They're chunks that the tokenizer learned are common:
- Common words: "the" → 1 token
- Uncommon words: "backpropagation" → "back", "prop", "ag", "ation" → 4 tokens
- Numbers: "42" might be 1 or 2 tokens depending on the tokenizer
- Math symbols: "\\boxed{" → could be 2-3 tokens

### Vocabulary size

The tokenizer has a fixed set of tokens it knows. Qwen2.5-0.5B has 151,936 tokens. Qwen2.5-Math-7B has 152,064 (128 extra math tokens). This mismatch caused our vocab crash — the teacher generated token ID 152000, but the student's embedding table only has 151,936 rows.

### Why token count matters for memory

```
Sequence length = prompt tokens + completion tokens = 256 + 786 = 1042

Per sample, the model needs to store:
- 1042 token embeddings
- Attention scores: 1042 × 1042 per layer per head
- Intermediate activations at each layer

With batch_size=2: all of this × 2
```

Longer sequences use quadratically more memory (because of attention) and linearly more compute.

---

## 17. Log-Probabilities (Logprobs)

### Probability vs log-probability

```
Probability:      P("Paris") = 0.85
Log-probability:  log(0.85) = -0.163
```

We use log-probabilities because:

1. **Multiplying probabilities is dangerous**. A sequence of 100 tokens:
   ```
   P(sequence) = P(token1) × P(token2) × ... × P(token100)
               = 0.8 × 0.7 × 0.9 × ...
               ≈ 0.0000000000001  ← underflows to 0 in floating point
   ```

2. **Adding log-probabilities is safe**:
   ```
   log P(sequence) = log P(token1) + log P(token2) + ... + log P(token100)
                   = -0.22 + -0.36 + -0.11 + ...
                   = -29.7  ← perfectly representable
   ```

3. **The ratio becomes a subtraction**:
   ```
   ratio = P_student / P_teacher = exp(log P_student - log P_teacher)
   ```

### In our pipeline

- Teacher logprobs: saved in `rollouts.jsonl`, one float per token (e.g., `-0.302`)
- Student logprobs: computed during training via forward pass
- Both are negative numbers (log of a probability between 0 and 1 is always negative)
- More negative = less confident. `-0.01` means ~99% confident. `-4.6` means ~1% confident.

---

## 18. KL Divergence

**KL divergence** (Kullback-Leibler divergence) measures how different two probability distributions are. In our case: how far the student has drifted from its original base model.

### Intuition

```
Base model (before training):  P("Factor") = 0.10
Student (after some training): P("Factor") = 0.25

The student is now 2.5× more likely to say "Factor" than it was before training.
KL measures this kind of drift across ALL tokens.
```

### The formula (per token)

```
KL = student_logprob - base_logprob
```

If the student assigns higher probability than the base: KL is positive (student has diverged).
If they agree: KL ≈ 0.

### Why we penalize KL

Without the KL penalty, the student could overfit to the teacher's solutions — it might become great at producing those specific solutions but terrible at everything else. The KL penalty says: "you can learn from the teacher, but don't stray too far from who you were."

### Beta (β)

```
total_loss = policy_gradient_loss + β × KL_penalty
```

| Beta | Effect |
|---|---|
| β = 0.0 | No KL penalty. Student can change freely. Risk of mode collapse. |
| β = 0.1 (ours) | Mild penalty. Student can learn but stays somewhat close to base. |
| β = 1.0 | Strong penalty. Student barely changes. Very conservative. |

### In our exp02 logs

KL grew from 0.0007 to 0.0027 during training — a very small divergence. The student's probability distribution changed by less than 0.3% overall. This might be too conservative (β=0.1 too high), which is why exp02 suggests trying β=0.01.

---

## 19. Importance Sampling

### The problem

We want the **student** to learn, but the solutions were generated by the **teacher**. The student and teacher are different models — they assign different probabilities to the same tokens.

If we just trained the student on teacher solutions directly (supervised fine-tuning), the student would learn to imitate the teacher blindly — including the teacher's wrong solutions.

### The solution: importance sampling ratio

```
ratio = π_student(token) / π_teacher(token)
```

This ratio adjusts the gradient based on how much the student and teacher agree:

| ratio | Meaning | Effect on gradient |
|---|---|---|
| ≈ 1.0 | Student and teacher agree on this token | Full gradient signal |
| < 1.0 (e.g., 0.3) | Student is less confident than teacher | Dampened gradient — don't force the student to drastically change |
| > 1.0 (e.g., 3.0) | Student is more confident than teacher | Amplified gradient — student already likes this, reinforce or suppress more |

### Why this matters

Without the ratio, a token the teacher loves but the student has never seen would get the same gradient as a token both models agree on. The ratio says: "weight the gradient by how relevant this token is to the student."

### In code

```python
ratio = exp(student_logprob - teacher_logprob)
loss = -advantage × ratio
```

The `exp(a - b) = exp(a)/exp(b) = P_student/P_teacher` identity converts the log-space subtraction into a probability-space ratio.

---

## 20. PPO-Style Clipping

### The problem with unbounded ratios

If the student and teacher strongly disagree on a token, the ratio can be extreme:

```
Teacher: P("Factor") = 0.80, logprob = -0.22
Student: P("Factor") = 0.01, logprob = -4.60

ratio = exp(-4.60 - (-0.22)) = exp(-4.38) = 0.013
```

Or the other direction:
```
Teacher: P("clearly") = 0.01
Student: P("clearly") = 0.50

ratio = 50.0  ← huge!
```

A ratio of 50 means the gradient is 50× stronger for this token. One weird token could dominate the entire training step.

### The solution: clip the ratio

```
clipped_ratio = clip(ratio, 1-ε, 1+ε)

where ε = 0.2 (default in TRL)
so ratio is clamped to [0.8, 1.2]
```

The actual loss uses the minimum of clipped and unclipped:
```
loss = -advantage × min(ratio, clipped_ratio)
```

This creates a "trust region" — the model can't change too much in one step. If the ratio goes outside [0.8, 1.2], the gradient is capped.

### Why "PPO-style"

PPO (Proximal Policy Optimization) introduced this clipping idea for reinforcement learning. GRPO borrows it. The "proximal" means "keep changes close to where you started."

---

## 21. LoRA — Low-Rank Adaptation

### The problem

Fine-tuning means updating model weights. A 7B model has 7 billion weights. Storing weights + optimizer states requires ~70 GB. Doesn't fit on our 48 GB GPU.

### The idea

When you fine-tune a model, the weight changes (ΔW) tend to be **low-rank** — they can be approximated by two small matrices multiplied together.

```
Original weight matrix W: 4096 × 4096 = 16.7 million numbers (FROZEN)

Instead of learning ΔW (another 16.7M numbers), learn:
  A: 4096 × 16 = 65,536 numbers
  B: 16 × 4096 = 65,536 numbers

ΔW ≈ A × B  (131,072 numbers instead of 16.7M — 128× fewer)

Forward pass: output = W × input + (A × B) × input
                       ↑ frozen     ↑ trainable (small!)
```

### Our LoRA config

| Parameter | Value | Meaning |
|---|---|---|
| **r = 16** | Rank of A and B matrices. Higher = more expressive but more memory. |
| **alpha = 64** | Scaling factor. LoRA update is multiplied by alpha/r = 4.0. Higher = bigger updates. |
| **dropout = 0.05** | 5% of LoRA values randomly zeroed during training (regularization). |
| **target_modules** | q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj | Apply LoRA to all attention projections + MLP projections in every layer. |

### What target_modules means

A transformer layer has these weight matrices:

```
Attention block:
  q_proj (Query):   transforms input → "what am I looking for?"
  k_proj (Key):     transforms input → "what do I contain?"
  v_proj (Value):   transforms input → "what information do I carry?"
  o_proj (Output):  transforms attention output → next layer input

MLP block:
  gate_proj:  first transformation (with gating)
  up_proj:    projects to higher dimension
  down_proj:  projects back down
```

We add LoRA adapters to **all 7** of these in **every layer**. For the 0.5B model (~24 layers):
```
7 matrices × 24 layers × ~131K params = ~22M trainable parameters
Total model: 500M, trainable: 22M (4.4%)
```

### The "free reference model" trick

With LoRA, getting the base model's predictions is free:

```python
# LoRA ON:  output = (W + A×B) × input    ← student
# LoRA OFF: output = W × input            ← base model (reference)

with model.disable_adapter():
    ref_logprobs = model(input)   # base model predictions, no extra memory!
```

Without LoRA, you'd need to load a **second copy** of the entire model as the reference — 14 GB wasted for a 7B model.

---

## 22. Multi-GPU Training

### Why use multiple GPUs?

1. **Model doesn't fit**: 7B model + optimizer ≈ 70 GB, but GPU has 48 GB
2. **Faster training**: split the data across GPUs, process in parallel

### DDP (Distributed Data Parallel)

The simplest strategy. Every GPU gets a **full copy** of the model.

```
GPU 0: Full model copy → processes batch A → gradient A
GPU 1: Full model copy → processes batch B → gradient B
                     ↓
              Average gradients (all-reduce)
                     ↓
              Same weight update on both GPUs
```

- **Pros**: Simple, always works, LoRA compatible
- **Cons**: Each GPU needs enough memory for the full model + optimizer
- **Speedup**: 2× GPUs ≈ 2× faster (each processes half the data)

### DeepSpeed ZeRO (Zero Redundancy Optimizer)

In DDP, every GPU stores the full model, full gradients, AND full optimizer states. That's redundant. ZeRO eliminates the redundancy:

```
                    DDP          ZeRO-2
GPU 0:
  Model weights:    Full (14GB)  Full (14GB)      ← still full
  Gradients:        Full (14GB)  Half (7GB)       ← split!
  Optimizer:        Full (28GB)  Half (14GB)      ← split!
  Total:            56 GB        35 GB            ← fits on 48GB GPU now!

GPU 1:
  Model weights:    Full (14GB)  Full (14GB)
  Gradients:        Full (14GB)  Other half (7GB)
  Optimizer:        Full (28GB)  Other half (14GB)
  Total:            56 GB        35 GB
```

**ZeRO-2** (what we tested): splits optimizer states + gradients across GPUs, but keeps full model weights on each GPU. This is why `disable_adapter()` still works — full weights are present.

**ZeRO-3**: also splits model weights. Most memory savings, but `disable_adapter()` operates on partial weights.

### FSDP (Fully Sharded Data Parallel)

PyTorch's built-in version of ZeRO-3. Shards everything (weights, gradients, optimizer states). During forward pass, weights are temporarily gathered from all GPUs, used, then discarded.

### When to use what

```
Model fits on 1 GPU?
  YES → single GPU, no distribution needed
  NO  → How big is it?
        ≤ 14B  → ZeRO-2 (split optimizer/gradients, keep full weights)
        > 14B  → ZeRO-3 or FSDP (split everything)
```

---

## 23. Gradient Checkpointing

### The memory problem

During the forward pass, the model stores **intermediate results** (activations) at every layer. These are needed for the backward pass to compute gradients.

```
Forward:  input → [layer 1 output saved] → [layer 2 output saved] → ... → [layer 24 output saved] → loss
Backward: loss → use saved layer 24 output → use saved layer 23 output → ... → gradients
```

For a 7B model with long sequences, these saved activations can use 15-30 GB of memory.

### The solution

**Gradient checkpointing** throws away most intermediate results during the forward pass. During the backward pass, when it needs them, it **recomputes** them on the fly.

```
Without checkpointing:
  Forward:  save all 24 layer outputs (15 GB)
  Backward: use saved outputs (fast)

With checkpointing:
  Forward:  save every 4th layer output (3.75 GB)    ← 4× less memory
  Backward: recompute missing layers from nearest checkpoint (slower)
```

### Trade-off

```
Memory:  ~60-75% reduction in activation memory
Speed:   ~25-33% slower (recomputing parts of the forward pass)
```

### When to use it

We don't currently use gradient checkpointing (our 0.5B model fits easily). But if we scale to a 7B or 14B student, we'll likely need it:

```python
training_args = GRPOConfig(
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    ...
)
```

---

## 24. Checkpointing and Resuming

### Checkpointing

Every `save_steps=500` steps, the trainer saves everything needed to resume later:

```
checkpoint-3500/
├── adapter_model.safetensors    ← LoRA weights (the actual model update)
├── adapter_config.json          ← LoRA configuration
├── optimizer.pt                 ← Adam states (m and v for every parameter)
├── scheduler.pt                 ← Learning rate scheduler state
├── rng_state.pth                ← Random number generator states
├── trainer_state.json           ← Step count, loss history, etc.
└── training_args.bin            ← All hyperparameters
```

### Resuming from checkpoint

```python
trainer.train(resume_from_checkpoint=True)    # auto-find latest in output_dir
trainer.train(resume_from_checkpoint="path/to/checkpoint-3500")  # specific checkpoint
```

When resuming, the trainer:
1. Loads model weights from the checkpoint
2. Loads optimizer states (so Adam's momentum/variance are preserved)
3. Loads the LR scheduler state (continues cosine from where it stopped)
4. **Skips** data that was already processed (replays the data loader to the right position)
5. Continues training from step 3501

### Why we needed this

Our exp02 training timed out at step 3500 (2-hour SLURM limit). We needed to resume from checkpoint-3500. But `--resume_from_checkpoint` wasn't implemented in `train.py` — we had to add it. Then we also found that `"latest"` is not a valid value (must use `True` for auto-detection).

### `save_total_limit`

If you set `save_total_limit=5`, only the 5 most recent checkpoints are kept. Older ones are deleted automatically. Useful because 24 checkpoints × ~50 MB each = 1.2 GB for a 0.5B model, but for a 7B model each checkpoint would be much larger.

---

## 25. Putting It All Together — Our Pipeline

Here's how every concept connects in one training step of our offline GRPO:

```
1. DATA LOADING (CPU)
   Load teacher's solution from rollouts.jsonl
   - completion_ids: [2647, 264, 14285, ...]  (token IDs)
   - behavior_logprobs: [-0.005, -0.302, ...]  (teacher's confidence per token)
   - advantage: +0.58 (pre-computed from reward vs group mean)

2. STUDENT FORWARD PASS (GPU, bf16)
   Feed [prompt + teacher's tokens] through student model
   - Uses causal attention mask (single pass, not autoregressive)
   - Uses Flash Attention 2 (memory-efficient attention computation)
   - Output: student_logprobs for each token

3. REFERENCE FORWARD PASS (GPU, bf16, no gradients)
   If beta > 0:
     disable_adapter() → turns off LoRA → pure base model
     Same input → base_logprobs
     Re-enable LoRA

4. LOSS COMPUTATION
   For each token:
     ratio = exp(student_logprob - teacher_logprob)     ← importance sampling
     clipped = clip(ratio, 0.8, 1.2)                    ← PPO-style safety
     token_loss = -advantage × min(ratio, clipped)

   KL = student_logprob - base_logprob
   total_loss = mean(token_losses) + 0.1 × KL

5. BACKWARD PASS (GPU)
   Compute gradient of total_loss with respect to every LoRA weight
   - Only LoRA weights (13M) have gradients, not the frozen base (500M)

6. GRADIENT ACCUMULATION
   Add this gradient to the running sum
   Repeat steps 1-5 for 8 mini-batches (gradient_accumulation_steps=8)

7. GRADIENT CLIPPING
   If ||accumulated_gradient|| > 0.1:
     Scale down to magnitude 0.1

8. OPTIMIZER STEP (Adam)
   For each LoRA weight:
     m = 0.9 × m_old + 0.1 × gradient
     v = 0.99 × v_old + 0.01 × gradient²
     weight -= 5e-6 × m / (√v + 1e-8)

   Also apply weight_decay: weight -= 0.1 × weight

9. LR SCHEDULER UPDATE
   Adjust learning rate according to cosine schedule
   (warm up for first 10% of steps, then decay toward 0)

10. LOGGING
    Every 10 steps: log loss, grad_norm, reward, KL, entropy to wandb

11. CHECKPOINTING
    Every 500 steps: save LoRA weights + optimizer states + scheduler state

12. REPEAT
    Go back to step 1 with the next batch. Continue for ~12,000 steps (1 epoch).
```

---

*Created: 2026-03-09*
