# Offline GRPO Project — Catch-Up Guide

---

## 1. The Task

We want to make a **small language model better at solving math problems**.

Specifically:
- We have a **student model** (Qwen2.5-0.5B-Instruct) — small, fast, but not great at math
- We have a **teacher model** (Qwen2.5-Math-7B-Instruct) — large, slow, good at math
- We want the student to **learn from the teacher's work**

The dataset is **MATH** (hendrycks-MATH-benchmark): 12,000 math problems for training, 500 for testing. Each problem has a text question, a step-by-step solution, and a final answer.

---

## 2. What Does the Data Look Like?

### Input (X): A math problem formatted as a chat

```
System: Please reason step by step, and put your final answer within \boxed{}.
User:   If 5x - 3 = 12, what is the value of 5x + 3?
```

### Output (Y): A step-by-step solution ending with \boxed{answer}

```
To solve the equation 5x - 3 = 12 and find the value of 5x + 3:

1. Start with: 5x - 3 = 12
2. Add 3 to both sides: 5x = 15
3. Substitute into 5x + 3: 15 + 3 = 18

Therefore, the value of 5x + 3 is \boxed{18}.
```

### But we don't train on (X, Y) pairs directly!

Unlike supervised fine-tuning where you show the model correct answers, GRPO works differently. We show the model **multiple attempts** (some correct, some wrong) and tell it: "do more of the correct ones, less of the wrong ones."

---

## 3. The Algorithm: GRPO (Group Relative Policy Optimization)

### The Idea in Plain English

1. Give a math problem to the teacher model
2. The teacher generates **4 different solutions** (some right, some wrong)
3. Check each solution: correct → reward = 2.0, wrong → reward = 0.0
4. Within each group of 4 solutions, compute **advantage** = how much better/worse than average
   - If 3 out of 4 are correct: correct ones get small positive advantage, wrong one gets large negative advantage
   - If all 4 are correct: everyone gets advantage ≈ 0 (nothing to learn)
5. Train the student: **increase probability** of high-advantage solutions, **decrease probability** of low-advantage ones

### Why "Offline"?

In **online GRPO**, the student generates its own solutions and learns from them. The ratio is:
```
ratio = π_student_new(solution) / π_student_old(solution)
```
Both numerator and denominator are the same model at different times.

In **offline GRPO**, the teacher generates solutions, and the student learns from the teacher's work. The ratio becomes:
```
ratio = π_student(solution) / π_teacher(solution)
```
This is called **importance sampling** — we're correcting for the fact that the solutions came from a different model.

### Why LoRA?

Full fine-tuning updates ALL model weights (500M+ parameters). **LoRA** (Low-Rank Adaptation) adds small trainable matrices to the model and only updates those (~0.5% of weights). This means:
- Much less GPU memory needed
- Faster training
- The saved checkpoint is tiny (MBs instead of GBs)
- Can scale to larger models that wouldn't fit in GPU memory with full fine-tuning

---

## 4. The Pipeline: Step by Step

```
┌─────────────────────────────────────────────────────────┐
│  Step 1: GENERATE ROLLOUTS (generate_rollouts.py)       │
│                                                         │
│  Teacher model (7B) solves 12,000 math problems         │
│  4 solutions per problem = 48,000 completions           │
│  Saves: solution text + token IDs + per-token logprobs  │
│                                                         │
│  Output: rollouts.jsonl                                 │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Step 2: TRAIN STUDENT (train.py + trainer.py)          │
│                                                         │
│  1. Load rollouts, compute rewards & advantages         │
│  2. Load student model (0.5B) with LoRA                 │
│  3. For each batch:                                     │
│     - Look up teacher's pre-computed completions         │
│     - Compute student's logprobs on those completions    │
│     - Compute IS ratio = π_student / π_teacher           │
│     - Clip ratio, multiply by advantage, compute loss    │
│     - Update LoRA weights                                │
│                                                         │
│  Output: LoRA checkpoint (adapter_model.safetensors)    │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Step 3: EVALUATE (evaluate.py)                         │
│                                                         │
│  Load student + LoRA, solve 500 test problems           │
│  Check answers, report accuracy                         │
│                                                         │
│  Output: accuracy score (printed)                       │
└─────────────────────────────────────────────────────────┘
```

---

## 5. Code Files — What Does Each One Do?

```
offline_grpo/
├── configs.py              ← Constants and utilities
├── generate_rollouts.py    ← Step 1: Teacher generates solutions
├── data.py                 ← Data loading, rewards, advantages
├── trainer.py              ← The custom offline GRPO trainer
├── train.py                ← Step 2: Train student with LoRA
├── evaluate.py             ← Step 3: Evaluate on test set
├── diagnose.py             ← Debug tool: check if training signal is healthy
├── run_test.sh             ← Shell script to run small test (41 problems)
└── run_full.sh             ← Shell script to run full scale (12,000 problems)
```

### configs.py
Shared constants used everywhere:
- `SYSTEM_PROMPT` = "Please reason step by step, and put your final answer within \\boxed{}."
- Default model names and LoRA hyperparameters (rank=16, alpha=64)
- `extract_boxed_answer()` — parses `\boxed{18}` from model output to get `18`

### generate_rollouts.py
**Purpose**: Use the teacher model to generate example solutions.

Uses **vLLM** (a fast inference engine) to generate `N` solutions per problem. For each solution, it saves:
- The text of the solution
- The **token IDs** (the solution as numbers the model sees internally)
- The **per-token log-probabilities** — how confident the teacher was about each word

The logprobs are critical: during training, we need to compute `π_student / π_teacher`, and the teacher's logprobs are the denominator.

**Output format** (rollouts.jsonl — one line per problem):
```json
{
  "question_id": 1,
  "original_problem": "If 5x - 3 = 12, what is the value of 5x + 3?",
  "ground_truth_answer": "18",
  "system_prompt": "Please reason step by step...",
  "runs": [
    {
      "run_id": 0,
      "response": "To solve... \\boxed{18}",
      "boxed_answer": "18",
      "logprobs": [-0.005, -0.302, -0.697, ...],
      "completion_ids": [2647, 264, 14285, ...]
    },
    { "run_id": 1, ... },
    { "run_id": 2, ... },
    { "run_id": 3, ... }
  ]
}
```

### data.py
**Purpose**: Load rollouts and compute rewards + advantages.

Four functions:
1. `load_rollouts()` — reads the JSONL file, flattens it into one record per completion
2. `compute_rewards_and_advantages()` — checks if each answer is correct (reward = 2.0 or 0.0), then computes group-normalized advantage:
   ```
   advantage = (reward - group_mean) / group_std
   ```
3. `build_training_dataset()` — creates a HuggingFace Dataset with prompts sorted so that each group of 4 completions for the same problem is contiguous
4. `build_offline_lookup()` — creates a dictionary keyed by `(question_id, run_id)` so the trainer can quickly find pre-computed data

### trainer.py — THE CORE
**Purpose**: Custom trainer that overrides TRL's GRPOTrainer for offline mode.

The only method it overrides is `_generate_and_score_completions()`. In normal (online) GRPO, this method would:
1. Generate new completions from the student
2. Score them with a reward function

Our offline version instead:
1. **Looks up** the teacher's pre-computed completions from the offline data dict
2. **Sets `old_per_token_logps`** to the teacher's logprobs

This is the key trick: TRL's loss function internally computes:
```python
ratio = exp(current_logps - old_per_token_logps)
```
In online GRPO, this is `π_new / π_old` (same model, different snapshots).
In our offline version, this becomes `π_student / π_teacher` (importance sampling ratio).

TRL then clips this ratio (PPO-style) and multiplies by the advantage to get the loss. We don't need to change the loss function at all.

### train.py
**Purpose**: Orchestrates everything for Step 2.

1. Loads rollouts → computes rewards/advantages → builds dataset + lookup
2. Loads student model (0.5B) with Flash Attention
3. Creates LoRA config (rank=16, applied to all attention + MLP projections)
4. Creates `GRPOConfig` with training hyperparameters
5. Creates `OfflineGRPOTrainer` and calls `.train()`
6. Saves the LoRA checkpoint

### evaluate.py
**Purpose**: Test how well the trained model does.

Can optionally merge a LoRA adapter into the base model first (`--merge_lora`). Then uses vLLM to generate solutions for 500 test problems and checks accuracy.

### diagnose.py
**Purpose**: Debug tool to check if the training pipeline is producing meaningful signal.

Checks:
- Are rewards diverse? (If all solutions for a problem are correct, advantage=0, no learning)
- Are student and teacher logprobs similar? (If too different, IS ratios explode)
- Are IS ratios reasonable? (Per-token should be ~1.0; sequence-level will always be ~0 due to multiplication)

---

## 6. Key Training Concepts

### Batch Size and Steps

```
per_device_train_batch_size = 2      ← 2 samples per forward pass
gradient_accumulation_steps = 8      ← accumulate 8 forward passes before updating
effective_batch_size = 2 × 8 = 16   ← 16 samples per weight update

total_samples = 48,000               ← 12,000 problems × 4 completions
steps_per_epoch = 48,000 / 16 = 3,000  ← 3,000 weight updates per pass through data
num_train_epochs = 1                 ← 1 pass through all data
total_training_steps = 3,000
```

### What Happens in One Training Step

```
1. Take 2 samples from the dataset
2. Look up their teacher completions, logprobs, and advantages
3. Feed [prompt + teacher's completion] through the student model
4. Get student's logprobs for each token of the teacher's completion
5. Compute ratio = exp(student_logprob - teacher_logprob) for each token
6. Clip the ratio to [1-ε, 1+ε] to prevent wild updates
7. Loss = -advantage × clipped_ratio  (summed over tokens)
8. Also add KL penalty: β × (student_logprob - base_model_logprob)
   This prevents the student from drifting too far from its original behavior
9. Accumulate this gradient (repeat 8 times for gradient_accumulation)
10. Update LoRA weights
```

### The Loss Function (Simplified)

```
For each token t in the completion:
    ratio_t = exp(log π_student(token_t) - log π_teacher(token_t))
    clipped_ratio_t = clip(ratio_t, 1-ε, 1+ε)
    loss_t = -advantage × min(ratio_t, clipped_ratio_t)

Total loss = mean(loss over all tokens) + β × KL_penalty
```

- If advantage > 0 (good solution): loss pushes student to increase probability of these tokens
- If advantage < 0 (bad solution): loss pushes student to decrease probability of these tokens
- If advantage ≈ 0 (all solutions same quality): no gradient signal

---

## 7. File Locations on the Cluster

### Code (backed up, safe)
```
/home/shuai14/projects/aip-szepesva/shuai14/backup_dongheng/offline_grpo/
```

### Models (in scratch, NOT backed up)
```
Student (0.5B): /home/shuai14/scratch/huggingface_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/
                snapshots/7ae557604adf67be50417f59c2c2f167def9a775

Teacher (7B):   /home/shuai14/scratch/huggingface_cache/hub/models--Qwen--Qwen2.5-Math-7B-Instruct/
                snapshots/ef9926d75ab1d54532f6a30dd5e760355eb9aa4d
```

### Dataset
```
/home/shuai14/scratch/datasets/MATH/nlile___hendrycks-math-benchmark/
  train: 12,000 problems
  test:  500 problems
```

### Outputs (in scratch)
```
Rollouts:    /home/shuai14/scratch/dongheng/rollouts_full.jsonl
Checkpoints: /home/shuai14/scratch/dongheng/offline_grpo_full/
```

---

## 8. How to Run

### Test Run (small, ~5 min total)
```bash
# 1. Get a GPU
salloc --account=aip-szepesva --time=1:00:00 --gpus-per-node=l40s:1 --cpus-per-task=16 --mem=64G

# 2. Set up environment
module load python/3.11 cuda/12.6
source /home/shuai14/projects/aip-szepesva/shuai14/verifiers/.venv/bin/activate
cd /home/shuai14/projects/aip-szepesva/shuai14/backup_dongheng/offline_grpo

# 3. Run
bash run_test.sh rollouts    # Teacher generates solutions (~2 min)
bash run_test.sh train       # Student learns from them (~1 min)
bash run_test.sh diagnose    # Check if signal is healthy (~5 min)

# 4. Release GPU
exit
```

### Full Run (from login node, runs in background)
```bash
cd /home/shuai14/projects/aip-szepesva/shuai14/backup_dongheng/offline_grpo

sbatch --job-name=grpo-rollouts run_full.sh rollouts   # ~10 hours
# Wait for it to finish (check with: squeue -u $USER)
sbatch --job-name=grpo-train run_full.sh train         # ~1 hour
# Wait for it to finish
sbatch --job-name=grpo-eval run_full.sh eval           # ~30 min
```

---

## 9. LoRA Deep Dive

### The Problem LoRA Solves

A language model is made of many weight matrices. A 7B model has ~7 billion numbers in these matrices. During training, for each trainable parameter the optimizer (Adam) stores:
- The **gradient** (same size as the parameter)
- **Momentum** (running average of gradients — same size)
- **Variance** (running average of squared gradients — same size)

So full fine-tuning of a 7B model needs:
```
Model weights:     14 GB  (7B × 2 bytes in bf16)
Gradients:         14 GB
Optimizer states:  28 GB  (2 × 14 GB for Adam momentum + variance, stored in fp32)
                   ──────
Total:             ~56 GB  just for parameters — doesn't fit on one 48GB GPU
```

### How LoRA Works

LoRA stands for **Low-Rank Adaptation**. The key insight: when you fine-tune a model, the weight change (ΔW) tends to be **low-rank** — meaning it can be approximated by the product of two much smaller matrices.

Instead of updating a weight matrix W directly:
```
Before LoRA:
  W is 4096 × 4096 = 16,777,216 parameters
  During training: W ← W + ΔW    (ΔW has 16M parameters to learn)

With LoRA:
  Freeze W entirely
  Add two small matrices:
    A: 4096 × r    (r = rank, e.g. 16)
    B: r × 4096

  ΔW ≈ A × B      (only 2 × 4096 × 16 = 131,072 parameters to learn)

  Forward pass: output = (W + A × B) × input
                        = W × input + A × B × input
                          ↑ frozen    ↑ trainable
```

Visually:
```
        4096 columns
       ┌─────────────┐
4096   │             │
rows   │      W      │   16,777,216 params (FROZEN)
       │             │
       └─────────────┘

is approximated by:

       ┌──┐   4096       ┌──┐
4096   │  │  ──────►      │  │ 65,536 params (A)
rows   │A │    r=16       │  │
       │  │               └──┘
       └──┘                ×
                          ┌─────────────┐
                     r=16 │      B      │ 65,536 params (B)
                          └─────────────┘
                            4096 columns

Total trainable: 131,072  (0.8% of original)
```

### Where LoRA Is Applied

Our config applies LoRA to 7 projection matrices in every transformer layer:

```python
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",   # attention
                  "up_proj", "down_proj", "gate_proj"]       # MLP (feed-forward)
```

For a 7B model with 32 layers:
```
7 matrices × 32 layers × ~131K params each ≈ 29M trainable parameters
```

Compare the memory:
```
                    Full Fine-Tune    LoRA (r=16)
Trainable params:   7,000M            29M
Optimizer memory:   ~56 GB            ~240 MB
Checkpoint size:    ~14 GB            ~120 MB
```

### LoRA Hyperparameters

- **r (rank)**: Size of the A and B matrices. Higher = more expressive but more memory. Common values: 8, 16, 32, 64.
- **lora_alpha**: A scaling factor. The LoRA update is scaled by `alpha/r`. Higher alpha = larger updates. Our config: `alpha=64, r=16`, so scale = 4.
- **lora_dropout**: Dropout applied to LoRA layers during training. Ours: 0.05.
- **target_modules**: Which weight matrices get LoRA adapters. We apply to all attention + MLP projections.

### What PEFT Is

**PEFT** (Parameter-Efficient Fine-Tuning) is a HuggingFace library that implements LoRA and other techniques:

| Technique | How it works | Trainable params |
|-----------|-------------|-----------------|
| **LoRA** | Adds low-rank A×B matrices alongside frozen weights | ~0.5% |
| **QLoRA** | LoRA but the frozen weights are quantized to 4-bit | ~0.5% (+ less memory for frozen weights) |
| **Prefix Tuning** | Adds trainable "virtual tokens" to the input | Very few |
| **Adapter** | Inserts small trainable layers between existing layers | ~1-5% |
| **IA3** | Learns scaling vectors instead of matrices | ~0.01% |

We use **LoRA** through PEFT. The code looks like:
```python
from peft import LoraConfig

peft_config = LoraConfig(
    r=16,                    # rank
    lora_alpha=64,           # scaling factor
    target_modules=[...],    # which layers get LoRA
    task_type="CAUSAL_LM",   # language model task
    lora_dropout=0.05,
)

# TRL's trainer applies it automatically:
trainer = OfflineGRPOTrainer(
    model=model,
    peft_config=peft_config,   # ← PEFT wraps the model
    ...
)
```

After training, only the small LoRA matrices are saved (~120 MB). To use the model:
1. Load the original base model
2. Load the LoRA adapter on top
3. Optionally **merge** them into a single model: `W_new = W + A × B`

### How LoRA Connects to the GRPO Loss

LoRA does NOT change the algorithm or math. The GRPO loss is:
```
L = -advantage × min(ratio, clip(ratio, 1-ε, 1+ε)) + β × KL
```

The only things that change:

| | Without LoRA | With LoRA |
|---|---|---|
| `π_student(token)` | Full model, all weights updated | Base model + LoRA adapters (only adapters updated) |
| `π_base(token)` for KL | A separate frozen copy of the model (2× memory!) | Same model with `disable_adapter()` (free!) |
| What gets updated | All 7B parameters | Only ~29M LoRA parameters |
| Checkpoint | Entire model (~14 GB) | Just the LoRA matrices (~120 MB) |

The "free reference model" is a major benefit: `disable_adapter()` turns off the LoRA matrices, giving us the original base model's predictions without loading a second copy. This saves ~14 GB of memory.

---

## 10. Distributed Training (Multi-GPU)

### Why Multi-GPU?

When a model is too large to fit on one GPU, you need to split the work across multiple GPUs. Even when it fits, multiple GPUs can speed up training.

### DDP (Distributed Data Parallel)

The simplest multi-GPU strategy. **Every GPU has a complete copy of the model.**

```
┌─────────────┐    ┌─────────────┐
│    GPU 0     │    │    GPU 1     │
│             │    │             │
│ Full Model  │    │ Full Model  │
│ (copy)      │    │ (copy)      │
│             │    │             │
│ Batch A     │    │ Batch B     │
│ → gradient A│    │ → gradient B│
└──────┬──────┘    └──────┬──────┘
       │                  │
       └───── average ────┘
              gradients
              │
       ┌──────┴──────┐
       │ Same weight  │
       │ update on    │
       │ both GPUs    │
       └─────────────┘
```

How it works:
1. Each GPU gets the full model
2. The training data is split — GPU 0 gets batch A, GPU 1 gets batch B
3. Each GPU computes gradients independently
4. Gradients are **averaged** across GPUs (all-reduce)
5. Each GPU applies the same update, so models stay in sync

**Pros**: Simple, well-supported, LoRA works perfectly
**Cons**: Each GPU must hold the entire model. If the model doesn't fit on 1 GPU, DDP can't help.

### DeepSpeed ZeRO (Zero Redundancy Optimizer)

The key insight: in DDP, every GPU stores the full model, full gradients, AND full optimizer states. That's redundant. ZeRO eliminates redundancy in stages:

```
What each GPU stores:

                    DDP        ZeRO-1     ZeRO-2     ZeRO-3
Model weights:      Full       Full       Full       Shard (1/N)
Gradients:          Full       Full       Shard      Shard (1/N)
Optimizer states:   Full       Shard      Shard      Shard (1/N)

Memory per GPU:     ~4× model  ~2× model  ~1.5× model  ~1/N of total
```

**ZeRO-1**: Each GPU stores only 1/N of the optimizer states (momentum, variance). Still needs full model on each GPU.

**ZeRO-2**: Also shards gradients. Each GPU only computes and stores gradients for its portion of parameters.

**ZeRO-3**: Also shards the model weights themselves. No single GPU has the full model — weights are gathered on-the-fly when needed for forward/backward pass.

Example with a 7B model on 4 GPUs:
```
                    DDP          ZeRO-2         ZeRO-3
Per-GPU memory:
  Model weights:    14 GB        14 GB          3.5 GB
  Gradients:        14 GB         3.5 GB        3.5 GB
  Optimizer:        28 GB         7 GB          7 GB
  ─────────────     ─────        ─────          ─────
  Total:            56 GB        24.5 GB        14 GB
```

### FSDP (Fully Sharded Data Parallel)

PyTorch's built-in version of ZeRO-3. Same concept — shards weights, gradients, and optimizer states across GPUs. The difference is implementation:
- ZeRO is from Microsoft (DeepSpeed library)
- FSDP is from Meta (built into PyTorch)

Both have the same `disable_adapter()` issue with LoRA because both shard model weights.

### Tensor Parallelism

Splits individual weight matrices across GPUs:
```
GPU 0 gets: left half of W_q    (4096 × 2048)
GPU 1 gets: right half of W_q   (4096 × 2048)
```

Both GPUs process the SAME input, but each computes only part of the output. Results are combined with communication.

**Pros**: No redundancy, works with LoRA (handled at the model level, not PEFT level)
**Cons**: Requires fast GPU-to-GPU interconnect (NVLink), more complex setup

### Pipeline Parallelism

Puts different layers on different GPUs:
```
GPU 0: Layers 0-15   (first half of model)
GPU 1: Layers 16-31  (second half of model)

Input → GPU 0 → GPU 1 → Output
```

**Pros**: Simple concept, works with LoRA
**Cons**: GPUs sit idle waiting for each other ("pipeline bubble"), needs careful scheduling

---

## 11. LoRA + Multi-GPU: The Compatibility Issue

### The Specific Problem

Our `trainer.py` computes the KL penalty like this:
```python
# Turn off LoRA to get base model predictions
with self.accelerator.unwrap_model(self.model).disable_adapter():
    ref_logprobs = self._get_per_token_logps_and_entropies(self.model, ...)
```

This works perfectly on 1 GPU: the full model is right there, you just toggle LoRA off and on.

On ZeRO-3 / FSDP, the model weights are **sharded** — split across GPUs. When you call `disable_adapter()`, the system tries to switch from `W + A×B` to just `W`. But `W` is not fully present on any GPU — it's split into pieces. This can crash or produce wrong results.

```
1 GPU:
  disable_adapter() → toggle LoRA off on complete W → ✓

ZeRO-3 with 4 GPUs:
  GPU 0 has W[0:25%], GPU 1 has W[25:50%], ...
  disable_adapter() → tries to toggle on partial W → ✗ can break
```

### Which Strategies Work with LoRA?

| Strategy | Splits weights? | disable_adapter() works? | LoRA compatible? |
|----------|----------------|-------------------------|-----------------|
| DDP | No | Yes | Yes |
| ZeRO-1 | No | Yes | Yes |
| ZeRO-2 | No | Yes | Yes |
| ZeRO-3 | Yes | Problematic | Needs workaround |
| FSDP | Yes | Problematic | Needs workaround |
| Tensor Parallel | Yes (but at model level) | N/A (different mechanism) | Yes |
| Pipeline Parallel | No (each GPU has full layers) | Yes | Yes |

### Workarounds for ZeRO-3 / FSDP + LoRA

**1. Set `beta=0` (skip KL penalty)**
If you don't need the KL penalty, `disable_adapter()` is never called. Simplest fix, but you lose regularization.

**2. Use a separate reference model**
Instead of toggling adapters, load a second copy of the base model as the reference. Avoids `disable_adapter()` entirely. Costs more memory, but the reference can be sharded too.

**3. ZeRO-2 + gradient checkpointing**
ZeRO-2 doesn't shard weights, so LoRA works perfectly. To fit larger models, enable gradient checkpointing — it recomputes activations during the backward pass instead of storing them, trading compute time for memory.
```
Without checkpointing: stores all intermediate activations (~5-15 GB)
With checkpointing:    recomputes them on the fly (~1 GB, but ~30% slower)
```

**4. Test recent PEFT/TRL versions first**
We're on PEFT 0.17.1 and TRL 0.21.0. Recent versions may have fixed the `disable_adapter()` issue with ZeRO-3. Always test before assuming it's broken.

### Recommended Path for Scaling Up

```
Model Size       Strategy                    Why
─────────────    ─────────────────────────    ────────────────────────
≤ 7B             1 GPU + LoRA                Fits easily (14GB model + 240MB optimizer)
7B-14B           ZeRO-2 + LoRA              Optimizer sharding frees memory
14B-70B          ZeRO-2 + LoRA + grad ckpt  Grad checkpointing saves activation memory
70B+             ZeRO-3 + workaround        Must shard weights; use beta=0 or separate ref
                 OR Tensor Parallel + LoRA   If fast interconnect available
```

---

## 12. Why `train.py` Needs `.to("cuda")` Removed for Multi-GPU

### The Problem

Our `train.py` line 82 does:
```python
model = AutoModelForCausalLM.from_pretrained(...).to("cuda")
```

`.to("cuda")` means: "put the entire model on GPU 0 right now." This works on 1 GPU but breaks multi-GPU.

### What Happens with `accelerate launch`

When you run `accelerate launch --num_processes 2 train.py`, two separate Python processes start — one for GPU 0, one for GPU 1. **Both processes execute the same `train.py` code.** Both hit `.to("cuda")`.

**With DDP** (copies model to each GPU):
```
Process 0: loads model to CPU → .to("cuda") → full model on GPU 0 → accelerate keeps it on GPU 0
Process 1: loads model to CPU → .to("cuda") → full model on GPU 0 → accelerate moves to GPU 1
                                                        ↑
                                          waste: briefly 2 full copies on GPU 0
```
This wastes memory but might not crash.

**With FSDP / ZeRO-3** (splits model across GPUs):
```
Process 0: loads model to CPU → .to("cuda") → FULL model on GPU 0
Process 1: loads model to CPU → .to("cuda") → FULL model on GPU 0
                                                     ↑
                                    crash: GPU 0 can't hold 2 full copies

Then FSDP tries to shard... but the model is already on the wrong device
```

### The Fix

Remove `.to("cuda")`. The model stays on CPU after loading:
```python
model = AutoModelForCausalLM.from_pretrained(
    args.target_model,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=None,
)
# No .to("cuda") — let the Trainer handle it
```

Then the Trainer calls `accelerator.prepare(model)` internally, which places weights correctly:
- **DDP**: copies full model to each GPU's own memory
- **FSDP**: shards model, each GPU gets only its portion
- **ZeRO-3**: same as FSDP
- **Single GPU**: moves to `cuda:0` (same behavior as before)

This change is safe for single-GPU too — the Trainer always handles device placement in `prepare()`.

---

## 13. Multi-GPU Test Results (2026-03-05)

### What We Tested

We tested 3 distributed training strategies × 2 beta values = 6 combinations on 2× L40s GPUs with the 0.5B student model. The goal: verify which strategies work with LoRA + the `disable_adapter()` mechanism for the KL penalty.

**Test setup**:
- Model: Qwen2.5-0.5B-Instruct with LoRA (r=16)
- Data: rollouts_test.jsonl (41 problems, 164 completions)
- GPUs: 2× L40s (48 GB each)
- Training: 1 epoch, batch_size=2, grad_accum=4
- Config files: `configs/accelerate_{ddp,zero2,fsdp}_2gpu.yaml`
- Scripts: `run_multigpu_test.sh`, `submit_multigpu_test.sh`

### Results

| Strategy | beta=0.0 (no KL) | beta=0.1 (KL + disable_adapter) |
|----------|:-:|:-:|
| **DDP** | PASS (66s) | PASS (51s) |
| **ZeRO-2** | PASS (65s) | PASS (51s) |
| **FSDP** | PASS (127s) | PASS (134s) |

**All 6 tests passed.** Every strategy works with LoRA + KL penalty.

### What We Found

**DDP**: Worked as expected. Each GPU holds a full model copy, no sharding. Fastest at ~51-66s.

**ZeRO-2**: Initially failed due to a config bug — `gradient_accumulation_steps: auto` in the accelerate YAML caused `ValueError: invalid literal for int() with base 10: 'auto'`. Accelerate 1.10.1 doesn't support `auto` here. Fixed by setting it to `4` (matching the training argument). After the fix, both tests passed.

**FSDP**: Both tests passed, **including beta=0.1** — this was a surprise. We predicted `disable_adapter()` would break when weights are sharded across GPUs, but PEFT 0.17.1 + TRL 0.21.0 + PyTorch 2.8.0 handle it correctly with `fsdp_use_orig_params: true`. Slower than DDP/ZeRO-2 (~127-134s vs ~51-66s) due to weight gather/scatter overhead.

**Important caveat**: The 0.5B model is very small. The FSDP + `disable_adapter()` compatibility should be re-verified with a 7B+ model, as larger models may trigger edge cases that don't appear at 0.5B.

### Changes Made for Multi-GPU Support

1. **`train.py`**: Removed `.to("cuda")` on line 82 so `accelerate` can control device placement (see Section 12)
2. **Created `configs/accelerate_ddp_2gpu.yaml`**: DDP config (`distributed_type: MULTI_GPU`)
3. **Created `configs/accelerate_zero2_2gpu.yaml`**: DeepSpeed ZeRO-2 config (`zero_stage: 2`, `gradient_accumulation_steps: 4`)
4. **Created `configs/accelerate_fsdp_2gpu.yaml`**: FSDP config (`FULL_SHARD`, `fsdp_use_orig_params: true`)
5. **Created `run_multigpu_test.sh`**: Test runner with subcommands `{ddp|zero2|fsdp|all}`
6. **Created `submit_multigpu_test.sh`**: SLURM batch wrapper (2× L40s, 32 CPUs, 96G RAM)

### Why `grad_norm = 0.0` on Many Steps

In the FSDP beta=0.0 logs (and other runs), you'll see `grad_norm: 0.0` on many steps. This is **not a multi-GPU bug** — it's the same data issue from diagnose.py:

1. GRPO computes advantages *relative to the group*: `advantage = (reward - group_mean) / group_std`
2. If all 4 completions for a problem have the same reward (all correct or all wrong), then `group_std = 0` → advantage = 0
3. With advantage = 0, the loss is 0, so the gradient is 0
4. 90.2% of our test groups have all-same rewards → ~90% of steps have zero gradient

This is a property of the GRPO algorithm combined with our data (7B teacher too accurate on easy MATH problems with only 4 generations). Fixes: more generations per problem (16+) or harder problems.

### Revised Recommendations for Scaling

All three strategies are viable. Updated recommendation table:

```
Model Size       Strategy                    Why
─────────────    ─────────────────────────    ────────────────────────
≤ 7B             1 GPU + LoRA                Fits easily
7B-14B           ZeRO-2 + LoRA              Optimizer sharding frees memory, proven compatible
14B-70B          ZeRO-2 + LoRA + grad ckpt  Grad checkpointing saves activation memory
                 OR FSDP + LoRA             Now confirmed to work (test at target scale first)
70B+             FSDP + LoRA                Must shard weights; verify disable_adapter() at scale
                 OR Tensor Parallel + LoRA   If fast interconnect (NVLink) available
```

Detailed results in: `test_plan_1.md`, logs in `logs/multigpu_test_20260305_*/`

---

## 14. What We've Done So Far

1. Set up the environment on Vulcan (modules + virtualenv)
2. Ran the test pipeline end-to-end on 1 GPU (41 problems, rollouts + training succeeded)
3. Ran diagnostics and found that 90% of groups had all-same rewards → almost no learning signal
   - Root cause: 7B teacher is too accurate on easy problems with only 4 attempts
   - Fix: more attempts per problem (16+) or harder problems
4. Prepared `run_full.sh` for full-scale run (12,000 problems)
5. Fixed `train.py` for multi-GPU compatibility (removed `.to("cuda")`)
6. Ran multi-GPU test matrix: **6/6 passed** — DDP, ZeRO-2, and FSDP all work with LoRA + KL penalty
7. **Not yet done**: Submit the full-scale run, evaluate before/after
8. **Not yet done**: Test multi-GPU with 7B+ student model

---

## 15. Known Issues

1. **Weak signal with 4 generations**: The 7B teacher gets 73% right, so most groups of 4 are all-correct → advantage=0 → no gradient. Consider increasing `--num_generations` to 16.

2. **Existing rollouts in scratch lack `completion_ids`**: The older rollout files at `/home/shuai14/scratch/rollouts_*.jsonl` don't have token IDs, so they can't be used with our pipeline. We need to re-generate with `generate_rollouts.py`.

3. **Scratch purge**: Files on `$SCRATCH` are deleted after 60 days of no access. Copy important results to `$PROJECT`.

4. **FSDP + LoRA at large scale**: Passed at 0.5B, but `disable_adapter()` should be re-verified with 7B+ models where sharding is more aggressive.

---

*Last updated: 2026-03-05*
