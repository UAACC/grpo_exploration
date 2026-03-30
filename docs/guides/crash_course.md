# Offline GRPO: Build the Entire Pipeline from Scratch

A crash course that teaches you to implement offline GRPO (Group Relative Policy Optimization) end-to-end. After this course, you'll be able to train a small student model using a large teacher model's pre-generated rollouts.

---

## Part 0: What Are We Building and Why?

### The Goal

Train a small model (0.5B) to be better at math by learning from a large model's (7B) solutions.

### Why "Offline"?

Standard GRPO (online) generates completions during training — the student writes answers, checks them, and learns from its own mistakes. This is expensive because generation is slow.

Offline GRPO separates the two phases:
1. **Phase 1 (Offline)**: A strong teacher generates solutions ahead of time
2. **Phase 2 (Training)**: The student learns from these pre-generated solutions

This is ~2x faster because generation only happens once.

### The Problem: Distribution Mismatch

The teacher wrote these solutions, not the student. When we use the teacher's solutions to train the student, we need to account for the fact that **the student would have written different things**. This is called the **off-policy problem**, and we fix it with **importance sampling**.

### The Math (Simplified)

Standard GRPO loss (on-policy):
```
L = -E[min(ratio × advantage, clip(ratio) × advantage)]

where: ratio = π_current(token) / π_old(token)
       π_old = the policy that generated the completion (= π_current from last step)
```

Offline GRPO loss (off-policy):
```
Same formula, but:
       π_old = π_teacher    (the teacher model's probability for each token)
       ratio = π_student(token) / π_teacher(token)    ← importance sampling ratio
```

The `ratio` corrects for the fact that student and teacher would assign different probabilities to the same token. If the student thinks a token is unlikely but the teacher thought it was likely, the ratio is small, and that token contributes less to the gradient. This is mathematically principled — it's the same importance sampling trick used in PPO.

### Pipeline Overview

```
┌─────────────────┐     ┌──────────────┐     ┌───────────────┐     ┌──────────┐
│ generate_rollouts│ ──→ │  data.py     │ ──→ │  trainer.py   │ ──→ │evaluate.py│
│ (vLLM + teacher) │     │  (process)   │     │  (train)      │     │ (test)   │
└─────────────────┘     └──────────────┘     └───────────────┘     └──────────┘
     JSONL file         rewards, advantages    LoRA checkpoint       accuracy
```

---

## Part 1: Configuration (`configs.py`)

Start by defining shared constants. This prevents magic strings scattered across files.

```python
# configs.py
import re

# System prompts tell the model how to format its answer
GSM8K_SYSTEM_PROMPT = (
    "Please solve this math problem step by step. "
    "Put your final numerical answer after ####."
)

# Model paths
DEFAULT_TARGET_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"    # student
DEFAULT_BEHAVIOR_MODEL = "Qwen/Qwen2.5-7B-Instruct"    # teacher

# LoRA configuration
DEFAULT_LORA_CONFIG = dict(
    r=16,                    # LoRA rank — controls parameter count
    lora_alpha=64,           # scaling factor (effective lr multiplier = alpha/r = 4)
    target_modules=[         # which layers get LoRA adapters
        "q_proj", "k_proj", "v_proj", "o_proj",   # attention
        "up_proj", "down_proj", "gate_proj",       # MLP
    ],
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
)
```

### Answer Extraction

You need to pull the final answer out of the model's response. This is dataset-specific:

```python
def extract_gsm8k_answer(text: str) -> str | None:
    """Extract numerical answer from GSM8K output.

    GSM8K convention: answer follows '####'
    Example: "... so the total is 42. #### 42"  →  "42"
    """
    # Try #### pattern first (most reliable)
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")

    # Fallback: \boxed{} pattern
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        num_match = re.search(r"[-]?[\d,]+(?:\.\d+)?", boxed[-1])
        if num_match:
            return num_match.group(0).replace(",", "")

    # Last resort: last number in text
    numbers = re.findall(r"[-]?[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None
```

**Why multiple fallbacks?** Models don't always follow instructions perfectly. The #### pattern is what GSM8K uses, but models sometimes use \boxed{} instead (MATH convention), or just write the number without any special formatting.

---

## Part 2: Generate Teacher Rollouts (`generate_rollouts.py`)

This is the most expensive step. The teacher model generates multiple solutions per problem, and we record **every token's probability**.

### Why Record Logprobs?

The importance sampling ratio needs π_teacher(token). We can't re-compute this later because:
1. Running the 7B teacher again is expensive
2. We want the exact logprobs from the generation process, not a forward pass (which would give slightly different values due to KV cache behavior)

### Implementation

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset

def generate_rollouts(model_name, output_path, num_generations=4):
    # 1. Load the teacher model with vLLM (fast batch generation)
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,      # GPUs for this model
        dtype="auto",                # bfloat16 auto-detected
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.8,  # leave some GPU memory free
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # 2. Load dataset and format prompts
    data = load_dataset("openai/gsm8k", "main", split="train")

    eval_data = []
    for i, item in enumerate(data):
        # Apply chat template: system prompt + user question → formatted string
        chat = [
            {"role": "system", "content": GSM8K_SYSTEM_PROMPT},
            {"role": "user", "content": item["question"]},
        ]
        formatted = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        eval_data.append({
            "question_id": i,
            "problem": item["question"],
            "prompt": formatted,
            "answer": item["answer"],
        })
```

**What does `apply_chat_template` do?** Converts the chat format to the model's expected input. For Qwen2.5 it looks like:
```
<|im_start|>system
Please solve this math problem step by step. Put your final numerical answer after ####.<|im_end|>
<|im_start|>user
If John has 5 apples and buys 3 more, how many does he have?<|im_end|>
<|im_start|>assistant
```

The `add_generation_prompt=True` adds the final `assistant` header so the model knows to start generating.

```python
    # 3. Configure sampling
    sampling_params = SamplingParams(
        temperature=0.7,       # randomness (higher = more diverse)
        top_p=1.0,             # no nucleus sampling cutoff
        max_tokens=1024,       # max completion length
        seed=42,               # reproducibility
        n=num_generations,     # generate N completions per prompt
        logprobs=1,            # KEY: record top-1 logprob per token
    )
```

**Why `n=num_generations`?** We want multiple solutions per problem. With `temperature=0.7`, each solution will be different because sampling is random. This gives us a **group** of solutions to compare against each other (the "Group" in GRPO).

**Why `logprobs=1`?** This tells vLLM to return the log-probability of the chosen token at each step. This is π_teacher(token) — the probability the teacher assigned to each token it generated. We need this for importance sampling later.

```python
    # 4. Generate!
    prompts = [item["prompt"] for item in eval_data]
    outputs = llm.generate(prompts, sampling_params)

    # 5. Save results
    with open(output_path, "w") as f:
        for item, request_output in zip(eval_data, outputs):
            record = {
                "question_id": item["question_id"],
                "original_problem": item["problem"],
                "ground_truth_answer": item["answer"],
                "system_prompt": GSM8K_SYSTEM_PROMPT,
                "dataset_type": "gsm8k",
                "runs": [],
            }

            for run_id, seq in enumerate(request_output.outputs[:num_generations]):
                token_ids = list(seq.token_ids)
                step_logprobs = seq.logprobs  # list of dicts
```

**Understanding vLLM's logprob format:**

`seq.logprobs` is a list (one entry per generated token). Each entry is a dict mapping token_id → Logprob object. We extract the logprob of the token that was actually chosen:

```python
                # Flatten logprobs: for each token, get its logprob
                logprob_list = []
                for step, tid in enumerate(token_ids):
                    lp_dict = step_logprobs[step]    # {token_id: Logprob, ...}
                    lp_obj = lp_dict.get(tid, None)  # get the chosen token's logprob
                    logprob_list.append(
                        float(lp_obj.logprob) if lp_obj is not None else None
                    )
```

Example: if the teacher generated token 3421 at step 0, and `logprobs[0] = {3421: Logprob(-0.5), 892: Logprob(-2.1)}`, then `logprob_list[0] = -0.5`.

```python
                # Strip EOS token — we don't want the model to learn
                # to predict EOS from teacher data
                if token_ids and token_ids[-1] == tokenizer.eos_token_id:
                    token_ids = token_ids[:-1]
                    logprob_list = logprob_list[:-1]

                record["runs"].append({
                    "run_id": run_id,
                    "response": seq.text,                       # decoded text
                    "extracted_answer": extract_gsm8k_answer(seq.text),
                    "logprobs": logprob_list,                   # π_teacher per token
                    "completion_ids": token_ids,                 # token IDs
                })

            f.write(json.dumps(record) + "\n")
```

### Output Format

Each line of the JSONL file looks like:
```json
{
  "question_id": 0,
  "original_problem": "If John has 5 apples...",
  "ground_truth_answer": "John originally had...\n#### 8",
  "runs": [
    {
      "run_id": 0,
      "response": "Step 1: John has 5 apples...\n#### 8",
      "extracted_answer": "8",
      "logprobs": [-0.5, -1.2, -0.3, -0.8, ...],
      "completion_ids": [3421, 892, 15200, 7788, ...]
    },
    {
      "run_id": 1,
      "response": "Let me solve this...\n#### 8",
      "extracted_answer": "8",
      "logprobs": [-0.7, -0.9, -0.4, ...],
      "completion_ids": [2911, 455, 8823, ...]
    },
    // ... more runs
  ]
}
```

**Each run is a different solution to the same problem.** The teacher generates N diverse solutions (due to temperature sampling). Some are correct, some are wrong. GRPO will learn to prefer the correct ones.

---

## Part 3: Data Processing (`data.py`)

### Step 3a: Load Rollouts

```python
def load_rollouts(jsonl_path: str, vocab_size: int | None = None) -> list[dict]:
    """Load JSONL → flat list of per-completion records."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            item = json.loads(line)
            for run in item["runs"]:
                cids = run["completion_ids"]
                lps = run["logprobs"]

                # Handle vocab mismatch between teacher and student
                if vocab_size is not None:
                    for idx, tid in enumerate(cids):
                        if tid >= vocab_size:
                            cids = cids[:idx]
                            lps = lps[:idx]
                            break

                records.append({
                    "question_id": item["question_id"],
                    "run_id": run["run_id"],
                    "problem": item["original_problem"],
                    "ground_truth": item["ground_truth_answer"],
                    "response": run["response"],
                    "extracted_answer": run["extracted_answer"],
                    "behavior_logprobs": lps,    # renamed for clarity
                    "completion_ids": cids,
                })
    return records
```

**Vocab mismatch truncation:** The teacher (7B) has 152064 embedding rows, the student (0.5B) has 151936. Both use the same tokenizer (151643 real tokens), but the models pad their embedding matrices to different multiples of 128 for GPU efficiency. If the teacher happens to generate a token ID in the 151936-152063 range (rare, but possible), the student can't process it — so we truncate the completion at that point.

### Step 3b: Compute Rewards and Advantages

This is the core of GRPO's "Group Relative" approach.

```python
def compute_rewards_and_advantages(records, eps=1e-4):
    """Assign reward per completion, then normalize within groups."""

    # Step 1: Binary reward — did it get the right answer?
    for rec in records:
        extracted = rec["extracted_answer"]
        gold = extract_gsm8k_answer(rec["ground_truth"])
        try:
            rec["reward"] = 1.0 if float(extracted) == float(gold) else 0.0
        except (ValueError, TypeError):
            rec["reward"] = 0.0
```

**Why binary reward?** For math, the answer is either right or wrong. No partial credit. (For MATH dataset, correct gets 2.0 to give stronger signal.)

```python
    # Step 2: Group by question and normalize
    groups = defaultdict(list)
    for rec in records:
        groups[rec["question_id"]].append(rec)

    for group in groups.values():
        rewards = [r["reward"] for r in group]
        mean_r = sum(rewards) / len(rewards)
        std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5

        for rec in group:
            rec["advantage"] = (rec["reward"] - mean_r) / (std_r + eps)

    return records
```

**What does "Group Relative" mean?** Instead of using absolute rewards, we compare each completion against other completions **for the same question**.

Example with 4 completions for question 0:

```
Run 0: correct (reward=1.0)
Run 1: wrong   (reward=0.0)
Run 2: correct (reward=1.0)
Run 3: wrong   (reward=0.0)

mean = 0.5, std = 0.5

advantage[0] = (1.0 - 0.5) / 0.5 = +1.0   ← "better than average, reinforce this"
advantage[1] = (0.0 - 0.5) / 0.5 = -1.0   ← "worse than average, discourage this"
advantage[2] = (1.0 - 0.5) / 0.5 = +1.0
advantage[3] = (0.0 - 0.5) / 0.5 = -1.0
```

**Why not just use reward directly?** Group normalization provides:
1. **Centered signal**: Mean advantage is ~0, so the model is pushed toward better solutions and away from worse ones equally.
2. **Scale invariance**: A question where the teacher gets 4/4 right has zero std → all advantages become 0 (nothing to learn). A question where it gets 2/4 right has maximum contrast.

**Edge case**: If all 4 runs are correct (or all wrong), `std = 0`, and advantage = 0 for all. This is intentional — there's no relative signal to learn from when all outputs have the same reward.

### Step 3c: Build Training Dataset

```python
def build_training_dataset(records):
    """Convert to HuggingFace Dataset format."""
    # CRITICAL: sort by (question_id, run_id)
    records = sorted(records, key=lambda r: (r["question_id"], r["run_id"]))

    prompts, answers, qids = [], [], []
    for rec in records:
        prompts.append([
            {"role": "system", "content": rec["system_prompt"]},
            {"role": "user", "content": rec["problem"]},
        ])
        answers.append(rec["ground_truth"])
        qids.append(rec["question_id"])

    return Dataset.from_dict({
        "prompt": prompts,
        "answer": answers,
        "question_id": qids,
    })
```

**Why sort by (question_id, run_id)?** TRL's dataloader uses `RepeatSampler` which expects consecutive `num_generations` rows to be the same question. After sorting:

```
Row 0: question_0, run_0    ← group of 4
Row 1: question_0, run_1
Row 2: question_0, run_2
Row 3: question_0, run_3
Row 4: question_1, run_0    ← next group
Row 5: question_1, run_1
...
```

**Wait, but TRL's RepeatSampler duplicates prompts — isn't this redundant?**

TRL's RepeatSampler takes each unique prompt and repeats it `num_generations` times. Since our dataset ALREADY has `num_generations` copies per question (one per run), the sampler sees each question_id `num_generations` times and says "these are my repeats." The sorted order makes this alignment work.

### Step 3d: Build Lookup Table

```python
def build_offline_lookup(records):
    """Dict for O(1) access during training."""
    lookup = {}
    for rec in records:
        lookup[(rec["question_id"], rec["run_id"])] = {
            "behavior_logprobs": rec["behavior_logprobs"],
            "completion_ids": rec["completion_ids"],
            "advantage": rec["advantage"],
            "reward": rec["reward"],
            "response": rec["response"],
        }
    return lookup
```

During training, the trainer gets a batch of prompts. Each prompt has a `question_id` and (implicitly via batch position) a `run_id`. The lookup table lets us instantly retrieve the pre-computed teacher completion, logprobs, and advantage.

---

## Part 4: The Trainer (`trainer.py`)

This is the most important file. We extend TRL's `GRPOTrainer` by overriding **one method**: `_generate_and_score_completions`.

### What Does Standard GRPOTrainer Do?

```
Standard GRPOTrainer loop:
  for each batch of prompts:
    1. _generate_and_score_completions():
       - model.generate() → student creates completions (SLOW)
       - reward_func() → compute rewards
       - normalize advantages
       - compute old_per_token_logps (student's current logprobs)
       - return output dict

    2. _compute_loss():
       - forward pass → get current logprobs
       - ratio = exp(current_logps - old_per_token_logps)
       - clipped surrogate loss
       - KL penalty
       - return loss

    3. loss.backward() → update parameters
```

### What Does Our Override Do?

We replace step 1 — instead of generating, we look up pre-computed teacher data:

```python
class OfflineGRPOTrainer(GRPOTrainer):

    def __init__(self, *args, offline_data: dict, ref_sync_steps: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self._offline_data = offline_data      # the lookup table
        self._ref_sync_steps = ref_sync_steps  # reference model sync frequency
```

```python
    def _generate_and_score_completions(self, inputs):
        device = self.accelerator.device

        # ---- 1. Tokenize prompts ----
        # Same as standard GRPOTrainer
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",    # left-pad for generation compatibility
            add_special_tokens=False,
        )
        prompt_ids = prompt_inputs["input_ids"]
        prompt_mask = prompt_inputs["attention_mask"]
```

**Why `padding_side="left"`?** Language models generate left-to-right. If we pad on the right, the padding tokens would be in the middle of the sequence when we concatenate prompt + completion. Left-padding keeps the actual content right-aligned and contiguous.

```python
        # ---- 2. Look up offline completions (NO GENERATION!) ----
        batch_size = len(inputs)
        num_gen = self.num_generations

        # TRL sends duplicated prompts: [q0, q0, q0, q0, q1, q1, q1, q1, ...]
        # We assign run_ids by position within each group
        question_ids = [x.get("question_id") for x in inputs]
        run_ids = [i % num_gen for i in range(batch_size)]

        completion_id_lists = []
        behavior_logprob_lists = []
        advantages_list = []

        for qid, rid in zip(question_ids, run_ids):
            rec = self._offline_data[(qid, rid)]    # O(1) lookup!
            completion_id_lists.append(rec["completion_ids"])
            behavior_logprob_lists.append(rec["behavior_logprobs"])
            advantages_list.append(rec["advantage"])
```

**How do we know which run_id each row corresponds to?** TRL's RepeatSampler sends `num_generations` copies of each prompt. Within a group, `i % num_gen` gives us 0, 1, 2, 3 — exactly matching our run_ids.

```python
        # ---- 3. Pad completions to equal length ----
        # Different completions have different lengths. We need tensors,
        # so we pad shorter ones with pad_token_id.
        max_comp_len = max(len(c) for c in completion_id_lists)
        if self.max_completion_length is not None:
            max_comp_len = min(max_comp_len, self.max_completion_length)

        completion_ids_tensors = []
        completion_mask_tensors = []
        old_logps_tensors = []

        for cids, blps in zip(completion_id_lists, behavior_logprob_lists):
            cids = cids[:max_comp_len]         # truncate if too long
            blps = blps[:max_comp_len]
            seq_len = len(cids)

            # Handle None logprobs (shouldn't happen, but defensive)
            blps = [lp if lp is not None else 0.0 for lp in blps]

            cid_t = torch.tensor(cids, dtype=torch.long, device=device)
            mask_t = torch.ones(seq_len, dtype=torch.int, device=device)
            lp_t = torch.tensor(blps, dtype=torch.float32, device=device)

            # Pad to max_comp_len
            pad_len = max_comp_len - seq_len
            if pad_len > 0:
                cid_t = torch.cat([cid_t, torch.full((pad_len,), self.pad_token_id, ...)])
                mask_t = torch.cat([mask_t, torch.zeros(pad_len, ...)])   # 0 = ignore
                lp_t = torch.cat([lp_t, torch.zeros(pad_len, ...)])      # 0 = ignore

            completion_ids_tensors.append(cid_t)
            completion_mask_tensors.append(mask_t)
            old_logps_tensors.append(lp_t)

        completion_ids = torch.stack(completion_ids_tensors)       # (B, C)
        completion_mask = torch.stack(completion_mask_tensors)     # (B, C)
        old_per_token_logps = torch.stack(old_logps_tensors)      # (B, C)
        advantages = torch.tensor(advantages_list, device=device) # (B,)
```

**The mask is crucial.** Padding tokens must not contribute to the loss. The mask is 1 for real tokens, 0 for padding. TRL's `_compute_loss` uses this mask automatically.

```python
        # ---- 4. Compute reference logprobs (for KL penalty) ----
        # KL penalty: don't let student drift too far from base model
        ref_per_token_logps = None
        if self.beta != 0.0:
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            logits_to_keep = completion_ids.size(1)  # only need logprobs for completion part

            with torch.no_grad():
                if self.ref_model is not None:
                    # Separate reference model (expensive, rarely used)
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep,
                    )
                else:
                    # LoRA trick: disable adapter → get base model logprobs FOR FREE
                    ref_per_token_logps = self._get_ref_logprobs(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep,
                    )
```

**LoRA reference model trick:** With LoRA, the base model weights are untouched — LoRA only adds small adapter weights on top. By calling `model.disable_adapter()`, we temporarily get the original base model's behavior without loading a second model. This saves ~50% GPU memory.

**What is `logits_to_keep`?** We only need logprobs for the completion tokens, not the prompt. This parameter tells the function to only compute logprobs for the last N tokens, saving compute.

```python
        # ---- 5. Build output dict ----
        # This dict is consumed by TRL's _compute_loss
        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,  # ← THE KEY LINE
        }
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        return output
```

**THE KEY LINE: `old_per_token_logps = teacher_logprobs`**

This is where the magic happens. In standard GRPO, `old_per_token_logps` is the **student's own logprobs** from the generation step. By setting it to the **teacher's logprobs**, we transform the importance sampling ratio:

```
Standard:  ratio = π_student_current / π_student_old     (on-policy)
Ours:      ratio = π_student_current / π_teacher          (off-policy IS correction)
```

TRL's `_compute_loss` doesn't know the difference — it just computes `exp(current - old)` and applies clipping. We get off-policy correction for free without modifying the loss function at all.

### What TRL Does With This Output (You Don't Write This)

TRL's `_compute_loss` (which we DON'T override) does:

```python
# Inside TRL's _compute_loss (simplified):

# 1. Forward pass: get student's CURRENT logprobs on teacher's completion
current_logps, entropy = self._get_per_token_logps_and_entropies(
    model, prompt_completion_ids, attention_mask, logits_to_keep
)

# 2. Per-token importance sampling ratio
#    In standard GRPO: π_current / π_old_student
#    In our offline GRPO: π_current_student / π_teacher
log_ratio = current_logps - old_per_token_logps
ratio = torch.exp(log_ratio)                    # per-token, not sequence-level!

# 3. Clipped surrogate objective (PPO-style)
clipped_ratio = torch.clamp(ratio, 1 - eps, 1 + eps)   # eps=0.2
loss1 = ratio * advantages.unsqueeze(1)
loss2 = clipped_ratio * advantages.unsqueeze(1)
surrogate_loss = -torch.min(loss1, loss2)       # per-token loss

# 4. KL penalty (optional)
kl = current_logps - ref_per_token_logps        # drift from base model
loss = surrogate_loss + beta * kl

# 5. Mask and average
loss = ((loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
```

### Reference Adapter Sync (Advanced)

An optional feature: periodically update what "reference model" means.

```python
def _sync_ref_adapter(self):
    """Save current LoRA weights as the new reference."""
    unwrapped = self.accelerator.unwrap_model(self.model)
    self._ref_adapter_state = {
        k: v.detach().clone()
        for k, v in unwrapped.named_parameters()
        if "lora_" in k
    }

def _get_ref_logprobs(self, model, prompt_completion_ids, attention_mask, logits_to_keep):
    if self._ref_sync_steps > 0 and self._ref_adapter_state is not None:
        # Temporarily swap in old LoRA weights
        unwrapped = self.accelerator.unwrap_model(model)
        current_state = {k: v.detach().clone() for k, v in unwrapped.named_parameters() if "lora_" in k}

        # Load reference weights
        for name, param in unwrapped.named_parameters():
            if name in self._ref_adapter_state:
                param.data.copy_(self._ref_adapter_state[name])

        # Compute ref logprobs with old weights
        ref_logps, _ = self._get_per_token_logps_and_entropies(...)

        # Restore current weights
        for name, param in unwrapped.named_parameters():
            if name in current_state:
                param.data.copy_(current_state[name])
        return ref_logps
    else:
        # Default: disable LoRA entirely → base model
        with unwrapped.disable_adapter():
            ref_logps, _ = self._get_per_token_logps_and_entropies(...)
        return ref_logps
```

**Why sync?**

- `ref_sync_steps=0` (default): KL penalty measures drift from the **original base model**. This can become very restrictive as training progresses.
- `ref_sync_steps=N`: Every N steps, snapshot the current LoRA weights. KL now measures drift from a **recent checkpoint**. This allows the model to evolve more freely over time while still preventing catastrophic updates within each N-step window.

---

## Part 5: Training Entry Point (`train.py`)

This script wires everything together:

```python
def main():
    args = parse_args()

    # 1. Load and process data
    model_config = AutoConfig.from_pretrained(args.target_model)
    records = load_rollouts(args.rollout_path, vocab_size=model_config.vocab_size)
    records = compute_rewards_and_advantages(records)
    dataset = build_training_dataset(records)
    offline_data = build_offline_lookup(records)

    # 2. Load student model
    model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.bfloat16,          # half precision to save memory
        attn_implementation="flash_attention_2",  # fast attention
        device_map=None,                     # let accelerate handle device placement
    )
    model.config.use_cache = False  # disable KV cache (incompatible with gradient computation)

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    tokenizer.pad_token = tokenizer.eos_token  # need a pad token for batching

    # 3. LoRA config
    peft_config = LoraConfig(
        r=16, lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )

    # 4. Training config
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=5e-6,
        beta=0.1,                              # KL penalty weight
        per_device_train_batch_size=2,         # per GPU
        gradient_accumulation_steps=8,         # effective batch = 2 × 4 GPUs × 8 = 64
        num_generations=4,                     # completions per prompt
        max_prompt_length=256,
        max_completion_length=786,
        num_train_epochs=1,
        max_grad_norm=0.1,                     # clip gradients (conservative)
        warmup_ratio=0.1,                      # 10% warmup
        lr_scheduler_type="cosine",
        bf16=True,
        save_steps=500,
        logging_steps=1,
        report_to="wandb",
    )
```

**Why `use_cache = False`?** KV cache speeds up generation by caching intermediate computations. But during training with backpropagation, we need all activations for gradient computation — caching interferes with this. Always disable for training.

**Why a dummy reward function?**

```python
    # TRL requires reward_funcs but we never call them (advantages are pre-computed)
    def _dummy_reward(prompts, completions, **kwargs):
        return [0.0] * len(completions)

    trainer = OfflineGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[_dummy_reward],      # never called, but TRL requires it
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        offline_data=offline_data,         # our pre-computed data
        ref_sync_steps=0,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
```

TRL's GRPOTrainer constructor requires `reward_funcs`. In standard GRPO, this is called to score generated completions. In our offline version, rewards are pre-computed in `data.py`, so this function is never called. We pass a dummy to satisfy the API.

---

## Part 6: Evaluation (`evaluate.py`)

After training, evaluate on the test set:

```python
def evaluate(model_path, base_model, merge_lora=True):
    # 1. If LoRA, merge adapter into base model
    if merge_lora:
        base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()     # merge LoRA weights into base
        model.save_pretrained(merged_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer.save_pretrained(merged_path)
```

**Why merge?** LoRA saves the adapter separately from the base model. For inference with vLLM (which doesn't natively support LoRA adapters), we merge the adapter weights into the base model: `W_merged = W_base + alpha/r × A × B`.

```python
    # 2. Load merged model with vLLM for fast inference
    llm = LLM(model=merged_path, gpu_memory_utilization=0.95, ...)

    # 3. Generate on test set (multiple runs for statistical significance)
    data = load_dataset("openai/gsm8k", "main", split="test")
    for run_idx in range(num_runs):
        sampling_params = SamplingParams(
            temperature=0.0,      # greedy for deterministic eval
            max_tokens=2048,
            seed=42 + run_idx,
        )
        outputs = llm.generate(prompts, sampling_params)

        # Check each answer
        correct = 0
        for i, output in enumerate(outputs):
            predicted = extract_gsm8k_answer(output.outputs[0].text)
            gold = extract_gsm8k_answer(data[i]["answer"])
            if float(predicted) == float(gold):
                correct += 1

        accuracy = correct / len(outputs)
        print(f"Run {run_idx}: {accuracy:.4f}")
```

**Why multiple runs?** Even with `temperature=0.0` (greedy), different seeds can give slightly different results due to floating point non-determinism in parallel computation. 5-10 runs with mean/std gives a reliable number.

---

## Part 7: Running It All (`run_gsm8k_offline.sh`)

SLURM batch script for cluster execution:

```bash
#!/bin/bash
#SBATCH --gpus-per-node=l40s:4          # 4 GPUs
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00

# Environment setup
module load python/3.11 cuda/12.6
source /path/to/venv/bin/activate
export TRANSFORMERS_OFFLINE=1            # no internet on compute nodes
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE="/path/to/cached/datasets"

case "${1}" in

rollouts)
    # Phase 1: Generate teacher rollouts (~2 hours)
    python generate_rollouts.py \
        --model_name /path/to/Qwen2.5-Math-7B-Instruct \
        --dataset_type gsm8k \
        --num_generations 4 \
        --temperature 0.7 \
        --max_tokens 1024
    ;;

train)
    # Phase 2: Train student with offline GRPO (~1 hour)
    accelerate launch \
        --config_file configs/accelerate_ddp_4gpu.yaml \
        train.py \
        --rollout_path /path/to/rollouts.jsonl \
        --target_model /path/to/Qwen2.5-0.5B-Instruct \
        --num_generations 4 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --learning_rate 5e-6 \
        --beta 0.1 \
        --num_train_epochs 1 \
        --report_to wandb
    ;;

eval)
    # Phase 3: Evaluate
    python evaluate.py \
        --model_path /path/to/checkpoint \
        --base_model /path/to/Qwen2.5-0.5B-Instruct \
        --merge_lora \
        --runs 10 \
        --temperature 0.0
    ;;
esac
```

---

## Part 8: Putting It All Together — The Full Data Flow

```
INPUT: GSM8K train set (7473 problems)

Step 1: GENERATE ROLLOUTS
  Teacher 7B processes 7473 problems × 4 generations = 29892 completions
  Each completion has: text, token_ids, per-token logprobs
  Output: rollouts_gsm8k.jsonl (~500MB)

Step 2: LOAD & PROCESS
  load_rollouts()
    → 29892 records
    → truncate any OOV tokens (rare)

  compute_rewards_and_advantages()
    → check each answer: correct=1.0, wrong=0.0
    → group by question, normalize: advantage = (reward - mean) / std
    → ~70% correct (teacher is strong), so most groups have mixed rewards

  build_training_dataset()
    → HF Dataset with 29892 rows, sorted by (qid, rid)
    → columns: prompt (chat format), answer, question_id

  build_offline_lookup()
    → dict[(qid, rid)] → {completion_ids, behavior_logprobs, advantage, reward}

Step 3: TRAIN (1 epoch)
  For each batch (e.g., 2 questions × 4 runs = 8 rows per GPU):

    _generate_and_score_completions():
      → Look up 8 teacher completions from offline_data
      → Get teacher logprobs (π_behavior)
      → Get pre-computed advantages
      → Compute ref logprobs (base model, via disable_adapter)
      → Return output dict (NO GENERATION, just lookup)

    TRL's _compute_loss():
      → Forward pass: compute π_student(each token of teacher's completion)
      → ratio = π_student / π_teacher    (per-token importance sampling)
      → clip ratio to [0.8, 1.2]
      → loss = -min(ratio × advantage, clipped_ratio × advantage)
      → add KL penalty: β × (log π_student - log π_base)
      → loss.backward()

    optimizer.step()

  Save LoRA checkpoint

Step 4: EVALUATE
  Merge LoRA into base → merged model
  vLLM generate on 1319 test problems
  Compare against ground truth
  Report accuracy (e.g., 48.79% vs 49.6% baseline)
```

---

## Part 9: Common Pitfalls

### 1. Batch Structure Mismatch
TRL's `RepeatSampler` duplicates each prompt `num_generations` times. Your `per_device_train_batch_size` MUST be divisible by `num_generations`. If `num_gen=4` and `batch_size=5`, you'll get silent misalignment.

### 2. Sequence-Level vs Per-Token Clipping
NEVER sum log ratios across tokens before exp. `exp(sum of 100 terms of -2.5) = exp(-250) = 0`. Always do per-token: `exp(-2.5) = 0.08` per token, then clip individually.

### 3. Vocab Size Mismatch
Teacher and student may have different embedding matrix sizes even with the same tokenizer. Always pass `vocab_size` when loading rollouts.

### 4. Side-Channel Data Bypasses TRL's Split
TRL splits the output dict for gradient accumulation. If you store data in `self._some_variable` instead of the output dict, it won't be split — you'll process all data every step instead of 1/N. This causes OOM.

### 5. KV Cache During Training
`model.config.use_cache = False` — forget this and you get cryptic errors during backward pass.

### 6. Offline Mode on Clusters
Set `TRANSFORMERS_OFFLINE=1` and `HF_DATASETS_OFFLINE=1`. Download everything on the login node first. The dataset cache path must be exact.

---

## Part 10: What to Monitor During Training

Watch these metrics on wandb:

| Metric | Healthy Range | Red Flag |
|--------|--------------|----------|
| `loss` | 0.01 ~ 0.1, decreasing | Negative = fine (GRPO loss can be negative). Spikes > 50 occasionally OK if they recover |
| `reward` | Should increase | Flat after many epochs = not learning |
| `kl` | 0.001 ~ 0.05, slowly increasing | > 0.1 = model drifting too far, increase β |
| `grad_norm` | 0.1 ~ 1.0 | > 10 = instability, reduce lr |
| `entropy` | Slowly decreasing | → 0 too fast = mode collapse |
| `clip_ratio/region_mean` | 0.01 ~ 0.05 | > 0.3 = too many tokens being clipped, policy changing too fast |

---

## Summary: Files You Need to Write

```
1. configs.py            (~70 lines)  Constants + answer extraction
2. generate_rollouts.py  (~160 lines) vLLM teacher generation
3. data.py               (~180 lines) Load, reward, advantage, dataset
4. trainer.py            (~275 lines) OfflineGRPOTrainer (1 override)
5. train.py              (~185 lines) Wiring + argparse
6. evaluate.py           (~135 lines) Merge LoRA + vLLM eval
7. run.sh                (~60 lines)  SLURM script
```

Total: ~1065 lines of Python + shell. The core insight is in ~20 lines of `trainer.py`: set `old_per_token_logps = teacher_logprobs` and let TRL handle the rest.
