# PEFT LoRA `disable_adapter()` Fix for Multi-GPU (FSDP/ZeRO-3)

---

## The Problem

Our `trainer.py` computes the KL penalty by temporarily turning off LoRA to get base model predictions:

```python
with self.accelerator.unwrap_model(self.model).disable_adapter():
    ref_logprobs = self._get_per_token_logps_and_entropies(self.model, ...)
```

### Why this works on a single GPU

With LoRA, the forward pass computes:

```
output = (W + A × B) × input
          ↑ frozen    ↑ LoRA adapters
```

`disable_adapter()` temporarily sets the LoRA contribution to zero:

```
output = W × input    (pure base model)
```

This gives the **reference model's predictions for free** — no need to load a second copy of the model. On one GPU, the full weight matrix `W` lives in memory, so toggling LoRA off is trivial.

### Why it broke on multi-GPU (FSDP / ZeRO-3)

With FSDP or ZeRO-3, model weights are **sharded across GPUs**:

```
GPU 0 has: W[0:25%], A[0:25%], B[0:25%]
GPU 1 has: W[25:50%], A[25:50%], B[25:50%]
GPU 2 has: W[50:75%], ...
GPU 3 has: W[75:100%], ...
```

During a normal forward pass, FSDP **all-gathers** the full weight onto each GPU temporarily:

```
1. all-gather W shards → full W on each GPU (temporary)
2. compute output = (W + A×B) × input
3. discard the gathered W (free memory)
```

The problem in older PEFT versions was **when** `disable_adapter()` toggled LoRA relative to this gather/scatter cycle:

- If PEFT tried to modify the weight **before** the all-gather, it was operating on a shard, not the full weight
- If the toggling interfered with FSDP's internal bookkeeping of which parameters are "original" vs "added," the gather could fail or return garbage
- The adapter scaling (`alpha/r × A×B`) might not be correctly removed from sharded parameters

---

## The Fix (Two Parts)

### Part 1: `fsdp_use_orig_params: true`

Without `use_orig_params`, FSDP flattens and concatenates all parameters in a layer into a single `FlatParameter`. This destroys the identity of individual parameter tensors — PEFT can't distinguish "the LoRA A matrix" from "the base weight" because they're fused into one blob.

With `use_orig_params: true`, FSDP keeps references to the **original parameter tensors**. Even though they're stored as shards under the hood, PEFT can still identify which tensor is the base weight `W` and which are the LoRA adapters `A`, `B`. This allows `disable_adapter()` to correctly:

1. Tell FSDP "I need to do a forward pass with just `W`"
2. FSDP gathers the full `W` (without `A×B`)
3. Forward pass runs on the pure base model
4. FSDP scatters back
5. LoRA is re-enabled

Our FSDP config (`configs/accelerate_fsdp_2gpu.yaml`) includes this setting:

```yaml
fsdp_config:
  fsdp_use_orig_params: true
```

### Part 2: PEFT 0.17.1 internal fix

Older PEFT versions didn't coordinate with FSDP's gather/scatter lifecycle. The fix in recent PEFT versions ensures that:

- `disable_adapter()` sets a flag that's checked **during** the forward pass (after weights are gathered), not by modifying the sharded weights directly
- The LoRA scaling is applied/removed at the right point in the compute graph
- The context manager properly restores state even if FSDP has resharded between the forward pass and the exit of `disable_adapter()`

The key insight: PEFT learned to **cooperate with FSDP's all-gather mechanism** and toggle the adapter at the computation level rather than the storage level.

---

## Our Test Results

We tested 3 strategies × 2 beta values on 2× L40s with Qwen2.5-0.5B-Instruct + LoRA (r=16):

| Strategy | beta=0.0 (no KL, no `disable_adapter`) | beta=0.1 (KL, calls `disable_adapter`) |
|----------|:-:|:-:|
| **DDP** | PASS (66s) | PASS (51s) |
| **ZeRO-2** | PASS (65s) | PASS (51s) |
| **FSDP** | PASS (127s) | PASS (134s) |

**All 6 passed.** We predicted FSDP + beta=0.1 would fail, but PEFT 0.17.1 + `fsdp_use_orig_params: true` handles it correctly.

### Caveat

This was verified at 0.5B scale. FSDP + `disable_adapter()` should be retested with 7B+ models where sharding is more aggressive and edge cases may surface.

---

## Environment

| Package | Version |
|---------|---------|
| PEFT | 0.17.1 |
| TRL | 0.21.0 |
| PyTorch | 2.8.0 |
| Transformers | 4.56.1 |
| Accelerate | 1.10.1 |

---

## Related Files

- Test plan and results: `test_plan_1.md`
- Experiment analysis: `experiment_analysis/exp01_multigpu_lora.md`
- FSDP config: `configs/accelerate_fsdp_2gpu.yaml`
- Trainer code with `disable_adapter()`: `trainer.py` (line 148)

---

*Created: 2026-03-06*
