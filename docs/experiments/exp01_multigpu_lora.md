# Experiment 01: Multi-GPU LoRA Compatibility Test

**Date**: 2026-03-05
**Jobs**: 4251937 (initial), 4251960 (ZeRO-2 retest)
**Status**: Complete

---

## Objective

Verify which multi-GPU distributed training strategies work with LoRA + KL penalty (`disable_adapter()`) for our offline GRPO pipeline. This determines how we scale to larger models.

## Setup

- **Model**: Qwen2.5-0.5B-Instruct with LoRA (r=16, alpha=64)
- **Data**: rollouts_test.jsonl (41 problems, 164 completions)
- **GPUs**: 2× L40s (48 GB each)
- **Training**: 1 epoch, batch_size=2, grad_accum=4
- **Test matrix**: 3 strategies × 2 beta values = 6 tests

## Results

| Strategy | beta=0.0 (no KL) | beta=0.1 (KL + disable_adapter) |
|----------|:-:|:-:|
| **DDP** | PASS (66s) | PASS (51s) |
| **ZeRO-2** | PASS (65s) | PASS (51s) |
| **FSDP** | PASS (127s) | PASS (134s) |

**6/6 passed.** All strategies are compatible with LoRA + KL penalty.

## Analysis

### DDP (Distributed Data Parallel)
- Fastest strategy (~51-66s)
- Full model copy per GPU — no sharding complexity
- Works exactly as expected

### DeepSpeed ZeRO-2
- Same speed as DDP for this small model (overhead is negligible at 0.5B)
- Initial run failed due to config bug (`gradient_accumulation_steps: auto` → needs integer). Fixed and retested successfully
- Shards optimizer states + gradients but not weights — `disable_adapter()` works because full weights are on each GPU

### FSDP (Fully Sharded Data Parallel)
- 2× slower than DDP/ZeRO-2 due to weight gather/scatter overhead
- **Surprising result**: beta=0.1 passed. We predicted `disable_adapter()` would break with sharded weights, but PEFT 0.17.1 handles it correctly with `fsdp_use_orig_params: true`
- Caveat: 0.5B model is small — sharding may behave differently at 7B+ scale

### Key observations from training logs
- `grad_norm: 0.0` on ~90% of steps across all strategies — this is the data issue (all-same rewards), not a multi-GPU issue
- KL metrics logged correctly in beta=0.1 runs — `disable_adapter()` produces valid reference logprobs
- FSDP shows identical loss values to DDP on same data — confirms correctness of weight gathering

## Conclusion

**All three strategies are viable.** Recommended path:
- **7B-14B models**: ZeRO-2 (saves optimizer memory, proven compatible)
- **14B+ models**: FSDP as fallback (verify `disable_adapter()` at target scale first)

## Artifacts

- Configs: `configs/accelerate_{ddp,zero2,fsdp}_2gpu.yaml`
- Logs: `logs/multigpu_test_20260305_001307/`
- Summary: `logs/multigpu_test_20260305_001307/summary.txt`
- Test plan: `test_plan_1.md`
