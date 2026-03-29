"""Verify stored behavior logprobs match raw teacher model logprobs."""
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda:0"
model_path = "/scratch/mrli/models/Qwen2.5-Math-7B-Instruct"

tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=None,
).to(device)
model.eval()

import sys
sys.path.insert(0, "/project/aip-szepesva/mrli/backup_dongheng/offline_grpo")
from configs import MATH_SYSTEM_PROMPT

with open("/scratch/mrli/rollouts/math_teacher/rollouts_full.jsonl") as f:
    item = json.loads(next(f))

run = item["runs"][0]
stored_logprobs = run["logprobs"]
comp_ids = run["completion_ids"]

# Build prompt
chat = [
    {"role": "system", "content": item.get("system_prompt", MATH_SYSTEM_PROMPT)},
    {"role": "user", "content": item["original_problem"]},
]
prompt_text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
prompt_ids = tok.encode(prompt_text, add_special_tokens=False)

# Full sequence
input_ids = prompt_ids + comp_ids
input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

with torch.no_grad():
    logits = model(input_ids=input_tensor).logits  # (1, T, V)

# Compute per-token logprobs for completion part
# logits[:, :-1, :] predicts tokens at positions 1..T-1
# For completion token i (at position prompt_len + i),
# we use logits at position prompt_len + i - 1
prompt_len = len(prompt_ids)
comp_len = len(comp_ids)

recomputed_logprobs = []
for i in range(comp_len):
    pos = prompt_len + i - 1  # logit position that predicts completion token i
    logit_vec = logits[0, pos, :].float()
    log_softmax = logit_vec - logit_vec.logsumexp(dim=0)
    token_logprob = log_softmax[comp_ids[i]].item()
    recomputed_logprobs.append(token_logprob)

print("=== Stored vs Recomputed logprobs (first 20 tokens) ===")
print(f"{'Pos':>4} {'Token':>8} {'Stored':>12} {'Recomputed':>12} {'Diff':>12} {'TokenText':>15}")
max_diff = 0
for i in range(min(20, comp_len)):
    s = stored_logprobs[i]
    r = recomputed_logprobs[i]
    d = abs(s - r)
    max_diff = max(max_diff, d)
    txt = repr(tok.decode([comp_ids[i]]))
    print(f"{i:4d} {comp_ids[i]:8d} {s:12.6f} {r:12.6f} {d:12.6f} {txt:>15}")

print(f"\n=== Summary over all {comp_len} tokens ===")
diffs = [abs(stored_logprobs[i] - recomputed_logprobs[i]) for i in range(comp_len)]
print(f"  Max diff: {max(diffs):.6f}")
print(f"  Mean diff: {sum(diffs)/len(diffs):.6f}")
print(f"  Tokens with diff > 0.01: {sum(1 for d in diffs if d > 0.01)}/{comp_len}")
print(f"  Tokens with diff > 0.1: {sum(1 for d in diffs if d > 0.1)}/{comp_len}")

if max(diffs) > 0.1:
    print("\n  WARNING: Significant logprob mismatch detected!")
    print("  Stored logprobs may be temperature-scaled or from a different model.")
else:
    print("\n  Logprobs match (within bf16 precision). No temperature scaling issue.")
