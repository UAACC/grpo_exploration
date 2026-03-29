"""
Deep diagnostic: run the FULL BC pipeline on a small batch manually,
verify every intermediate step against HF Trainer's behavior.
"""
import json
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model

sys.path.insert(0, "/project/aip-szepesva/mrli/backup_dongheng/offline_grpo")
from configs import MATH_SYSTEM_PROMPT, DEFAULT_LORA_CONFIG

torch.manual_seed(42)
device = "cuda:0"

# ---- Load model ----
model_path = "/scratch/mrli/models/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_path)
print(f"Native pad_token: {repr(tok.pad_token)} id={tok.pad_token_id}")
tok.pad_token = tok.eos_token  # Same as BC train_bc.py does
print(f"After override: {repr(tok.pad_token)} id={tok.pad_token_id}")

model = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=None,
).to(device)
model.config.use_cache = False
model.eval()

# ---- Load rollout samples ----
with open("/scratch/mrli/rollouts/math_teacher/rollouts_full.jsonl") as f:
    items = [json.loads(next(f)) for _ in range(3)]

# Build BC-style inputs: 4 completions from 2 problems
samples = []
for item in items[:2]:
    for run in item["runs"][:2]:
        chat = [
            {"role": "system", "content": item.get("system_prompt", MATH_SYSTEM_PROMPT)},
            {"role": "user", "content": item["original_problem"]},
        ]
        prompt_text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
        comp_ids = run["completion_ids"]

        input_ids = prompt_ids + comp_ids
        labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]

        samples.append({
            "input_ids": input_ids,
            "labels": labels,
            "prompt_len": len(prompt_ids),
            "comp_len": len(comp_ids),
        })

print(f"\n=== Built {len(samples)} samples ===")
for i, s in enumerate(samples):
    print(f"  Sample {i}: prompt_len={s['prompt_len']}, comp_len={s['comp_len']}, total={len(s['input_ids'])}")

# ---- Pad (like BCCollator) ----
max_len = max(len(s["input_ids"]) for s in samples)
batch_input_ids = []
batch_labels = []
batch_attn_mask = []

for s in samples:
    seq_len = len(s["input_ids"])
    pad_len = max_len - seq_len
    ids = torch.tensor(s["input_ids"] + [tok.pad_token_id] * pad_len, dtype=torch.long)
    labs = torch.tensor(s["labels"] + [-100] * pad_len, dtype=torch.long)
    mask = torch.tensor([1] * seq_len + [0] * pad_len, dtype=torch.long)
    batch_input_ids.append(ids)
    batch_labels.append(labs)
    batch_attn_mask.append(mask)

batch_input_ids = torch.stack(batch_input_ids).to(device)
batch_labels = torch.stack(batch_labels).to(device)
batch_attn_mask = torch.stack(batch_attn_mask).to(device)

print(f"\nBatch shape: {batch_input_ids.shape}")

# ---- CHECK 1: Label verification ----
print("\n=== CHECK 1: Label verification ===")
for i in range(len(samples)):
    pl = samples[i]["prompt_len"]
    cl = samples[i]["comp_len"]
    prompt_labels = batch_labels[i, :pl]
    assert (prompt_labels == -100).all(), f"Sample {i}: prompt labels not all -100!"
    comp_labels = batch_labels[i, pl:pl + cl]
    comp_inputs = batch_input_ids[i, pl:pl + cl]
    assert (comp_labels == comp_inputs).all(), f"Sample {i}: comp labels != input_ids!"
    pad_labels = batch_labels[i, pl + cl:]
    assert (pad_labels == -100).all(), f"Sample {i}: padding labels not -100!"
    print(f"  Sample {i}: OK (prompt={pl} masked, comp={cl} labeled, pad={max_len - pl - cl} masked)")
print("  All label checks PASSED")

# ---- CHECK 2: HF auto loss ----
print("\n=== CHECK 2: HF auto loss ===")
with torch.no_grad():
    outputs_with_mask = model(
        input_ids=batch_input_ids,
        attention_mask=batch_attn_mask,
        labels=batch_labels,
    )
print(f"  HF loss (with attn_mask): {outputs_with_mask.loss.item():.6f}")

# ---- CHECK 3: Manual loss ----
print("\n=== CHECK 3: Manual loss computation ===")
with torch.no_grad():
    logits = model(
        input_ids=batch_input_ids,
        attention_mask=batch_attn_mask,
    ).logits

shift_logits = logits[:, :-1, :].contiguous()
shift_labels = batch_labels[:, 1:].contiguous()

loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
per_token_loss = loss_fct(
    shift_logits.view(-1, shift_logits.size(-1)),
    shift_labels.view(-1),
).view(shift_logits.size(0), shift_logits.size(1))

valid_mask = (shift_labels != -100).float()
manual_loss = (per_token_loss * valid_mask).sum() / valid_mask.sum()
print(f"  Manual loss: {manual_loss.item():.6f}")
print(f"  Diff from HF: {abs(outputs_with_mask.loss.item() - manual_loss.item()):.6f}")

# ---- CHECK 4: Per-sample loss ----
print("\n=== CHECK 4: Per-sample loss ===")
for i in range(len(samples)):
    sample_loss = (per_token_loss[i] * valid_mask[i]).sum() / valid_mask[i].sum()
    n_valid = int(valid_mask[i].sum().item())
    print(f"  Sample {i}: loss={sample_loss.item():.4f}, n_tokens={n_valid}")

# ---- CHECK 5: Top-1 accuracy on completion tokens ----
print("\n=== CHECK 5: Top-1 accuracy on teacher completion tokens ===")
for i in range(len(samples)):
    pl = samples[i]["prompt_len"]
    cl = samples[i]["comp_len"]
    # After causal shift: shift_logits[i, pl-1 : pl+cl-1] predicts comp tokens
    comp_logits = shift_logits[i, pl - 1:pl + cl - 1]
    comp_targets = shift_labels[i, pl - 1:pl + cl - 1]
    predictions = comp_logits.argmax(dim=-1)
    correct = (predictions == comp_targets).float().mean().item()
    print(f"  Sample {i}: top1={correct:.4f} ({int(correct * cl)}/{cl})")

# ---- CHECK 6: flash_attention_2 + right-padding correctness ----
# Run same batch WITHOUT flash_attention_2 and compare logits
print("\n=== CHECK 6: Flash Attention 2 vs Eager attention (right-padding) ===")
model_eager = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.bfloat16,
    attn_implementation="eager",
    device_map=None,
).to(device)
model_eager.config.use_cache = False
model_eager.eval()

with torch.no_grad():
    outputs_eager = model_eager(
        input_ids=batch_input_ids,
        attention_mask=batch_attn_mask,
        labels=batch_labels,
    )
    logits_eager = model_eager(
        input_ids=batch_input_ids,
        attention_mask=batch_attn_mask,
    ).logits

print(f"  Flash loss:  {outputs_with_mask.loss.item():.6f}")
print(f"  Eager loss:  {outputs_eager.loss.item():.6f}")
print(f"  Diff:        {abs(outputs_with_mask.loss.item() - outputs_eager.loss.item()):.6f}")

# Compare logits on non-padded positions
for i in range(len(samples)):
    seq_len = len(samples[i]["input_ids"])
    l_flash = logits[i, :seq_len]
    l_eager = logits_eager[i, :seq_len]
    max_diff = (l_flash.float() - l_eager.float()).abs().max().item()
    mean_diff = (l_flash.float() - l_eager.float()).abs().mean().item()
    print(f"  Sample {i} logit diff: max={max_diff:.6f}, mean={mean_diff:.6f}")

del model_eager
torch.cuda.empty_cache()

# ---- CHECK 7: Without attention_mask (check if mask matters) ----
print("\n=== CHECK 7: Effect of attention_mask ===")
with torch.no_grad():
    outputs_no_mask = model(
        input_ids=batch_input_ids,
        # NO attention_mask
        labels=batch_labels,
    )
print(f"  Loss WITH attn_mask:    {outputs_with_mask.loss.item():.6f}")
print(f"  Loss WITHOUT attn_mask: {outputs_no_mask.loss.item():.6f}")
print(f"  Diff: {abs(outputs_with_mask.loss.item() - outputs_no_mask.loss.item()):.6f}")
if abs(outputs_with_mask.loss.item() - outputs_no_mask.loss.item()) > 0.01:
    print("  WARNING: attention_mask changes the loss significantly!")
    print("  This means padding tokens are affecting the computation.")

# ---- CHECK 8: Single sample (no padding needed) ----
print("\n=== CHECK 8: Single sample (no padding) ===")
s = samples[0]
single_ids = torch.tensor([s["input_ids"]], dtype=torch.long, device=device)
single_labels = torch.tensor([s["labels"]], dtype=torch.long, device=device)
single_mask = torch.ones_like(single_ids)
with torch.no_grad():
    out_single = model(input_ids=single_ids, attention_mask=single_mask, labels=single_labels)
print(f"  Single-sample loss: {out_single.loss.item():.6f}")
# Compare with the per-sample loss from CHECK 4
sample0_loss = (per_token_loss[0] * valid_mask[0]).sum() / valid_mask[0].sum()
print(f"  Batched sample-0 loss: {sample0_loss.item():.6f}")
print(f"  Diff: {abs(out_single.loss.item() - sample0_loss.item()):.6f}")
if abs(out_single.loss.item() - sample0_loss.item()) > 0.01:
    print("  WARNING: Batching with padding changes per-sample loss!")
    print("  This indicates padding is leaking into the computation.")

# ---- CHECK 9: Verify BCCollator produces identical results ----
print("\n=== CHECK 9: BCCollator output verification ===")
sys.path.insert(0, "/project/aip-szepesva/mrli/backup_dongheng/bc")
from data import BCDataset, BCCollator

ds = BCDataset(
    rollout_path="/scratch/mrli/rollouts/math_teacher/rollouts_full.jsonl",
    tokenizer=tok,
    max_length=2304,
    vocab_size=AutoConfig.from_pretrained(model_path).vocab_size,
)
collator = BCCollator(pad_token_id=tok.pad_token_id)

# Get first 4 samples and collate
batch_items = [ds[i] for i in range(4)]
collated = collator(batch_items)
print(f"  Collated input_ids shape: {collated['input_ids'].shape}")
print(f"  Collated labels shape: {collated['labels'].shape}")
print(f"  Collated attention_mask shape: {collated['attention_mask'].shape}")

# Check labels
for i in range(4):
    mask = collated["attention_mask"][i]
    labs = collated["labels"][i]
    ids = collated["input_ids"][i]
    seq_len = mask.sum().item()
    n_masked = (labs == -100).sum().item()
    n_labeled = (labs != -100).sum().item()
    # Check: labeled positions should have labels == input_ids
    labeled_mask = (labs != -100)
    if labeled_mask.any():
        match = (labs[labeled_mask] == ids[labeled_mask]).all().item()
    else:
        match = True
    print(f"  Sample {i}: seq_len={int(seq_len)}, masked={n_masked}, labeled={n_labeled}, labels==input_ids: {match}")

# Run through model
collated_gpu = {k: v.to(device) for k, v in collated.items()}
with torch.no_grad():
    out_collated = model(**collated_gpu)
print(f"  Collated batch loss: {out_collated.loss.item():.6f}")

print("\n=== ALL CHECKS COMPLETE ===")
