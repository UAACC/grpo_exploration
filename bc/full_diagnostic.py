"""
Full BC diagnostic covering all priorities:
  P1: Single-sample overfit test
  P2: Per-token input / label / mask printout
  P3: Prompt length, padding, truncation, chat template check
  P4: Teacher-token logprob before/after BC (using saved checkpoint)
  P5: Incorrect sample ratio and EOS check
Also: Verify stored behavior logprobs match raw teacher model output
"""
import json
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel

sys.path.insert(0, "/project/aip-szepesva/mrli/backup_dongheng/offline_grpo")
from configs import MATH_SYSTEM_PROMPT, DEFAULT_LORA_CONFIG

device = "cuda:0"
STUDENT_PATH = "/scratch/mrli/models/Qwen2.5-0.5B-Instruct"
TEACHER_PATH = "/scratch/mrli/models/Qwen2.5-Math-7B-Instruct"
BC_CKPT = "/scratch/mrli/checkpoints/bc_math"
ROLLOUT_PATH = "/scratch/mrli/rollouts/math_teacher/rollouts_full.jsonl"


def load_sample(idx=0):
    """Load a single rollout sample."""
    with open(ROLLOUT_PATH) as f:
        for i, line in enumerate(f):
            if i == idx:
                return json.loads(line)


def build_bc_input(item, run, tok):
    """Build BC-style (input_ids, labels) from a rollout record."""
    chat = [
        {"role": "system", "content": item.get("system_prompt", MATH_SYSTEM_PROMPT)},
        {"role": "user", "content": item["original_problem"]},
    ]
    prompt_text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    comp_ids = run["completion_ids"]
    input_ids = prompt_ids + comp_ids
    labels = [-100] * len(prompt_ids) + list(comp_ids)
    return input_ids, labels, len(prompt_ids), prompt_text


# =========================================================================
# P2: Per-token input / label / mask printout
# =========================================================================
print("=" * 70)
print("P2: Per-token input / label / mask printout")
print("=" * 70)

tok = AutoTokenizer.from_pretrained(STUDENT_PATH)
item = load_sample(0)
run = item["runs"][0]
input_ids, labels, prompt_len, prompt_text = build_bc_input(item, run, tok)

print(f"Prompt length: {prompt_len}")
print(f"Completion length: {len(run['completion_ids'])}")
print(f"Total length: {len(input_ids)}")
print()

# Print boundary region: last 5 prompt tokens + first 10 completion tokens
print("--- Boundary: last 5 prompt + first 10 completion ---")
print(f"{'Pos':>5} {'InputID':>8} {'Label':>8} {'Mask':>5} {'InputToken':>20} {'LabelToken':>20}")
start = max(0, prompt_len - 5)
end = min(len(input_ids), prompt_len + 10)
for i in range(start, end):
    iid = input_ids[i]
    lab = labels[i]
    mask = "COMP" if lab != -100 else "MASK"
    itok = repr(tok.decode([iid]))
    ltok = repr(tok.decode([lab])) if lab != -100 else "---"
    marker = " <-- boundary" if i == prompt_len else ""
    print(f"{i:5d} {iid:8d} {lab:8d} {mask:>5} {itok:>20} {ltok:>20}{marker}")

print()
print("--- Last 5 tokens ---")
for i in range(len(input_ids) - 5, len(input_ids)):
    iid = input_ids[i]
    lab = labels[i]
    mask = "COMP" if lab != -100 else "MASK"
    itok = repr(tok.decode([iid]))
    ltok = repr(tok.decode([lab])) if lab != -100 else "---"
    print(f"{i:5d} {iid:8d} {lab:8d} {mask:>5} {itok:>20} {ltok:>20}")

# Verify: labels[i] == input_ids[i] for all completion positions
for i in range(prompt_len, len(input_ids)):
    assert labels[i] == input_ids[i], f"MISMATCH at pos {i}: label={labels[i]}, input={input_ids[i]}"
print("\nAll completion labels == input_ids: VERIFIED")

# =========================================================================
# P3: Prompt length, padding, truncation, chat template check
# =========================================================================
print()
print("=" * 70)
print("P3: Prompt length, padding, truncation, chat template")
print("=" * 70)

# Check prompt lengths across multiple samples
print("\nPrompt lengths for first 20 problems:")
prompt_lengths = []
comp_lengths = []
with open(ROLLOUT_PATH) as f:
    for i, line in enumerate(f):
        if i >= 20:
            break
        it = json.loads(line)
        r = it["runs"][0]
        ids, labs, pl, _ = build_bc_input(it, r, tok)
        cl = len(r["completion_ids"])
        prompt_lengths.append(pl)
        comp_lengths.append(cl)
        print(f"  Problem {i}: prompt_len={pl}, comp_len={cl}, total={pl+cl}")

print(f"\nPrompt length range: {min(prompt_lengths)}-{max(prompt_lengths)}")
print(f"Completion length range: {min(comp_lengths)}-{max(comp_lengths)}")

# Check: any truncation happening with max_length=2304?
print(f"\nMax total (prompt+comp): {max(pl+cl for pl, cl in zip(prompt_lengths, comp_lengths))}")
print(f"Max allowed: 2304")

# Check broader truncation stats
n_truncated = 0
total_checked = 0
with open(ROLLOUT_PATH) as f:
    for line in f:
        it = json.loads(line)
        for r in it["runs"]:
            chat = [
                {"role": "system", "content": it.get("system_prompt", MATH_SYSTEM_PROMPT)},
                {"role": "user", "content": it["original_problem"]},
            ]
            pt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            pids = tok.encode(pt, add_special_tokens=False)
            total_len = len(pids) + len(r["completion_ids"])
            if total_len > 2304:
                n_truncated += 1
            total_checked += 1
print(f"\nSamples exceeding max_length=2304: {n_truncated}/{total_checked} ({100*n_truncated/total_checked:.1f}%)")

# Chat template check
print(f"\nChat template output (first sample):")
print(f"  {repr(prompt_text)}")

# =========================================================================
# P5: Incorrect sample ratio and EOS
# =========================================================================
print()
print("=" * 70)
print("P5: Incorrect sample ratio and EOS")
print("=" * 70)

from math_verify import parse, verify
from configs import extract_boxed_answer

correct = 0
incorrect = 0
has_eos = 0
eos_id = tok.eos_token_id
total = 0

with open(ROLLOUT_PATH) as f:
    for line in f:
        it = json.loads(line)
        gt = it["ground_truth_answer"]
        for r in it["runs"]:
            total += 1
            cids = r["completion_ids"]
            if cids and cids[-1] == eos_id:
                has_eos += 1
            extracted = r.get("extracted_answer") or r.get("boxed_answer")
            if extracted is None:
                extracted = extract_boxed_answer(r["response"])
            try:
                if verify(parse(extracted), parse(gt)) is True:
                    correct += 1
                else:
                    incorrect += 1
            except:
                incorrect += 1

print(f"Total completions: {total}")
print(f"Correct: {correct} ({100*correct/total:.1f}%)")
print(f"Incorrect: {incorrect} ({100*incorrect/total:.1f}%)")
print(f"Has EOS at end: {has_eos} ({100*has_eos/total:.1f}%)")

# =========================================================================
# P1: Single-sample overfit test
# =========================================================================
print()
print("=" * 70)
print("P1: Single-sample overfit test (50 steps on 1 sample)")
print("=" * 70)

# Load student model with LoRA
model = AutoModelForCausalLM.from_pretrained(
    STUDENT_PATH, dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=None,
).to(device)
model.config.use_cache = False

lora_config = LoraConfig(
    r=32, lora_alpha=32,
    target_modules=DEFAULT_LORA_CONFIG["target_modules"],
    task_type="CAUSAL_LM",
    lora_dropout=DEFAULT_LORA_CONFIG["lora_dropout"],
)
model = get_peft_model(model, lora_config)

# Prepare single sample
item = load_sample(0)
run = item["runs"][0]
input_ids_list, labels_list, pl, _ = build_bc_input(item, run, tok)
input_t = torch.tensor([input_ids_list], dtype=torch.long, device=device)
labels_t = torch.tensor([labels_list], dtype=torch.long, device=device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

model.train()
print(f"{'Step':>5} {'Loss':>10} {'CompLogprob':>14} {'Top1Acc':>10}")
for step in range(51):
    outputs = model(input_ids=input_t, labels=labels_t)
    loss = outputs.loss

    # Compute teacher-token logprob and top-1 accuracy
    with torch.no_grad():
        logits = outputs.logits  # (1, T, V)
        shift_logits = logits[:, :-1, :].float()
        shift_labels = labels_t[:, 1:]
        valid = (shift_labels != -100)

        # Per-token logprobs
        log_probs = shift_logits.log_softmax(dim=-1)
        comp_token_logprobs = log_probs[0, valid[0]].gather(
            1, shift_labels[0, valid[0]].unsqueeze(1)
        ).squeeze(1)
        mean_logprob = comp_token_logprobs.mean().item()

        # Top-1 accuracy
        preds = shift_logits[0, valid[0]].argmax(dim=-1)
        targets = shift_labels[0, valid[0]]
        top1 = (preds == targets).float().mean().item()

    if step % 5 == 0:
        print(f"{step:5d} {loss.item():10.6f} {mean_logprob:14.6f} {top1:10.4f}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

del model, optimizer
torch.cuda.empty_cache()

# =========================================================================
# P4: Teacher-token logprob before/after BC
# =========================================================================
print()
print("=" * 70)
print("P4: Teacher-token logprob BEFORE vs AFTER BC training")
print("=" * 70)

# Load base model (no LoRA)
base_model = AutoModelForCausalLM.from_pretrained(
    STUDENT_PATH, dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=None,
).to(device)
base_model.config.use_cache = False
base_model.eval()

# Load BC-trained model (base + LoRA)
bc_model = AutoModelForCausalLM.from_pretrained(
    STUDENT_PATH, dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=None,
)
bc_model = PeftModel.from_pretrained(bc_model, BC_CKPT).to(device)
bc_model.config.use_cache = False
bc_model.eval()

# Evaluate on first 100 completions
n_eval = 100
mean_logprob_before = 0.0
mean_logprob_after = 0.0
mean_top1_before = 0.0
mean_top1_after = 0.0
count = 0

with open(ROLLOUT_PATH) as f:
    for line_idx, line in enumerate(f):
        it = json.loads(line)
        for r in it["runs"][:1]:  # 1 completion per problem
            if count >= n_eval:
                break
            ids, labs, pl, _ = build_bc_input(it, r, tok)
            inp = torch.tensor([ids], dtype=torch.long, device=device)
            lab = torch.tensor([labs], dtype=torch.long, device=device)

            with torch.no_grad():
                # Before BC
                out_before = base_model(input_ids=inp, labels=lab)
                logits_b = out_before.logits[:, :-1, :].float()
                shift_lab = lab[:, 1:]
                valid = (shift_lab != -100)
                lp_b = logits_b.log_softmax(dim=-1)
                comp_lp_b = lp_b[0, valid[0]].gather(
                    1, shift_lab[0, valid[0]].unsqueeze(1)
                ).squeeze(1).mean().item()
                top1_b = (logits_b[0, valid[0]].argmax(-1) == shift_lab[0, valid[0]]).float().mean().item()

                # After BC
                out_after = bc_model(input_ids=inp, labels=lab)
                logits_a = out_after.logits[:, :-1, :].float()
                lp_a = logits_a.log_softmax(dim=-1)
                comp_lp_a = lp_a[0, valid[0]].gather(
                    1, shift_lab[0, valid[0]].unsqueeze(1)
                ).squeeze(1).mean().item()
                top1_a = (logits_a[0, valid[0]].argmax(-1) == shift_lab[0, valid[0]]).float().mean().item()

            mean_logprob_before += comp_lp_b
            mean_logprob_after += comp_lp_a
            mean_top1_before += top1_b
            mean_top1_after += top1_a
            count += 1

        if count >= n_eval:
            break

mean_logprob_before /= count
mean_logprob_after /= count
mean_top1_before /= count
mean_top1_after /= count

print(f"Evaluated on {count} completions")
print(f"{'':>20} {'Before BC':>12} {'After BC':>12} {'Delta':>12}")
print(f"{'Mean teacher logprob':>20} {mean_logprob_before:12.6f} {mean_logprob_after:12.6f} {mean_logprob_after - mean_logprob_before:+12.6f}")
print(f"{'Top-1 accuracy':>20} {mean_top1_before:12.4f} {mean_top1_after:12.4f} {mean_top1_after - mean_top1_before:+12.4f}")

if mean_logprob_after > mean_logprob_before:
    print("\nTeacher-token logprob INCREASED after BC (learning is happening)")
else:
    print("\nWARNING: Teacher-token logprob DECREASED after BC!")

del base_model, bc_model
torch.cuda.empty_cache()

# =========================================================================
# Bonus: Verify stored behavior logprobs match raw teacher model
# =========================================================================
print()
print("=" * 70)
print("BONUS: Verify stored logprobs vs teacher model recomputation")
print("=" * 70)

teacher = AutoModelForCausalLM.from_pretrained(
    TEACHER_PATH, dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=None,
).to(device)
teacher.eval()

item = load_sample(0)
run = item["runs"][0]
stored_lps = run["logprobs"]
comp_ids = run["completion_ids"]

tok_t = AutoTokenizer.from_pretrained(TEACHER_PATH)
chat = [
    {"role": "system", "content": item.get("system_prompt", MATH_SYSTEM_PROMPT)},
    {"role": "user", "content": item["original_problem"]},
]
prompt_text = tok_t.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
prompt_ids = tok_t.encode(prompt_text, add_special_tokens=False)
full_ids = prompt_ids + comp_ids
input_t = torch.tensor([full_ids], dtype=torch.long, device=device)

with torch.no_grad():
    logits = teacher(input_ids=input_t).logits[0].float()

recomputed = []
pl = len(prompt_ids)
for i in range(len(comp_ids)):
    lp = logits[pl + i - 1].log_softmax(dim=0)[comp_ids[i]].item()
    recomputed.append(lp)

print(f"{'Pos':>4} {'Stored':>12} {'Recomputed':>12} {'Diff':>10}")
for i in range(min(20, len(comp_ids))):
    d = abs(stored_lps[i] - recomputed[i])
    print(f"{i:4d} {stored_lps[i]:12.6f} {recomputed[i]:12.6f} {d:10.6f}")

diffs = [abs(stored_lps[i] - recomputed[i]) for i in range(len(comp_ids))]
print(f"\nMax diff: {max(diffs):.6f}")
print(f"Mean diff: {sum(diffs)/len(diffs):.6f}")
print(f"Diff > 0.01: {sum(1 for d in diffs if d > 0.01)}/{len(comp_ids)}")
print(f"Diff > 0.1: {sum(1 for d in diffs if d > 0.1)}/{len(comp_ids)}")

print()
print("=" * 70)
print("ALL DIAGNOSTICS COMPLETE")
print("=" * 70)
