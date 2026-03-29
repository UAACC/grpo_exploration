"""
Deep investigation: WHY does BC degrade the student?

Hypotheses:
  H1: Wrong completions (29%) poison the model — it learns incorrect reasoning
  H2: Exposure bias — model is trained on teacher context but generates from its own
  H3: Teacher reasoning style doesn't fit 0.5B capacity — distribution mismatch
  H4: Catastrophic forgetting — LoRA fine-tuning erases useful student knowledge

Tests:
  T1: Compare loss on correct vs incorrect completions
  T2: Generate from both models on same problems, compare outputs side-by-side
  T3: Check if BC model's OWN-generated token logprobs decrease (forgetting)
  T4: Teacher-token logprob breakdown: where does BC improve vs hurt?
"""
import json
import sys
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel

sys.path.insert(0, "/project/aip-szepesva/mrli/backup_dongheng/offline_grpo")
from configs import MATH_SYSTEM_PROMPT, extract_boxed_answer
from math_verify import parse, verify

device = "cuda:0"
STUDENT_PATH = "/scratch/mrli/models/Qwen2.5-0.5B-Instruct"
BC_CKPT = "/scratch/mrli/checkpoints/bc_math"
ROLLOUT_PATH = "/scratch/mrli/rollouts/math_teacher/rollouts_full.jsonl"

tok = AutoTokenizer.from_pretrained(STUDENT_PATH)
VOCAB_SIZE = AutoConfig.from_pretrained(STUDENT_PATH).vocab_size


def build_bc_input(item, run):
    chat = [
        {"role": "system", "content": item.get("system_prompt", MATH_SYSTEM_PROMPT)},
        {"role": "user", "content": item["original_problem"]},
    ]
    pt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    pids = tok.encode(pt, add_special_tokens=False)
    cids = list(run["completion_ids"])
    for idx, tid in enumerate(cids):
        if tid >= VOCAB_SIZE:
            cids = cids[:idx]
            break
    if not cids:
        return None, None, 0
    ids = pids + cids
    labels = [-100] * len(pids) + list(cids)
    return ids, labels, len(pids)


def compute_sample_metrics(model, ids, labels):
    """Compute loss, logprob, top1 for a single sample."""
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    lab = torch.tensor([labels], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids=inp, labels=lab)
        logits = out.logits[:, :-1, :].float()
        shift_lab = lab[:, 1:]
        valid = (shift_lab != -100)
        lp = logits.log_softmax(dim=-1)
        comp_lp = lp[0, valid[0]].gather(1, shift_lab[0, valid[0]].unsqueeze(1)).squeeze(1)
        top1 = (logits[0, valid[0]].argmax(-1) == shift_lab[0, valid[0]]).float().mean().item()
    return out.loss.item(), comp_lp.mean().item(), top1


# =====================================================================
# T1: Loss on CORRECT vs INCORRECT completions
# =====================================================================
print("=" * 70)
print("T1: Loss on correct vs incorrect completions (base model)")
print("=" * 70)

base_model = AutoModelForCausalLM.from_pretrained(
    STUDENT_PATH, dtype=torch.bfloat16,
    attn_implementation="sdpa", device_map=None,
).to(device)
base_model.config.use_cache = False
base_model.eval()

correct_losses = []
incorrect_losses = []
correct_logprobs = []
incorrect_logprobs = []
n_check = 200

count = 0
with open(ROLLOUT_PATH) as f:
    for line in f:
        if count >= n_check:
            break
        it = json.loads(line)
        gt = it["ground_truth_answer"]
        for r in it["runs"][:1]:
            ids, labs, pl = build_bc_input(it, r)
            if ids is None:
                continue

            extracted = r.get("extracted_answer") or r.get("boxed_answer")
            if extracted is None:
                extracted = extract_boxed_answer(r["response"])
            try:
                is_correct = verify(parse(extracted), parse(gt)) is True
            except:
                is_correct = False

            loss, lp, t1 = compute_sample_metrics(base_model, ids, labs)

            if is_correct:
                correct_losses.append(loss)
                correct_logprobs.append(lp)
            else:
                incorrect_losses.append(loss)
                incorrect_logprobs.append(lp)

            count += 1

print(f"  Correct completions: n={len(correct_losses)}")
print(f"    Mean loss: {sum(correct_losses)/len(correct_losses):.4f}")
print(f"    Mean logprob: {sum(correct_logprobs)/len(correct_logprobs):.4f}")
print(f"  Incorrect completions: n={len(incorrect_losses)}")
print(f"    Mean loss: {sum(incorrect_losses)/len(incorrect_losses):.4f}")
print(f"    Mean logprob: {sum(incorrect_logprobs)/len(incorrect_logprobs):.4f}")
print(f"  Student finds incorrect completions {'harder' if sum(incorrect_losses)/len(incorrect_losses) > sum(correct_losses)/len(correct_losses) else 'easier'} to predict")

del base_model
torch.cuda.empty_cache()

# =====================================================================
# T2: Generate from both models on same problems, compare
# =====================================================================
print()
print("=" * 70)
print("T2: Side-by-side generation comparison (10 problems)")
print("=" * 70)

# Load base model for generation
base_model = AutoModelForCausalLM.from_pretrained(
    STUDENT_PATH, dtype=torch.bfloat16,
    attn_implementation="sdpa", device_map=None,
).to(device)
base_model.config.use_cache = True
base_model.eval()

# Load BC model
bc_model = AutoModelForCausalLM.from_pretrained(
    STUDENT_PATH, dtype=torch.bfloat16,
    attn_implementation="sdpa", device_map=None,
)
bc_model = PeftModel.from_pretrained(bc_model, BC_CKPT).to(device)
bc_model.config.use_cache = True
bc_model.eval()

# Pick 10 problems
random.seed(42)
with open(ROLLOUT_PATH) as f:
    all_items = [json.loads(l) for l in f]
samples = random.sample(all_items, 10)

base_correct = 0
bc_correct = 0
base_only = []
bc_only = []

for i, item in enumerate(samples):
    gt = item["ground_truth_answer"]
    chat = [
        {"role": "system", "content": item.get("system_prompt", MATH_SYSTEM_PROMPT)},
        {"role": "user", "content": item["original_problem"]},
    ]
    prompt_text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    gen_kwargs = dict(
        max_new_tokens=512,
        temperature=0.6,
        top_p=1.0,
        do_sample=True,
    )
    torch.manual_seed(42 + i)
    with torch.no_grad():
        base_out = base_model.generate(prompt_tensor, **gen_kwargs)
    base_text = tok.decode(base_out[0][len(prompt_ids):], skip_special_tokens=True)

    torch.manual_seed(42 + i)
    with torch.no_grad():
        bc_out = bc_model.generate(prompt_tensor, **gen_kwargs)
    bc_text = tok.decode(bc_out[0][len(prompt_ids):], skip_special_tokens=True)

    # Check correctness
    base_ans = extract_boxed_answer(base_text)
    bc_ans = extract_boxed_answer(bc_text)
    try:
        base_ok = verify(parse(base_ans), parse(gt)) is True
    except:
        base_ok = False
    try:
        bc_ok = verify(parse(bc_ans), parse(gt)) is True
    except:
        bc_ok = False

    if base_ok:
        base_correct += 1
    if bc_ok:
        bc_correct += 1
    if base_ok and not bc_ok:
        base_only.append(i)
    if bc_ok and not base_ok:
        bc_only.append(i)

    print(f"\n--- Problem {i} (GT: {gt[:30]}) ---")
    print(f"  Base: {'CORRECT' if base_ok else 'WRONG'} (ans={base_ans}, len={len(base_text)})")
    print(f"  BC:   {'CORRECT' if bc_ok else 'WRONG'} (ans={bc_ans}, len={len(bc_text)})")
    if base_ok != bc_ok:
        print(f"  >>> DIVERGENT!")
        # Print first 200 chars of each
        print(f"  Base output: {base_text[:200]}...")
        print(f"  BC output:   {bc_text[:200]}...")

print(f"\n  Summary: Base={base_correct}/10, BC={bc_correct}/10")
print(f"  Base-only correct: {base_only}")
print(f"  BC-only correct: {bc_only}")

# =====================================================================
# T3: Does BC hurt the student's OWN logprobs?
# =====================================================================
print()
print("=" * 70)
print("T3: Student's OWN generation logprobs before/after BC")
print("    (Generate with base model, measure logprobs with both)")
print("=" * 70)

# Use base model generations from T2 and measure logprobs under both models
base_model.config.use_cache = False
bc_model.config.use_cache = False

own_lp_before = []
own_lp_after = []

for i, item in enumerate(samples[:20]):
    chat = [
        {"role": "system", "content": item.get("system_prompt", MATH_SYSTEM_PROMPT)},
        {"role": "user", "content": item["original_problem"]},
    ]
    prompt_text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # Generate with base model
    base_model.config.use_cache = True
    torch.manual_seed(42 + i)
    with torch.no_grad():
        gen = base_model.generate(prompt_tensor, max_new_tokens=256, temperature=0.6, do_sample=True)
    base_model.config.use_cache = False

    gen_ids = gen[0].tolist()
    comp_ids = gen_ids[len(prompt_ids):]
    if not comp_ids:
        continue

    # Build labels for this generation
    labels = [-100] * len(prompt_ids) + comp_ids
    full_ids = gen_ids

    # Measure logprobs under base model
    loss_b, lp_b, _ = compute_sample_metrics(base_model, full_ids, labels)
    # Measure logprobs under BC model
    loss_a, lp_a, _ = compute_sample_metrics(bc_model, full_ids, labels)

    own_lp_before.append(lp_b)
    own_lp_after.append(lp_a)

mean_before = sum(own_lp_before) / len(own_lp_before)
mean_after = sum(own_lp_after) / len(own_lp_after)
print(f"  Base model's own generations measured under:")
print(f"    Base model logprob: {mean_before:.4f}")
print(f"    BC model logprob:   {mean_after:.4f}")
print(f"    Delta:              {mean_after - mean_before:+.4f}")

if mean_after < mean_before:
    print("  >>> BC model assigns LOWER probability to base model's outputs!")
    print("  >>> This indicates the BC model has shifted away from the student distribution.")
else:
    print("  BC model still agrees with base model's outputs.")

# =====================================================================
# T4: Per-problem breakdown — where does BC model lose accuracy?
# =====================================================================
print()
print("=" * 70)
print("T4: Generation length comparison")
print("=" * 70)

# Already have data from T2
base_model.config.use_cache = True
bc_model.config.use_cache = True

base_lens = []
bc_lens = []
for i, item in enumerate(samples[:10]):
    chat = [
        {"role": "system", "content": item.get("system_prompt", MATH_SYSTEM_PROMPT)},
        {"role": "user", "content": item["original_problem"]},
    ]
    prompt_text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    torch.manual_seed(42 + i)
    with torch.no_grad():
        base_out = base_model.generate(prompt_tensor, max_new_tokens=512, temperature=0.6, do_sample=True)
    torch.manual_seed(42 + i)
    with torch.no_grad():
        bc_out = bc_model.generate(prompt_tensor, max_new_tokens=512, temperature=0.6, do_sample=True)

    base_lens.append(base_out.shape[1] - len(prompt_ids))
    bc_lens.append(bc_out.shape[1] - len(prompt_ids))

print(f"  Base model avg gen length: {sum(base_lens)/len(base_lens):.0f}")
print(f"  BC model avg gen length:   {sum(bc_lens)/len(bc_lens):.0f}")

print()
print("=" * 70)
print("INVESTIGATION COMPLETE")
print("=" * 70)
