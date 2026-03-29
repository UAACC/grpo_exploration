"""P4: Teacher-token logprob BEFORE vs AFTER BC. Runs sequentially to avoid OOM."""
import json
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel

sys.path.insert(0, "/project/aip-szepesva/mrli/backup_dongheng/offline_grpo")
from configs import MATH_SYSTEM_PROMPT

device = "cuda:0"
STUDENT_PATH = "/scratch/mrli/models/Qwen2.5-0.5B-Instruct"
BC_CKPT = "/scratch/mrli/checkpoints/bc_math"
ROLLOUT_PATH = "/scratch/mrli/rollouts/math_teacher/rollouts_full.jsonl"

tok = AutoTokenizer.from_pretrained(STUDENT_PATH)
STUDENT_VOCAB = AutoConfig.from_pretrained(STUDENT_PATH).vocab_size  # 151936
N_EVAL = 200


def build_bc_input(item, run):
    chat = [
        {"role": "system", "content": item.get("system_prompt", MATH_SYSTEM_PROMPT)},
        {"role": "user", "content": item["original_problem"]},
    ]
    pt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    pids = tok.encode(pt, add_special_tokens=False)
    cids = list(run["completion_ids"])
    # Truncate at first OOV token (same as BC training does)
    for idx, tid in enumerate(cids):
        if tid >= STUDENT_VOCAB:
            cids = cids[:idx]
            break
    if not cids:
        return None, None
    ids = pids + cids
    labels = [-100] * len(pids) + list(cids)
    return ids, labels


def evaluate_model(model, tag):
    model.eval()
    total_lp = 0.0
    total_top1 = 0.0
    count = 0
    with open(ROLLOUT_PATH) as f:
        for line in f:
            if count >= N_EVAL:
                break
            it = json.loads(line)
            r = it["runs"][0]
            ids, labs = build_bc_input(it, r)
            if ids is None:
                continue
            inp = torch.tensor([ids], dtype=torch.long, device=device)
            lab = torch.tensor([labs], dtype=torch.long, device=device)

            with torch.no_grad():
                logits = model(input_ids=inp).logits[:, :-1, :].float()
                shift_lab = lab[:, 1:]
                valid = (shift_lab != -100)
                lp = logits.log_softmax(dim=-1)
                comp_lp = lp[0, valid[0]].gather(
                    1, shift_lab[0, valid[0]].unsqueeze(1)
                ).squeeze(1).mean().item()
                top1 = (logits[0, valid[0]].argmax(-1) == shift_lab[0, valid[0]]).float().mean().item()

            total_lp += comp_lp
            total_top1 += top1
            count += 1

    print(f"  [{tag}] n={count}, mean_teacher_logprob={total_lp/count:.6f}, mean_top1={total_top1/count:.4f}")
    return total_lp / count, total_top1 / count


# ---- Before BC (base model) ----
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    STUDENT_PATH, dtype=torch.bfloat16,
    attn_implementation="sdpa", device_map=None,
).to(device)
model.config.use_cache = False
lp_before, top1_before = evaluate_model(model, "BEFORE BC")
del model
torch.cuda.empty_cache()

# ---- After BC (base + LoRA) ----
print("Loading BC model...")
model = AutoModelForCausalLM.from_pretrained(
    STUDENT_PATH, dtype=torch.bfloat16,
    attn_implementation="sdpa", device_map=None,
)
model = PeftModel.from_pretrained(model, BC_CKPT).to(device)
model.config.use_cache = False
lp_after, top1_after = evaluate_model(model, "AFTER BC")
del model
torch.cuda.empty_cache()

# ---- Summary ----
print(f"\n{'':>25} {'Before':>12} {'After':>12} {'Delta':>12}")
print(f"{'Teacher logprob':>25} {lp_before:12.6f} {lp_after:12.6f} {lp_after - lp_before:+12.6f}")
print(f"{'Top-1 accuracy':>25} {top1_before:12.4f} {top1_after:12.4f} {top1_after - top1_before:+12.4f}")

# ---- Also verify stored logprobs vs teacher recomputation ----
print("\n--- Bonus: Verify stored logprobs vs teacher model ---")
print("Loading teacher model...")
TEACHER_PATH = "/scratch/mrli/models/Qwen2.5-Math-7B-Instruct"
tok_t = AutoTokenizer.from_pretrained(TEACHER_PATH)
teacher = AutoModelForCausalLM.from_pretrained(
    TEACHER_PATH, dtype=torch.bfloat16,
    attn_implementation="sdpa", device_map=None,
).to(device)
teacher.eval()

with open(ROLLOUT_PATH) as f:
    item = json.loads(next(f))
run = item["runs"][0]
stored = run["logprobs"]
cids = run["completion_ids"]

chat = [
    {"role": "system", "content": item.get("system_prompt", MATH_SYSTEM_PROMPT)},
    {"role": "user", "content": item["original_problem"]},
]
pt = tok_t.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
pids = tok_t.encode(pt, add_special_tokens=False)
full = pids + cids
inp = torch.tensor([full], dtype=torch.long, device=device)

with torch.no_grad():
    logits = teacher(input_ids=inp).logits[0].float()

pl = len(pids)
recomputed = []
for i in range(len(cids)):
    lp = logits[pl + i - 1].log_softmax(dim=0)[cids[i]].item()
    recomputed.append(lp)

diffs = [abs(stored[i] - recomputed[i]) for i in range(len(cids))]
print(f"  Max diff: {max(diffs):.6f}")
print(f"  Mean diff: {sum(diffs)/len(diffs):.6f}")
print(f"  Diff > 0.01: {sum(1 for d in diffs if d > 0.01)}/{len(cids)}")
