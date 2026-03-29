"""Check: WHICH tokens does the student disagree with the teacher on?
Are they reasoning-critical or mundane?"""
import json
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

sys.path.insert(0, "/project/aip-szepesva/mrli/backup_dongheng/offline_grpo")
from configs import MATH_SYSTEM_PROMPT

device = "cuda:0"
STUDENT_PATH = "/scratch/mrli/models/Qwen2.5-0.5B-Instruct"
ROLLOUT_PATH = "/scratch/mrli/rollouts/math_teacher/rollouts_full.jsonl"

tok = AutoTokenizer.from_pretrained(STUDENT_PATH)
VOCAB_SIZE = AutoConfig.from_pretrained(STUDENT_PATH).vocab_size

model = AutoModelForCausalLM.from_pretrained(
    STUDENT_PATH, dtype=torch.bfloat16,
    attn_implementation="sdpa", device_map=None,
).to(device)
model.eval()

with open(ROLLOUT_PATH) as f:
    items = [json.loads(next(f)) for _ in range(5)]

for item_idx, item in enumerate(items[:3]):
    run = item["runs"][0]
    cids = run["completion_ids"]

    # Truncate OOV
    for i, tid in enumerate(cids):
        if tid >= VOCAB_SIZE:
            cids = cids[:i]
            break

    chat = [
        {"role": "system", "content": item.get("system_prompt", MATH_SYSTEM_PROMPT)},
        {"role": "user", "content": item["original_problem"]},
    ]
    prompt_text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)

    input_ids = prompt_ids + cids
    input_t = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_ids=input_t).logits[0].float()

    pl = len(prompt_ids)
    cl = len(cids)

    print(f"\n{'#' * 80}")
    print(f"Problem {item_idx}: {item['original_problem'][:100]}")
    print(f"Completion length: {cl} tokens")
    print(f"{'#' * 80}")

    agree = 0
    disagree = 0
    disagree_examples = []

    for i in range(cl):
        pos = pl + i - 1  # logit position predicting completion token i
        teacher_token = cids[i]
        student_pred = logits[pos].argmax().item()

        if student_pred == teacher_token:
            agree += 1
        else:
            disagree += 1
            # Get context (5 tokens before)
            ctx_start = max(0, pl + i - 5)
            ctx_ids = input_ids[ctx_start:pl + i]
            ctx_text = tok.decode(ctx_ids)

            teacher_text = tok.decode([teacher_token])
            student_text = tok.decode([student_pred])

            # Get student's probability for the teacher token
            probs = logits[pos].softmax(dim=0)
            teacher_prob = probs[teacher_token].item()
            student_prob = probs[student_pred].item()

            disagree_examples.append({
                "pos": i,
                "context": ctx_text,
                "teacher_token": teacher_text,
                "student_pred": student_text,
                "teacher_prob": teacher_prob,
                "student_prob": student_prob,
            })

    print(f"\nAgree: {agree}/{cl} ({100*agree/cl:.1f}%)")
    print(f"Disagree: {disagree}/{cl} ({100*disagree/cl:.1f}%)")

    print(f"\n--- ALL {len(disagree_examples)} disagreement tokens ---")
    print(f"{'Pos':>5} {'Context':>40} {'Teacher':>15} {'Student':>15} {'T_prob':>8} {'S_prob':>8}")
    for ex in disagree_examples:
        ctx = repr(ex['context'])
        if len(ctx) > 40:
            ctx = ctx[-40:]
        print(f"{ex['pos']:5d} {ctx:>40} {repr(ex['teacher_token']):>15} {repr(ex['student_pred']):>15} {ex['teacher_prob']:8.4f} {ex['student_prob']:8.4f}")
