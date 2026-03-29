"""Side-by-side: Teacher rollout vs Student baseline generation on same problems."""
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
sys.path.insert(0, "/project/aip-szepesva/mrli/backup_dongheng/offline_grpo")
from configs import MATH_SYSTEM_PROMPT, extract_boxed_answer
from math_verify import parse, verify

device = "cuda:0"
tok = AutoTokenizer.from_pretrained("/scratch/mrli/models/Qwen2.5-0.5B-Instruct")

model = AutoModelForCausalLM.from_pretrained(
    "/scratch/mrli/models/Qwen2.5-0.5B-Instruct",
    dtype=torch.bfloat16, attn_implementation="sdpa", device_map=None,
).to(device)
model.eval()

with open("/scratch/mrli/rollouts/math_teacher/rollouts_full.jsonl") as f:
    items = [json.loads(next(f)) for _ in range(5)]

for idx, item in enumerate(items):
    gt = item["ground_truth_answer"]
    problem = item["original_problem"]

    # Pick first correct teacher completion (or first if none correct)
    teacher_text = item["runs"][0]["response"]
    teacher_ans = item["runs"][0].get("extracted_answer") or item["runs"][0].get("boxed_answer")
    for r in item["runs"]:
        ans = r.get("extracted_answer") or r.get("boxed_answer")
        try:
            if verify(parse(ans), parse(gt)):
                teacher_text = r["response"]
                teacher_ans = ans
                break
        except:
            pass

    # Generate student completion
    chat = [
        {"role": "system", "content": item.get("system_prompt", MATH_SYSTEM_PROMPT)},
        {"role": "user", "content": problem},
    ]
    prompt_text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    torch.manual_seed(42)
    with torch.no_grad():
        out = model.generate(prompt_tensor, max_new_tokens=2048, temperature=0.6, do_sample=True)
    student_text = tok.decode(out[0][len(prompt_ids):], skip_special_tokens=True)
    student_ans = extract_boxed_answer(student_text)

    try:
        t_ok = verify(parse(teacher_ans), parse(gt))
    except:
        t_ok = False
    try:
        s_ok = verify(parse(student_ans), parse(gt))
    except:
        s_ok = False

    print(f"\n{'#' * 80}")
    print(f"Problem {idx}: {problem[:150]}")
    print(f"Ground truth: {gt}")
    print(f"{'#' * 80}")

    print(f"\n>>> TEACHER (7B) — {'CORRECT' if t_ok else 'WRONG'}, answer={teacher_ans}, len={len(teacher_text)} chars")
    print(teacher_text[:2000])
    if len(teacher_text) > 2000:
        print(f"[...truncated...]")

    print(f"\n>>> STUDENT (0.5B) — {'CORRECT' if s_ok else 'WRONG'}, answer={student_ans}, len={len(student_text)} chars")
    print(student_text[:2000])
    if len(student_text) > 2000:
        print(f"[...truncated...]")
