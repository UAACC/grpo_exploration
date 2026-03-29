"""Exp 3B: Token agreement measured on student-generated context.

Compares the 93% teacher-context agreement (from check_p4.py) against
agreement measured when the student generates its own prefix.

For each problem:
1. Student generates a completion (greedy)
2. Teacher model scores each position in the student's completion
3. At each position: does teacher's argmax match student's actual token?

Usage:
    python agreement_on_student_ctx.py \
        --student_model /path/to/student \
        --teacher_model /path/to/teacher \
        --num_problems 100
"""

import argparse
import json
import os
import sys

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "offline_grpo"))
from configs import MATH_SYSTEM_PROMPT

MATH_DATASET = "nlile/hendrycks-MATH-benchmark"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--student_model", type=str, required=True)
    p.add_argument("--teacher_model", type=str, required=True)
    p.add_argument("--num_problems", type=int, default=100)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--max_model_len", type=int, default=3072)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda:0"

    # Load dataset
    data = load_dataset(MATH_DATASET)["test"]
    problems = [{"problem": data[i]["problem"], "answer": data[i]["answer"]}
                for i in range(min(args.num_problems, len(data)))]
    print(f"Loaded {len(problems)} MATH test problems")

    # ---- Step 1: Generate student completions with vLLM ----
    print(f"\n=== Generating student completions: {args.student_model} ===")
    student_llm = LLM(
        model=args.student_model,
        dtype="auto",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.90,
        enforce_eager=False,
    )
    student_tok = student_llm.get_tokenizer()

    prompts = []
    prompt_ids_list = []
    for prob in problems:
        chat = [
            {"role": "system", "content": MATH_SYSTEM_PROMPT},
            {"role": "user", "content": prob["problem"]},
        ]
        text = student_tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompts.append(text)
        prompt_ids_list.append(student_tok.encode(text, add_special_tokens=False))

    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=args.seed)
    outputs = student_llm.generate(prompts, params)

    student_completions = []
    for i, output in enumerate(outputs):
        completion_ids = list(output.outputs[0].token_ids)
        student_completions.append({
            "prompt_ids": prompt_ids_list[i],
            "completion_ids": completion_ids,
            "text": output.outputs[0].text,
        })

    del student_llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Step 2: Load teacher model with HF (need logits, not just generation) ----
    print(f"\n=== Loading teacher model: {args.teacher_model} ===")
    teacher_tok = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=None,
    ).to(device)
    teacher_model.eval()

    # ---- Step 3: Compute agreement on student context ----
    print(f"\n=== Computing token agreement on student-generated context ===")

    all_agreements = []
    per_position_agreements = {}  # position -> list of 0/1

    for i, comp in enumerate(student_completions):
        cids = comp["completion_ids"]
        if len(cids) < 5:
            continue

        # Build full sequence: prompt + student completion
        # Use student tokenizer's prompt (both should produce same prompt for same chat)
        full_ids = comp["prompt_ids"] + cids
        prompt_len = len(comp["prompt_ids"])

        # Truncate to max model length
        if len(full_ids) > args.max_model_len:
            full_ids = full_ids[:args.max_model_len]
            cids = full_ids[prompt_len:]

        inp = torch.tensor([full_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = teacher_model(input_ids=inp).logits[0].float()

        # For each completion token position, check if teacher argmax matches student token
        matches = 0
        total = 0
        for t in range(len(cids)):
            pos = prompt_len + t - 1  # logits at pos predict token at pos+1
            if pos < 0 or pos >= logits.size(0):
                continue
            teacher_pred = logits[pos].argmax().item()
            student_actual = cids[t]
            match = int(teacher_pred == student_actual)
            matches += match
            total += 1

            # Track per-position (binned by relative position)
            rel_pos = t  # absolute position in completion
            if rel_pos not in per_position_agreements:
                per_position_agreements[rel_pos] = []
            per_position_agreements[rel_pos].append(match)

        if total > 0:
            agreement = matches / total
            all_agreements.append(agreement)

        if (i + 1) % 20 == 0:
            running_mean = sum(all_agreements) / len(all_agreements)
            print(f"  [{i+1}/{len(student_completions)}] running agreement: {100*running_mean:.2f}%")

    del teacher_model
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Results ----
    print(f"\n{'='*70}")
    print(f"TOKEN AGREEMENT ON STUDENT-GENERATED CONTEXT")
    print(f"{'='*70}")
    mean_agreement = sum(all_agreements) / len(all_agreements)
    print(f"  Problems analyzed: {len(all_agreements)}")
    print(f"  Mean agreement: {100*mean_agreement:.2f}%")
    print(f"  Min agreement:  {100*min(all_agreements):.2f}%")
    print(f"  Max agreement:  {100*max(all_agreements):.2f}%")
    print(f"\n  COMPARISON:")
    print(f"    Teacher context (check_p4.py): 93.06%")
    print(f"    Student context (this script): {100*mean_agreement:.2f}%")
    print(f"    Gap: {93.06 - 100*mean_agreement:.2f} percentage points")

    # Agreement vs position (binned)
    print(f"\n  Agreement by position in completion:")
    print(f"  {'Position':>10} {'Agreement':>12} {'Samples':>10}")
    bin_size = 50
    max_pos = max(per_position_agreements.keys()) if per_position_agreements else 0
    for bin_start in range(0, min(max_pos + 1, 600), bin_size):
        bin_matches = []
        for pos in range(bin_start, min(bin_start + bin_size, max_pos + 1)):
            if pos in per_position_agreements:
                bin_matches.extend(per_position_agreements[pos])
        if bin_matches:
            bin_agreement = sum(bin_matches) / len(bin_matches)
            print(f"  {bin_start:>4}-{bin_start+bin_size-1:<4} {100*bin_agreement:>11.2f}% {len(bin_matches):>10}")


if __name__ == "__main__":
    main()
