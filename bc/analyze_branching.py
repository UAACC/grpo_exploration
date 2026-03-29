"""Analyze probability distributions at branching points.

For each "gap" problem (teacher correct, student wrong), compares the full
probability distribution of both models at the first divergence token.

Key questions:
1. Is the teacher's chosen token in the student's top-K? At what rank/probability?
2. Is the student's chosen token in the teacher's top-K?
3. How different are the full distributions (KL divergence) at the branching point?
4. If we sample N times from the student, how often would it pick the teacher's path?

Usage:
    python analyze_branching.py \
        --student_model /path/to/student \
        --teacher_model /path/to/teacher \
        --num_problems 200
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from math_verify import parse, verify

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "offline_grpo"))
from configs import MATH_SYSTEM_PROMPT, extract_boxed_answer

MATH_DATASET = "nlile/hendrycks-MATH-benchmark"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--student_model", type=str, required=True)
    p.add_argument("--teacher_model", type=str, required=True)
    p.add_argument("--num_problems", type=int, default=200)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--max_model_len", type=int, default=3072)
    p.add_argument("--top_k_report", type=int, default=20,
                    help="Report top-K tokens at branching points.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def check_correct(gen_text, gold_answer):
    try:
        pred = parse(extract_boxed_answer(gen_text))
        gold = parse(gold_answer)
        return verify(gold, pred) is True
    except Exception:
        return False


def main():
    args = parse_args()
    device = "cuda:0"

    # Load dataset
    data = load_dataset(MATH_DATASET)["test"]
    problems = [{"problem": data[i]["problem"], "answer": data[i]["answer"]}
                for i in range(min(args.num_problems, len(data)))]
    print(f"Loaded {len(problems)} MATH test problems")

    # ---- Step 1: Generate completions from both models with vLLM ----
    student_tok = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)

    def build_prompts(tokenizer):
        prompts = []
        prompt_ids_list = []
        for prob in problems:
            chat = [
                {"role": "system", "content": MATH_SYSTEM_PROMPT},
                {"role": "user", "content": prob["problem"]},
            ]
            text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            prompts.append(text)
            prompt_ids_list.append(tokenizer.encode(text, add_special_tokens=False))
        return prompts, prompt_ids_list

    prompts, prompt_ids_list = build_prompts(student_tok)
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=args.seed)

    print(f"\n=== Generating student completions ===")
    student_llm = LLM(model=args.student_model, dtype="auto", trust_remote_code=True,
                       max_model_len=args.max_model_len, gpu_memory_utilization=0.90)
    student_outputs = student_llm.generate(prompts, params)
    student_results = []
    for i, out in enumerate(student_outputs):
        text = out.outputs[0].text
        token_ids = list(out.outputs[0].token_ids)
        student_results.append({
            "text": text, "token_ids": token_ids,
            "correct": check_correct(text, problems[i]["answer"]),
        })
    del student_llm

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n=== Generating teacher completions ===")
    teacher_tok = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    teacher_prompts, teacher_prompt_ids_list = build_prompts(teacher_tok)
    teacher_llm = LLM(model=args.teacher_model, dtype="auto", trust_remote_code=True,
                       max_model_len=args.max_model_len, gpu_memory_utilization=0.90)
    teacher_outputs = teacher_llm.generate(teacher_prompts, params)
    teacher_results = []
    for i, out in enumerate(teacher_outputs):
        text = out.outputs[0].text
        token_ids = list(out.outputs[0].token_ids)
        teacher_results.append({
            "text": text, "token_ids": token_ids,
            "correct": check_correct(text, problems[i]["answer"]),
        })
    del teacher_llm
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Step 2: Identify gap problems and find divergence tokens ----
    gap_problems = []
    for i in range(len(problems)):
        s = student_results[i]
        t = teacher_results[i]
        if t["correct"] and not s["correct"]:
            # Find first diverging token (using student tokenizer for both texts)
            s_ids = s["token_ids"]
            # Retokenize teacher text with student tokenizer for fair comparison
            t_ids = student_tok.encode(t["text"], add_special_tokens=False)
            div_pos = 0
            for j in range(min(len(s_ids), len(t_ids))):
                if s_ids[j] != t_ids[j]:
                    div_pos = j
                    break
            else:
                div_pos = min(len(s_ids), len(t_ids))

            gap_problems.append({
                "idx": i,
                "problem": problems[i]["problem"],
                "answer": problems[i]["answer"],
                "student_ids": s_ids,
                "teacher_ids": t_ids,
                "student_text": s["text"],
                "teacher_text": t["text"],
                "div_pos": div_pos,
                "prompt_ids": prompt_ids_list[i],
            })

    print(f"\nFound {len(gap_problems)} gap problems (teacher correct, student wrong)")

    # ---- Step 3: Load both models with HF for logit analysis ----
    print(f"\n=== Loading student model for logit analysis ===")
    student_model = AutoModelForCausalLM.from_pretrained(
        args.student_model, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map=None,
    ).to(device)
    student_model.eval()

    print(f"=== Loading teacher model for logit analysis ===")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map=None,
    ).to(device)
    teacher_model.eval()

    # ---- Step 4: Analyze branching points ----
    print(f"\n{'='*70}")
    print(f"BRANCHING POINT ANALYSIS ({len(gap_problems)} gap problems)")
    print(f"{'='*70}")

    K = args.top_k_report
    teacher_token_in_student_topk = []
    teacher_token_student_ranks = []
    teacher_token_student_probs = []
    student_token_teacher_ranks = []
    kl_divs = []

    # How many tokens to analyze at the branching region (not just 1)
    BRANCH_WINDOW = 5

    for gi, gp in enumerate(gap_problems):
        div = gp["div_pos"]
        prompt_ids = gp["prompt_ids"]
        s_ids = gp["student_ids"]
        t_ids = gp["teacher_ids"]

        # Build input: prompt + shared prefix (tokens before divergence)
        shared_prefix = s_ids[:div]  # same as t_ids[:div] by construction
        input_ids = prompt_ids + shared_prefix
        prompt_len = len(prompt_ids)

        if len(input_ids) < 2:
            continue

        inp = torch.tensor([input_ids], dtype=torch.long, device=device)

        # Get logits from both models at the divergence point
        with torch.no_grad():
            s_logits = student_model(input_ids=inp).logits[0, -1, :].float()
            t_logits = teacher_model(input_ids=inp).logits[0, -1, :].float()

        # Truncate to shared vocab size (student=151936, teacher=152064)
        shared_vocab = min(s_logits.size(0), t_logits.size(0))
        s_probs = F.softmax(s_logits[:shared_vocab], dim=-1)
        t_probs = F.softmax(t_logits[:shared_vocab], dim=-1)

        # Student's chosen token and teacher's chosen token at divergence
        s_token = s_ids[div] if div < len(s_ids) else None
        t_token = t_ids[div] if div < len(t_ids) else None

        if s_token is None or t_token is None:
            continue

        # Rank of teacher's token in student's distribution
        s_sorted_indices = s_probs.argsort(descending=True)
        t_token_rank_in_student = (s_sorted_indices == t_token).nonzero(as_tuple=True)[0].item()
        t_token_prob_in_student = s_probs[t_token].item()

        # Rank of student's token in teacher's distribution
        t_sorted_indices = t_probs.argsort(descending=True)
        s_token_rank_in_teacher = (t_sorted_indices == s_token).nonzero(as_tuple=True)[0].item()

        teacher_token_in_student_topk.append(t_token_rank_in_student < K)
        teacher_token_student_ranks.append(t_token_rank_in_student)
        teacher_token_student_probs.append(t_token_prob_in_student)
        student_token_teacher_ranks.append(s_token_rank_in_teacher)

        # KL divergence at this point
        # KL(teacher || student) — how much info is lost using student instead of teacher
        kl = F.kl_div(s_probs.log(), t_probs, reduction='sum').item()
        kl_divs.append(kl)

        # Print details for first 15 gap problems
        if gi < 15:
            s_token_text = student_tok.decode([s_token])
            t_token_text = student_tok.decode([t_token])

            # Shared prefix text (last 50 chars for context)
            prefix_text = student_tok.decode(shared_prefix)
            context = prefix_text[-80:] if len(prefix_text) > 80 else prefix_text

            print(f"\n  --- Problem {gp['idx']} (div at completion token {div}) ---")
            print(f"  Ground truth: {gp['answer']}")
            print(f"  Context: ...{context}")
            print(f"  Student chose: '{s_token_text}' (teacher rank: {s_token_rank_in_teacher})")
            print(f"  Teacher chose: '{t_token_text}' (student rank: {t_token_rank_in_student}, "
                  f"prob: {t_token_prob_in_student:.4f})")
            print(f"  KL(teacher||student): {kl:.4f}")

            # Print student's top-10
            print(f"  Student top-10:")
            for r in range(min(10, len(s_sorted_indices))):
                tok_id = s_sorted_indices[r].item()
                tok_text = student_tok.decode([tok_id])
                prob = s_probs[tok_id].item()
                marker = " <-- TEACHER" if tok_id == t_token else ""
                marker += " <-- STUDENT" if tok_id == s_token else ""
                print(f"    rank {r}: '{tok_text}' (p={prob:.4f}){marker}")

            # Print teacher's top-10
            print(f"  Teacher top-10:")
            for r in range(min(10, len(t_sorted_indices))):
                tok_id = t_sorted_indices[r].item()
                tok_text = student_tok.decode([tok_id])
                prob = t_probs[tok_id].item()
                marker = " <-- TEACHER" if tok_id == t_token else ""
                marker += " <-- STUDENT" if tok_id == s_token else ""
                print(f"    rank {r}: '{tok_text}' (p={prob:.4f}){marker}")

    del student_model, teacher_model
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Summary statistics ----
    print(f"\n{'='*70}")
    print(f"SUMMARY ({len(teacher_token_student_ranks)} gap problems analyzed)")
    print(f"{'='*70}")

    n = len(teacher_token_student_ranks)
    if n == 0:
        print("  No gap problems to analyze.")
        return

    # Teacher's token rank in student's distribution
    mean_rank = sum(teacher_token_student_ranks) / n
    median_rank = sorted(teacher_token_student_ranks)[n // 2]
    in_top1 = sum(1 for r in teacher_token_student_ranks if r == 0)
    in_top5 = sum(1 for r in teacher_token_student_ranks if r < 5)
    in_top10 = sum(1 for r in teacher_token_student_ranks if r < 10)
    in_top20 = sum(1 for r in teacher_token_student_ranks if r < 20)
    in_top50 = sum(1 for r in teacher_token_student_ranks if r < 50)
    in_top100 = sum(1 for r in teacher_token_student_ranks if r < 100)

    print(f"\n  Teacher's chosen token rank in STUDENT's distribution:")
    print(f"    Mean rank: {mean_rank:.1f}")
    print(f"    Median rank: {median_rank}")
    print(f"    In student top-1:   {in_top1:3d}/{n} ({100*in_top1/n:.1f}%)")
    print(f"    In student top-5:   {in_top5:3d}/{n} ({100*in_top5/n:.1f}%)")
    print(f"    In student top-10:  {in_top10:3d}/{n} ({100*in_top10/n:.1f}%)")
    print(f"    In student top-20:  {in_top20:3d}/{n} ({100*in_top20/n:.1f}%)")
    print(f"    In student top-50:  {in_top50:3d}/{n} ({100*in_top50/n:.1f}%)")
    print(f"    In student top-100: {in_top100:3d}/{n} ({100*in_top100/n:.1f}%)")

    # Probability of teacher's token under student
    mean_prob = sum(teacher_token_student_probs) / n
    print(f"\n  Teacher's token probability under student: mean={mean_prob:.4f}")

    # Student's token rank in teacher's distribution
    s_mean_rank = sum(student_token_teacher_ranks) / n
    s_in_top5 = sum(1 for r in student_token_teacher_ranks if r < 5)
    s_in_top10 = sum(1 for r in student_token_teacher_ranks if r < 10)
    print(f"\n  Student's chosen token rank in TEACHER's distribution:")
    print(f"    Mean rank: {s_mean_rank:.1f}")
    print(f"    In teacher top-5:  {s_in_top5:3d}/{n} ({100*s_in_top5/n:.1f}%)")
    print(f"    In teacher top-10: {s_in_top10:3d}/{n} ({100*s_in_top10/n:.1f}%)")

    # KL divergence
    mean_kl = sum(kl_divs) / n
    print(f"\n  KL(teacher||student) at branching point: mean={mean_kl:.4f}")

    # Implication for best-of-N
    print(f"\n  IMPLICATION FOR BEST-OF-N:")
    print(f"    If teacher's token is in student's top-5 with p={mean_prob:.4f},")
    print(f"    then sampling N times at temp>0 would pick the teacher's path")
    print(f"    roughly {100*mean_prob:.1f}% of the time per sample.")
    print(f"    With N=16 samples: P(at least one correct path) ≈ "
          f"{100*(1-(1-mean_prob)**16):.1f}%")
    print(f"    With N=64 samples: P(at least one correct path) ≈ "
          f"{100*(1-(1-mean_prob)**64):.1f}%")


if __name__ == "__main__":
    main()
