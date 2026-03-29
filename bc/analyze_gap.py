"""Exp 3A+3C: Large-scale generation comparison between student and teacher.

For each MATH problem, generates completions from both models, checks correctness,
and for "gap" problems (teacher correct, student wrong) analyzes where the student
diverges from the teacher.

Usage:
    python analyze_gap.py \
        --student_model /path/to/student \
        --teacher_model /path/to/teacher \
        --num_problems 200
"""

import argparse
import json
import os
import sys

from datasets import load_dataset
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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    return p.parse_args()


def check_correct(gen_text, gold_answer):
    try:
        pred = parse(extract_boxed_answer(gen_text))
        gold = parse(gold_answer)
        return verify(gold, pred) is True
    except Exception:
        return False


def find_divergence_point(tokens_a, tokens_b):
    """Find the first index where two token lists differ."""
    min_len = min(len(tokens_a), len(tokens_b))
    for i in range(min_len):
        if tokens_a[i] != tokens_b[i]:
            return i
    return min_len


def generate_all(llm, tokenizer, problems, params):
    """Generate completions for all problems."""
    prompts = []
    for prob in problems:
        chat = [
            {"role": "system", "content": MATH_SYSTEM_PROMPT},
            {"role": "user", "content": prob["problem"]},
        ]
        prompts.append(tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        ))
    outputs = llm.generate(prompts, params)
    results = []
    for i, output in enumerate(outputs):
        text = output.outputs[0].text
        token_ids = list(output.outputs[0].token_ids)
        correct = check_correct(text, problems[i]["answer"])
        results.append({
            "text": text,
            "token_ids": token_ids,
            "correct": correct,
            "length": len(token_ids),
        })
    return results


def main():
    args = parse_args()

    # Load dataset
    data = load_dataset(MATH_DATASET)["test"]
    problems = [{"problem": data[i]["problem"], "answer": data[i]["answer"]}
                for i in range(min(args.num_problems, len(data)))]
    print(f"Loaded {len(problems)} MATH test problems")

    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    # ---- Generate student completions ----
    print(f"\n=== Generating student completions: {args.student_model} ===")
    student_llm = LLM(
        model=args.student_model,
        dtype="auto",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
    )
    student_tok = student_llm.get_tokenizer()
    student_results = generate_all(student_llm, student_tok, problems, params)
    del student_llm

    # Force GPU cleanup before loading teacher
    import torch
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Generate teacher completions ----
    print(f"\n=== Generating teacher completions: {args.teacher_model} ===")
    teacher_llm = LLM(
        model=args.teacher_model,
        dtype="auto",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
    )
    teacher_tok = teacher_llm.get_tokenizer()
    teacher_results = generate_all(teacher_llm, teacher_tok, problems, params)
    del teacher_llm
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Categorize into quadrants ----
    both_correct = []
    teacher_only = []  # THE GAP
    student_only = []
    both_wrong = []

    for i in range(len(problems)):
        s = student_results[i]
        t = teacher_results[i]
        entry = {
            "idx": i,
            "problem": problems[i]["problem"][:200],
            "answer": problems[i]["answer"],
            "student": s,
            "teacher": t,
        }
        if s["correct"] and t["correct"]:
            both_correct.append(entry)
        elif t["correct"] and not s["correct"]:
            teacher_only.append(entry)
        elif s["correct"] and not t["correct"]:
            student_only.append(entry)
        else:
            both_wrong.append(entry)

    # ---- Summary ----
    n = len(problems)
    print(f"\n{'='*70}")
    print(f"QUADRANT ANALYSIS ({n} problems)")
    print(f"{'='*70}")
    print(f"  Both correct:       {len(both_correct):4d} ({100*len(both_correct)/n:.1f}%)")
    print(f"  Teacher only (GAP): {len(teacher_only):4d} ({100*len(teacher_only)/n:.1f}%)")
    print(f"  Student only:       {len(student_only):4d} ({100*len(student_only)/n:.1f}%)")
    print(f"  Both wrong:         {len(both_wrong):4d} ({100*len(both_wrong)/n:.1f}%)")
    print(f"  Student accuracy:   {100*(len(both_correct)+len(student_only))/n:.1f}%")
    print(f"  Teacher accuracy:   {100*(len(both_correct)+len(teacher_only))/n:.1f}%")

    # ---- Analyze GAP problems ----
    print(f"\n{'='*70}")
    print(f"GAP ANALYSIS: {len(teacher_only)} problems where teacher correct, student wrong")
    print(f"{'='*70}")

    if not teacher_only:
        print("  No gap problems found.")
        return

    # Completion length stats
    gap_student_lens = [e["student"]["length"] for e in teacher_only]
    gap_teacher_lens = [e["teacher"]["length"] for e in teacher_only]
    print(f"\n  Completion lengths (gap problems):")
    print(f"    Student: mean={sum(gap_student_lens)/len(gap_student_lens):.0f}, "
          f"min={min(gap_student_lens)}, max={max(gap_student_lens)}")
    print(f"    Teacher: mean={sum(gap_teacher_lens)/len(gap_teacher_lens):.0f}, "
          f"min={min(gap_teacher_lens)}, max={max(gap_teacher_lens)}")

    # Divergence analysis (use student tokenizer for both — approximate for teacher)
    # This is token-id level comparison, only meaningful when both use same tokenizer
    # For cross-model comparison, we compare at the text level instead
    print(f"\n  --- Side-by-side for first 10 gap problems ---")
    for entry in teacher_only[:10]:
        i = entry["idx"]
        s = entry["student"]
        t = entry["teacher"]
        print(f"\n  {'#'*60}")
        print(f"  Problem {i}: {entry['problem'][:120]}...")
        print(f"  Ground truth: {entry['answer']}")
        print(f"  Student: WRONG, len={s['length']} tokens")
        print(f"  Teacher: CORRECT, len={t['length']} tokens")

        # Find text-level divergence (first differing line)
        s_lines = s["text"].split("\n")
        t_lines = t["text"].split("\n")
        div_line = 0
        for li in range(min(len(s_lines), len(t_lines))):
            if s_lines[li].strip() != t_lines[li].strip():
                div_line = li
                break
        else:
            div_line = min(len(s_lines), len(t_lines))

        total_lines = max(len(s_lines), len(t_lines))
        print(f"  Text diverges at line {div_line}/{total_lines} "
              f"({100*div_line/max(total_lines,1):.0f}% through)")

        # Show first few lines of each
        print(f"\n  STUDENT (first 15 lines):")
        for li, line in enumerate(s_lines[:15]):
            marker = ">>>" if li == div_line else "   "
            print(f"    {marker} {line}")
        if len(s_lines) > 15:
            print(f"    ... ({len(s_lines) - 15} more lines)")

        print(f"\n  TEACHER (first 15 lines):")
        for li, line in enumerate(t_lines[:15]):
            marker = ">>>" if li == div_line else "   "
            print(f"    {marker} {line}")
        if len(t_lines) > 15:
            print(f"    ... ({len(t_lines) - 15} more lines)")

    # ---- Divergence statistics across all gap problems ----
    print(f"\n{'='*70}")
    print(f"DIVERGENCE STATISTICS (all {len(teacher_only)} gap problems)")
    print(f"{'='*70}")
    div_fractions = []
    for entry in teacher_only:
        s_lines = entry["student"]["text"].split("\n")
        t_lines = entry["teacher"]["text"].split("\n")
        div_line = 0
        for li in range(min(len(s_lines), len(t_lines))):
            if s_lines[li].strip() != t_lines[li].strip():
                div_line = li
                break
        else:
            div_line = min(len(s_lines), len(t_lines))
        total = max(len(s_lines), len(t_lines), 1)
        div_fractions.append(div_line / total)

    mean_div = sum(div_fractions) / len(div_fractions)
    early = sum(1 for f in div_fractions if f < 0.2)
    mid = sum(1 for f in div_fractions if 0.2 <= f < 0.6)
    late = sum(1 for f in div_fractions if f >= 0.6)

    print(f"  Mean divergence point: {100*mean_div:.1f}% through the completion")
    print(f"  Early divergence (<20%):  {early} ({100*early/len(teacher_only):.0f}%)")
    print(f"  Mid divergence (20-60%):  {mid} ({100*mid/len(teacher_only):.0f}%)")
    print(f"  Late divergence (>60%):   {late} ({100*late/len(teacher_only):.0f}%)")

    # ---- Also analyze "both correct" for comparison ----
    if both_correct:
        print(f"\n{'='*70}")
        print(f"COMPARISON: Both-correct problems ({len(both_correct)} problems)")
        print(f"{'='*70}")
        bc_student_lens = [e["student"]["length"] for e in both_correct]
        bc_teacher_lens = [e["teacher"]["length"] for e in both_correct]
        print(f"  Student length: mean={sum(bc_student_lens)/len(bc_student_lens):.0f}")
        print(f"  Teacher length: mean={sum(bc_teacher_lens)/len(bc_teacher_lens):.0f}")

        bc_div_fractions = []
        for entry in both_correct:
            s_lines = entry["student"]["text"].split("\n")
            t_lines = entry["teacher"]["text"].split("\n")
            div_line = 0
            for li in range(min(len(s_lines), len(t_lines))):
                if s_lines[li].strip() != t_lines[li].strip():
                    div_line = li
                    break
            else:
                div_line = min(len(s_lines), len(t_lines))
            total = max(len(s_lines), len(t_lines), 1)
            bc_div_fractions.append(div_line / total)

        bc_mean_div = sum(bc_div_fractions) / len(bc_div_fractions)
        print(f"  Mean divergence point: {100*bc_mean_div:.1f}% through (vs {100*mean_div:.1f}% for gap)")


if __name__ == "__main__":
    main()
