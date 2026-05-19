"""Standalone eval of an R1-distilled teacher on MATH-500.

Faithful to the DeepSeek-Math/DeepSeek-R1 official eval pipeline:
  - No system prompt; directive lives in the user message.
  - The chat template auto-prefixes the assistant turn with `<think>\\n`.
  - temp=0.6, top_p=0.95.
  - Answer extraction via DeepSeek's `extract_math_answer` (multi-candidate, exhaust=True).
  - Equivalence via DeepSeek's `math_equal` after `strip_string` (sympy-based).
  - Multiple runs averaged for pass@1.

Source ports of `math_equal`, `strip_string`, `extract_answer` etc. live in
`DG-offline/math_equal.py`. See top of that file for upstream URLs.

Usage:
    python Math_Verifier/eval_r1_teacher.py \\
        --model_path /scratch/mrli/models/DeepSeek-R1-Distill-Qwen-7B \\
        --runs 16 --max_tokens 32768
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import sys
import time

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Multi-candidate extractor + sympy-based equivalence (peer module in same package)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from math_eval import is_equiv_multi, extract_math_answer


MATH_DIRECTIVE = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

MATH_DATASET = "nlile/hendrycks-MATH-benchmark"


def truncated_count(outputs, max_tokens: int) -> int:
    """Number of completions that emitted exactly max_tokens tokens (likely truncated)."""
    return sum(1 for o in outputs if len(o.outputs[0].token_ids) >= max_tokens - 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--runs", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=16384)
    p.add_argument("--max_model_len", type=int, default=17408)
    p.add_argument("--tensor_parallel_size", type=int, default=4)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_problems", type=int, default=None,
                   help="If set, eval on the first N problems only (for quick diagnostics).")
    p.add_argument("--save_completions", type=str, default=None,
                   help="Optional path: save first-run per-problem completions for inspection.")
    args = p.parse_args()

    print(f"=== R1-Teacher MATH-500 eval ===")
    print(f"  model:           {args.model_path}")
    print(f"  runs:            {args.runs}")
    print(f"  temp:            {args.temperature}")
    print(f"  top_p:           {args.top_p}")
    print(f"  max_tokens:      {args.max_tokens}")
    print(f"  max_model_len:   {args.max_model_len}")
    print(f"  TP:              {args.tensor_parallel_size}")
    if args.max_problems:
        print(f"  max_problems:    {args.max_problems} (subset)")
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Build chat WITHOUT a system prompt — directive in user message, per DeepSeek's model card.
    print("Loading MATH test split...")
    data = load_dataset(MATH_DATASET)["test"]
    if args.max_problems:
        data = data.select(range(min(args.max_problems, len(data))))
    print(f"  {len(data)} problems")

    prompts = []
    answers = []
    questions = []
    for item in data:
        chat = [
            {"role": "user", "content": f"{item['problem']}\n{MATH_DIRECTIVE}"},
        ]
        formatted = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        prompts.append(formatted)
        answers.append(item["answer"])
        questions.append(item["problem"])

    # Show one prompt for debug verification
    print("=== Sample prompt (problem 0) ===")
    print(repr(prompts[0][:400]))
    print("=== End sample ===\n")

    print("Loading vLLM engine...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="auto",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
    )
    print()

    all_accs = []
    all_lens = []
    all_truncated = []
    all_run_records = []  # save per-problem completions across all runs (for re-checks)

    for run_idx in range(args.runs):
        seed = args.seed + run_idx
        params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=seed,
        )
        t0 = time.time()
        outputs = llm.generate(prompts, params)
        gen_secs = time.time() - t0

        correct = 0
        lengths = []
        run_records = []
        for i, out in enumerate(outputs):
            text = out.outputs[0].text
            ok = is_equiv_multi(questions[i], text, answers[i])
            if ok:
                correct += 1
            lengths.append(len(out.outputs[0].token_ids))
            run_records.append({
                "run": run_idx,
                "seed": seed,
                "problem_id": i,
                "ground_truth": answers[i],
                "completion_text": text,
                "completion_token_count": len(out.outputs[0].token_ids),
                "candidates": extract_math_answer(questions[i], text, task="math"),
                "correct": ok,
            })

        acc = correct / len(outputs)
        avg_len = sum(lengths) / len(lengths)
        max_len = max(lengths)
        n_trunc = truncated_count(outputs, args.max_tokens)

        all_accs.append(acc)
        all_lens.append(avg_len)
        all_truncated.append(n_trunc)
        all_run_records.extend(run_records)

        print(
            f"  Run {run_idx+1}: acc={acc:.4f}, avg_len={avg_len:.0f}, "
            f"max_len={max_len}, truncated={n_trunc}/{len(outputs)}, "
            f"time={gen_secs:.1f}s, seed={seed}"
        )

    mean_acc = sum(all_accs) / args.runs
    mean_len = sum(all_lens) / args.runs
    print()
    print(f"Model:        {args.model_path}")
    print(f"Mean acc:     {mean_acc:.4f}  (n_runs={args.runs})")
    if args.runs > 1:
        std_acc = statistics.stdev(all_accs)
        print(f"Std / range:  {std_acc:.4f} / [{min(all_accs):.4f}, {max(all_accs):.4f}]")
    print(f"Avg length:   {mean_len:.0f} tokens")
    print(f"Truncated:    {sum(all_truncated)}/{args.runs * len(prompts)} runs hit max_tokens cap")

    if args.save_completions:
        os.makedirs(os.path.dirname(args.save_completions) or ".", exist_ok=True)
        import json
        with open(args.save_completions, "w") as f:
            for rec in all_run_records:
                f.write(json.dumps(rec) + "\n")
        print(f"Saved {len(all_run_records)} per-problem records ({args.runs} runs) to {args.save_completions}")


if __name__ == "__main__":
    main()
