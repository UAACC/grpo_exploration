"""Evaluate best-of-N: generate N completions per problem, report pass@1 and pass@N.

If the branching analysis is correct (teacher's path is in student's top-5 with
~22% probability), then best-of-N should dramatically improve accuracy.

Usage:
    python eval_best_of_n.py \
        --model_path /path/to/model \
        --n_samples 16 \
        --temperature 0.6
"""

import argparse
import os
import sys

from datasets import load_dataset
from vllm import LLM, SamplingParams
from math_verify import parse, verify

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "offline_grpo"))
from configs import MATH_SYSTEM_PROMPT, extract_boxed_answer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mixture_grpo"))
from configs import GSM8K_SYSTEM_PROMPT, extract_gsm8k_answer

MATH_DATASET = "nlile/hendrycks-MATH-benchmark"
GSM8K_DATASET = "openai/gsm8k"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--n_samples", type=int, default=16,
                    help="Number of completions per problem.")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--dataset_type", type=str, default="math",
                    choices=["math", "gsm8k"])
    p.add_argument("--num_problems", type=int, default=500,
                    help="Number of test problems to evaluate.")
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--max_model_len", type=int, default=3072)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    return p.parse_args()


def check_correct_math(gen_text, gold_answer):
    try:
        pred = parse(extract_boxed_answer(gen_text))
        gold = parse(gold_answer)
        return verify(gold, pred) is True
    except Exception:
        return False


def check_correct_gsm8k(gen_text, gold_answer):
    pred = extract_gsm8k_answer(gen_text)
    gold = extract_gsm8k_answer(gold_answer)
    if pred is not None and gold is not None:
        try:
            return float(pred) == float(gold)
        except ValueError:
            pass
    return False


def main():
    args = parse_args()

    is_math = args.dataset_type == "math"
    if is_math:
        data = load_dataset(MATH_DATASET)["test"]
        question_field = "problem"
        system_prompt = MATH_SYSTEM_PROMPT
        checker = check_correct_math
    else:
        data = load_dataset(GSM8K_DATASET, "main", split="test")
        question_field = "question"
        system_prompt = GSM8K_SYSTEM_PROMPT
        checker = check_correct_gsm8k

    num_problems = min(args.num_problems, len(data))
    problems = [{"problem": data[i][question_field], "answer": data[i]["answer"]}
                for i in range(num_problems)]
    print(f"Loaded {num_problems} {args.dataset_type.upper()} test problems")
    print(f"Generating {args.n_samples} completions per problem (temp={args.temperature})")

    llm = LLM(
        model=args.model_path,
        dtype="auto",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
    )
    tokenizer = llm.get_tokenizer()

    # Build prompts — one per problem (vLLM handles n_samples internally)
    prompts = []
    for prob in problems:
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prob["problem"]},
        ]
        prompts.append(tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        ))

    params = SamplingParams(
        temperature=args.temperature,
        n=args.n_samples,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    print(f"\nGenerating {num_problems} × {args.n_samples} = {num_problems * args.n_samples} completions...")
    outputs = llm.generate(prompts, params)

    # Evaluate
    pass_at_1_total = 0
    pass_at_n_total = 0
    per_problem_correct_counts = []

    for i, output in enumerate(outputs):
        gold = problems[i]["answer"]
        n_correct = 0
        first_correct = False

        for j, completion in enumerate(output.outputs):
            correct = checker(completion.text, gold)
            if correct:
                n_correct += 1
                if j == 0:
                    first_correct = True

        if first_correct:
            pass_at_1_total += 1
        if n_correct > 0:
            pass_at_n_total += 1

        per_problem_correct_counts.append(n_correct)

    pass_at_1 = pass_at_1_total / num_problems
    pass_at_n = pass_at_n_total / num_problems
    mean_correct = sum(per_problem_correct_counts) / num_problems

    print(f"\n{'='*60}")
    print(f"BEST-OF-{args.n_samples} RESULTS ({num_problems} MATH problems)")
    print(f"{'='*60}")
    print(f"  Temperature: {args.temperature}")
    print(f"  pass@1:  {100*pass_at_1:.2f}%")
    print(f"  pass@{args.n_samples}: {100*pass_at_n:.2f}%")
    print(f"  Mean correct per problem: {mean_correct:.2f} / {args.n_samples}")
    print(f"  Lift (pass@{args.n_samples} / pass@1): {pass_at_n/max(pass_at_1,0.001):.2f}x")

    # Distribution of correct counts
    print(f"\n  Distribution of correct counts per problem:")
    for c in range(args.n_samples + 1):
        count = sum(1 for x in per_problem_correct_counts if x == c)
        if count > 0:
            bar = "#" * (count * 40 // num_problems)
            print(f"    {c:2d}/{args.n_samples}: {count:4d} ({100*count/num_problems:5.1f}%) {bar}")


if __name__ == "__main__":
    main()
