"""Unified teacher rollout generation for any registered dataset.

Generates N completions per problem from the teacher model using vLLM,
stores token IDs, log-probabilities, and extracted answers to JSONL.

This version writes incrementally in small prompt batches so that
`output_path` grows during the run instead of only after all generations
finish.

Usage:
    python shared/generate_rollouts_by_batch.py \
        --dataset svamp \
        --teacher_model /path/to/7B \
        --output_path /path/to/output.jsonl
"""

import argparse
import json
import os
import sys
from typing import List

from vllm import LLM, SamplingParams

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets_registry import get_dataset_config, load_eval_data, list_datasets


def parse_args():
    p = argparse.ArgumentParser(description="Generate teacher rollouts.")
    p.add_argument("--dataset", type=str, required=True, choices=list_datasets())
    p.add_argument("--teacher_model", type=str, required=True)
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--num_generations", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_problems", type=int, default=None)
    p.add_argument(
        "--manual_split",
        type=str,
        default=None,
        choices=["train", "test"],
        help="For datasets without proper train split (e.g., asdiv).",
    )
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--prompt_batch_size",
        type=int,
        default=8,
        help=(
            "Number of prompts to send to vLLM per batch. Smaller values make "
            "output_path update more frequently; larger values improve throughput."
        ),
    )
    p.add_argument(
        "--fsync_every_batch",
        action="store_true",
        help="Force data to disk after each written batch. Useful on some filesystems.",
    )
    return p.parse_args()


def extract_logprobs(completion):
    """Extract per-token logprobs from a vLLM completion output."""
    token_ids = list(completion.token_ids)
    logprobs_list = []
    if not completion.logprobs:
        return [0.0] * len(token_ids)

    for idx, lp_dict in enumerate(completion.logprobs):
        if not lp_dict:
            logprobs_list.append(0.0)
            continue
        tid = token_ids[idx] if idx < len(token_ids) else None
        if tid is not None and tid in lp_dict:
            logprobs_list.append(lp_dict[tid].logprob)
        else:
            top = list(lp_dict.values())
            logprobs_list.append(top[0].logprob if top else 0.0)

    return logprobs_list


def batched(items: List, batch_size: int):
    if batch_size <= 0:
        raise ValueError("prompt_batch_size must be positive.")
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def build_prompt(tokenizer, system_prompt: str, problem_text: str) -> str:
    chat = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem_text},
    ]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def main():
    args = parse_args()
    cfg = get_dataset_config(args.dataset)

    print(f"=== Generating rollouts: {cfg.name.upper()} ===")
    print(f"  Teacher: {args.teacher_model}")
    print(f"  Output: {args.output_path}")
    print(f"  Generations per problem: {args.num_generations}")
    print(f"  Prompt batch size: {args.prompt_batch_size}")

    problems = load_eval_data(
        cfg,
        split=cfg.split_train,
        manual_split=args.manual_split,
        max_problems=args.max_problems,
    )
    print(f"  Problems: {len(problems)}")

    llm = LLM(
        model=args.teacher_model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="auto",
        trust_remote_code=True,
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
    )
    tokenizer = llm.get_tokenizer()

    params = SamplingParams(
        temperature=args.temperature,
        max_tokens=cfg.max_tokens,
        seed=args.seed,
        logprobs=1,
        n=args.num_generations,
    )

    prompts = [
        build_prompt(tokenizer, cfg.system_prompt, prob["problem"])
        for prob in problems
    ]

    total_expected = len(prompts) * args.num_generations
    print(
        f"  Generating {len(prompts)} x {args.num_generations} = "
        f"{total_expected} completions in batches..."
    )

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    total_correct = 0
    total_completions = 0
    problems_written = 0

    with open(args.output_path, "w") as f:
        for batch_start, prompt_batch in batched(prompts, args.prompt_batch_size):
            batch_end = batch_start + len(prompt_batch)
            print(
                f"  Generating batch {batch_start // args.prompt_batch_size + 1}: "
                f"problems {batch_start + 1}-{batch_end}/{len(problems)}"
            )

            outputs = llm.generate(prompt_batch, params)

            for local_idx, output in enumerate(outputs):
                problem_idx = batch_start + local_idx
                prob = problems[problem_idx]

                runs = []
                for j, completion in enumerate(output.outputs):
                    text = completion.text
                    extracted = cfg.extract_answer(text)
                    extracted_str = str(extracted) if extracted is not None else None

                    if cfg.check_answer(text, prob["answer"]):
                        total_correct += 1
                    total_completions += 1

                    runs.append(
                        {
                            "run_id": j,
                            "response": text,
                            "extracted_answer": extracted_str,
                            "logprobs": extract_logprobs(completion),
                            "completion_ids": list(completion.token_ids),
                        }
                    )

                f.write(
                    json.dumps(
                        {
                            "question_id": problem_idx,
                            "original_problem": prob["problem"],
                            "ground_truth_answer": prob["answer"],
                            "system_prompt": cfg.system_prompt,
                            "dataset_type": cfg.name,
                            "runs": runs,
                        }
                    )
                    + "\n"
                )
                problems_written += 1

            f.flush()
            if args.fsync_every_batch:
                os.fsync(f.fileno())

            print(
                f"  Wrote {problems_written}/{len(problems)} problems "
                f"to {args.output_path}"
            )

    print(f"  Written {problems_written} problems to {args.output_path}")
    print(
        f"  Teacher accuracy: {total_correct}/{total_completions} "
        f"({100 * total_correct / total_completions:.1f}%)"
    )


if __name__ == "__main__":
    main()
