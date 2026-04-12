"""Unified teacher rollout generation for any registered dataset.

Generates N completions per problem from the teacher model using vLLM,
stores token IDs, log-probabilities, and extracted answers to JSONL.

Usage:
    python shared/generate_rollouts.py --dataset svamp --teacher_model /path/to/7B --output_path /path/to/output.jsonl
"""

import argparse
import json
import os
import sys

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

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
    p.add_argument("--manual_split", type=str, default=None, choices=["train", "test"],
                    help="For datasets without proper train split (e.g., asdiv).")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--seed", type=int, default=42)
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


def main():
    args = parse_args()
    cfg = get_dataset_config(args.dataset)

    print(f"=== Generating rollouts: {cfg.name.upper()} ===")
    print(f"  Teacher: {args.teacher_model}")
    print(f"  Output: {args.output_path}")
    print(f"  Generations per problem: {args.num_generations}")

    # Load training problems
    problems = load_eval_data(
        cfg,
        split=cfg.split_train,
        manual_split=args.manual_split,
        max_problems=args.max_problems,
    )
    print(f"  Problems: {len(problems)}")

    # Load teacher
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

    # Build prompts
    prompts = []
    for prob in problems:
        chat = [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": prob["problem"]},
        ]
        prompts.append(tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        ))

    print(f"  Generating {len(prompts)} x {args.num_generations} = "
          f"{len(prompts) * args.num_generations} completions...")

    outputs = llm.generate(prompts, params)

    # Write JSONL and count accuracy
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    total_correct = 0
    total_completions = 0

    with open(args.output_path, "w") as f:
        for i, output in enumerate(outputs):
            runs = []
            for j, completion in enumerate(output.outputs):
                text = completion.text
                extracted = cfg.extract_answer(text)
                extracted_str = str(extracted) if extracted is not None else None

                if cfg.check_answer(text, problems[i]["answer"]):
                    total_correct += 1
                total_completions += 1

                runs.append({
                    "run_id": j,
                    "response": text,
                    "extracted_answer": extracted_str,
                    "logprobs": extract_logprobs(completion),
                    "completion_ids": list(completion.token_ids),
                })

            f.write(json.dumps({
                "question_id": i,
                "original_problem": problems[i]["problem"],
                "ground_truth_answer": problems[i]["answer"],
                "system_prompt": cfg.system_prompt,
                "dataset_type": cfg.name,
                "runs": runs,
            }) + "\n")

    print(f"  Written {len(outputs)} problems to {args.output_path}")
    print(f"  Teacher accuracy: {total_correct}/{total_completions} "
          f"({100*total_correct/total_completions:.1f}%)")


if __name__ == "__main__":
    main()
