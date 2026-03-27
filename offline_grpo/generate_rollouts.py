"""Step 1: Generate rollouts from a behavior policy using vLLM.

Produces a JSONL file where each line corresponds to one MATH problem and
contains N completions with per-token logprobs from the behavior policy.
"""

import argparse
import json

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

from configs import (
    MATH_SYSTEM_PROMPT, MATH_DATASET, GSM8K_SYSTEM_PROMPT, GSM8K_DATASET,
    DEFAULT_BEHAVIOR_MODEL, extract_boxed_answer, extract_gsm8k_answer,
)


def parse_args():
    p = argparse.ArgumentParser(description="Generate behavior-policy rollouts.")
    p.add_argument("--model_name", type=str, default=DEFAULT_BEHAVIOR_MODEL)
    p.add_argument("--dataset_type", type=str, default="math", choices=["math", "gsm8k"],
                    help="Dataset type: 'math' for MATH benchmark, 'gsm8k' for GSM8K.")
    p.add_argument("--dataset", type=str, default=None,
                    help="Override dataset name (default: auto from dataset_type).")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=-1)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--max_model_len", type=int, default=3072)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    p.add_argument("--output_path", type=str, default="rollouts.jsonl")
    p.add_argument("--test_mode", action="store_true", help="Only process first 40 problems.")
    p.add_argument("--num_shards", type=int, default=1, help="Total number of data-parallel shards.")
    p.add_argument("--shard_id", type=int, default=0, help="Which shard this process handles (0-indexed).")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve dataset config
    is_gsm8k = args.dataset_type == "gsm8k"
    if args.dataset is None:
        args.dataset = GSM8K_DATASET if is_gsm8k else MATH_DATASET
    system_prompt = GSM8K_SYSTEM_PROMPT if is_gsm8k else MATH_SYSTEM_PROMPT
    # GSM8K uses "question" field; MATH uses "problem"
    question_field = "question" if is_gsm8k else "problem"
    answer_extract_fn = extract_gsm8k_answer if is_gsm8k else extract_boxed_answer

    # Load model
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="auto",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Load dataset and format prompts
    if is_gsm8k:
        data = load_dataset(args.dataset, "main", split=args.split)
    else:
        data = load_dataset(args.dataset)[args.split]
    print(f"Dataset ({args.dataset_type}): {len(data)} problems")

    all_items = list(enumerate(data))
    if args.test_mode:
        all_items = all_items[:41]

    # Shard the data for data-parallel inference
    if args.num_shards > 1:
        all_items = [item for idx, item in enumerate(all_items) if idx % args.num_shards == args.shard_id]
        print(f"Shard {args.shard_id}/{args.num_shards}: processing {len(all_items)} problems")

    eval_data = []
    for i, item in all_items:
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item[question_field]},
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        eval_data.append({
            "question_id": i,
            "problem": item[question_field],
            "prompt": formatted_prompt,
            "answer": item["answer"],
        })

    prompts = [item["prompt"] for item in eval_data]
    print(f"Prepared {len(prompts)} prompts")

    # Sampling params
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
        n=args.num_generations,
        logprobs=1,       # top-1 logprob (enough to recover chosen-token logprob)
        prompt_logprobs=0,
    )

    # Generate
    print("Starting generation with vLLM...")
    outputs = llm.generate(prompts, sampling_params)

    # Save to JSONL
    with open(args.output_path, "w", encoding="utf-8") as f:
        for item, request_output in tqdm(zip(eval_data, outputs), total=len(eval_data), desc="Saving"):
            record = {
                "question_id": item["question_id"],
                "original_problem": item["problem"],
                "ground_truth_answer": item["answer"],
                "system_prompt": system_prompt,
                "dataset_type": args.dataset_type,
                "runs": [],
            }

            for run_id, seq in enumerate(request_output.outputs[: args.num_generations]):
                token_ids = list(seq.token_ids)
                step_logprobs = seq.logprobs  # list[dict[token_id -> Logprob]]

                # Flatten logprobs to a simple float list
                logprob_list = []
                for step, tid in enumerate(token_ids):
                    lp_dict = step_logprobs[step]
                    lp_obj = lp_dict.get(tid, None)
                    logprob_list.append(float(lp_obj.logprob) if lp_obj is not None else None)

                # Strip EOS token from completion_ids and logprobs
                if token_ids and token_ids[-1] == tokenizer.eos_token_id:
                    token_ids = token_ids[:-1]
                    logprob_list = logprob_list[:-1]

                record["runs"].append({
                    "run_id": run_id,
                    "response": seq.text,
                    "extracted_answer": answer_extract_fn(seq.text),
                    "logprobs": logprob_list,
                    "completion_ids": token_ids,
                })

            f.write(json.dumps(record) + "\n")

    print(f"Done. Saved {len(eval_data)} records to {args.output_path}")


if __name__ == "__main__":
    main()
