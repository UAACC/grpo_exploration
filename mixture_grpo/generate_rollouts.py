"""Generate teacher rollouts for GSM8K using vLLM.

Produces a JSONL file where each line corresponds to one GSM8K problem and
contains N completions with per-token logprobs from the teacher model.

Usage:
    python generate_rollouts.py \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --output_path rollouts_gsm8k.jsonl
"""

import argparse
import json

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

from configs import (
    GSM8K_SYSTEM_PROMPT, GSM8K_DATASET,
    DEFAULT_BEHAVIOR_MODEL, extract_gsm8k_answer,
)


def parse_args():
    p = argparse.ArgumentParser(description="Generate teacher rollouts for GSM8K.")
    p.add_argument("--model_name", type=str, default=DEFAULT_BEHAVIOR_MODEL)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=-1)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--max_model_len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    p.add_argument("--output_path", type=str, default="rollouts_gsm8k.jsonl")
    p.add_argument("--test_mode", action="store_true", help="Only process first 40 problems.")
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_id", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

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

    data = load_dataset(args.dataset if hasattr(args, 'dataset') else GSM8K_DATASET, "main", split=args.split)
    print(f"Dataset (GSM8K): {len(data)} problems")

    all_items = list(enumerate(data))
    if args.test_mode:
        all_items = all_items[:41]

    if args.num_shards > 1:
        all_items = [item for idx, item in enumerate(all_items) if idx % args.num_shards == args.shard_id]
        print(f"Shard {args.shard_id}/{args.num_shards}: processing {len(all_items)} problems")

    eval_data = []
    for i, item in all_items:
        chat = [
            {"role": "system", "content": GSM8K_SYSTEM_PROMPT},
            {"role": "user", "content": item["question"]},
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        eval_data.append({
            "question_id": i,
            "problem": item["question"],
            "prompt": formatted_prompt,
            "answer": item["answer"],
        })

    prompts = [item["prompt"] for item in eval_data]
    print(f"Prepared {len(prompts)} prompts")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
        n=args.num_generations,
        logprobs=1,
        prompt_logprobs=0,
    )

    print("Starting generation with vLLM...")
    outputs = llm.generate(prompts, sampling_params)

    with open(args.output_path, "w", encoding="utf-8") as f:
        for item, request_output in tqdm(zip(eval_data, outputs), total=len(eval_data), desc="Saving"):
            record = {
                "question_id": item["question_id"],
                "original_problem": item["problem"],
                "ground_truth_answer": item["answer"],
                "system_prompt": GSM8K_SYSTEM_PROMPT,
                "dataset_type": "gsm8k",
                "runs": [],
            }

            for run_id, seq in enumerate(request_output.outputs[:args.num_generations]):
                token_ids = list(seq.token_ids)
                step_logprobs = seq.logprobs

                logprob_list = []
                for step, tid in enumerate(token_ids):
                    lp_dict = step_logprobs[step]
                    lp_obj = lp_dict.get(tid, None)
                    logprob_list.append(float(lp_obj.logprob) if lp_obj is not None else None)

                if token_ids and token_ids[-1] == tokenizer.eos_token_id:
                    token_ids = token_ids[:-1]
                    logprob_list = logprob_list[:-1]

                record["runs"].append({
                    "run_id": run_id,
                    "response": seq.text,
                    "extracted_answer": extract_gsm8k_answer(seq.text),
                    "logprobs": logprob_list,
                    "completion_ids": token_ids,
                })

            f.write(json.dumps(record) + "\n")

    print(f"Done. Saved {len(eval_data)} records to {args.output_path}")


if __name__ == "__main__":
    main()
