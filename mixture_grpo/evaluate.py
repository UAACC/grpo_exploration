"""Evaluate a (possibly LoRA) checkpoint on GSM8K or MATH test split using vLLM."""

import argparse

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from vllm import LLM, SamplingParams
from tqdm import tqdm

from configs import (
    GSM8K_SYSTEM_PROMPT, GSM8K_DATASET,
    MATH_SYSTEM_PROMPT, MATH_DATASET,
    DEFAULT_TARGET_MODEL,
    extract_gsm8k_answer, extract_boxed_answer,
)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate on GSM8K or MATH test set.")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--base_model", type=str, default=DEFAULT_TARGET_MODEL)
    p.add_argument("--merge_lora", action="store_true")
    p.add_argument("--merged_output", type=str, default=None)
    p.add_argument("--dataset_type", type=str, default="gsm8k",
                    choices=["gsm8k", "math"], help="Which dataset to evaluate on.")
    p.add_argument("--dataset", type=str, default=None,
                    help="Override dataset path. Defaults based on dataset_type.")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=-1)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--max_model_len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    return p.parse_args()


def merge_lora(base_model: str, adapter_path: str, output_path: str) -> str:
    import torch
    print(f"Merging LoRA: base={base_model}, adapter={adapter_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model = model.merge_and_unload()
    model.save_pretrained(output_path)
    tok = AutoTokenizer.from_pretrained(base_model)
    tok.save_pretrained(output_path)
    print(f"Merged model saved to {output_path}")
    return output_path


def check_gsm8k(pred_text: str, gold_text: str) -> bool:
    pred = extract_gsm8k_answer(pred_text)
    gold = extract_gsm8k_answer(gold_text)
    if pred is not None and gold is not None:
        try:
            return float(pred) == float(gold)
        except ValueError:
            pass
    return False


def check_math(pred_text: str, gold_text: str) -> bool:
    from math_verify import parse, verify
    try:
        pred = parse(extract_boxed_answer(pred_text))
        gold = parse(gold_text)
        return verify(gold, pred) is True
    except Exception:
        return False


def main():
    args = parse_args()

    # Resolve dataset defaults based on type
    is_math = args.dataset_type == "math"
    if args.dataset is None:
        args.dataset = MATH_DATASET if is_math else GSM8K_DATASET
    system_prompt = MATH_SYSTEM_PROMPT if is_math else GSM8K_SYSTEM_PROMPT
    question_field = "problem" if is_math else "question"
    answer_field = "answer"
    checker = check_math if is_math else check_gsm8k

    if is_math:
        args.max_tokens = max(args.max_tokens, 2048)
        args.max_model_len = max(args.max_model_len, 3072)

    model_name = args.model_path
    if args.merge_lora:
        import tempfile
        output = args.merged_output or tempfile.mkdtemp(prefix="merged_lora_")
        model_name = merge_lora(args.base_model, args.model_path, output)

    llm = LLM(
        model=model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="auto",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if is_math:
        data = load_dataset(args.dataset)[args.split]
    else:
        data = load_dataset(args.dataset, "main", split=args.split)
    print(f"Dataset ({args.dataset_type}): {len(data)} problems ({args.split} split)")

    eval_data = []
    for i, item in enumerate(data):
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item[question_field]},
        ]
        formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        eval_data.append({"prompt": formatted, "answer": item[answer_field]})

    prompts = [item["prompt"] for item in eval_data]

    total_accuracy = 0.0
    total_length = 0.0
    all_accs = []
    for run_idx in tqdm(range(args.runs), desc="Eval runs"):
        run_seed = args.seed + run_idx
        run_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            seed=run_seed,
        )
        outputs = llm.generate(prompts, run_params)
        correct = 0
        lengths = []
        for i, output in enumerate(outputs):
            gen_text = output.outputs[0].text
            if checker(gen_text, eval_data[i]["answer"]):
                correct += 1
            lengths.append(len(output.outputs[0].token_ids))

        acc = correct / len(outputs)
        avg_len = sum(lengths) / len(lengths)
        total_accuracy += acc
        total_length += avg_len
        all_accs.append(acc)
        print(f"  Run {run_idx+1}: accuracy={acc:.4f}, avg_length={avg_len:.1f} (seed={run_seed})")

    mean_acc = total_accuracy / args.runs
    print(f"\nModel: {model_name}")
    print(f"Dataset: {args.dataset_type}")
    print(f"Accuracy: {mean_acc:.4f}  (over {args.runs} run(s))")
    print(f"Avg length: {total_length / args.runs:.1f}")
    if args.runs > 1:
        import statistics
        std_acc = statistics.stdev(all_accs)
        print(f"Std: {std_acc:.4f}, Min: {min(all_accs):.4f}, Max: {max(all_accs):.4f}")


if __name__ == "__main__":
    main()
