"""Evaluate a (possibly LoRA) checkpoint on GSM8K test set using vLLM."""

import argparse
import re

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from vllm import LLM, SamplingParams
from tqdm import tqdm


SYSTEM_PROMPT = (
    "Please solve this math problem step by step. "
    "Put your final numerical answer after ####."
)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def extract_gsm8k_answer(text: str) -> str | None:
    """Extract the final numerical answer from model output or ground truth.

    Tries in order:
    1. #### NUMBER (GSM8K ground truth format and GRPO-trained model format)
    2. \\boxed{NUMBER} (base model typical output format)
    3. Last standalone number in the text (fallback)
    """
    # Try #### format first
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)\s*$", text, re.MULTILINE)
    if match:
        return match.group(1).replace(",", "")
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")

    # Try \boxed{} format
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        content = boxed[-1].strip()
        num_match = re.search(r"[-]?[\d,]+(?:\.\d+)?", content)
        if num_match:
            return num_match.group(0).replace(",", "")

    # Fallback: last standalone number in the text
    numbers = re.findall(r"[-]?[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate on GSM8K test set.")
    p.add_argument("--model_path", type=str, required=True,
                    help="Path to checkpoint (or HF model name).")
    p.add_argument("--base_model", type=str, default=DEFAULT_MODEL,
                    help="Base model (needed when --merge_lora is set).")
    p.add_argument("--merge_lora", action="store_true",
                    help="Load base + LoRA adapter, merge, then evaluate.")
    p.add_argument("--merged_output", type=str, default=None,
                    help="Where to save the merged model.")
    p.add_argument("--dataset", type=str, default="openai/gsm8k")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--runs", type=int, default=1,
                    help="Repeat evaluation N times and average.")
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
    """Merge LoRA adapter into base model and save to output_path."""
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


def main():
    args = parse_args()

    # Resolve model path (optionally merge LoRA first)
    model_name = args.model_path
    if args.merge_lora:
        import tempfile
        output = args.merged_output or tempfile.mkdtemp(prefix="merged_lora_")
        model_name = merge_lora(args.base_model, args.model_path, output)

    # Load vLLM
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

    # Load test data
    data = load_dataset(args.dataset, "main", split=args.split)
    print(f"Dataset: {len(data)} problems ({args.split} split)")

    eval_data = []
    for item in data:
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["question"]},
        ]
        formatted = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        eval_data.append({"prompt": formatted, "answer": item["answer"]})

    prompts = [item["prompt"] for item in eval_data]

    # Evaluate
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
            pred = extract_gsm8k_answer(gen_text)
            gold = extract_gsm8k_answer(eval_data[i]["answer"])

            if pred is not None and gold is not None:
                try:
                    if float(pred) == float(gold):
                        correct += 1
                except ValueError:
                    pass
            lengths.append(len(output.outputs[0].token_ids))

        acc = correct / len(outputs)
        avg_len = sum(lengths) / len(lengths)
        total_accuracy += acc
        total_length += avg_len
        all_accs.append(acc)
        print(f"  Run {run_idx+1}: accuracy={acc:.4f} ({correct}/{len(outputs)}), "
              f"avg_length={avg_len:.1f} (seed={run_seed})")

    mean_acc = total_accuracy / args.runs
    print(f"\nModel: {model_name}")
    print(f"Accuracy: {mean_acc:.4f}  (over {args.runs} run(s))")
    print(f"Avg length: {total_length / args.runs:.1f}")
    if args.runs > 1:
        import statistics
        std_acc = statistics.stdev(all_accs)
        print(f"Std: {std_acc:.4f}, Min: {min(all_accs):.4f}, Max: {max(all_accs):.4f}")


if __name__ == "__main__":
    main()
