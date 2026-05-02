"""Evaluate a (possibly LoRA) checkpoint on the MATH test split using vLLM.

Math accuracy uses the project-canonical Math_Verifier (DeepSeek-Math port)
via `is_equiv_multi`. See docs/eval_methodology.md for the upgrade story.
"""

import argparse
import os
import sys

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from vllm import LLM, SamplingParams
from tqdm import tqdm

from configs import SYSTEM_PROMPT, DEFAULT_DATASET, DEFAULT_TARGET_MODEL, extract_boxed_answer

# Project-wide canonical math equivalence checker.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from Math_Verifier import is_equiv_multi  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate on MATH test set.")
    p.add_argument("--model_path", type=str, required=True,
                    help="Path to checkpoint (or HF model name).")
    p.add_argument("--base_model", type=str, default=DEFAULT_TARGET_MODEL,
                    help="Base model (needed when --merge_lora is set).")
    p.add_argument("--merge_lora", action="store_true",
                    help="Load base_model + LoRA adapter, merge, then evaluate.")
    p.add_argument("--merged_output", type=str, default=None,
                    help="Where to save the merged model. Temp dir if omitted.")
    p.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--runs", type=int, default=1, help="Repeat evaluation N times and average.")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=-1)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--max_model_len", type=int, default=3072)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    return p.parse_args()


def merge_lora(base_model: str, adapter_path: str, output_path: str) -> str:
    """Merge LoRA adapter into base model and save to *output_path*."""
    import torch
    print(f"Merging LoRA: base={base_model}, adapter={adapter_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model = model.merge_and_unload()
    model.save_pretrained(output_path)
    # Also copy tokenizer
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
    data = load_dataset(args.dataset)[args.split]
    print(f"Dataset: {len(data)} problems ({args.split} split)")

    eval_data = []
    for i, item in enumerate(data):
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["problem"]},
        ]
        formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        eval_data.append({
            "prompt": formatted,
            "answer": item["answer"],
            "question": item["problem"],  # passed to is_equiv_multi for question-aware extraction
        })

    prompts = [item["prompt"] for item in eval_data]

    # Evaluate
    total_accuracy = 0.0
    total_length = 0.0
    all_accs = []
    for run_idx in tqdm(range(args.runs), desc="Eval runs"):
        # Use different seed per run so temp>0 gives different samples
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
            if is_equiv_multi(eval_data[i].get("question", ""), gen_text, eval_data[i]["answer"]):
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
    print(f"Accuracy: {mean_acc:.4f}  (over {args.runs} run(s))")
    print(f"Avg length: {total_length / args.runs:.1f}")
    if args.runs > 1:
        import statistics
        std_acc = statistics.stdev(all_accs)
        print(f"Std: {std_acc:.4f}, Min: {min(all_accs):.4f}, Max: {max(all_accs):.4f}")


if __name__ == "__main__":
    main()
