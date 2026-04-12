"""Unified evaluation script for all registered datasets.

Supports greedy eval, best-of-N, or both in a single run.

Usage:
    python shared/eval_unified.py --model_path /path/to/model --dataset math --mode both
    python shared/eval_unified.py --model_path /path/to/lora --base_model /path/to/base --merge_lora --dataset svamp --mode greedy
"""

import argparse
import os
import sys

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets_registry import get_dataset_config, load_eval_data, list_datasets


def parse_args():
    p = argparse.ArgumentParser(description="Unified eval for all math benchmarks.")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--base_model", type=str, default=None)
    p.add_argument("--merge_lora", action="store_true")
    p.add_argument("--merged_output", type=str, default=None)
    p.add_argument("--dataset", type=str, required=True, choices=list_datasets())
    p.add_argument("--mode", type=str, default="greedy",
                    choices=["greedy", "best_of_n", "both"])
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max_problems", type=int, default=None)
    p.add_argument("--manual_split", type=str, default=None, choices=["train", "test"],
                    help="For datasets without proper test split (e.g., asdiv).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    return p.parse_args()


def merge_lora(base_model, adapter_path, output_path):
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


def build_prompts(problems, cfg, tokenizer):
    prompts = []
    for prob in problems:
        chat = [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": prob["problem"]},
        ]
        prompts.append(tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        ))
    return prompts


def run_greedy_eval(llm, prompts, problems, cfg, runs, seed):
    all_accs = []
    for run_idx in tqdm(range(runs), desc="Greedy runs"):
        run_seed = seed + run_idx
        params = SamplingParams(temperature=0.0, max_tokens=cfg.max_tokens, seed=run_seed)
        outputs = llm.generate(prompts, params)
        correct = sum(
            1 for i, out in enumerate(outputs)
            if cfg.check_answer(out.outputs[0].text, problems[i]["answer"])
        )
        acc = correct / len(outputs)
        all_accs.append(acc)
        print(f"  Run {run_idx+1}: accuracy={acc:.4f} (seed={run_seed})")

    mean_acc = sum(all_accs) / len(all_accs)
    print(f"\nGreedy accuracy: {mean_acc:.4f} (over {runs} run(s))")
    if runs > 1:
        import statistics
        print(f"Std: {statistics.stdev(all_accs):.4f}, "
              f"Min: {min(all_accs):.4f}, Max: {max(all_accs):.4f}")
    return mean_acc


def run_best_of_n(llm, prompts, problems, cfg, n_samples, temperature, seed):
    params = SamplingParams(
        temperature=temperature, n=n_samples, max_tokens=cfg.max_tokens, seed=seed,
    )
    print(f"Generating {len(prompts)} x {n_samples} = {len(prompts)*n_samples} completions...")
    outputs = llm.generate(prompts, params)

    pass_at_1_total = 0
    pass_at_n_total = 0
    for i, output in enumerate(outputs):
        gold = problems[i]["answer"]
        any_correct = False
        for j, completion in enumerate(output.outputs):
            if cfg.check_answer(completion.text, gold):
                any_correct = True
                if j == 0:
                    pass_at_1_total += 1
                break  # found one correct, no need to keep checking for pass@N
        if any_correct:
            pass_at_n_total += 1
        elif not any_correct:
            # need to check all completions for pass@N (the break above was for pass@1 optimization)
            pass

    # Recount properly — the break above only helps pass@1
    pass_at_1_total = 0
    pass_at_n_total = 0
    per_problem_correct = []
    for i, output in enumerate(outputs):
        gold = problems[i]["answer"]
        n_correct = 0
        first_correct = False
        for j, completion in enumerate(output.outputs):
            if cfg.check_answer(completion.text, gold):
                n_correct += 1
                if j == 0:
                    first_correct = True
        if first_correct:
            pass_at_1_total += 1
        if n_correct > 0:
            pass_at_n_total += 1
        per_problem_correct.append(n_correct)

    n = len(problems)
    pass_at_1 = pass_at_1_total / n
    pass_at_n = pass_at_n_total / n

    print(f"\nBest-of-{n_samples} results ({n} {cfg.name.upper()} problems):")
    print(f"  Temperature: {temperature}")
    print(f"  pass@1:  {100*pass_at_1:.2f}%")
    print(f"  pass@{n_samples}: {100*pass_at_n:.2f}%")
    print(f"  Mean correct per problem: {sum(per_problem_correct)/n:.2f} / {n_samples}")
    return pass_at_1, pass_at_n


def main():
    args = parse_args()
    cfg = get_dataset_config(args.dataset)

    print(f"=== Unified eval: {cfg.name.upper()} ({cfg.difficulty}) ===")
    print(f"  Model: {args.model_path}")
    print(f"  Mode: {args.mode}")

    # Resolve model path (merge LoRA if needed)
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
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load data
    problems = load_eval_data(cfg, max_problems=args.max_problems,
                              manual_split=args.manual_split)
    print(f"  Problems: {len(problems)}")

    prompts = build_prompts(problems, cfg, tokenizer)

    # Eval
    if args.mode in ("greedy", "both"):
        run_greedy_eval(llm, prompts, problems, cfg, args.runs, args.seed)

    if args.mode in ("best_of_n", "both"):
        run_best_of_n(llm, prompts, problems, cfg, args.n_samples,
                      args.temperature, args.seed)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
