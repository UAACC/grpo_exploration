"""Detailed ASDiv baseline check: run inference on every test problem and print results."""
import os, sys
os.environ['HF_DATASETS_CACHE'] = '/scratch/mrli/datasets/ASDIV'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets_registry import get_dataset_config, load_eval_data, extract_numeric_answer, check_numeric


def main():
    cfg = get_dataset_config('asdiv')
    problems = load_eval_data(cfg, manual_split='test')
    print(f"Loaded {len(problems)} ASDiv test problems")

    from vllm import LLM, SamplingParams
    llm = LLM(model='/scratch/mrli/models/Qwen2.5-0.5B-Instruct', max_model_len=2048,
              gpu_memory_utilization=0.90, dtype='auto', trust_remote_code=True)
    tok = llm.get_tokenizer()
    params = SamplingParams(temperature=0.0, max_tokens=1024, seed=42)

    prompts = []
    for prob in problems:
        chat = [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": prob["problem"]},
        ]
        prompts.append(tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))

    print(f"Generating {len(prompts)} completions...")
    outputs = llm.generate(prompts, params)

    correct = 0
    wrong = 0
    extraction_failures = 0

    print(f"\n{'='*80}")
    print(f"PER-PROBLEM RESULTS")
    print(f"{'='*80}\n")

    for i, output in enumerate(outputs):
        text = output.outputs[0].text
        gold = problems[i]["answer"]
        pred = extract_numeric_answer(text)
        is_correct = check_numeric(text, gold)

        if is_correct:
            correct += 1
        else:
            wrong += 1
            gold_num = extract_numeric_answer(gold)
            if pred is None:
                extraction_failures += 1

            print(f"[WRONG #{wrong}] Problem {i}")
            print(f"  Q: {problems[i]['problem'][:120]}")
            print(f"  Gold: '{gold}' (extracted: '{gold_num}')")
            print(f"  Pred: '{pred}'")
            print(f"  Response (last 200): ...{text[-200:]}")
            print()

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total: {len(outputs)}")
    print(f"Correct: {correct} ({100*correct/len(outputs):.1f}%)")
    print(f"Wrong: {wrong} ({100*wrong/len(outputs):.1f}%)")
    print(f"Extraction failures (pred=None): {extraction_failures}")
    print(f"\nFirst 5 correct examples:")

    correct_count = 0
    for i, output in enumerate(outputs):
        text = output.outputs[0].text
        gold = problems[i]["answer"]
        if check_numeric(text, gold):
            correct_count += 1
            if correct_count <= 5:
                pred = extract_numeric_answer(text)
                print(f"  [{i}] Gold='{gold}' Pred='{pred}' Response tail: ...{text[-80:]}")


if __name__ == "__main__":
    main()
