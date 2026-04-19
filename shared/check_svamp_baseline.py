"""Quick diagnostic: check SVAMP baseline model outputs."""
import os, sys
os.environ['HF_DATASETS_CACHE'] = '/scratch/mrli/datasets/SVAMP'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets_registry import get_dataset_config, load_eval_data, extract_numeric_answer, check_numeric


def main():
    cfg = get_dataset_config('svamp')
    problems = load_eval_data(cfg, split='test')
    print(f"Loaded {len(problems)} SVAMP test problems")

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

    outputs = llm.generate(prompts, params)

    correct = 0
    wrong_examples = []
    right_examples = []
    for i, output in enumerate(outputs):
        text = output.outputs[0].text
        gold = problems[i]["answer"]
        pred = extract_numeric_answer(text)
        is_correct = check_numeric(text, gold)
        if is_correct:
            correct += 1
            if len(right_examples) < 3:
                right_examples.append({"problem": problems[i]["problem"][:80], "gold": gold, "pred": pred, "tail": text[-100:]})
        else:
            wrong_examples.append({"problem": problems[i]["problem"][:80], "gold": gold, "pred": pred, "tail": text[-150:]})

    print(f"\n{'='*60}")
    print(f"SVAMP BASELINE (max_tokens=1024)")
    print(f"{'='*60}")
    print(f"Accuracy: {correct}/{len(outputs)} ({100*correct/len(outputs):.1f}%)")
    print(f"\nCORRECT examples (first 3):")
    for r in right_examples:
        print(f"  Q: {r['problem']}...")
        print(f"  Gold: {r['gold']}, Extracted: {r['pred']}")
        print(f"  Response: ...{r['tail']}")
        print()
    print(f"WRONG examples (first 10):")
    for w in wrong_examples[:10]:
        print(f"  Q: {w['problem']}...")
        print(f"  Gold: {w['gold']}, Extracted: {w['pred']}")
        print(f"  Response: ...{w['tail']}")
        print()


if __name__ == "__main__":
    main()
