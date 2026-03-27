"""Quick check: what does the base model output on GSM8K with our system prompt?"""
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

model_path = "/scratch/mrli/models/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)

problems = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
    "A store sells apples for $2 each. If you buy 5 apples, how much do you pay?",
]

sys_prompt = "Please solve this math problem step by step. Put your final numerical answer after ####."

prompts = []
for p in problems:
    chat = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": p},
    ]
    prompts.append(tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))

llm = LLM(model=model_path, max_model_len=2048, gpu_memory_utilization=0.5)
outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=512))

for i, out in enumerate(outputs):
    print(f"\n{'='*60}")
    print(f"Problem: {problems[i][:80]}...")
    print(f"{'='*60}")
    print(out.outputs[0].text)
    print(f"{'='*60}")
