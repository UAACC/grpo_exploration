"""Online GRPO training on GSM8K or MATH using TRL's native GRPOTrainer.

The student model generates its own completions via vLLM during training.
No teacher model or pre-computed rollouts needed.

Usage:
    accelerate launch --config_file configs/accelerate_ddp_4gpu.yaml \
        train.py --dataset_type gsm8k --output_dir ./outputs
    accelerate launch --config_file configs/accelerate_ddp_4gpu.yaml \
        train.py --dataset_type math --output_dir ./outputs
"""

import argparse
import re
from datetime import datetime

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


# ── Constants ────────────────────────────────────────────────────────────
GSM8K_SYSTEM_PROMPT = (
    "Please solve this math problem step by step. "
    "Put your final numerical answer after ####."
)
MATH_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
GSM8K_DATASET = "openai/gsm8k"
MATH_DATASET = "nlile/hendrycks-MATH-benchmark"


# ── Answer extraction ──────────────────────────────────────────────────

def extract_gsm8k_answer(text: str) -> str | None:
    """Extract the final number after #### from model output or ground truth."""
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)\s*$", text, re.MULTILINE)
    if match:
        return match.group(1).replace(",", "")
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    return None


def extract_boxed_answer(text: str) -> str | None:
    """Extract content from the last \\boxed{...} in text, handling nested braces."""
    if "\\boxed{" not in text:
        return None
    idx = text.rfind("\\boxed{")
    i = idx + 7
    brace_count = 1
    content_start = i
    while i < len(text):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
        if brace_count == 0:
            return text[content_start:i]
        i += 1
    return None


# ── Reward functions ───────────────────────────────────────────────────

def gsm8k_reward_func(completions, answer, **kwargs):
    """Reward: 1.0 if predicted answer matches ground truth, else 0.0."""
    rewards = []
    for completion, gold in zip(completions, answer):
        if isinstance(completion, list):
            gen_text = completion[-1]["content"] if completion else ""
        else:
            gen_text = str(completion)

        pred = extract_gsm8k_answer(gen_text)
        gold_answer = extract_gsm8k_answer(gold) if "####" in str(gold) else str(gold).strip()

        if pred is not None and gold_answer is not None:
            try:
                rewards.append(1.0 if float(pred) == float(gold_answer) else 0.0)
            except ValueError:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards


def math_reward_func(completions, answer, **kwargs):
    """Reward: 1.0 if predicted answer matches ground truth via math_verify, else 0.0."""
    from math_verify import parse, verify
    rewards = []
    for completion, gold in zip(completions, answer):
        if isinstance(completion, list):
            gen_text = completion[-1]["content"] if completion else ""
        else:
            gen_text = str(completion)

        try:
            pred = parse(extract_boxed_answer(gen_text))
            gold_parsed = parse(gold)
            if verify(gold_parsed, pred) is True:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


# ── Dataset preparation ────────────────────────────────────────────────

def prepare_dataset(dataset_name: str, dataset_type: str, system_prompt: str,
                    split: str = "train"):
    """Load dataset and format for GRPOTrainer.

    GRPOTrainer expects a dataset with a 'prompt' column containing
    chat-formatted messages.
    """
    if dataset_type == "math":
        dataset = load_dataset(dataset_name)[split]
        question_field = "problem"
    else:
        dataset = load_dataset(dataset_name, "main", split=split)
        question_field = "question"

    def format_example(example):
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example[question_field]},
        ]
        return {"prompt": prompt, "answer": example["answer"]}

    cols_to_remove = [c for c in dataset.column_names if c not in ("answer",)]
    dataset = dataset.map(format_example, remove_columns=cols_to_remove)
    return dataset


# ── Main ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Online GRPO on GSM8K or MATH with LoRA.")
    # Model
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    # Data
    p.add_argument("--dataset_type", type=str, default="gsm8k",
                    choices=["gsm8k", "math"], help="Which dataset to train on.")
    p.add_argument("--dataset", type=str, default=None,
                    help="Override dataset path. Defaults based on dataset_type.")
    # Output
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    # Training hyperparams (matching VERL benchmark)
    p.add_argument("--learning_rate", type=float, default=3e-6)
    p.add_argument("--beta", type=float, default=0.001,
                    help="KL penalty coefficient (VERL uses 0.001).")
    p.add_argument("--num_generations", type=int, default=5,
                    help="Number of completions per prompt for GRPO.")
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--num_train_epochs", type=int, default=15)
    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--max_completion_length", type=int, default=1024)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--logging_steps", type=int, default=10)
    # LoRA (matching VERL: rank=32, alpha=32, all-linear)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--no_lora", action="store_true", help="Disable LoRA.")
    # Generation config
    p.add_argument("--no_vllm", action="store_true",
                    help="Disable vLLM; use native PyTorch generation instead.")
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=-1)
    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--report_to", type=str, default="wandb")
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    # ---- 1. Resolve dataset defaults ------------------------------------
    is_math = args.dataset_type == "math"
    if args.dataset is None:
        args.dataset = MATH_DATASET if is_math else GSM8K_DATASET
    system_prompt = MATH_SYSTEM_PROMPT if is_math else GSM8K_SYSTEM_PROMPT
    reward_func = math_reward_func if is_math else gsm8k_reward_func

    # Adjust defaults for MATH (longer completions)
    if is_math:
        args.max_completion_length = max(args.max_completion_length, 2048)
        args.max_prompt_length = max(args.max_prompt_length, 512)

    # ---- 2. Load dataset ------------------------------------------------
    print(f"Loading dataset: {args.dataset} (type={args.dataset_type})")
    train_dataset = prepare_dataset(args.dataset, args.dataset_type,
                                     system_prompt, split="train")
    print(f"  {len(train_dataset)} training examples")

    # ---- 3. Load model --------------------------------------------------
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 4. LoRA config -------------------------------------------------
    peft_config = None
    if not args.no_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
        )
        print(f"  LoRA: r={args.lora_r}, alpha={args.lora_alpha}, target=all-linear")

    # ---- 5. Training config ---------------------------------------------
    if args.output_dir is None:
        args.output_dir = f"./outputs/online_grpo_{args.dataset_type}_{time_str}"
    if args.run_name is None:
        args.run_name = f"online-grpo-{args.dataset_type}-{args.model.split('/')[-1]}-{time_str}"

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        # Optimizer
        learning_rate=args.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        # GRPO specific
        beta=args.beta,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        # Generation
        use_vllm=not args.no_vllm,
        **({"vllm_mode": "colocate",
            "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization}
           if not args.no_vllm else {}),
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        # Batching
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        # Logging & saving
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to=args.report_to,
        log_on_each_node=False,
        # Precision
        bf16=args.bf16,
        seed=args.seed,
    )

    # ---- 6. Create trainer & train --------------------------------------
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )
    print(f"Beta (KL coeff): {trainer.beta}")
    print(f"Generation backend: {'vLLM colocate' if not args.no_vllm else 'native PyTorch'}")
    print(f"Num generations per prompt: {args.num_generations}")
    print(f"Effective batch size: {args.per_device_train_batch_size} * "
          f"{torch.cuda.device_count()} * {args.gradient_accumulation_steps} = "
          f"{args.per_device_train_batch_size * torch.cuda.device_count() * args.gradient_accumulation_steps}")

    # Log config to wandb
    if args.report_to == "wandb" and trainer.accelerator.is_main_process:
        import wandb
        if wandb.run is not None:
            wandb.config.update({
                "model": args.model,
                "dataset": args.dataset,
                "dataset_type": args.dataset_type,
                "num_train_examples": len(train_dataset),
                "beta": args.beta,
                "num_generations": args.num_generations,
                "temperature": args.temperature,
                "lora_r": args.lora_r if not args.no_lora else None,
                "lora_alpha": args.lora_alpha if not args.no_lora else None,
                "lora_enabled": not args.no_lora,
                "max_prompt_length": args.max_prompt_length,
                "max_completion_length": args.max_completion_length,
                "use_vllm": not args.no_vllm,
            })

    print("Starting online GRPO training...")

    resume = args.resume_from_checkpoint
    if resume == "latest":
        resume = True
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.output_dir)
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
