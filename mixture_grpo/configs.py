"""Shared constants, prompts, and utilities for mixture GRPO."""

import re

# -- MATH dataset --
MATH_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
MATH_DATASET = "nlile/hendrycks-MATH-benchmark"

# -- GSM8K dataset --
GSM8K_SYSTEM_PROMPT = (
    "Please solve this math problem step by step. "
    "Put your final numerical answer after ####."
)
GSM8K_DATASET = "openai/gsm8k"

# Default to GSM8K
SYSTEM_PROMPT = GSM8K_SYSTEM_PROMPT
DEFAULT_DATASET = GSM8K_DATASET

DEFAULT_TARGET_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_BEHAVIOR_MODEL = "Qwen/Qwen2.5-7B-Instruct"

DEFAULT_LORA_CONFIG = dict(
    r=32,
    lora_alpha=32,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
)


def extract_boxed_answer(text: str) -> str | None:
    """Extract content from the last \\boxed{...} in *text*, handling nested braces."""
    if "\\boxed{" not in text:
        return None
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    i = idx + 7  # len("\\boxed{")
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


def extract_gsm8k_answer(text: str) -> str | None:
    """Extract the final numerical answer from GSM8K model output or ground truth."""
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)\s*$", text, re.MULTILINE)
    if match:
        return match.group(1).replace(",", "")
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        content = boxed[-1].strip()
        num_match = re.search(r"[-]?[\d,]+(?:\.\d+)?", content)
        if num_match:
            return num_match.group(0).replace(",", "")
    numbers = re.findall(r"[-]?[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None
