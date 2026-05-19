"""Unified dataset registry for all math reasoning benchmarks.

Single source of truth for dataset paths, field names, system prompts,
answer extraction, and reward functions. All eval/training scripts import
from here instead of hardcoding per-dataset logic.

Usage:
    from shared.datasets import get_dataset_config, load_eval_data, check_answer

    cfg = get_dataset_config("math")
    problems = load_eval_data(cfg, split="test")
    correct = check_answer(cfg, model_output_text, gold_answer)
"""

import re
from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

MATH_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

GSM8K_SYSTEM_PROMPT = (
    "Please solve this math problem step by step. "
    "Put your final numerical answer after ####."
)

# SVAMP and ASDiv use the same prompt as GSM8K (all numeric-answer math)
NUMERIC_SYSTEM_PROMPT = GSM8K_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_boxed_answer(text: str) -> str | None:
    """Extract content from the last \\boxed{...} in text, handling nested braces."""
    if "\\boxed{" not in text:
        return None
    idx = text.rfind("\\boxed{")
    i = idx + len("\\boxed{")
    depth = 1
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return text[idx + len("\\boxed{"):i - 1]
    return None


def extract_numeric_answer(text: str) -> str | None:
    """Extract a numeric answer from text. Tries multiple formats:
    1. After #### delimiter (GSM8K style)
    2. Inside \\boxed{} (MATH style — the 7B Math teacher uses this even on GSM8K-like tasks)
    3. Last number in the text (fallback)
    """
    # Try #### first
    if "####" in text:
        after = text.split("####")[-1].strip()
        after = after.replace(",", "").strip()
        match = re.search(r"-?[\d.]+", after)
        if match:
            return match.group()
    # Try \boxed{}
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        boxed = boxed.replace(",", "").strip()
        match = re.search(r"-?[\d.]+", boxed)
        if match:
            return match.group()
    # Fallback: last number in the text
    numbers = re.findall(r"-?[\d.]+", text)
    return numbers[-1] if numbers else None


# ---------------------------------------------------------------------------
# Answer checking
# ---------------------------------------------------------------------------

def check_math(pred_text: str, gold_answer: str) -> bool:
    """MATH-style: extract boxed answer, verify with math_verify."""
    from math_verify import parse, verify
    try:
        pred = parse(extract_boxed_answer(pred_text))
        gold = parse(gold_answer)
        return verify(gold, pred) is True
    except Exception:
        return False


def check_numeric(pred_text: str, gold_answer: str) -> bool:
    """GSM8K/SVAMP-style: extract numeric answer, compare floats."""
    pred = extract_numeric_answer(pred_text)
    gold = extract_numeric_answer(gold_answer)
    if pred is not None and gold is not None:
        try:
            return float(pred) == float(gold)
        except ValueError:
            pass
    return False


def reward_math(pred_text: str, gold_answer: str) -> float:
    """MATH reward: 2.0 for correct, 0.0 otherwise."""
    return 2.0 if check_math(pred_text, gold_answer) else 0.0


def reward_numeric(pred_text: str, gold_answer: str) -> float:
    """GSM8K/SVAMP/ASDiv reward: 1.0 for correct, 0.0 otherwise."""
    return 1.0 if check_numeric(pred_text, gold_answer) else 0.0


# ---------------------------------------------------------------------------
# Dataset config
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    name: str
    hf_path: str
    hf_config: str | None        # HF dataset config name (e.g., "main" for GSM8K)
    split_test: str               # test split name
    split_train: str              # train split name
    question_field: str           # field name for the problem text
    answer_field: str             # field name for the gold answer
    system_prompt: str
    extract_answer: Callable      # function to extract answer from model output
    check_answer: Callable        # function to check correctness
    reward_func: Callable         # function returning float reward
    max_tokens: int               # max generation tokens
    max_model_len: int            # max model context length
    num_test: int                 # expected test set size (for reference)
    difficulty: str               # "easy", "medium", "hard" (relative)
    build_question: Callable | None = None  # custom question builder (if fields need combining)


DATASET_REGISTRY: dict[str, DatasetConfig] = {}


def _register(cfg: DatasetConfig):
    DATASET_REGISTRY[cfg.name] = cfg


# --- MATH (our current main benchmark) ---
_register(DatasetConfig(
    name="math",
    hf_path="nlile/hendrycks-MATH-benchmark",
    hf_config=None,
    split_test="test",
    split_train="train",
    question_field="problem",
    answer_field="answer",
    system_prompt=MATH_SYSTEM_PROMPT,
    extract_answer=extract_boxed_answer,
    check_answer=check_math,
    reward_func=reward_math,
    max_tokens=2048,
    max_model_len=3072,
    num_test=500,
    difficulty="hard",
))

# --- GSM8K ---
_register(DatasetConfig(
    name="gsm8k",
    hf_path="openai/gsm8k",
    hf_config="main",
    split_test="test",
    split_train="train",
    question_field="question",
    answer_field="answer",
    system_prompt=GSM8K_SYSTEM_PROMPT,
    extract_answer=extract_numeric_answer,
    check_answer=check_numeric,
    reward_func=reward_numeric,
    max_tokens=1024,
    max_model_len=2048,
    num_test=1319,
    difficulty="easy",
))

# --- SVAMP (easy, similar to GSM8K) ---
def _svamp_build_question(item: dict) -> str:
    """SVAMP has separate Body and Question fields that need combining."""
    return item["Body"].strip() + " " + item["Question"].strip()

_register(DatasetConfig(
    name="svamp",
    hf_path="ChilleD/SVAMP",
    hf_config=None,
    split_test="test",
    split_train="train",
    question_field="question_concat",   # pre-concatenated field
    answer_field="Answer",
    system_prompt=NUMERIC_SYSTEM_PROMPT,
    extract_answer=extract_numeric_answer,
    check_answer=check_numeric,
    reward_func=reward_numeric,
    max_tokens=512,
    max_model_len=1024,
    num_test=300,
    difficulty="easy",
    build_question=_svamp_build_question,
))

# --- ASDiv (easy-medium, elementary curriculum math, manual 80/20 split) ---
def _asdiv_build_question(item: dict) -> str:
    """ASDiv has separate body and question fields."""
    return item["body"].strip() + " " + item["question"].strip()

def _asdiv_clean_answer(answer: str) -> str:
    """Strip units from ASDiv answers: '9 (apples)' -> '9'."""
    import re
    match = re.match(r"(-?[\d.]+)", answer.strip())
    return match.group(1) if match else answer.strip()

_register(DatasetConfig(
    name="asdiv",
    hf_path="EleutherAI/asdiv",
    hf_config=None,
    split_test="validation",   # only has validation; we split it manually
    split_train="validation",  # same — load_eval_data handles the split
    question_field="body",     # used by build_question
    answer_field="answer",
    system_prompt=GSM8K_SYSTEM_PROMPT,
    extract_answer=extract_numeric_answer,
    check_answer=check_numeric,
    reward_func=reward_numeric,
    max_tokens=512,
    max_model_len=1024,
    num_test=461,              # 2305 * 0.2 ≈ 461
    difficulty="easy-medium",
    build_question=_asdiv_build_question,
))


# --- AceReason-Math (NVIDIA, MATH-style boxed answers, train-only, manual 80/20 split) ---
_register(DatasetConfig(
    name="acereason",
    hf_path="nvidia/AceReason-Math",
    hf_config=None,
    split_test="train",        # only has train; we split it manually
    split_train="train",       # same — load_eval_data handles the split
    question_field="problem",
    answer_field="answer",
    system_prompt=MATH_SYSTEM_PROMPT,
    extract_answer=extract_boxed_answer,
    check_answer=check_math,
    reward_func=reward_math,
    max_tokens=2048,
    max_model_len=3072,
    num_test=9917,             # 49585 * 0.2 ≈ 9917
    difficulty="hard",
))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_dataset_config(name: str) -> DatasetConfig:
    """Get dataset config by name. Raises KeyError if not found."""
    if name not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")
    return DATASET_REGISTRY[name]


def list_datasets() -> list[str]:
    """Return list of registered dataset names."""
    return sorted(DATASET_REGISTRY.keys())


def load_eval_data(cfg: DatasetConfig, split: str | None = None,
                   max_problems: int | None = None,
                   manual_split: str | None = None,
                   manual_split_seed: int = 42) -> list[dict]:
    """Load evaluation data as a list of {"problem": ..., "answer": ...} dicts.

    Uses the dataset config to handle field name differences across datasets.

    Args:
        cfg: dataset config from the registry
        split: HF split to load (default: cfg.split_test)
        max_problems: limit number of problems (for testing)
        manual_split: for datasets without proper train/test splits (e.g., ASDiv),
            pass "train" or "test" to get an 80/20 deterministic split of the
            loaded data. The split is shuffled with manual_split_seed for
            reproducibility.
        manual_split_seed: seed for the manual split shuffle (default: 42)
    """
    import glob as _glob
    import os
    from datasets import load_dataset, Dataset

    split = split or cfg.split_test
    try:
        if cfg.hf_config:
            ds = load_dataset(cfg.hf_path, cfg.hf_config, split=split)
        else:
            ds = load_dataset(cfg.hf_path, split=split)
    except ConnectionError:
        # Newer Hub datasets (no custom .py script) require a Hub API call even
        # in offline mode to discover data files. Fall back to loading the
        # cached arrow file directly from HF_DATASETS_CACHE.
        cache_dir = os.environ.get("HF_DATASETS_CACHE", "")
        matches = [
            f for f in _glob.glob(os.path.join(cache_dir, "**", "*.arrow"), recursive=True)
            if f"-{split}." in os.path.basename(f)
        ]
        if not matches:
            raise
        ds = Dataset.from_file(matches[0])

    problems = []
    for i, item in enumerate(ds):
        if cfg.build_question:
            question = cfg.build_question(item)
        else:
            question = item[cfg.question_field]

        answer = str(item[cfg.answer_field])

        # Clean answer if the dataset needs it (e.g., ASDiv "9 (apples)" -> "9")
        if cfg.name == "asdiv":
            answer = _asdiv_clean_answer(answer)

        problems.append({
            "problem": question,
            "answer": answer,
        })

    # Manual train/test split for datasets without proper splits
    if manual_split in ("train", "test"):
        import random
        rng = random.Random(manual_split_seed)
        indices = list(range(len(problems)))
        rng.shuffle(indices)
        split_point = int(len(indices) * 0.8)
        if manual_split == "train":
            indices = indices[:split_point]
        else:
            indices = indices[split_point:]
        problems = [problems[i] for i in indices]

    if max_problems:
        problems = problems[:max_problems]

    return problems
