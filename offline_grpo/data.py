"""Data loading, reward computation, and advantage normalization for offline GRPO."""

import json
from collections import defaultdict

from datasets import Dataset

from configs import MATH_SYSTEM_PROMPT, extract_boxed_answer, extract_gsm8k_answer


# ---------------------------------------------------------------------------
# 1. Load rollouts
# ---------------------------------------------------------------------------

def load_rollouts(jsonl_path: str, vocab_size: int | None = None) -> list[dict]:
    """Load rollout JSONL into a flat list of per-completion records.

    Each record contains:
        question_id, run_id, problem, ground_truth, system_prompt,
        dataset_type, response, extracted_answer, behavior_logprobs, completion_ids

    If ``vocab_size`` is provided, completions are truncated at the first
    token ID that exceeds the student model's vocabulary (can happen when
    teacher and student have different vocab sizes).
    """
    records = []
    truncated = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            qid = item["question_id"]
            dataset_type = item.get("dataset_type", "math")
            for run in item["runs"]:
                cids = run["completion_ids"]
                lps = run["logprobs"]

                # Truncate at first out-of-vocab token
                if vocab_size is not None:
                    for idx, tid in enumerate(cids):
                        if tid >= vocab_size:
                            cids = cids[:idx]
                            lps = lps[:idx]
                            truncated += 1
                            break

                records.append({
                    "question_id": qid,
                    "run_id": run["run_id"],
                    "problem": item["original_problem"],
                    "ground_truth": item["ground_truth_answer"],
                    "system_prompt": item.get("system_prompt", MATH_SYSTEM_PROMPT),
                    "dataset_type": dataset_type,
                    "response": run["response"],
                    # Support both old ("boxed_answer") and new ("extracted_answer") field names
                    "extracted_answer": run.get("extracted_answer") or run.get("boxed_answer"),
                    "behavior_logprobs": lps,
                    "completion_ids": cids,
                })
    if truncated > 0:
        print(f"  Warning: {truncated} completions truncated at out-of-vocab tokens (vocab_size={vocab_size})")
    return records


# ---------------------------------------------------------------------------
# 2. Compute rewards and GRPO advantages
# ---------------------------------------------------------------------------

def _compute_correctness_math(extracted: str | None, ground_truth: str) -> float:
    """Return 2.0 for correct, 0.0 otherwise, using math_verify (MATH dataset)."""
    from math_verify import parse, verify
    try:
        if verify(parse(extracted), parse(ground_truth)) is True:
            return 2.0
    except Exception:
        pass
    return 0.0


def _compute_correctness_gsm8k(extracted: str | None, ground_truth: str) -> float:
    """Return 1.0 for correct, 0.0 otherwise (GSM8K dataset)."""
    if extracted is None:
        return 0.0
    gold = extract_gsm8k_answer(ground_truth)
    if gold is None:
        return 0.0
    try:
        if float(extracted) == float(gold):
            return 1.0
    except ValueError:
        pass
    return 0.0


def compute_rewards_and_advantages(records: list[dict], eps: float = 1e-4) -> list[dict]:
    """Compute per-completion rewards and GRPO-style group-normalized advantages.

    Mutates *records* in-place (adds ``reward`` and ``advantage`` keys) and
    returns the same list.
    """
    # Compute rewards (auto-detect dataset type from first record)
    dataset_type = records[0].get("dataset_type", "math") if records else "math"
    for rec in records:
        ds = rec.get("dataset_type", dataset_type)
        extracted = rec["extracted_answer"]
        if ds in ("gsm8k", "svamp", "asdiv"):
            # Try stored extracted_answer, then try boxed format
            # (the 7B Math teacher outputs \boxed{} even on numeric tasks)
            if extracted is None:
                extracted = extract_gsm8k_answer(rec["response"])
            rec["reward"] = _compute_correctness_gsm8k(extracted, rec["ground_truth"])
            if rec["reward"] == 0.0:
                boxed = extract_boxed_answer(rec["response"])
                if boxed is not None:
                    rec["reward"] = _compute_correctness_gsm8k(boxed, rec["ground_truth"])
        else:
            if extracted is None:
                extracted = extract_boxed_answer(rec["response"])
            rec["reward"] = _compute_correctness_math(extracted, rec["ground_truth"])

    # Group by question_id for advantage normalization
    groups: dict[int, list[dict]] = defaultdict(list)
    for rec in records:
        groups[rec["question_id"]].append(rec)

    for group in groups.values():
        rewards = [r["reward"] for r in group]
        mean_r = sum(rewards) / len(rewards)
        std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5
        for rec in group:
            rec["advantage"] = (rec["reward"] - mean_r) / (std_r + eps)

    return records


# ---------------------------------------------------------------------------
# 3. Build HuggingFace Dataset for TRL
# ---------------------------------------------------------------------------

def build_training_dataset(records: list[dict]) -> Dataset:
    """Build an HF Dataset with ``prompt`` (chat format) and ``answer`` columns.

    Sorted by (question_id, run_id) so that groups of completions for the same
    question are contiguous — matching TRL's expectation that consecutive
    ``num_generations`` rows belong to the same prompt.
    """
    records = sorted(records, key=lambda r: (r["question_id"], r["run_id"]))

    prompts, answers, qids = [], [], []
    for rec in records:
        prompts.append([
            {"role": "system", "content": rec["system_prompt"]},
            {"role": "user", "content": rec["problem"]},
        ])
        answers.append(rec["ground_truth"])
        qids.append(rec["question_id"])

    return Dataset.from_dict({
        "prompt": prompts,
        "answer": answers,
        "question_id": qids,
    })


# ---------------------------------------------------------------------------
# 4. Build offline lookup (keyed by (qid, rid))
# ---------------------------------------------------------------------------

def build_offline_lookup(records: list[dict]) -> dict[tuple[int, int], dict]:
    """Return a dict keyed by ``(question_id, run_id)`` for the trainer to look up
    pre-computed behavior logprobs, completion_ids, and advantages."""
    lookup = {}
    for rec in records:
        lookup[(rec["question_id"], rec["run_id"])] = {
            "behavior_logprobs": rec["behavior_logprobs"],
            "completion_ids": rec["completion_ids"],
            "advantage": rec["advantage"],
            "reward": rec["reward"],
            "response": rec["response"],
        }
    return lookup
