"""DG-offline teacher-agnostic data loader (Path A).

Loads pre-collected teacher rollouts as TEXT and re-tokenizes each completion
under the STUDENT tokenizer. This is the conceptually correct DG-offline data
path: the student computes its surprisal on whatever the teacher emitted, with
no assumption that the teacher and student share a tokenizer.

This module is fully self-contained. It does NOT import from
`offline_grpo/data.py` or `offline_grpo/configs.py`. The small reward-parsing
helpers (`extract_boxed_answer`, `extract_gsm8k_answer`) are inlined below so
the entire reward + advantage pipeline lives in one file.

Inputs
------
A rollout JSONL where each line has at least:
    question_id, original_problem, ground_truth_answer, system_prompt,
    dataset_type (optional), runs[] = [{run_id, response, ...}, ...]

This loader uses: question_id, original_problem, ground_truth_answer,
system_prompt, dataset_type, runs[].run_id, runs[].response, and either
runs[].extracted_answer or the older runs[].boxed_answer.

Other fields commonly present in rollout JSONL (logprobs, completion_ids) are
intentionally IGNORED. Student-vocab completion_ids are produced by
re-tokenizing `response` at load time, so any teacher token IDs on disk play
no role.

Output
------
A flat list of per-completion records carrying student-vocab `completion_ids`
plus `reward` and `advantage` (added by `compute_rewards_and_advantages`).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

from datasets import Dataset


# ---------------------------------------------------------------------------
# 0. Inlined constants and reward parsers (kept here for self-containment)
# ---------------------------------------------------------------------------

MATH_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def extract_boxed_answer(text: str) -> str | None:
    """Return content of the LAST \\boxed{...} in *text*, handling nested braces."""
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
    """Extract a final numerical answer: #### N -> last \\boxed{N} -> last number."""
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


# ---------------------------------------------------------------------------
# 1. Load rollouts (Path A: re-tokenize teacher response under STUDENT)
# ---------------------------------------------------------------------------

def load_rollouts_text(jsonl_path: str, student_tokenizer) -> list[dict]:
    """Load a rollout JSONL and re-tokenize each completion under *student_tokenizer*.

    Each output record contains:
        question_id, run_id, problem, ground_truth, system_prompt,
        dataset_type, response, extracted_answer, completion_ids

    `completion_ids` are STUDENT-vocab IDs produced by tokenizing `response`
    with *student_tokenizer* using `add_special_tokens=False`. Any teacher
    `completion_ids` and `logprobs` on disk are ignored.
    """
    records: list[dict] = []
    n_empty = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            qid = item["question_id"]
            dataset_type = item.get("dataset_type", "math")
            for run in item["runs"]:
                response_text = run["response"]
                # Re-tokenize the teacher's text under the student tokenizer.
                # No special tokens — the prompt side carries those at training time.
                student_ids = student_tokenizer(
                    response_text, add_special_tokens=False
                )["input_ids"]
                if len(student_ids) == 0:
                    n_empty += 1
                    # Keep the record so num_generations grouping stays intact.
                    # The trainer pads short rows; an empty row becomes all-pad
                    # with zero mask and contributes nothing to the loss.

                records.append({
                    "question_id": qid,
                    "run_id": run["run_id"],
                    "problem": item["original_problem"],
                    "ground_truth": item["ground_truth_answer"],
                    "system_prompt": item.get("system_prompt", MATH_SYSTEM_PROMPT),
                    "dataset_type": dataset_type,
                    "response": response_text,
                    # Newer rollouts use "extracted_answer"; older ones use "boxed_answer".
                    "extracted_answer": run.get("extracted_answer") or run.get("boxed_answer"),
                    "completion_ids": student_ids,
                })
    if n_empty > 0:
        print(
            f"  Warning: {n_empty} completions re-tokenized to empty under the student "
            f"tokenizer (kept as zero-length rows)"
        )
    return records


# ---------------------------------------------------------------------------
# 2. Compute rewards and GRPO advantages (tokenizer-agnostic)
# ---------------------------------------------------------------------------

def _compute_correctness_math(extracted: str | None, ground_truth: str, *,
                              question: str | None = None,
                              response: str | None = None) -> float:
    """Return 2.0 if MATH-style answer matches ground truth, else 0.0.

    Uses DeepSeek's port (`is_equiv_multi` from Math_Verifier) when the
    full *question* and *response* are available — gives multi-candidate
    extraction (every \\boxed{} in the response is tried). Falls back to
    single-candidate `is_equiv` when only the pre-extracted answer is on hand
    (e.g. older rollouts with extracted_answer already populated).
    """
    import os as _os, sys as _sys
    _root = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    from Math_Verifier import is_equiv, is_equiv_multi

    if question is not None and response is not None:
        return 2.0 if is_equiv_multi(question, response, ground_truth) else 0.0
    return 2.0 if is_equiv(extracted, ground_truth) else 0.0


def _compute_correctness_gsm8k(extracted: str | None, ground_truth: str) -> float:
    """Return 1.0 if extracted numeric answer matches ground truth, else 0.0."""
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
    """Add `reward` and group-normalized `advantage` to each record. Mutates in place."""
    default_ds = records[0].get("dataset_type", "math") if records else "math"
    for rec in records:
        ds = rec.get("dataset_type", default_ds)
        extracted = rec["extracted_answer"]
        if ds in ("gsm8k", "svamp", "asdiv"):
            if extracted is None:
                extracted = extract_gsm8k_answer(rec["response"])
            rec["reward"] = _compute_correctness_gsm8k(extracted, rec["ground_truth"])
            if rec["reward"] == 0.0:
                # Fallback: the 7B math teacher also emits \boxed{} on numeric tasks.
                boxed = extract_boxed_answer(rec["response"])
                if boxed is not None:
                    rec["reward"] = _compute_correctness_gsm8k(boxed, rec["ground_truth"])
        else:
            if extracted is None:
                extracted = extract_boxed_answer(rec["response"])
            rec["reward"] = _compute_correctness_math(
                extracted,
                rec["ground_truth"],
                question=rec.get("problem"),
                response=rec.get("response"),
            )

    groups: dict[Any, list[dict]] = defaultdict(list)
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
# 3. Build a HuggingFace Dataset for TRL
# ---------------------------------------------------------------------------

def build_training_dataset(records: list[dict]) -> Dataset:
    """Sorted by (question_id, run_id) so consecutive `num_generations` rows
    share a prompt — matching TRL's grouping expectation."""
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
# 4. Build offline lookup keyed by (question_id, run_id)
# ---------------------------------------------------------------------------

def build_offline_lookup(records: list[dict]) -> dict[tuple, dict]:
    """Lookup for the trainer.

    Note: behavior_logprobs are intentionally NOT included. DG-offline does
    not use them, and omitting them keeps the Path A vs Path B distinction
    obvious at the call site.
    """
    lookup: dict[tuple, dict] = {}
    for rec in records:
        lookup[(rec["question_id"], rec["run_id"])] = {
            "completion_ids": rec["completion_ids"],
            "advantage": rec["advantage"],
            "reward": rec["reward"],
            "response": rec["response"],
        }
    return lookup
