"""Data loading, reward computation, and advantage normalization for offline GRPO."""

import json
from collections import defaultdict

from datasets import Dataset

from configs import MATH_SYSTEM_PROMPT, extract_boxed_answer, extract_gsm8k_answer


# ---------------------------------------------------------------------------
# Consumer-side Path-A safety check.
#
# Offline-GRPO's IS ratio depends on `completion_ids` being aligned with
# whatever tokenization the student forward-passes on. Direct re-use of
# teacher-vocab IDs is silently corrupt when teacher and student tokenizers
# diverge at any ID (e.g., R1's `<think>` (151648) / `</think>` (151649)
# colliding with unrelated student-vocab special tokens).
#
# The check below decodes each rollout's stored `completion_ids` under the
# *student* tokenizer and compares the result to the rollout's `response`
# text. If they round-trip cleanly, those IDs are safe to feed to the
# student. If not, refuse to load — caller should run the rollout file
# through `shared/prepare_cleaned_og_rollouts.py` (or use a teacher whose
# tokenizer agrees with the student's).
#
# Tolerance: BPE round-trip can introduce trivial whitespace/special-token
# differences; we strip whitespace before comparison. A handful of mismatches
# is allowed (default 1%) before raising — set MAX_INVALID_FRACTION higher
# to be more permissive.
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    return "".join(s.split())


def assert_completion_ids_safe_for_student(
    records: list[dict],
    student_tokenizer,
    sample_n: int = 200,
    max_invalid_fraction: float = 0.01,
) -> None:
    """Fail loud if rollout `completion_ids` aren't student-vocab-safe.

    Compares `student_tokenizer.decode(completion_ids)` against `response`
    on a random sample. Passes when the decoded text equals the response OR
    is a prefix of it (the OOV-truncation case in legacy loaders only chops
    from the tail; the *bomb* is when `decoded` is unrelated text, which
    happens when `completion_ids` are teacher-vocab IDs the student doesn't
    have at the same string mapping). See docs/eval_methodology.md.
    """
    if not records:
        return
    import random
    rng = random.Random(0)
    sample = rng.sample(records, min(sample_n, len(records)))
    invalid = []
    for rec in sample:
        cids = rec.get("completion_ids")
        resp = rec.get("response", "")
        if not cids or not resp:
            continue
        decoded = student_tokenizer.decode(cids, skip_special_tokens=True)
        ndec, nresp = _normalize(decoded), _normalize(resp)
        # Safe iff decoded equals or is a prefix of response (tolerates
        # legacy OOV-truncation that cuts only from the tail).
        if ndec != nresp and not nresp.startswith(ndec):
            invalid.append((rec.get("question_id"), rec.get("run_id"),
                            decoded[:200], resp[:200]))
    threshold = max(1, int(max_invalid_fraction * len(sample)))
    if len(invalid) > threshold:
        first = invalid[0]
        raise RuntimeError(
            f"Rollout `completion_ids` are NOT student-vocab-safe: "
            f"{len(invalid)}/{len(sample)} sampled completions failed the "
            f"student-decode round-trip. Feeding these to the student forward "
            f"pass would silently corrupt training (Path-B bomb).\n"
            f"  First mismatch: question_id={first[0]}, run_id={first[1]}\n"
            f"  Decoded under student (head): {first[2]!r}\n"
            f"  Original response (head):    {first[3]!r}\n"
            f"Run the rollout file through `shared/prepare_cleaned_og_rollouts.py` "
            f"to produce a student-aligned version before training, OR use a teacher "
            f"whose tokenizer agrees with the student's at every emitted ID."
        )


# ---------------------------------------------------------------------------
# 1. Load rollouts
# ---------------------------------------------------------------------------

def load_rollouts(
    jsonl_path: str,
    vocab_size: int | None = None,
    student_tokenizer=None,
) -> list[dict]:
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
    if student_tokenizer is not None:
        assert_completion_ids_safe_for_student(records, student_tokenizer)
        print("  Path-A check passed: completion_ids round-trip cleanly under the student tokenizer.")
    return records


# ---------------------------------------------------------------------------
# 2. Compute rewards and GRPO advantages
# ---------------------------------------------------------------------------

def _compute_correctness_math(extracted: str | None, ground_truth: str) -> float:
    """Return 2.0 for correct, 0.0 otherwise, via Math_Verifier (DeepSeek-Math port).

    Uses single-candidate `is_equiv` since callers pass a pre-extracted answer.
    See docs/eval_methodology.md.
    """
    import os as _os, sys as _sys
    _root = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    from Math_Verifier import is_equiv
    try:
        return 2.0 if is_equiv(extracted, ground_truth) else 0.0
    except Exception:
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
