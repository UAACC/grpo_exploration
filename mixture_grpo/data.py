"""Data loading for teacher rollouts (offline component of mixture GRPO).

DEPRECATED: this module loads teacher-vocab `completion_ids` and feeds them
through to the student forward pass (Path B). It is unsafe for any teacher
whose tokenizer disagrees with the student's at any in-range ID — concretely,
it silently corrupts training on R1-Distill-style teachers (which emit
`<think>` at 151648 / `</think>` at 151649, IDs that mean different strings
in the student's vocab). Mixture A/B and DG-Mixture are deprecated active
methods (per 2026-04-18 decision); for any new training using teacher
rollouts, route through `shared/prepare_cleaned_og_rollouts.py` first to
align IDs with the student tokenizer.

The `assert_completion_ids_safe_for_student` check below is the consumer-side
gate: it refuses to load rollouts whose `completion_ids` don't round-trip
under the student tokenizer, so future use catches the bomb at load time
rather than during training.
"""

import json
from configs import SYSTEM_PROMPT, GSM8K_SYSTEM_PROMPT, MATH_SYSTEM_PROMPT, extract_gsm8k_answer, extract_boxed_answer


def _normalize(s: str) -> str:
    return "".join(s.split())


def assert_completion_ids_safe_for_student(
    flat_records: list[dict],
    student_tokenizer,
    sample_n: int = 200,
    max_invalid_fraction: float = 0.01,
) -> None:
    """Path-A safety check. Same logic as offline_grpo/data.py — see that file.

    Decodes a sample of `completion_ids` under the student tokenizer and
    compares to the rollout's `response` text. Refuses to proceed if too
    many fail the round-trip — those IDs are NOT student-vocab and would
    silently corrupt the student forward pass.
    """
    if not flat_records:
        return
    import random
    rng = random.Random(0)
    sample = rng.sample(flat_records, min(sample_n, len(flat_records)))
    invalid = []
    for rec in sample:
        cids = rec.get("completion_ids")
        resp = rec.get("response", "")
        if not cids or not resp:
            continue
        decoded = student_tokenizer.decode(cids, skip_special_tokens=True)
        ndec, nresp = _normalize(decoded), _normalize(resp)
        # Safe iff decoded equals or is a prefix of response — see the
        # offline_grpo/data.py copy of this check for the rationale.
        if ndec != nresp and not nresp.startswith(ndec):
            invalid.append((decoded[:200], resp[:200]))
    threshold = max(1, int(max_invalid_fraction * len(sample)))
    if len(invalid) > threshold:
        first = invalid[0]
        raise RuntimeError(
            f"Mixture-grpo rollout `completion_ids` are NOT student-vocab-safe: "
            f"{len(invalid)}/{len(sample)} sampled completions failed the "
            f"student-decode round-trip. Refusing to feed teacher IDs through "
            f"the student forward pass (would silently corrupt training).\n"
            f"  Decoded under student (head): {first[0]!r}\n"
            f"  Original response (head):    {first[1]!r}\n"
            f"Run the rollout file through `shared/prepare_cleaned_og_rollouts.py` "
            f"or use a teacher whose tokenizer agrees with the student's."
        )


def load_teacher_rollouts(jsonl_path: str, vocab_size: int | None = None,
                          dataset_type: str = "gsm8k",
                          student_tokenizer=None) -> dict:
    """Load teacher rollouts and return a dict keyed by question_id.

    Args:
        jsonl_path: Path to teacher rollout JSONL file.
        vocab_size: If provided, truncate completions at first OOV token.
        dataset_type: "gsm8k" or "math" — determines reward computation.

    Returns:
        dict[int, dict]: keyed by question_id, each value has keys:
            runs (list of dicts), problem, ground_truth, system_prompt.
    """
    default_prompt = MATH_SYSTEM_PROMPT if dataset_type == "math" else GSM8K_SYSTEM_PROMPT
    teacher_data = {}
    truncated = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            qid = item["question_id"]
            ground_truth = item["ground_truth_answer"]
            runs = []
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

                # Compute reward based on dataset type
                if dataset_type == "math":
                    extracted = run.get("extracted_answer") or run.get("boxed_answer") or extract_boxed_answer(run.get("response", ""))
                    reward = compute_math_correctness(extracted, ground_truth)
                else:
                    extracted = run.get("extracted_answer") or extract_gsm8k_answer(run.get("response", ""))
                    reward = compute_gsm8k_correctness(extracted, ground_truth)

                runs.append({
                    "completion_ids": cids,
                    "behavior_logprobs": lps,
                    "reward": reward,
                    "response": run.get("response", ""),
                })
            teacher_data[qid] = {
                "runs": runs,
                "problem": item["original_problem"],
                "ground_truth": ground_truth,
                "system_prompt": item.get("system_prompt", default_prompt),
            }
    if truncated > 0:
        print(f"  Warning: {truncated} completions truncated at out-of-vocab tokens (vocab_size={vocab_size})")
    if student_tokenizer is not None:
        # Flatten runs across questions so the safety check sees the full set.
        flat = [r for q in teacher_data.values() for r in q["runs"]]
        assert_completion_ids_safe_for_student(flat, student_tokenizer)
        print("  Path-A check passed: completion_ids round-trip cleanly under the student tokenizer.")
    return teacher_data


def compute_math_correctness(extracted: str | None, ground_truth: str) -> float:
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


def compute_gsm8k_correctness(extracted: str | None, ground_truth: str) -> float:
    """Return 1.0 if predicted answer matches ground truth, 0.0 otherwise.

    Handles GSM8K format where ground truth may contain #### prefix.
    """
    if extracted is None:
        return 0.0
    gold = extract_gsm8k_answer(ground_truth) if "####" in ground_truth else ground_truth.strip()
    if gold is None:
        return 0.0
    try:
        if float(extracted) == float(gold):
            return 1.0
    except ValueError:
        pass
    return 0.0
