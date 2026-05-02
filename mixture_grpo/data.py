"""Data loading for teacher rollouts (offline component of mixture GRPO)."""

import json
from configs import SYSTEM_PROMPT, GSM8K_SYSTEM_PROMPT, MATH_SYSTEM_PROMPT, extract_gsm8k_answer, extract_boxed_answer


def load_teacher_rollouts(jsonl_path: str, vocab_size: int | None = None,
                          dataset_type: str = "gsm8k") -> dict:
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
