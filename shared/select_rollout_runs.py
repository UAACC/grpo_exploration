"""Select a fixed number of runs per rollout record.

For MATH rollouts this uses the same answer extraction path as evaluation:
``Math_Verifier.extract_math_answer(question, response, task="math")``. A run
is considered interpretable when that extractor returns at least one non-empty
candidate answer.

Selection policy for the common 5 -> 4 case:
  * all runs interpretable: keep the shortest 4;
  * exactly one run uninterpretable: drop it;
  * more than one run uninterpretable: keep interpretable runs first, then fill
    with the shortest uninterpretable runs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser(description="Select rollout runs per question.")
    p.add_argument("--input_path", required=True, help="Input rollout JSONL.")
    p.add_argument("--output_path", required=True, help="Output rollout JSONL.")
    p.add_argument("--keep_runs", type=int, default=4,
                   help="Number of runs to keep per problem.")
    p.add_argument("--length_metric", choices=["tokens", "chars"], default="tokens",
                   help="How to decide which runs are shortest.")
    p.add_argument("--max_problems", type=int, default=None,
                   help="Optional limit for smoke tests.")
    return p.parse_args()


def _extract_boxed_answers(text: str) -> list[str]:
    answers = []
    for piece in text.split("boxed{")[1:]:
        brace_count = 1
        for i, ch in enumerate(piece):
            if ch == "{":
                brace_count += 1
            elif ch == "}":
                brace_count -= 1
            if brace_count == 0:
                answers.append(piece[:i].strip())
                break
    return answers


def _extract_math_candidates(question: str, response: str) -> list[str]:
    try:
        from Math_Verifier import extract_math_answer

        return extract_math_answer(question, response, task="math")
    except Exception:
        return _extract_boxed_answers(response)


def _numeric_answer(text: str) -> str | None:
    match = re.search(r"####\s*([-]?[\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1)
    numbers = re.findall(r"[-]?[\d,]+(?:\.\d+)?", text)
    return numbers[-1] if numbers else None


def _has_interpretable_answer(item: dict, run: dict) -> bool:
    response = run.get("response", "")
    if item.get("dataset_type", "math") == "math":
        return any(
            candidate.strip()
            for candidate in _extract_math_candidates(
                item.get("original_problem", ""), response
            )
        )
    return _numeric_answer(response) is not None


def _run_length(run: dict, metric: str) -> int:
    if metric == "tokens":
        completion_ids = run.get("completion_ids")
        if completion_ids is not None:
            return len(completion_ids)
    return len(run.get("response", ""))


def _sort_key(run_with_index: tuple[int, dict], metric: str) -> tuple[int, int]:
    idx, run = run_with_index
    return (_run_length(run, metric), idx)


def select_runs(item: dict, keep_runs: int, length_metric: str) -> tuple[list[dict], str]:
    runs = item.get("runs", [])
    if len(runs) < keep_runs:
        raise ValueError(
            f"question_id={item.get('question_id')} has only {len(runs)} runs; "
            f"requested {keep_runs}."
        )

    interpretable = []
    uninterpretable = []
    for idx, run in enumerate(runs):
        if _has_interpretable_answer(item, run):
            interpretable.append((idx, run))
        else:
            uninterpretable.append((idx, run))

    if len(interpretable) == len(runs):
        selected = sorted(interpretable, key=lambda x: _sort_key(x, length_metric))[:keep_runs]
        reason = "all_interpretable_shortest"
    elif len(uninterpretable) == 1 and len(interpretable) >= keep_runs:
        selected = sorted(interpretable, key=lambda x: x[0])[:keep_runs]
        reason = "drop_one_uninterpretable"
    else:
        selected = sorted(interpretable, key=lambda x: x[0])
        if len(selected) < keep_runs:
            selected.extend(
                sorted(uninterpretable, key=lambda x: _sort_key(x, length_metric))[
                    : keep_runs - len(selected)
                ]
            )
        else:
            selected = sorted(interpretable, key=lambda x: _sort_key(x, length_metric))[
                :keep_runs
            ]
        reason = "fill_with_shortest_uninterpretable"

    selected_runs = []
    for new_run_id, (_, run) in enumerate(selected):
        new_run = dict(run)
        new_run["run_id"] = new_run_id
        selected_runs.append(new_run)
    return selected_runs, reason


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    stats = Counter()
    total_input_runs = 0
    total_output_runs = 0

    with open(args.input_path, "r", encoding="utf-8") as in_f, open(
        args.output_path, "w", encoding="utf-8"
    ) as out_f:
        for line_idx, line in enumerate(in_f):
            if args.max_problems is not None and line_idx >= args.max_problems:
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            total_input_runs += len(item.get("runs", []))

            selected_runs, reason = select_runs(item, args.keep_runs, args.length_metric)
            stats[reason] += 1
            item["runs"] = selected_runs
            total_output_runs += len(selected_runs)
            out_f.write(json.dumps(item) + "\n")

    print(f"Wrote {sum(stats.values())} problems to {args.output_path}")
    print(f"Runs: {total_input_runs} -> {total_output_runs}")
    for key in sorted(stats):
        print(f"{key}: {stats[key]}")


if __name__ == "__main__":
    main()
