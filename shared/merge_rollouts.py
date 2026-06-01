"""Merge teacher rollout JSONL files into one rollout dataset.

Each input line is expected to be one problem record with a `runs` list, as
written by `shared/generate_rollouts.py`. Matching `question_id`s are merged by
concatenating selected runs and remapping `run_id` to 0..N-1.

Usage:
    python shared/merge_rollouts.py \
        --input_paths /path/teacher_a.jsonl /path/teacher_b.jsonl \
        --runs_per_file 4 5 \
        --selection interpretable \
        --output_path /path/merged.jsonl
"""

import argparse
import json
import os
import re
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser(description="Merge teacher rollout JSONL files.")
    p.add_argument("--input_paths", nargs="+", required=True,
                   help="Two or more rollout JSONL files to merge.")
    p.add_argument("--runs_per_file", nargs="+", type=int, default=None,
                   help=(
                       "Number of runs to take from each input file per question. "
                       "Pass one value for all files, or one value per input file. "
                       "Defaults to all runs."
                   ))
    p.add_argument("--selection", choices=["interpretable", "first"],
                   default="interpretable",
                   help=(
                       "When selecting a subset, prefer runs whose final answer "
                       "can be extracted by the eval verifier, or keep first N."
                   ))
    p.add_argument("--output_path", required=True,
                   help="Path to write the merged rollout JSONL.")
    return p.parse_args()


def _same_problem(a: dict, b: dict) -> bool:
    keys = (
        "original_problem",
        "ground_truth_answer",
    )
    return all(a.get(k) == b.get(k) for k in keys)


def _resolve_runs_per_file(args):
    if args.runs_per_file is None:
        return [None] * len(args.input_paths)
    if len(args.runs_per_file) == 1:
        return args.runs_per_file * len(args.input_paths)
    if len(args.runs_per_file) != len(args.input_paths):
        raise ValueError(
            "--runs_per_file must have either one value or one value per input file."
        )
    return args.runs_per_file



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
        try:
            return any(c.strip() for c in _extract_math_candidates(
                item.get("original_problem", ""), response
            ))
        except Exception:
            return False
    return _numeric_answer(response) is not None


def _select_runs(item: dict, keep_runs: int | None, selection: str) -> list[dict]:
    runs = item["runs"]
    if keep_runs is None:
        return runs
    if len(runs) < keep_runs:
        qid = item.get("question_id")
        raise ValueError(
            f"question_id={qid} has only {len(runs)} runs; "
            f"requested {keep_runs}."
        )
    if selection == "first":
        return runs[:keep_runs]

    interpretable = [run for run in runs if _has_interpretable_answer(item, run)]
    selected = interpretable[:keep_runs]
    if len(selected) < keep_runs:
        selected_ids = {id(run) for run in selected}
        selected.extend(run for run in runs if id(run) not in selected_ids)
    return selected[:keep_runs]


def main():
    args = parse_args()
    if len(args.input_paths) < 2:
        raise ValueError("--input_paths should contain at least two rollout files.")

    runs_per_file = _resolve_runs_per_file(args)
    merged = OrderedDict()
    total_runs = 0

    for path, keep_runs in zip(args.input_paths, runs_per_file):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                qid = item["question_id"]

                if qid not in merged:
                    merged[qid] = dict(item)
                    merged[qid]["runs"] = []
                elif not _same_problem(merged[qid], item):
                    raise ValueError(
                        f"Question metadata mismatch for question_id={qid} in {path}"
                    )

                runs = _select_runs(item, keep_runs, args.selection)
                for run in runs:
                    new_run = dict(run)
                    new_run["run_id"] = len(merged[qid]["runs"])
                    merged[qid]["runs"].append(new_run)
                    total_runs += 1

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        for item in merged.values():
            f.write(json.dumps(item) + "\n")

    print(f"Wrote {len(merged)} problems and {total_runs} runs to {args.output_path}")


if __name__ == "__main__":
    main()
