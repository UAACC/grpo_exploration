"""Validate rollouts after multi-GPU generation.

Checks:
1. All expected question_ids are present (no missing problems)
2. No duplicate question_ids
3. Each problem has the expected number of runs (completions)
4. Every run has required fields: response, logprobs, completion_ids
5. Logprobs and completion_ids have matching lengths
6. No empty completions or logprobs
7. Reward distribution (correct vs incorrect answers)
8. Per-shard statistics (if shard files exist)

Usage:
    python validate_rollouts.py --rollout_path /path/to/rollouts_full.jsonl
    python validate_rollouts.py --rollout_path /path/to/rollouts_full.jsonl --shard_dir /path/to/teacher_rollouts/
"""

import argparse
import json
import glob
import os

from configs import extract_boxed_answer


def load_records(path):
    records = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ERROR: malformed JSON at line {line_num}: {e}")
    return records


def validate_file(path, expected_num_problems=None, num_generations=4, label=""):
    prefix = f"[{label}] " if label else ""
    print(f"\n{prefix}Validating: {path}")

    if not os.path.exists(path):
        print(f"  ERROR: file does not exist")
        return False

    file_size = os.path.getsize(path)
    print(f"  File size: {file_size / 1024 / 1024:.1f} MB")

    records = load_records(path)
    print(f"  Records: {len(records)}")

    if len(records) == 0:
        print(f"  ERROR: file is empty")
        return False

    errors = []
    warnings = []

    # Check question_ids
    qids = [r.get("question_id") for r in records]
    unique_qids = set(qids)

    if len(qids) != len(unique_qids):
        dups = [q for q in unique_qids if qids.count(q) > 1]
        errors.append(f"Duplicate question_ids: {sorted(dups)[:10]}{'...' if len(dups) > 10 else ''}")

    if expected_num_problems is not None:
        expected_ids = set(range(expected_num_problems))
        missing = expected_ids - unique_qids
        extra = unique_qids - expected_ids
        if missing:
            errors.append(f"Missing {len(missing)} question_ids: {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")
        if extra:
            warnings.append(f"Extra {len(extra)} question_ids outside expected range: {sorted(extra)[:10]}")

    # Validate each record
    total_completions = 0
    correct = 0
    incorrect = 0
    empty_responses = 0
    length_mismatches = 0
    missing_fields = 0
    completion_lengths = []

    for r in records:
        qid = r.get("question_id", "?")
        runs = r.get("runs", [])
        gt = r.get("ground_truth_answer", "")

        if len(runs) != num_generations:
            warnings.append(f"question_id={qid}: expected {num_generations} runs, got {len(runs)}")

        for run in runs:
            total_completions += 1

            # Check required fields
            for field in ["response", "logprobs", "completion_ids", "run_id"]:
                if field not in run:
                    missing_fields += 1
                    errors.append(f"question_id={qid}, run_id={run.get('run_id', '?')}: missing field '{field}'")
                    continue

            response = run.get("response", "")
            logprobs = run.get("logprobs", [])
            completion_ids = run.get("completion_ids", [])

            if not response:
                empty_responses += 1

            if len(logprobs) != len(completion_ids):
                length_mismatches += 1
                if length_mismatches <= 5:
                    errors.append(f"question_id={qid}, run_id={run.get('run_id')}: logprobs len={len(logprobs)} != completion_ids len={len(completion_ids)}")

            if len(completion_ids) > 0:
                completion_lengths.append(len(completion_ids))

            # Check reward
            boxed = run.get("boxed_answer", extract_boxed_answer(response))
            if boxed is not None and gt:
                if str(boxed).strip() == str(gt).strip():
                    correct += 1
                else:
                    incorrect += 1

    # Print summary
    print(f"  Total completions: {total_completions}")
    print(f"  Problems × generations: {len(records)} × {num_generations} = {len(records) * num_generations}")

    if completion_lengths:
        avg_len = sum(completion_lengths) / len(completion_lengths)
        min_len = min(completion_lengths)
        max_len = max(completion_lengths)
        print(f"  Completion lengths: avg={avg_len:.0f}, min={min_len}, max={max_len}")

    if correct + incorrect > 0:
        total_scored = correct + incorrect
        print(f"  Rewards: {correct}/{total_scored} correct ({100*correct/total_scored:.1f}%)")
    else:
        warnings.append("Could not compute reward statistics")

    if empty_responses > 0:
        warnings.append(f"{empty_responses} empty responses")

    if length_mismatches > 0:
        errors.append(f"{length_mismatches} total logprobs/completion_ids length mismatches")

    if missing_fields > 0:
        errors.append(f"{missing_fields} total missing required fields")

    # Print errors and warnings
    if warnings:
        print(f"\n  WARNINGS ({len(warnings)}):")
        for w in warnings[:20]:
            print(f"    - {w}")

    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors[:20]:
            print(f"    - {e}")
        if len(errors) > 20:
            print(f"    ... and {len(errors) - 20} more")
        return False
    else:
        print(f"\n  PASSED - no errors found")
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollout_path", type=str, required=True, help="Path to merged rollouts JSONL")
    p.add_argument("--shard_dir", type=str, default=None, help="Directory containing shard files (optional)")
    p.add_argument("--expected_problems", type=int, default=12000, help="Expected number of problems")
    p.add_argument("--num_generations", type=int, default=4)
    args = p.parse_args()

    all_passed = True

    # Validate individual shards if directory provided
    if args.shard_dir:
        shard_files = sorted(glob.glob(os.path.join(args.shard_dir, "rollouts_shard_*.jsonl")))
        if shard_files:
            print(f"{'='*60}")
            print(f"Validating {len(shard_files)} shard files")
            print(f"{'='*60}")

            total_shard_records = 0
            for sf in shard_files:
                label = os.path.basename(sf)
                records = load_records(sf)
                total_shard_records += len(records)
                passed = validate_file(sf, expected_num_problems=None,
                                       num_generations=args.num_generations, label=label)
                if not passed:
                    all_passed = False

            print(f"\nTotal records across shards: {total_shard_records}")
            if total_shard_records != args.expected_problems:
                print(f"  WARNING: expected {args.expected_problems}, got {total_shard_records}")
        else:
            print(f"No shard files found in {args.shard_dir}")

    # Validate merged file
    print(f"\n{'='*60}")
    print(f"Validating merged rollouts")
    print(f"{'='*60}")
    passed = validate_file(args.rollout_path, expected_num_problems=args.expected_problems,
                           num_generations=args.num_generations)
    if not passed:
        all_passed = False

    # Final verdict
    print(f"\n{'='*60}")
    if all_passed:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED - review errors above")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
