"""Repair a pick4 rollout file by regenerating missing interpretable runs.

Targets questions whose original rollout record contains more than
``--uninterpretable_threshold`` uninterpretable runs. For those questions, keep
the interpretable runs already present in the pick4 file, ask the teacher model
for more completions, and write a pick4-style JSONL containing only records
with ``--keep_runs`` interpretable runs.

For questions that had zero interpretable original runs, the script applies a
special early-discard rule: if no interpretable completion appears within
``--all_uninterpretable_first_success_limit`` new generations, discard the
question. This is meant for cases like question_id 9351 and 10909.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.datasets_registry import get_dataset_config, list_datasets
from shared.select_rollout_runs import _has_interpretable_answer


DEFAULT_TEACHER = "/scratch/shuai14/models/Qwen2.5-Math-7B-Instruct"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Regenerate teacher runs until targeted pick4 records have 4 interpretable runs."
    )
    p.add_argument("--original_path", required=True, help="Original 8-run rollout JSONL.")
    p.add_argument("--pick4_path", required=True, help="Existing pick4 JSONL.")
    p.add_argument(
        "--output_path",
        default=None,
        help="Output repaired JSONL. Defaults to <pick4_path>.repaired.",
    )
    p.add_argument(
        "--in_place",
        action="store_true",
        help="Replace pick4_path atomically after writing a temporary repaired file.",
    )
    p.add_argument("--dataset", default="math", choices=list_datasets())
    p.add_argument("--teacher_model", default=DEFAULT_TEACHER)
    p.add_argument("--keep_runs", type=int, default=4)
    p.add_argument(
        "--uninterpretable_threshold",
        type=int,
        default=4,
        help="Repair records whose original uninterpretable-run count is greater than this.",
    )
    p.add_argument(
        "--all_uninterpretable_first_success_limit",
        type=int,
        default=24,
        help=(
            "For records with zero original interpretable runs, discard if no "
            "new interpretable completion appears within this many generated runs."
        ),
    )
    p.add_argument(
        "--max_extra_runs",
        type=int,
        default=64,
        help="Maximum new runs to generate per repaired question. Use 0 for no cap.",
    )
    p.add_argument(
        "--generations_per_round",
        type=int,
        default=4,
        help="Number of completions to request per active prompt each generation round.",
    )
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_tokens", type=int, default=None)
    p.add_argument("--max_model_len", type=int, default=None)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no_system_prompt",
        action="store_true",
        help="Fold the system prompt into the user message instead of using a system role.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Only report which questions would be repaired; do not load the teacher.",
    )
    return p.parse_args()


def load_jsonl(path: str) -> list[dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str, records: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item) + "\n")


def extract_logprobs(completion) -> list[float]:
    token_ids = list(completion.token_ids)
    if not completion.logprobs:
        return [0.0] * len(token_ids)

    logprobs = []
    for idx, lp_dict in enumerate(completion.logprobs):
        if not lp_dict:
            logprobs.append(0.0)
            continue
        token_id = token_ids[idx] if idx < len(token_ids) else None
        if token_id is not None and token_id in lp_dict:
            logprobs.append(lp_dict[token_id].logprob)
        else:
            top = list(lp_dict.values())
            logprobs.append(top[0].logprob if top else 0.0)
    return logprobs


def renumber_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    renumbered = []
    for run_id, run in enumerate(runs):
        new_run = dict(run)
        new_run["run_id"] = run_id
        renumbered.append(new_run)
    return renumbered


def build_prompt(tokenizer, item: dict[str, Any], no_system_prompt: bool) -> str:
    problem = item["original_problem"]
    system_prompt = item.get(
        "system_prompt",
        "Please reason step by step, and put your final answer within \\boxed{}.",
    )
    if no_system_prompt:
        chat = [{"role": "user", "content": f"{problem}\n{system_prompt}"}]
    else:
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem},
        ]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def question_id(item: dict[str, Any]) -> Any:
    return item.get("question_id")


@dataclass
class RepairState:
    original_item: dict[str, Any]
    output_item: dict[str, Any]
    original_interpretable: int
    original_uninterpretable: int
    selected_runs: list[dict[str, Any]]
    attempts: int = 0
    generated_interpretable: int = 0
    discarded: bool = False
    discard_reason: str | None = None
    generated_run_ids: list[int] = field(default_factory=list)

    @property
    def initially_all_uninterpretable(self) -> bool:
        return self.original_interpretable == 0

    @property
    def complete(self) -> bool:
        return len(self.selected_runs) >= 4


def build_repair_states(
    original_records: list[dict[str, Any]],
    pick4_by_qid: dict[Any, dict[str, Any]],
    keep_runs: int,
    uninterpretable_threshold: int,
) -> dict[Any, RepairState]:
    states = {}
    for original in original_records:
        runs = original.get("runs", [])
        interpretable_runs = [
            run for run in runs if _has_interpretable_answer(original, run)
        ]
        uninterpretable_count = len(runs) - len(interpretable_runs)
        if uninterpretable_count <= uninterpretable_threshold:
            continue

        qid = question_id(original)
        if qid not in pick4_by_qid:
            raise ValueError(f"question_id={qid} is missing from pick4_path.")

        pick4_item = pick4_by_qid[qid]
        selected = [
            dict(run)
            for run in pick4_item.get("runs", [])
            if _has_interpretable_answer(pick4_item, run)
        ][:keep_runs]

        states[qid] = RepairState(
            original_item=original,
            output_item={k: v for k, v in pick4_item.items() if k != "runs"},
            original_interpretable=len(interpretable_runs),
            original_uninterpretable=uninterpretable_count,
            selected_runs=selected,
        )
    return states


def make_generated_run(
    cfg,
    completion,
    run_id: int,
) -> dict[str, Any]:
    text = completion.text
    extracted = cfg.extract_answer(text)
    return {
        "run_id": run_id,
        "response": text,
        "extracted_answer": str(extracted) if extracted is not None else None,
        "logprobs": extract_logprobs(completion),
        "completion_ids": list(completion.token_ids),
    }


def active_states(
    states: dict[Any, RepairState],
    keep_runs: int,
    first_success_limit: int,
    max_extra_runs: int,
) -> list[RepairState]:
    active = []
    for state in states.values():
        if state.discarded or len(state.selected_runs) >= keep_runs:
            continue

        if max_extra_runs > 0 and state.attempts >= max_extra_runs:
            state.discarded = True
            state.discard_reason = "max_extra_runs_without_enough_interpretable"
            continue

        if (
            state.initially_all_uninterpretable
            and state.generated_interpretable == 0
            and state.attempts >= first_success_limit
        ):
            state.discarded = True
            state.discard_reason = "no_interpretable_in_first_success_limit"
            continue

        active.append(state)
    return active


def generation_width(
    state: RepairState,
    requested_width: int,
    first_success_limit: int,
    max_extra_runs: int,
) -> int:
    width = requested_width
    if max_extra_runs > 0:
        width = min(width, max_extra_runs - state.attempts)
    if state.initially_all_uninterpretable and state.generated_interpretable == 0:
        width = min(width, first_success_limit - state.attempts)
    return max(0, width)


def repair_with_teacher(args: argparse.Namespace, states: dict[Any, RepairState]) -> None:
    from vllm import LLM, SamplingParams

    cfg = get_dataset_config(args.dataset)
    max_tokens = args.max_tokens if args.max_tokens is not None else cfg.max_tokens
    max_model_len = (
        args.max_model_len if args.max_model_len is not None else cfg.max_model_len
    )

    print(f"=== Repairing pick4 rollouts for {args.dataset.upper()} ===")
    print(f"  Teacher: {args.teacher_model}")
    print(f"  Target questions: {len(states)}")
    print(f"  keep_runs: {args.keep_runs}")
    print(f"  Sampling: temp={args.temperature}, top_p={args.top_p}")
    print(f"  Limits: first_success={args.all_uninterpretable_first_success_limit}, "
          f"max_extra_runs={args.max_extra_runs or 'unlimited'}")

    llm = LLM(
        model=args.teacher_model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="auto",
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
    )
    tokenizer = llm.get_tokenizer()
    prompts_by_qid = {
        qid: build_prompt(tokenizer, state.original_item, args.no_system_prompt)
        for qid, state in states.items()
    }

    round_idx = 0
    while True:
        current = active_states(
            states,
            args.keep_runs,
            args.all_uninterpretable_first_success_limit,
            args.max_extra_runs,
        )
        if not current:
            break

        by_width: dict[int, list[RepairState]] = {}
        for state in current:
            width = generation_width(
                state,
                args.generations_per_round,
                args.all_uninterpretable_first_success_limit,
                args.max_extra_runs,
            )
            if width > 0:
                by_width.setdefault(width, []).append(state)

        if not by_width:
            break

        round_idx += 1
        for width, group in sorted(by_width.items()):
            params = SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=max_tokens,
                seed=args.seed + round_idx,
                logprobs=1,
                n=width,
            )
            prompts = [prompts_by_qid[question_id(s.original_item)] for s in group]
            print(
                f"  Round {round_idx}: generating {len(group)} x {width} "
                f"= {len(group) * width} completions"
            )
            outputs = llm.generate(prompts, params)

            for state, output in zip(group, outputs):
                for completion in output.outputs:
                    state.attempts += 1
                    run = make_generated_run(
                        cfg,
                        completion,
                        run_id=len(state.selected_runs) + state.attempts - 1,
                    )
                    if _has_interpretable_answer(state.original_item, run):
                        state.generated_interpretable += 1
                        state.generated_run_ids.append(state.attempts)
                        state.selected_runs.append(run)
                        if len(state.selected_runs) >= args.keep_runs:
                            break

            complete = sum(
                1 for state in states.values() if len(state.selected_runs) >= args.keep_runs
            )
            discarded = sum(1 for state in states.values() if state.discarded)
            print(f"    Progress: complete={complete}, discarded={discarded}, total={len(states)}")


def finalize_records(
    original_pick4_records: list[dict[str, Any]],
    states: dict[Any, RepairState],
    keep_runs: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repaired_records = []
    repaired = []
    discarded = []

    for item in original_pick4_records:
        qid = question_id(item)
        state = states.get(qid)
        if state is None:
            repaired_records.append(item)
            continue

        if len(state.selected_runs) < keep_runs:
            state.discarded = True
            state.discard_reason = state.discard_reason or "not_enough_interpretable"

        if state.discarded:
            discarded.append(
                {
                    "question_id": qid,
                    "reason": state.discard_reason,
                    "original_interpretable": state.original_interpretable,
                    "original_uninterpretable": state.original_uninterpretable,
                    "generated_attempts": state.attempts,
                    "generated_interpretable": state.generated_interpretable,
                }
            )
            continue

        out = dict(state.output_item)
        out["runs"] = renumber_runs(state.selected_runs[:keep_runs])
        repaired_records.append(out)
        repaired.append(
            {
                "question_id": qid,
                "original_interpretable": state.original_interpretable,
                "original_uninterpretable": state.original_uninterpretable,
                "generated_attempts": state.attempts,
                "generated_interpretable": state.generated_interpretable,
                "kept_runs": len(out["runs"]),
            }
        )

    stats = {
        "input_records": len(original_pick4_records),
        "output_records": len(repaired_records),
        "targeted_questions": len(states),
        "repaired_questions": len(repaired),
        "discarded_questions": len(discarded),
        "repaired": repaired,
        "discarded": discarded,
    }
    return repaired_records, stats


def main() -> None:
    args = parse_args()
    if args.in_place and args.output_path:
        raise ValueError("Use either --in_place or --output_path, not both.")
    if args.generations_per_round <= 0:
        raise ValueError("--generations_per_round must be positive.")

    output_path = args.output_path
    if output_path is None:
        output_path = args.pick4_path + ".tmp_repaired" if args.in_place else args.pick4_path + ".repaired"

    original_records = load_jsonl(args.original_path)
    pick4_records = load_jsonl(args.pick4_path)
    pick4_by_qid = {question_id(item): item for item in pick4_records}

    states = build_repair_states(
        original_records,
        pick4_by_qid,
        args.keep_runs,
        args.uninterpretable_threshold,
    )

    all_uninterpretable = [
        question_id(s.original_item)
        for s in states.values()
        if s.initially_all_uninterpretable
    ]
    print(f"Found {len(states)} questions with >{args.uninterpretable_threshold} uninterpretable runs.")
    print(f"Initially all-uninterpretable question_ids: {all_uninterpretable}")

    if args.dry_run:
        return

    if states:
        repair_with_teacher(args, states)

    repaired_records, stats = finalize_records(pick4_records, states, args.keep_runs)
    write_jsonl(output_path, repaired_records)

    stats_path = output_path + ".stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")

    if args.in_place:
        os.replace(output_path, args.pick4_path)
        os.replace(stats_path, args.pick4_path + ".stats.json")
        output_path = args.pick4_path
        stats_path = args.pick4_path + ".stats.json"

    print(f"Wrote repaired pick4 file: {output_path}")
    print(f"Wrote repair stats: {stats_path}")
    print(
        f"Summary: targeted={stats['targeted_questions']}, "
        f"repaired={stats['repaired_questions']}, discarded={stats['discarded_questions']}, "
        f"output_records={stats['output_records']}"
    )


if __name__ == "__main__":
    main()
