"""Prepare a "cleaned" rollout dataset that Offline-GRPO can train on, even when
the teacher's tokenizer has special-token IDs that collide with the student's
vocabulary (the R1-Distill `<think>` / `</think>` collision case).

Pipeline (per rollout):

  1. Read the teacher's `response` text (vLLM already applied
     `skip_special_tokens=True`, so `<think>` / `</think>` markers are absent).
  2. Tokenize the response under the **student** tokenizer
     → `student_completion_ids` (all in 0–151,664, the shared real-text range).
  3. Build the student's training prompt (MATH_SYSTEM_PROMPT + problem, applied
     through the student's chat template) → `student_prompt_ids`.
  4. Concatenate `[student_prompt_ids; student_completion_ids]` and run the
     **teacher** model on this exact sequence (vLLM with `prompt_logprobs=1`
     returning the per-position logprob of the actual token).
  5. Slice out the per-token teacher logprobs aligned with the completion
     positions → `logprobs_aligned`.
  6. Emit a cleaned JSONL whose schema mirrors the original rollout JSONL:
     {question_id, original_problem, ground_truth_answer, system_prompt,
      dataset_type, runs: [{run_id, response, extracted_answer,
                            logprobs (= logprobs_aligned),
                            completion_ids (= student_completion_ids)}]}.

The output file is consumable by `offline_grpo/data.py:load_rollouts` with
`vocab_size=None`, since the IDs are already student-vocab.

Why we do this: Offline-GRPO's IS ratio is
    π_student(seq | prompt_student) / π_teacher(seq | prompt_teacher)
which requires both sides to be evaluated on the **same** tokenization of the
sequence. Direct re-use of teacher token IDs through the student forward pass
(the legacy loader behavior) silently corrupts when teacher and student vocabs
diverge at any ID, which they do for R1's `<think>`/`</think>` markers.

See docs/dg_offline/plans/multi_teacher_experiment.md §5.1.1 for context.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Iterable

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


# Hardcoded for our setup; matches what `shared/datasets_registry.py:math` defines.
MATH_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Recompute teacher logprobs aligned with student tokenization."
    )
    p.add_argument("--input_path", type=str, required=True,
                    help="Original rollout JSONL (e.g. R1 rollouts produced by shared/generate_rollouts.py).")
    p.add_argument("--output_path", type=str, required=True,
                    help="Where to write the cleaned JSONL.")
    p.add_argument("--teacher_model", type=str, required=True)
    p.add_argument("--student_model", type=str, required=True,
                    help="Student model whose tokenizer will be used for re-tokenization.")
    p.add_argument("--max_problems", type=int, default=None,
                    help="Limit (for smoke tests).")
    p.add_argument("--tensor_parallel_size", type=int, default=4)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--max_model_len", type=int, default=33280,
                    help="Must accommodate the longest student-tokenized "
                    "(prompt + completion) we expect.")
    p.add_argument("--batch_size", type=int, default=64,
                    help="vLLM dispatches everything via continuous batching; this is "
                    "just the chunk size for streaming write to disk.")
    return p.parse_args()


def iter_input_records(path: str, max_problems: int | None = None) -> Iterable[dict]:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            n += 1
            if max_problems is not None and n >= max_problems:
                break


def build_student_prompt_text(student_tokenizer, problem: str) -> str:
    chat = [
        {"role": "system", "content": MATH_SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    return student_tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )


def main():
    args = parse_args()

    print("=== Prepare cleaned OG rollouts ===")
    print(f"  input:        {args.input_path}")
    print(f"  output:       {args.output_path}")
    print(f"  teacher:      {args.teacher_model}")
    print(f"  student:      {args.student_model}")
    print(f"  TP:           {args.tensor_parallel_size}")
    print(f"  max_model_len: {args.max_model_len}")

    student_tok = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    student_tok.pad_token = student_tok.eos_token

    print("Loading teacher into vLLM ...")
    llm = LLM(
        model=args.teacher_model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="auto",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False,
    )
    print()

    # We feed the teacher pre-tokenized prompt_token_ids (so the teacher reads
    # the exact student-vocab IDs at the response positions). vLLM's
    # `prompt_logprobs=1` then returns, for each prompt position, a dict that
    # includes the logprob of the actual token at that position.
    sp_logprobs = SamplingParams(
        max_tokens=1,                # we don't generate; we want prompt logprobs
        prompt_logprobs=1,
        temperature=0.0,
    )

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    # Phase 1: tokenize all (prompt + response) pairs under student.
    print("Tokenizing all (prompt+completion) pairs under student tokenizer ...")
    t0 = time.time()
    flat = []  # [(item_idx, run_idx, prompt_ids, completion_ids, completion_text, run_dict, item_dict)]
    items = list(iter_input_records(args.input_path, args.max_problems))
    print(f"  {len(items)} prompts to process")

    for item_idx, item in enumerate(items):
        problem = item["original_problem"]
        prompt_text = build_student_prompt_text(student_tok, problem)
        prompt_ids = student_tok(prompt_text, add_special_tokens=False)["input_ids"]

        for run in item["runs"]:
            response = run["response"]
            completion_ids = student_tok(response, add_special_tokens=False)["input_ids"]
            if len(completion_ids) == 0:
                # Emit a degenerate record; OG can skip if needed.
                flat.append((item_idx, run["run_id"], prompt_ids, [], response, run, item))
                continue
            flat.append(
                (item_idx, run["run_id"], prompt_ids, completion_ids, response, run, item)
            )
    print(f"  [{time.time()-t0:.1f}s] {len(flat)} (prompt, completion) pairs ready")
    print()

    # Phase 2: query teacher for prompt_logprobs in chunks.
    # Each input is `prompt_token_ids = prompt_ids + completion_ids` (a single
    # token_id list). vLLM returns prompt_logprobs for each position.
    print("Querying teacher for student-aligned logprobs (continuous-batched) ...")
    t1 = time.time()
    # Build TokensPrompt objects (vLLM 0.16+ API; the older `prompt_token_ids=`
    # kwarg was removed). Each TokensPrompt feeds the teacher pre-tokenized
    # student-vocab IDs, so the teacher reads the exact same token sequence
    # the student will see at training time.
    token_prompts = [
        TokensPrompt(prompt_token_ids=(pids + cids))
        for (_, _, pids, cids, _, _, _) in flat
    ]
    outputs = llm.generate(token_prompts, sampling_params=sp_logprobs)
    print(f"  [{time.time()-t1:.1f}s] teacher forward done; {len(outputs)} outputs")
    print()

    # Phase 3: align logprobs to completion positions and write cleaned JSONL.
    print("Slicing logprobs to completion windows + writing JSONL ...")
    t2 = time.time()
    # Group results back by item, then by run_id within item.
    by_item: dict[int, dict[int, dict]] = {}
    for (item_idx, run_id, pids, cids, response, run_dict, item_dict), out in zip(flat, outputs):
        prompt_len = len(pids)
        comp_len = len(cids)
        # vLLM returns prompt_logprobs aligned with the entire prompt_token_ids
        # input (length prompt_len + comp_len). The logprob of token at position t
        # is conditioned on positions 0..t-1. We want logprobs for the completion
        # tokens, which sit at positions [prompt_len, prompt_len + comp_len).
        plogprobs = out.prompt_logprobs  # list of length (prompt_len + comp_len), index 0 is None
        comp_logprobs: list[float] = []
        for pos in range(prompt_len, prompt_len + comp_len):
            tid = (pids + cids)[pos]
            entry = plogprobs[pos] if plogprobs and pos < len(plogprobs) else None
            if entry is None:
                comp_logprobs.append(0.0)
                continue
            # entry is dict[int, Logprob]; key = token_id, value.logprob = scalar
            lp = entry.get(tid)
            comp_logprobs.append(lp.logprob if lp is not None else 0.0)

        cleaned_run = {
            "run_id": run_id,
            "response": response,
            "extracted_answer": run_dict.get("extracted_answer") or run_dict.get("boxed_answer"),
            "logprobs": comp_logprobs,        # teacher logprobs at student-vocab positions
            "completion_ids": cids,           # student-vocab IDs
        }
        if item_idx not in by_item:
            by_item[item_idx] = {"item": item_dict, "runs": {}}
        by_item[item_idx]["runs"][run_id] = cleaned_run

    written = 0
    with open(args.output_path, "w", encoding="utf-8") as f:
        for item_idx in sorted(by_item):
            item = by_item[item_idx]["item"]
            runs_dict = by_item[item_idx]["runs"]
            cleaned_runs = [runs_dict[rid] for rid in sorted(runs_dict)]
            f.write(json.dumps({
                "question_id": item["question_id"],
                "original_problem": item["original_problem"],
                "ground_truth_answer": item["ground_truth_answer"],
                "system_prompt": MATH_SYSTEM_PROMPT,  # student's system prompt going forward
                "dataset_type": item.get("dataset_type", "math"),
                "runs": cleaned_runs,
            }) + "\n")
            written += 1
    print(f"  [{time.time()-t2:.1f}s] wrote {written} prompts to {args.output_path}")
    print(f"=== Cleaned OG dataset prep complete ({time.time()-t0:.1f}s total) ===")


if __name__ == "__main__":
    main()
