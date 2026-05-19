"""Prepare a "cleaned" rollout dataset that Offline-GRPO can train on, even when
the teacher's tokenizer has special-token IDs that collide with the student's
vocabulary (the R1-Distill `<think>` (151648) / `</think>` (151649) collision
case).

Pipeline (per rollout):

  1. Read the teacher's `response` text (vLLM applied `skip_special_tokens=True`
     at rollout time, so `<think>` / `</think>` markers are absent).
  2. Tokenize the response under the **student** tokenizer
     -> `student_completion_ids` (all in 0..151,664, the shared real-text range).
  3. Build the student's training prompt (MATH_SYSTEM_PROMPT + problem, applied
     through the student's chat template) -> `student_prompt_ids`.
  4. Run a transformers forward pass on the concatenation
     `[student_prompt_ids; student_completion_ids]` and compute per-token
     teacher logprob via `gather + logsumexp`. We never materialize the full
     [seq_len, vocab_size] log_softmax distribution.
  5. Slice out the per-token teacher logprobs aligned with the completion
     positions -> `logprobs_aligned`.
  6. Emit a cleaned JSONL whose schema matches the original rollout JSONL,
     directly consumable by `offline_grpo/data.py:load_rollouts` with
     `vocab_size=None`.

Why transformers-direct rather than vLLM's `prompt_logprobs=1`:

    vLLM materializes `[seq_len, vocab_size]` in fp32 at every prompt position
    when `prompt_logprobs=1` is set (`compute_logprobs` -> `log_softmax(...,
    dtype=torch.float32)`). For R1 + 152K vocab + ~38K-token rollouts that's
    ~23 GiB *per request*, and continuous batching keeps several in flight ->
    OOM on 44 GiB L40s. We saw this fail 5 times.

    Computing `target_logit - logsumexp(logits, dim=-1)` directly only keeps
    a `[seq_len]` scalar tensor instead of the full distribution, so peak
    memory is dominated by the (unavoidable) bf16 logits buffer plus small
    fp32 chunks for stable logsumexp. Roughly 150,000x less memory than the
    vLLM path.

See docs/dg_offline/plans/multi_teacher_experiment.md sec 5.1.1 for context.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Hardcoded for our setup; matches what `shared/datasets_registry.py:math` defines.
MATH_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Recompute teacher logprobs aligned with student tokenization, "
                    "via a transformers-direct forward pass (no vLLM)."
    )
    p.add_argument("--input_path", type=str, required=True,
                    help="Original rollout JSONL.")
    p.add_argument("--output_path", type=str, required=True,
                    help="Where to write the cleaned JSONL.")
    p.add_argument("--teacher_model", type=str, required=True)
    p.add_argument("--student_model", type=str, required=True,
                    help="Tokenizer used to re-tokenize the response text.")
    p.add_argument("--max_problems", type=int, default=None,
                    help="Limit (for smoke tests).")
    p.add_argument("--max_seq_len", type=int, default=38000,
                    help="Drop (prompt+completion) pairs longer than this in "
                    "student tokens. Keeps peak logits memory bounded.")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--logsumexp_chunk", type=int, default=1024,
                    help="Process the [completion_len, vocab] logits slice in "
                    "chunks of this many positions when casting to fp32 for "
                    "logsumexp, to bound peak fp32 memory.")
    p.add_argument("--flush_every", type=int, default=20,
                    help="Flush output JSONL every N items written.")
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


@torch.no_grad()
def compute_completion_logprobs(
    model,
    prompt_ids: list[int],
    completion_ids: list[int],
    device: str,
    logsumexp_chunk: int,
) -> list[float]:
    """Run the teacher on (prompt + completion) and return per-token logprob
    over the completion positions only.

    Uses gather + chunked-fp32 logsumexp to avoid materializing the full
    [completion_len, vocab_size] log_softmax distribution.
    """
    if len(completion_ids) == 0:
        return []

    full_ids = prompt_ids + completion_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)

    # logits[t] predicts the token at position t+1 given prefix [0..t].
    # The logit that predicts target at absolute position `pos` is logits[pos-1].
    logits = model(input_ids).logits  # [1, L, V] bf16

    p = len(prompt_ids)
    c = len(completion_ids)

    # View into the predictor logits for completion targets and the targets.
    pred_logits = logits[0, p - 1 : p - 1 + c, :]   # [c, V] bf16 (view)
    target_ids = input_ids[0, p : p + c]             # [c]

    # Gather actual-token logit (bf16, tiny output).
    target_logits_bf16 = pred_logits.gather(
        -1, target_ids.unsqueeze(-1)
    ).squeeze(-1)                                     # [c] bf16

    # Chunked fp32 logsumexp: cap transient fp32 buffer at
    # logsumexp_chunk * V * 4 bytes (~0.6 GiB at chunk=1024, V=152K).
    parts = []
    for i in range(0, c, logsumexp_chunk):
        chunk_fp32 = pred_logits[i : i + logsumexp_chunk].float()
        parts.append(torch.logsumexp(chunk_fp32, dim=-1))
        del chunk_fp32
    logsumexp = torch.cat(parts) if parts else torch.empty(0, device=device)

    token_logprobs = (target_logits_bf16.float() - logsumexp).cpu().tolist()

    # Free the big logits buffer before returning.
    del logits, pred_logits, target_logits_bf16, logsumexp, input_ids, parts
    return token_logprobs


def main():
    args = parse_args()

    print("=== Prepare cleaned OG rollouts (transformers-direct) ===")
    print(f"  input:        {args.input_path}")
    print(f"  output:       {args.output_path}")
    print(f"  teacher:      {args.teacher_model}")
    print(f"  student:      {args.student_model}")
    print(f"  max_seq_len:  {args.max_seq_len}")
    print(f"  device:       {args.device}")
    print(f"  lse chunk:    {args.logsumexp_chunk}")
    print()

    student_tok = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    student_tok.pad_token = student_tok.eos_token

    print("Loading teacher (transformers + flash_attention_2) ...")
    t_load = time.time()
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to(args.device).eval()
    print(f"  loaded in {time.time()-t_load:.1f}s; "
          f"GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GiB")
    print()

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    items = list(iter_input_records(args.input_path, args.max_problems))
    total_items = len(items)
    print(f"Processing {total_items} items ...")
    print()

    n_dropped_too_long = 0
    n_runs_processed = 0
    n_items_written = 0
    t0 = time.time()

    out_fp = open(args.output_path, "w", encoding="utf-8")

    for item_idx, item in enumerate(items):
        problem = item["original_problem"]
        prompt_text = build_student_prompt_text(student_tok, problem)
        prompt_ids = student_tok(prompt_text, add_special_tokens=False)["input_ids"]

        cleaned_runs = []
        for run in item["runs"]:
            response = run["response"]
            completion_ids = student_tok(response, add_special_tokens=False)["input_ids"]
            total_len = len(prompt_ids) + len(completion_ids)
            if total_len > args.max_seq_len:
                n_dropped_too_long += 1
                continue

            comp_logprobs = compute_completion_logprobs(
                teacher, prompt_ids, completion_ids, args.device, args.logsumexp_chunk
            )
            n_runs_processed += 1

            cleaned_runs.append({
                "run_id": run["run_id"],
                "response": response,
                "extracted_answer": run.get("extracted_answer") or run.get("boxed_answer"),
                "logprobs": comp_logprobs,
                "completion_ids": completion_ids,
            })

        if cleaned_runs:
            cleaned_runs.sort(key=lambda r: r["run_id"])
            out_fp.write(json.dumps({
                "question_id": item["question_id"],
                "original_problem": item["original_problem"],
                "ground_truth_answer": item["ground_truth_answer"],
                "system_prompt": MATH_SYSTEM_PROMPT,
                "dataset_type": item.get("dataset_type", "math"),
                "runs": cleaned_runs,
            }) + "\n")
            n_items_written += 1

            if n_items_written % args.flush_every == 0:
                out_fp.flush()

        if (item_idx + 1) % 25 == 0 or item_idx == total_items - 1:
            elapsed = time.time() - t0
            rate = (item_idx + 1) / max(elapsed, 1e-9)
            eta = (total_items - item_idx - 1) / max(rate, 1e-9)
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            print(f"  [{elapsed:6.0f}s] item {item_idx+1}/{total_items} | "
                  f"runs ok={n_runs_processed} dropped={n_dropped_too_long} | "
                  f"items_written={n_items_written} | "
                  f"{rate:.2f} item/s | ETA {eta:.0f}s | "
                  f"peak GPU {peak_gb:.1f} GiB",
                  flush=True)

    out_fp.close()
    print()
    print(f"=== Cleaned OG dataset prep complete ===")
    print(f"  total wall:                    {time.time()-t0:.1f}s")
    print(f"  items written:                 {n_items_written}")
    print(f"  runs processed:                {n_runs_processed}")
    print(f"  runs dropped (>max_seq_len):   {n_dropped_too_long}")
    print(f"  peak GPU memory:               {torch.cuda.max_memory_allocated()/1e9:.1f} GiB")


if __name__ == "__main__":
    main()
