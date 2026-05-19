"""Diagnostic: measure the surprisal distribution of teacher rollouts under the student policy.

Question this script answers: is DG-offline's sigmoid gate actually in its saturating regime?

For each teacher completion we compute mean per-token surprisal under the student policy
(same formula DG-offline uses). We then compute per-completion advantages via GRPO group
normalization, and the delight = advantage * surprisal that feeds the gate. Finally we
report the distribution of delight and the resulting gate = sigmoid(delight / eta) for
several eta values. This tells us whether the four-quadrant noise-filter story applies
or whether the sigmoid is stuck in its linear regime.
"""

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
from datasets_registry import get_dataset_config  # noqa: E402


ROLLOUT_PATHS = {
    "math":  "/scratch/mrli/rollouts/math_teacher/rollouts_full.jsonl",
    "gsm8k": "/scratch/mrli/rollouts/gsm8k_teacher/rollouts_gsm8k.jsonl",
    "svamp": "/scratch/mrli/rollouts/svamp_teacher/rollouts_svamp.jsonl",
    "asdiv": "/scratch/mrli/rollouts/asdiv_teacher/rollouts_asdiv.jsonl",
}


def compute_surprisal(model, tokenizer, sys_prompt, problem, completion_ids, device):
    chat = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": problem},
    ]
    prompt_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    comp = torch.tensor(completion_ids, dtype=torch.long, device=device).unsqueeze(0)
    input_ids = torch.cat([prompt_ids, comp], dim=-1)

    with torch.no_grad():
        logits = model(input_ids).logits  # (1, L, V)

    prompt_len = prompt_ids.size(-1)
    shift_logits = logits[0, prompt_len - 1 : -1, :].float()
    target = comp[0]
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_logps = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    token_surprisal = -token_logps  # non-negative, in nats
    return token_surprisal.cpu().tolist()


def summarize(label, values):
    if not values:
        print(f"  {label}: no data")
        return
    vs = sorted(values)
    n = len(vs)
    print(f"  {label} (n={n}):  "
          f"mean={sum(vs)/n:.3f}  std={statistics.pstdev(vs):.3f}  "
          f"min={vs[0]:.3f}  p5={vs[max(0,int(n*0.05))]:.3f}  "
          f"p50={vs[n//2]:.3f}  "
          f"p95={vs[min(n-1,int(n*0.95))]:.3f}  max={vs[-1]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(ROLLOUT_PATHS))
    ap.add_argument("--student", default="/scratch/mrli/models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--max_questions", type=int, default=400,
                    help="cap on number of distinct questions sampled (5 completions each)")
    ap.add_argument("--output", default=None, help="optional json dump of per-completion records")
    args = ap.parse_args()

    cfg = get_dataset_config(args.dataset)
    rollout_path = ROLLOUT_PATHS[args.dataset]

    print(f"[setup] student = {args.student}")
    tokenizer = AutoTokenizer.from_pretrained(args.student)
    model = AutoModelForCausalLM.from_pretrained(
        args.student,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda",
    )
    model.eval()
    vocab_size = model.config.vocab_size
    print(f"[setup] student vocab_size (embedding rows) = {vocab_size}")

    print(f"[load] reading rollouts from {rollout_path}")
    records = []
    seen_qids = set()
    with open(rollout_path) as f:
        for line in f:
            item = json.loads(line)
            qid = item["question_id"]
            if len(seen_qids) >= args.max_questions and qid not in seen_qids:
                continue
            seen_qids.add(qid)

            problem = item["original_problem"]
            gt = item["ground_truth_answer"]
            sys_prompt = item.get("system_prompt", cfg.system_prompt)

            for run in item["runs"]:
                cids = run["completion_ids"]
                # truncate at first out-of-student-vocab id
                for idx, tid in enumerate(cids):
                    if tid >= vocab_size:
                        cids = cids[:idx]
                        break
                if len(cids) < 2:
                    continue

                response = run.get("response", "")
                reward = 1.0 if cfg.check_answer(response, gt) else 0.0

                records.append({
                    "qid": qid, "problem": problem, "sys_prompt": sys_prompt,
                    "completion_ids": cids, "reward": reward,
                })

            if len(seen_qids) >= args.max_questions:
                break

    print(f"[load] {len(records)} completions across {len(seen_qids)} questions")

    # ---- compute per-completion surprisal ----
    print(f"[measure] computing surprisal under student...")
    for i, rec in enumerate(records):
        token_surp = compute_surprisal(
            model, tokenizer, rec["sys_prompt"], rec["problem"], rec["completion_ids"], model.device,
        )
        rec["mean_surprisal"] = sum(token_surp) / len(token_surp)
        rec["n_tokens"] = len(token_surp)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(records)} done")

    # ---- group-normalize advantages (GRPO style, same as compute_rewards_and_advantages) ----
    groups = defaultdict(list)
    for rec in records:
        groups[rec["qid"]].append(rec)
    for group in groups.values():
        rewards = [r["reward"] for r in group]
        mu = sum(rewards) / len(rewards)
        var = sum((r - mu) ** 2 for r in rewards) / len(rewards)
        sd = var ** 0.5
        for rec in group:
            rec["advantage"] = (rec["reward"] - mu) / (sd + 1e-4)

    # ---- delight + gate at several eta ----
    for rec in records:
        rec["delight"] = rec["advantage"] * rec["mean_surprisal"]

    # ---- report ----
    correct = [r for r in records if r["reward"] > 0]
    wrong   = [r for r in records if r["reward"] == 0]

    print(f"\n=== SURPRISAL (mean per-token, nats) — dataset={args.dataset}, n={len(records)} ===")
    summarize("All      ", [r["mean_surprisal"] for r in records])
    summarize("Correct  ", [r["mean_surprisal"] for r in correct])
    summarize("Wrong    ", [r["mean_surprisal"] for r in wrong])

    print(f"\n=== ADVANTAGE (group-normalized) ===")
    summarize("All      ", [r["advantage"] for r in records])

    print(f"\n=== DELIGHT = advantage x mean_surprisal ===")
    summarize("All         ", [r["delight"] for r in records])
    summarize("Correct (A>0)", [r["delight"] for r in correct if r["advantage"] > 0])
    summarize("Wrong (A<0)  ", [r["delight"] for r in wrong if r["advantage"] < 0])

    print(f"\n=== GATE = sigmoid(delight / eta) for each eta ===")
    import math
    etas = [0.1, 0.5, 1.0, 2.0]
    print(f"  {'eta':>6}  {'mean':>7}  {'min':>7}  {'p5':>7}  {'p50':>7}  {'p95':>7}  {'max':>7}  "
          f"{'frac<0.1':>9}  {'frac>0.9':>9}  # fraction in saturated regions")
    for eta in etas:
        gates = [1.0 / (1.0 + math.exp(-r["delight"] / eta)) for r in records]
        gates_sorted = sorted(gates)
        n = len(gates)
        frac_low  = sum(1 for g in gates if g < 0.1) / n
        frac_high = sum(1 for g in gates if g > 0.9) / n
        print(f"  {eta:>6}  {sum(gates)/n:>7.3f}  "
              f"{gates_sorted[0]:>7.3f}  "
              f"{gates_sorted[max(0,int(n*0.05))]:>7.3f}  "
              f"{gates_sorted[n//2]:>7.3f}  "
              f"{gates_sorted[min(n-1,int(n*0.95))]:>7.3f}  "
              f"{gates_sorted[-1]:>7.3f}  "
              f"{frac_low:>9.1%}  {frac_high:>9.1%}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(records, f, default=str)
        print(f"\n[save] per-completion records written to {args.output}")


if __name__ == "__main__":
    main()
