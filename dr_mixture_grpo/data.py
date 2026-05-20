"""Data loading + per-rollout reward computation for Dr.Mixture-GRPO.

The Dr.Mixture advantage A_i = r_teacher_i - r_mean_student(qid) is computed
LIVE inside the trainer at every step (under the current LoRA-wrapped
policy), not here. This module only:

  1. Loads the teacher rollouts (re-tokenized under the student tokenizer,
     reusing offline_grpo's ``load_rollouts`` for Path-A safety).
  2. Computes the per-rollout teacher reward (Math_Verifier for MATH,
     numeric extraction for GSM8K/SVAMP/ASDiv), matching ``offline_grpo``.
  3. Builds the HF ``Dataset`` and ``(qid, rid)`` offline lookup the trainer
     consumes.

See ``trainer.py:_compute_live_student_baseline`` for the live r_mean_student
computation.
"""

from __future__ import annotations

import importlib.util
import os
import sys

from datasets import Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "offline_grpo"))  # for configs.py


def _load_module_from_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_offline_data = _load_module_from_path(
    "_offline_grpo_data",
    os.path.join(_PROJECT_ROOT, "offline_grpo", "data.py"),
)
_compute_correctness_gsm8k = _offline_data._compute_correctness_gsm8k
_compute_correctness_math = _offline_data._compute_correctness_math
extract_boxed_answer = _offline_data.extract_boxed_answer
extract_gsm8k_answer = _offline_data.extract_gsm8k_answer

# Use DG-offline's Path-A loader: re-tokenize teacher response under the
# student tokenizer (teacher-tokenizer-agnostic; no OOV truncation needed).
_dg_loader = _load_module_from_path(
    "_dg_offline_loader",
    os.path.join(_PROJECT_ROOT, "DG-offline", "teacher_agnostic_loader.py"),
)
load_rollouts_text = _dg_loader.load_rollouts_text


def compute_rewards(records: list[dict]) -> list[dict]:
    """Add the per-rollout teacher ``reward`` field, in place."""
    if not records:
        return records

    dataset_type = records[0].get("dataset_type", "math")
    for rec in records:
        ds = rec.get("dataset_type", dataset_type)
        extracted = rec.get("extracted_answer")
        if ds in ("gsm8k", "svamp", "asdiv"):
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
    return records


def build_training_dataset(records: list[dict]) -> Dataset:
    """HF Dataset sorted by (qid, rid) so TRL groups num_generations per prompt."""
    records = sorted(records, key=lambda r: (r["question_id"], r["run_id"]))
    prompts, answers, qids = [], [], []
    for rec in records:
        prompts.append([
            {"role": "system", "content": rec["system_prompt"]},
            {"role": "user", "content": rec["problem"]},
        ])
        answers.append(rec["ground_truth"])
        qids.append(rec["question_id"])
    return Dataset.from_dict({"prompt": prompts, "answer": answers, "question_id": qids})


def build_offline_lookup(records: list[dict]) -> dict[tuple[int, int], dict]:
    """Lookup keyed by (question_id, run_id) with what the trainer needs."""
    lookup: dict[tuple[int, int], dict] = {}
    for rec in records:
        lookup[(rec["question_id"], rec["run_id"])] = {
            "completion_ids": rec["completion_ids"],
            "reward": rec["reward"],
            "response": rec["response"],
            "problem": rec["problem"],
            "ground_truth": rec["ground_truth"],
        }
    return lookup
