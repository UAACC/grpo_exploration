"""Data loading for offline BC: load teacher rollouts into (input_ids, labels) pairs.

Labels have -100 on prompt tokens and padding, so loss is computed only on
teacher completion tokens.
"""

import json

import torch
from torch.utils.data import Dataset

from configs import MATH_SYSTEM_PROMPT


class BCDataset(Dataset):
    """Dataset that yields (input_ids, labels, attention_mask) for BC training.

    Each sample is one teacher completion: prompt_ids ++ completion_ids.
    Labels mask out the prompt (set to -100) so only completion tokens
    contribute to the cross-entropy loss.
    """

    def __init__(
        self,
        rollout_path: str,
        tokenizer,
        max_length: int = 2304,
        vocab_size: int | None = None,
        filter_correct_only: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        records = _load_rollouts(rollout_path, vocab_size=vocab_size)
        if filter_correct_only:
            before = len(records)
            records = [r for r in records if r["reward"] > 0]
            print(f"  Filtered to correct-only: {before} -> {len(records)}")

        for rec in records:
            prompt_ids, comp_ids = self._build_ids(rec)
            if len(comp_ids) == 0:
                continue

            input_ids = prompt_ids + comp_ids
            # Truncate from the right if exceeding max_length
            input_ids = input_ids[:max_length]
            prompt_len = len(prompt_ids)
            # Labels: -100 for prompt, actual token ids for completion
            labels = [-100] * min(prompt_len, len(input_ids))
            if len(input_ids) > prompt_len:
                labels += input_ids[prompt_len:]

            self.samples.append({
                "input_ids": input_ids,
                "labels": labels,
            })

        print(f"  BCDataset: {len(self.samples)} samples, max_length={max_length}")

    def _build_ids(self, rec: dict) -> tuple[list[int], list[int]]:
        """Tokenize the prompt via chat template, return (prompt_ids, completion_ids)."""
        chat = [
            {"role": "system", "content": rec["system_prompt"]},
            {"role": "user", "content": rec["problem"]},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        comp_ids = rec["completion_ids"]
        return prompt_ids, comp_ids

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "input_ids": torch.tensor(s["input_ids"], dtype=torch.long),
            "labels": torch.tensor(s["labels"], dtype=torch.long),
        }


class BCCollator:
    """Pads a batch of BC samples to the same length (right-padding)."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(f["input_ids"].size(0) for f in features)
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        for f in features:
            seq_len = f["input_ids"].size(0)
            pad_len = max_len - seq_len

            input_ids_list.append(
                torch.cat([f["input_ids"], torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
            )
            labels_list.append(
                torch.cat([f["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
            )
            attention_mask_list.append(
                torch.cat([torch.ones(seq_len, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
            )

        return {
            "input_ids": torch.stack(input_ids_list),
            "labels": torch.stack(labels_list),
            "attention_mask": torch.stack(attention_mask_list),
        }


def _load_rollouts(jsonl_path: str, vocab_size: int | None = None) -> list[dict]:
    """Load rollout JSONL into a flat list of per-completion records.

    Reuses logic from offline_grpo/data.py but also computes reward inline
    so we can optionally filter correct-only.
    """
    from configs import extract_boxed_answer
    from math_verify import parse, verify

    records = []
    truncated = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            qid = item["question_id"]
            gt = item["ground_truth_answer"]
            for run in item["runs"]:
                cids = run["completion_ids"]

                # Truncate at first out-of-vocab token
                if vocab_size is not None:
                    for idx, tid in enumerate(cids):
                        if tid >= vocab_size:
                            cids = cids[:idx]
                            truncated += 1
                            break

                # Compute reward for optional filtering
                extracted = run.get("extracted_answer") or run.get("boxed_answer")
                if extracted is None:
                    extracted = extract_boxed_answer(run["response"])
                try:
                    reward = 2.0 if verify(parse(extracted), parse(gt)) is True else 0.0
                except Exception:
                    reward = 0.0

                records.append({
                    "question_id": qid,
                    "problem": item["original_problem"],
                    "ground_truth": gt,
                    "system_prompt": item.get("system_prompt", MATH_SYSTEM_PROMPT),
                    "completion_ids": cids,
                    "reward": reward,
                })

    if truncated > 0:
        print(f"  Warning: {truncated} completions truncated at out-of-vocab tokens (vocab_size={vocab_size})")
    correct = sum(1 for r in records if r["reward"] > 0)
    print(f"  Loaded {len(records)} completions, {correct} correct ({100*correct/len(records):.1f}%)")
    return records
