"""Smoke-test the new teacher-agnostic loader against the legacy shortcut loader.

Runs both on a single MATH rollout shard and reports:
  1. Record counts match.
  2. Reward and advantage distributions match (both should be functions only of
     `response` text, not of how it was tokenized).
  3. How often student-vocab `completion_ids` from Path A differ from teacher
     `completion_ids` from Path B (expected: rarely, since Qwen-Math and Qwen
     student share IDs 0–151,935; differences should track the OOV-truncation
     events the legacy loader would have triggered).
  4. Spot-check that the teacher's response decoded under the *student*
     tokenizer matches the original `response` text (round-trip sanity).

Usage:
    python smoke_test_loader.py [/path/to/rollouts.jsonl]
"""

from __future__ import annotations

import os
import sys

# Make the legacy loader importable for comparison
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "offline_grpo"))

from transformers import AutoTokenizer, AutoConfig  # noqa: E402

import data as legacy  # offline_grpo/data.py  # noqa: E402
import teacher_agnostic_loader as new  # noqa: E402


STUDENT_MODEL = "/scratch/mrli/models/Qwen2.5-0.5B-Instruct"
DEFAULT_ROLLOUT = "/scratch/mrli/rollouts/math_teacher/rollouts_shard_0.jsonl"


def main():
    rollout_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROLLOUT
    print(f"=== Smoke test: teacher_agnostic_loader vs offline_grpo/data ===")
    print(f"Rollout file: {rollout_path}")
    print(f"Student model: {STUDENT_MODEL}")

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    student_vocab_size = AutoConfig.from_pretrained(STUDENT_MODEL).vocab_size
    print(f"Student vocab size (config): {student_vocab_size}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print()

    # ---- Legacy (Path B) ----
    print("Loading via legacy loader (offline_grpo/data.load_rollouts) ...")
    legacy_records = legacy.load_rollouts(rollout_path, vocab_size=student_vocab_size)
    legacy_records = legacy.compute_rewards_and_advantages(legacy_records)
    print(f"  -> {len(legacy_records)} records")

    # ---- New (Path A) ----
    print("Loading via new loader (teacher_agnostic_loader.load_rollouts_text) ...")
    new_records = new.load_rollouts_text(rollout_path, tokenizer)
    new_records = new.compute_rewards_and_advantages(new_records)
    print(f"  -> {len(new_records)} records")
    print()

    # ---- 1. Record counts ----
    assert len(legacy_records) == len(new_records), \
        f"Record count mismatch: legacy={len(legacy_records)} new={len(new_records)}"
    print(f"[1/4] Record counts match: {len(new_records)}")

    # Index by (qid, rid) for one-to-one comparison
    legacy_by_key = {(r["question_id"], r["run_id"]): r for r in legacy_records}
    new_by_key = {(r["question_id"], r["run_id"]): r for r in new_records}
    common_keys = sorted(set(legacy_by_key) & set(new_by_key))
    assert len(common_keys) == len(legacy_records), "Key sets differ"

    # ---- 2. Rewards and advantages match ----
    n_reward_mismatch = 0
    n_adv_mismatch = 0
    for k in common_keys:
        if abs(legacy_by_key[k]["reward"] - new_by_key[k]["reward"]) > 1e-9:
            n_reward_mismatch += 1
        if abs(legacy_by_key[k]["advantage"] - new_by_key[k]["advantage"]) > 1e-6:
            n_adv_mismatch += 1
    print(f"[2/4] Reward mismatches: {n_reward_mismatch} / {len(common_keys)}")
    print(f"      Advantage mismatches: {n_adv_mismatch} / {len(common_keys)}")
    assert n_reward_mismatch == 0, "Reward distributions diverge — would invalidate Stage 1 smoke test"

    # ---- 3. Completion-ID divergence between Path A and Path B ----
    # Path B IDs come from teacher; Path A IDs come from re-tokenizing the response.
    # For Qwen-Math/Qwen students they should match in most cases.
    n_id_match = 0
    n_id_diff = 0
    n_legacy_truncated = 0
    n_length_diff_examples_shown = 0
    for k in common_keys:
        leg_ids = legacy_by_key[k]["completion_ids"]
        new_ids = new_by_key[k]["completion_ids"]
        if leg_ids == new_ids:
            n_id_match += 1
        else:
            n_id_diff += 1
            # Legacy may have been truncated at the first OOV (>= vocab_size) token.
            full_response = legacy_by_key[k]["response"]
            full_response_new = new_by_key[k]["response"]
            assert full_response == full_response_new, "Response text mismatch — unexpected"
            # Was it OOV-truncated?
            re_tok_full = tokenizer(full_response, add_special_tokens=False)["input_ids"]
            if len(re_tok_full) > len(leg_ids):
                n_legacy_truncated += 1
            if n_length_diff_examples_shown < 3 and len(leg_ids) != len(new_ids):
                print(
                    f"      Example diff at qid={k[0]} rid={k[1]}: "
                    f"legacy_len={len(leg_ids)} new_len={len(new_ids)}"
                )
                n_length_diff_examples_shown += 1
    print(f"[3/4] Path A vs Path B completion_ids match: {n_id_match} / {len(common_keys)}")
    print(f"      Diverged: {n_id_diff} (of which {n_legacy_truncated} look like legacy OOV truncations)")

    # ---- 4. Round-trip: tokenize-then-decode of response should be lossless modulo whitespace ----
    n_roundtrip_examples = 0
    n_roundtrip_match = 0
    for k in common_keys[:50]:  # 50-record spot-check is enough
        text = new_by_key[k]["response"]
        ids = new_by_key[k]["completion_ids"]
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        n_roundtrip_examples += 1
        if decoded == text:
            n_roundtrip_match += 1
    print(f"[4/4] Round-trip exact match: {n_roundtrip_match} / {n_roundtrip_examples}")
    if n_roundtrip_match < n_roundtrip_examples:
        # Show one mismatch
        for k in common_keys[:50]:
            text = new_by_key[k]["response"]
            ids = new_by_key[k]["completion_ids"]
            decoded = tokenizer.decode(ids, skip_special_tokens=False)
            if decoded != text:
                print(f"      First non-exact roundtrip at qid={k[0]} rid={k[1]}:")
                print(f"        len(text)={len(text)} len(decoded)={len(decoded)}")
                # Show the first differing chunk
                for i, (a, b) in enumerate(zip(text, decoded)):
                    if a != b:
                        print(f"        first diff at char {i}: text={text[max(0,i-10):i+10]!r} decoded={decoded[max(0,i-10):i+10]!r}")
                        break
                break

    print()
    print("=== Smoke test PASSED ===")


if __name__ == "__main__":
    main()
