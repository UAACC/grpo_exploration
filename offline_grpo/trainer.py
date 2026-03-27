"""OfflineGRPOTrainer: off-policy GRPO with importance-sampling correction.

Overrides only ``_generate_and_score_completions`` so that TRL's standard
``_compute_loss`` handles the IS-corrected clipped objective automatically.

Key idea:
  - ``old_per_token_logps`` is set to the *behavior policy*'s logprobs.
  - TRL computes ``ratio = exp(current_logps - old_per_token_logps)``
    which becomes ``π_target / π_behavior`` — the IS correction ratio.
  - TRL then applies PPO-style clipping to this ratio, giving us a correct
    off-policy GRPO loss with zero changes to ``_compute_loss``.

LoRA reference model:
  - With ``peft_config`` supplied and ``ref_model=None``, TRL uses
    ``model.disable_adapter()`` to get base-model logprobs for the KL penalty.
  - With ``ref_sync_steps > 0``, the reference adapter is periodically
    updated to a snapshot of the current LoRA weights, so KL measures
    drift from a recent checkpoint rather than from the original base model.
"""

import copy
from typing import Any, Union

import torch
from accelerate.utils import gather_object, is_peft_model
from trl import GRPOTrainer
from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.trainer.utils import pad


class OfflineGRPOTrainer(GRPOTrainer):
    """GRPOTrainer that uses pre-computed offline rollouts instead of live generation."""

    def __init__(self, *args, offline_data: dict, ref_sync_steps: int = 0, **kwargs):
        """
        Args:
            offline_data: dict keyed by ``(question_id, run_id)`` with values
                containing ``behavior_logprobs``, ``completion_ids``,
                ``advantage``, ``reward``, ``response``.
            ref_sync_steps: If > 0, sync the reference LoRA adapter to the
                current weights every N training steps. 0 = never sync
                (reference is the original base model via disable_adapter).
        """
        super().__init__(*args, **kwargs)
        self._offline_data = offline_data
        self._ref_sync_steps = ref_sync_steps
        self._ref_adapter_state = None
        self._steps_since_ref_sync = 0

        # Initialize reference adapter state if syncing is enabled
        if self._ref_sync_steps > 0 and self.beta != 0.0:
            self._sync_ref_adapter()

    # ------------------------------------------------------------------
    # Reference adapter sync
    # ------------------------------------------------------------------
    def _sync_ref_adapter(self):
        """Snapshot current LoRA weights as the reference for KL computation."""
        unwrapped = self.accelerator.unwrap_model(self.model)
        if is_peft_model(unwrapped):
            self._ref_adapter_state = {
                k: v.detach().clone()
                for k, v in unwrapped.named_parameters()
                if "lora_" in k
            }
            self._steps_since_ref_sync = 0

    def _get_ref_logprobs(self, model, prompt_completion_ids, attention_mask, logits_to_keep):
        """Compute reference logprobs using the appropriate reference model.

        If ref_sync_steps > 0, temporarily swap in the reference LoRA weights.
        Otherwise, use disable_adapter() (original base model).
        """
        if self._ref_sync_steps > 0 and self._ref_adapter_state is not None:
            # Swap in reference LoRA weights, compute, swap back
            unwrapped = self.accelerator.unwrap_model(model)
            current_state = {
                k: v.detach().clone()
                for k, v in unwrapped.named_parameters()
                if "lora_" in k
            }
            # Load reference weights
            for name, param in unwrapped.named_parameters():
                if name in self._ref_adapter_state:
                    param.data.copy_(self._ref_adapter_state[name])

            ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                model, prompt_completion_ids, attention_mask, logits_to_keep,
            )

            # Restore current weights
            for name, param in unwrapped.named_parameters():
                if name in current_state:
                    param.data.copy_(current_state[name])

            return ref_per_token_logps
        else:
            # Original behavior: disable adapter to get base model logprobs
            with self.accelerator.unwrap_model(model).disable_adapter():
                ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    model, prompt_completion_ids, attention_mask, logits_to_keep,
                )
            return ref_per_token_logps

    # ------------------------------------------------------------------
    # The main method we override
    # ------------------------------------------------------------------
    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # ---- 1. Tokenize prompts (same as upstream) --------------------
        prompts = [x["prompt"] for x in inputs]
        original_prompts = copy.deepcopy(prompts)
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]

        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        from transformers import Trainer as _Trainer
        prompt_inputs = _Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids = prompt_inputs["input_ids"]
        prompt_mask = prompt_inputs["attention_mask"]

        # ---- 2. Look up offline completions ----------------------------
        batch_size = len(inputs)
        num_gen = self.num_generations
        # inputs arrive duplicated: num_gen copies per question
        # question_ids repeat like [q0,q0,q0,q0, q1,q1,q1,q1, ...]
        question_ids = [x.get("question_id") for x in inputs]
        run_ids = [i % num_gen for i in range(batch_size)]

        completion_id_lists = []
        behavior_logprob_lists = []
        advantages_list = []

        for qid, rid in zip(question_ids, run_ids):
            rec = self._offline_data.get((qid, rid))
            if rec is None:
                raise KeyError(
                    f"No offline data for (question_id={qid}, run_id={rid}). "
                    "Check that num_generations matches the rollout file."
                )
            completion_id_lists.append(rec["completion_ids"])
            behavior_logprob_lists.append(rec["behavior_logprobs"])
            advantages_list.append(rec["advantage"])

        # ---- 3. Pad completions & build masks --------------------------
        max_comp_len = max(len(c) for c in completion_id_lists)
        if self.max_completion_length is not None:
            max_comp_len = min(max_comp_len, self.max_completion_length)

        completion_ids_tensors = []
        completion_mask_tensors = []
        old_logps_tensors = []

        for cids, blps in zip(completion_id_lists, behavior_logprob_lists):
            # Truncate if needed
            cids = cids[:max_comp_len]
            blps = blps[:max_comp_len]
            seq_len = len(cids)

            # Replace None logprobs with 0.0 (shouldn't happen, but be safe)
            blps = [lp if lp is not None else 0.0 for lp in blps]

            cid_t = torch.tensor(cids, dtype=torch.long, device=device)
            mask_t = torch.ones(seq_len, dtype=torch.int, device=device)
            lp_t = torch.tensor(blps, dtype=torch.float32, device=device)

            # Pad to max_comp_len
            pad_len = max_comp_len - seq_len
            if pad_len > 0:
                cid_t = torch.cat([cid_t, torch.full((pad_len,), self.pad_token_id, dtype=torch.long, device=device)])
                mask_t = torch.cat([mask_t, torch.zeros(pad_len, dtype=torch.int, device=device)])
                lp_t = torch.cat([lp_t, torch.zeros(pad_len, dtype=torch.float32, device=device)])

            completion_ids_tensors.append(cid_t)
            completion_mask_tensors.append(mask_t)
            old_logps_tensors.append(lp_t)

        completion_ids = torch.stack(completion_ids_tensors)       # (B, C)
        completion_mask = torch.stack(completion_mask_tensors)     # (B, C)
        old_per_token_logps = torch.stack(old_logps_tensors)      # (B, C)
        advantages = torch.tensor(advantages_list, dtype=torch.float32, device=device)  # (B,)

        # ---- 4. Compute ref logprobs (for KL penalty) ------------------
        ref_per_token_logps = None
        if self.beta != 0.0:
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            logits_to_keep = completion_ids.size(1)

            with torch.no_grad():
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                    )
                else:
                    ref_per_token_logps = self._get_ref_logprobs(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                    )

            # Sync reference adapter if it's time
            if self._ref_sync_steps > 0 and mode == "train":
                self._steps_since_ref_sync += 1
                if self._steps_since_ref_sync >= self._ref_sync_steps:
                    self._sync_ref_adapter()

        # ---- 5. Decode completions for logging -------------------------
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        inputs_is_conversational = is_conversational(inputs[0])
        if inputs_is_conversational:
            completions = []
            for prompt, text in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + text}])
        else:
            completions = completions_text

        # ---- 6. Log metrics --------------------------------------------
        completion_lengths = completion_mask.sum(1)
        if mode == "train":
            full_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            self.state.num_input_tokens_seen += self.accelerator.gather(full_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Reward logging
        rewards = torch.tensor(
            [self._offline_data[(qid, rid)]["reward"] for qid, rid in zip(question_ids, run_ids)],
            device=device,
        )
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(rewards.std().item())

        # Log prompt/completion text
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        self._logs["advantages"].extend(advantages.tolist())

        # ---- 7. Build output dict --------------------------------------
        num_items_in_batch = agg_completion_lengths.sum()

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "num_items_in_batch": num_items_in_batch,
        }
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps

        return output
