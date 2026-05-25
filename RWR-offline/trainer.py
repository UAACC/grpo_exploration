"""Delightful offline GRPO trainer.

Replaces importance-sampling ratios and PPO clipping with a sigmoid gate
based on "delight" = advantage x surprisal. Surprisal is computed under the
learner's current policy, so no behavior policy logprobs are needed.

The key change from Reward Weighted Regression:
  - old_per_token_logps is set to the CURRENT policy's logprobs (not behavior's),
    neutralizing the IS ratio to 1.0 so PPO clipping becomes a no-op.

"""

import copy
from typing import Any, Union

import torch
from accelerate.utils import gather_object, is_peft_model
from trl import GRPOTrainer
from trl.data_utils import is_conversational, maybe_apply_chat_template


class RWROfflineTrainer(GRPOTrainer):
    """GRPOTrainer with reward weighted regression."""

    def __init__(
        self,
        *args,
        offline_data: dict,
        dg_temperature: float = 1.0,
        dg_gating: str = "completion",
        ref_sync_steps: int = 0,
        **kwargs,
    ):
        """
        Args:
            offline_data: dict keyed by (question_id, run_id) with values
                containing completion_ids, advantage, reward, response.
                Behavior logprobs are NOT required (ignored if present).
            dg_temperature: deprecated
            dg_gating: deprecated. How to compute surprisal for the gate.
                "completion" = mean per-token surprisal over the completion.
                "token" = per-token gating (each token gets its own gate).
            ref_sync_steps: Sync reference LoRA adapter every N steps.
                0 = never sync (use original base model via disable_adapter).
        """
        super().__init__(*args, **kwargs)
        self._offline_data = offline_data
        self._dg_temperature = dg_temperature
        self._dg_gating = dg_gating
        self._ref_sync_steps = ref_sync_steps
        self._ref_adapter_state = None
        self._steps_since_ref_sync = 0

        if self._ref_sync_steps > 0 and self.beta != 0.0:
            self._sync_ref_adapter()

    def _sync_ref_adapter(self):
        unwrapped = self.accelerator.unwrap_model(self.model)
        if is_peft_model(unwrapped):
            self._ref_adapter_state = {
                k: v.detach().clone()
                for k, v in unwrapped.named_parameters()
                if "lora_" in k
            }
            self._steps_since_ref_sync = 0

    def _get_ref_logprobs(self, model, prompt_completion_ids, attention_mask, logits_to_keep):
        if self._ref_sync_steps > 0 and self._ref_adapter_state is not None:
            unwrapped = self.accelerator.unwrap_model(model)
            current_state = {
                k: v.detach().clone()
                for k, v in unwrapped.named_parameters()
                if "lora_" in k
            }
            for name, param in unwrapped.named_parameters():
                if name in self._ref_adapter_state:
                    param.data.copy_(self._ref_adapter_state[name])
            ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                model, prompt_completion_ids, attention_mask, logits_to_keep,
            )
            for name, param in unwrapped.named_parameters():
                if name in current_state:
                    param.data.copy_(current_state[name])
            return ref_per_token_logps
        else:
            with self.accelerator.unwrap_model(model).disable_adapter():
                ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    model, prompt_completion_ids, attention_mask, logits_to_keep,
                )
            return ref_per_token_logps

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # ---- 1. Tokenize prompts ------------------------------------------
        prompts = [x["prompt"] for x in inputs]
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

        # ---- 2. Look up offline completions -------------------------------
        batch_size = len(inputs)
        num_gen = self.num_generations
        question_ids = [x.get("question_id") for x in inputs]
        run_ids = [i % num_gen for i in range(batch_size)]

        completion_id_lists = []
        advantages_list = []

        for qid, rid in zip(question_ids, run_ids):
            rec = self._offline_data.get((qid, rid))
            if rec is None:
                raise KeyError(
                    f"No offline data for (question_id={qid}, run_id={rid}). "
                    "Check that num_generations matches the rollout file."
                )
            completion_id_lists.append(rec["completion_ids"])
            advantages_list.append(rec["advantage"])

        # ---- 3. Pad completions & build masks -----------------------------
        max_comp_len = max(len(c) for c in completion_id_lists)
        if self.max_completion_length is not None:
            max_comp_len = min(max_comp_len, self.max_completion_length)

        completion_ids_tensors = []
        completion_mask_tensors = []

        for cids in completion_id_lists:
            cids = cids[:max_comp_len]
            seq_len = len(cids)
            cid_t = torch.tensor(cids, dtype=torch.long, device=device)
            mask_t = torch.ones(seq_len, dtype=torch.int, device=device)
            pad_len = max_comp_len - seq_len
            if pad_len > 0:
                cid_t = torch.cat([cid_t, torch.full((pad_len,), self.pad_token_id, dtype=torch.long, device=device)])
                mask_t = torch.cat([mask_t, torch.zeros(pad_len, dtype=torch.int, device=device)])
            completion_ids_tensors.append(cid_t)
            completion_mask_tensors.append(mask_t)

        completion_ids = torch.stack(completion_ids_tensors)
        completion_mask = torch.stack(completion_mask_tensors)
        advantages = torch.tensor(advantages_list, dtype=torch.float32, device=device)

        # ---- 4. Compute current policy logprobs ---------------------------
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        with torch.no_grad():
            current_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                self.model, prompt_completion_ids, attention_mask, logits_to_keep,
            )

        # ---- 5. Compute DG gate -------------------------------------------
        surprisal = -current_per_token_logps  # (B, C), non-negative

        if self._dg_gating == "completion":
            # Mean per-token surprisal over the completion
            lengths = completion_mask.sum(dim=1).clamp(min=1).float()
            completion_surprisal = (surprisal * completion_mask).sum(dim=1) / lengths  # (B,)
            delight = advantages * completion_surprisal
            gate = torch.sigmoid(delight / self._dg_temperature)  # (B,)
            gate = torch.ones_like(gate)
            gated_advantages = gate * advantages
        elif self._dg_gating == "token":
            # Per-token gating: each token gets its own gate
            per_token_delight = advantages.unsqueeze(1) * surprisal  # (B, C)
            per_token_gate = torch.sigmoid(per_token_delight / self._dg_temperature)  # (B, C)
            # For TRL compatibility, we fold per-token gates into advantages
            # by using a single scalar advantage weighted by the mean gate
            mean_gate = (per_token_gate * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)
            mean_gate = torch.ones_like(mean_gate)
            gated_advantages = mean_gate * advantages
        else:
            raise ValueError(f"Unknown dg_gating mode: {self._dg_gating}")

        # ---- 6. Set old_logps = current logps (neutralize IS ratio) -------
        # With ratio = exp(new - old) = exp(new - current) ≈ 1.0 at the
        # start of each step, PPO clipping becomes a no-op. The effective
        # loss is: -gated_advantage * log π_current, i.e. gated REINFORCE.
        old_per_token_logps = current_per_token_logps.detach()

        # ---- 7. Compute ref logprobs (for KL penalty) ---------------------
        ref_per_token_logps = None
        if self.beta != 0.0:
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep,
                    )
                else:
                    ref_per_token_logps = self._get_ref_logprobs(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep,
                    )
            if self._ref_sync_steps > 0 and mode == "train":
                self._steps_since_ref_sync += 1
                if self._steps_since_ref_sync >= self._ref_sync_steps:
                    self._sync_ref_adapter()

        # ---- 8. Log metrics -----------------------------------------------
        completion_lengths = completion_mask.sum(1)
        if mode == "train":
            full_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            self.state.num_input_tokens_seen += self.accelerator.gather(full_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        rewards = torch.tensor(
            [self._offline_data[(qid, rid)]["reward"] for qid, rid in zip(question_ids, run_ids)],
            device=device,
        )
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(rewards.std().item())

        # DG-specific metrics
        if self._dg_gating == "completion":
            self._metrics[mode]["dg/gate_mean"].append(gate.mean().item())
            self._metrics[mode]["dg/gate_min"].append(gate.min().item())
            self._metrics[mode]["dg/gate_max"].append(gate.max().item())
            self._metrics[mode]["dg/delight_mean"].append(delight.mean().item())
            self._metrics[mode]["dg/surprisal_mean"].append(completion_surprisal.mean().item())

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

        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        self._logs["advantages"].extend(gated_advantages.tolist())

        # ---- 9. Build output dict -----------------------------------------
        num_items_in_batch = agg_completion_lengths.sum()

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": gated_advantages,
            "old_per_token_logps": old_per_token_logps,
            "num_items_in_batch": num_items_in_batch,
        }
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps

        return output
