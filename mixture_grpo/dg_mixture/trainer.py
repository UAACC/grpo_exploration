"""DG-Mixture GRPO Trainer.

Combines online student rollouts (standard GRPO loss) with offline teacher
rollouts weighted by DG-offline's sigmoid gate on delight = advantage * surprisal.

Key differences from the Weighted Mixture trainer (`method_B_weighted/`):
  - Teacher advantage uses TEACHER-ONLY group statistics (matches DG-offline's
    calibration of the gate's eta parameter), NOT the student's online stats.
  - Teacher loss is gated REINFORCE: each completion's advantage is multiplied
    by sigma(advantage * mean_surprisal / eta) where surprisal is computed
    under the learner's CURRENT policy (no behavior logprobs needed).
  - IS ratio is neutralized by setting old_per_token_logps = current_logps
    at generation time, so the PPO clip becomes a no-op and the effective
    teacher loss is gated REINFORCE: -gate * advantage * log pi_current.

L_total = L_online (PPO clipped + KL on student)
        + lambda * L_dg_teacher (gated REINFORCE, no KL on teacher)

Reference: Osband (2026), arXiv:2603.20521 (DG paper).
DG-offline implementation: ../../DG-offline/trainer.py
"""

import copy
from typing import Any, Union

import torch
import torch.nn.functional as F
from accelerate.utils import gather_object, is_peft_model
from trl import GRPOTrainer
from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.trainer.utils import pad

# Reward function is passed via constructor (resolved from shared/datasets_registry.py)


class DGMixtureGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with online student loss + DG-gated offline teacher loss."""

    def __init__(self, *args, teacher_data: dict, dg_offline_weight: float = 0.3,
                 dg_temperature: float = 0.5, dg_gating: str = "completion",
                 num_teacher_per_prompt: int = 4, ref_sync_steps: int = 0,
                 dataset_type: str = "math", reward_func=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._teacher_data = teacher_data
        self._reward_func = reward_func
        self._dg_offline_weight = dg_offline_weight
        self._dg_temperature = dg_temperature
        self._dg_gating = dg_gating
        self._num_teacher = num_teacher_per_prompt
        self._ref_sync_steps = ref_sync_steps
        self._dataset_type = dataset_type
        self._ref_adapter_state = None
        self._steps_since_ref_sync = 0

        if self._ref_sync_steps > 0 and self.beta != 0.0:
            self._sync_ref_adapter()

    # ------------------------------------------------------------------
    # Reference adapter sync (verbatim from Method B)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Override _compute_loss to add DG teacher loss
    # ------------------------------------------------------------------
    def _compute_loss(self, model, inputs):
        """L = L_online (super) + lambda * L_dg_teacher."""
        # Online student loss (PPO clipped + KL vs ref) — standard TRL path
        loss = super()._compute_loss(model, inputs)

        mode = "train" if self.model.training else "eval"
        if "teacher_completion_ids" in inputs:
            dg_loss = self._compute_dg_teacher_loss(
                model,
                inputs["teacher_prompt_ids"], inputs["teacher_prompt_mask"],
                inputs["teacher_completion_ids"], inputs["teacher_completion_mask"],
                inputs["teacher_old_per_token_logps"], inputs["teacher_advantages"],
            )
            loss = loss + self._dg_offline_weight * dg_loss
            self._metrics[mode]["dg_teacher_loss"].append(dg_loss.detach().item())

        return loss

    def _compute_dg_teacher_loss(self, model, prompt_ids, prompt_mask,
                                  completion_ids, completion_mask,
                                  old_per_token_logps, gated_advantages):
        """DG-gated teacher loss.

        The advantages passed in are ALREADY pre-multiplied by the DG gate
        (gate is computed at generation time in _generate_and_score_completions).
        Since old_per_token_logps == current_logps at the start of each outer
        step, the PPO ratio ~= 1 and the clipped surrogate reduces to:

            loss = -gated_advantage * log pi_current

        which is gated REINFORCE — exactly DG's update rule.
        """
        mode = "train" if self.model.training else "eval"
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # Current policy logprobs on teacher completions (gradient flows here)
        per_token_logps, _ = self._get_per_token_logps_and_entropies(
            model, prompt_completion_ids, attention_mask, logits_to_keep,
        )

        # PPO-style surrogate (with ratio ~ 1, clip is a no-op for the first
        # microbatch; engages only if multiple inner PPO epochs drift the model)
        log_ratio = per_token_logps - old_per_token_logps
        ratio = torch.exp(log_ratio)
        eps = 0.2
        clipped_ratio = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
        per_token_loss1 = ratio * gated_advantages.unsqueeze(1)
        per_token_loss2 = clipped_ratio * gated_advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # Average over tokens, then over batch
        loss = ((per_token_loss * completion_mask).sum(dim=1)
                / completion_mask.sum(dim=1).clamp(min=1.0)).mean()

        return loss

    # ------------------------------------------------------------------
    # Main override: generate student rollouts + prepare DG-gated teacher batch
    # ------------------------------------------------------------------
    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # ---- 1. Tokenize prompts ----------------------------------------
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

        # ---- 2. Generate student completions (ONLINE) -------------------
        num_gen = self.num_generations
        batch_size = len(inputs)
        num_unique = batch_size // num_gen

        with torch.no_grad():
            student_outputs = self.model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                generation_config=self.generation_config,
                pad_token_id=self.pad_token_id,
            )

        prompt_len = prompt_ids.size(1)
        student_completion_ids = student_outputs[:, prompt_len:]

        student_full_ids = student_outputs
        student_full_mask = (student_full_ids != self.pad_token_id).int()
        student_comp_len = student_completion_ids.size(1)

        with torch.no_grad():
            student_old_logps, _ = self._get_per_token_logps_and_entropies(
                self.model, student_full_ids, student_full_mask, student_comp_len,
            )

        student_comp_mask = (student_completion_ids != self.pad_token_id).int()
        student_texts = self.processing_class.batch_decode(
            student_completion_ids, skip_special_tokens=True
        )

        question_ids = [x.get("question_id") for x in inputs]
        answers = [x.get("answer") for x in inputs]
        unique_qids = question_ids[::num_gen]

        # Student rewards (using reward function from dataset registry)
        student_rewards = []
        for text, answer in zip(student_texts, answers):
            student_rewards.append(self._reward_func(text, answer))

        # ---- 3. Compute ONLINE advantages (student-only group stats) ----
        online_advantages = []
        stu_idx = 0
        n_zero_std_groups = 0
        n_all_wrong_groups = 0
        n_all_correct_groups = 0
        for q in range(num_unique):
            group_rewards = []
            for g in range(num_gen):
                group_rewards.append(student_rewards[stu_idx + g])

            mean_r = sum(group_rewards) / len(group_rewards)
            std_r = (sum((r - mean_r) ** 2 for r in group_rewards) / len(group_rewards)) ** 0.5
            eps = 1e-4

            if std_r <= eps:
                n_zero_std_groups += 1
                if all(r == 0.0 for r in group_rewards):
                    n_all_wrong_groups += 1
                else:
                    n_all_correct_groups += 1

            for g in range(num_gen):
                adv = (group_rewards[g] - mean_r) / (std_r + eps) if std_r > eps else 0.0
                online_advantages.append(adv)
            stu_idx += num_gen

        advantages = torch.tensor(online_advantages, dtype=torch.float32, device=device)

        self._metrics[mode]["frac_reward_zero_std"].append(n_zero_std_groups / max(num_unique, 1))
        self._metrics[mode]["frac_all_wrong"].append(n_all_wrong_groups / max(num_unique, 1))
        self._metrics[mode]["frac_all_correct"].append(n_all_correct_groups / max(num_unique, 1))

        # ---- 4. Prepare teacher batch (OFFLINE, DG-gated) ---------------
        # Use TEACHER-ONLY group advantages from teacher_data (pre-computed at
        # load time via offline_grpo.data.compute_rewards_and_advantages).
        teacher_completion_id_lists = []
        teacher_raw_advantages = []
        teacher_prompt_ids_list = []
        teacher_prompt_mask_list = []

        for q in range(num_unique):
            qid = unique_qids[q]
            tdata = self._teacher_data.get(qid)
            if tdata is None:
                continue

            n_tea = min(self._num_teacher, len(tdata["runs"]))
            for k in range(n_tea):
                run = tdata["runs"][k]
                teacher_completion_id_lists.append(run["completion_ids"])
                teacher_raw_advantages.append(run["advantage"])
                teacher_prompt_ids_list.append(prompt_ids[q * num_gen])
                teacher_prompt_mask_list.append(prompt_mask[q * num_gen])

        # ---- 5. Build padded teacher tensors and apply DG gate ----------
        teacher_output_keys = {}
        if teacher_completion_id_lists:
            max_tea_len = max(len(c) for c in teacher_completion_id_lists)
            if self.max_completion_length is not None:
                max_tea_len = min(max_tea_len, self.max_completion_length)

            tea_cid_tensors = []
            tea_mask_tensors = []
            for cids in teacher_completion_id_lists:
                cids = cids[:max_tea_len]
                seq_len = len(cids)

                cid_t = torch.tensor(cids, dtype=torch.long, device=device)
                mask_t = torch.ones(seq_len, dtype=torch.int, device=device)

                pad_len = max_tea_len - seq_len
                if pad_len > 0:
                    cid_t = torch.cat([cid_t, torch.full((pad_len,), self.pad_token_id, dtype=torch.long, device=device)])
                    mask_t = torch.cat([mask_t, torch.zeros(pad_len, dtype=torch.int, device=device)])

                tea_cid_tensors.append(cid_t)
                tea_mask_tensors.append(mask_t)

            teacher_completion_ids_t = torch.stack(tea_cid_tensors)
            teacher_completion_mask_t = torch.stack(tea_mask_tensors)
            teacher_prompt_ids_t = torch.stack(teacher_prompt_ids_list)
            teacher_prompt_mask_t = torch.stack(teacher_prompt_mask_list)
            teacher_raw_advantages_t = torch.tensor(
                teacher_raw_advantages, dtype=torch.float32, device=device
            )

            # Compute current policy logprobs on teacher completions for the DG gate
            tea_full_ids = torch.cat([teacher_prompt_ids_t, teacher_completion_ids_t], dim=1)
            tea_attention_mask = torch.cat([teacher_prompt_mask_t, teacher_completion_mask_t], dim=1)
            tea_logits_to_keep = teacher_completion_ids_t.size(1)

            with torch.no_grad():
                current_teacher_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model, tea_full_ids, tea_attention_mask, tea_logits_to_keep,
                )

            # Per-token surprisal under current student policy
            surprisal = -current_teacher_logps  # (B_tea, T)

            if self._dg_gating == "completion":
                lengths = teacher_completion_mask_t.sum(dim=1).clamp(min=1).float()
                completion_surprisal = (surprisal * teacher_completion_mask_t).sum(dim=1) / lengths
                delight = teacher_raw_advantages_t * completion_surprisal
                gate = torch.sigmoid(delight / self._dg_temperature)
                gated_teacher_advantages = gate * teacher_raw_advantages_t
            elif self._dg_gating == "token":
                per_token_delight = teacher_raw_advantages_t.unsqueeze(1) * surprisal
                per_token_gate = torch.sigmoid(per_token_delight / self._dg_temperature)
                mean_gate = ((per_token_gate * teacher_completion_mask_t).sum(dim=1)
                             / teacher_completion_mask_t.sum(dim=1).clamp(min=1))
                gate = mean_gate
                completion_surprisal = (surprisal * teacher_completion_mask_t).sum(dim=1) / teacher_completion_mask_t.sum(dim=1).clamp(min=1).float()
                delight = teacher_raw_advantages_t * completion_surprisal
                gated_teacher_advantages = mean_gate * teacher_raw_advantages_t
            else:
                raise ValueError(f"Unknown dg_gating mode: {self._dg_gating}")

            # Neutralize IS ratio: setting old_logps = current_logps makes
            # the PPO ratio ~ 1 at the start of each step, so the clipped
            # surrogate reduces to gated REINFORCE.
            teacher_old_logps_t = current_teacher_logps.detach()

            # DG-specific metrics (matching DG-offline naming)
            self._metrics[mode]["dg/gate_mean"].append(gate.mean().item())
            self._metrics[mode]["dg/gate_min"].append(gate.min().item())
            self._metrics[mode]["dg/gate_max"].append(gate.max().item())
            self._metrics[mode]["dg/delight_mean"].append(delight.mean().item())
            self._metrics[mode]["dg/surprisal_mean"].append(completion_surprisal.mean().item())
            self._metrics[mode]["dg/teacher_advantage_mean"].append(teacher_raw_advantages_t.mean().item())
            self._metrics[mode]["dg/teacher_advantage_std"].append(teacher_raw_advantages_t.std().item())

            teacher_output_keys = {
                "teacher_prompt_ids": teacher_prompt_ids_t,
                "teacher_prompt_mask": teacher_prompt_mask_t,
                "teacher_completion_ids": teacher_completion_ids_t,
                "teacher_completion_mask": teacher_completion_mask_t,
                "teacher_old_per_token_logps": teacher_old_logps_t,
                "teacher_advantages": gated_teacher_advantages,
            }

        # ---- 6. Handle student completion padding -----------------------
        if self.max_completion_length is not None and student_completion_ids.size(1) > self.max_completion_length:
            student_completion_ids = student_completion_ids[:, :self.max_completion_length]
            student_comp_mask = student_comp_mask[:, :self.max_completion_length]
            student_old_logps = student_old_logps[:, :self.max_completion_length]

        # ---- 7. Compute ref logprobs (student only, for KL) ------------
        ref_per_token_logps = None
        if self.beta != 0.0:
            prompt_completion_ids = torch.cat([prompt_ids, student_completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, student_comp_mask], dim=1)
            logits_to_keep = student_completion_ids.size(1)

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

        # ---- 8. Decode for logging --------------------------------------
        completions_text = self.processing_class.batch_decode(
            student_completion_ids, skip_special_tokens=True
        )
        inputs_is_conversational = is_conversational(inputs[0])
        if inputs_is_conversational:
            completions = []
            for prompt, text in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + text}])
        else:
            completions = completions_text

        # ---- 9. Log metrics ---------------------------------------------
        completion_lengths = student_comp_mask.sum(1)
        if mode == "train":
            full_mask = torch.cat([prompt_mask, student_comp_mask], dim=1)
            self.state.num_input_tokens_seen += self.accelerator.gather(full_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())

        stu_rewards_t = torch.tensor(student_rewards, device=device)
        self._metrics[mode]["reward"].append(stu_rewards_t.mean().item())
        self._metrics[mode]["reward_std"].append(stu_rewards_t.std().item())

        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        self._logs["advantages"].extend(advantages.tolist())

        # ---- 10. Build output dict --------------------------------------
        num_items_in_batch = agg_completion_lengths.sum()

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": student_completion_ids,
            "completion_mask": student_comp_mask,
            "advantages": advantages,
            "old_per_token_logps": student_old_logps,
            "num_items_in_batch": num_items_in_batch,
        }
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps

        # Teacher data goes through TRL's split mechanism (must be in output dict)
        output.update(teacher_output_keys)

        return output
