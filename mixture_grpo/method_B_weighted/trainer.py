"""Method B: Weighted Mixture GRPO Trainer.

Extends TRL's GRPOTrainer to combine online student rollouts and offline
teacher rollouts with separate loss terms weighted by lambda.

Key design: teacher advantage uses the student's online group statistics
as baseline, linking teacher signal to student's current ability.

L_total = L_online + lambda * L_offline + beta * L_KL
"""

import copy
from typing import Any, Union

import torch
from accelerate.utils import gather_object, is_peft_model
from trl import GRPOTrainer
from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.trainer.utils import pad

from configs import extract_gsm8k_answer, extract_boxed_answer
from data import compute_gsm8k_correctness, compute_math_correctness


class WeightedMixtureGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with separate online/offline losses weighted by lambda."""

    def __init__(self, *args, teacher_data: dict, offline_weight: float = 0.3,
                 num_teacher_per_prompt: int = 5, ref_sync_steps: int = 0,
                 dataset_type: str = "gsm8k", **kwargs):
        """
        Args:
            teacher_data: dict keyed by question_id with teacher rollouts.
            offline_weight: lambda — weight for the offline teacher loss.
            num_teacher_per_prompt: number of teacher completions per prompt.
            ref_sync_steps: sync reference LoRA adapter every N steps. 0 = never.
            dataset_type: "gsm8k" or "math" — determines reward computation.
        """
        super().__init__(*args, **kwargs)
        self._teacher_data = teacher_data
        self._offline_weight = offline_weight
        self._num_teacher = num_teacher_per_prompt
        self._ref_sync_steps = ref_sync_steps
        self._dataset_type = dataset_type
        self._ref_adapter_state = None
        self._steps_since_ref_sync = 0

        if self._ref_sync_steps > 0 and self.beta != 0.0:
            self._sync_ref_adapter()

    # ------------------------------------------------------------------
    # Reference adapter sync
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
    # Override _compute_loss to add offline teacher loss
    # ------------------------------------------------------------------
    def _compute_loss(self, model, inputs):
        """Compute online loss (via super) + offline teacher loss."""
        # Parent _compute_loss returns a scalar loss tensor (not a tuple)
        loss = super()._compute_loss(model, inputs)

        # Add offline teacher loss if teacher data is present in inputs
        mode = "train" if self.model.training else "eval"
        if "teacher_completion_ids" in inputs:
            teacher_loss = self._compute_offline_loss(
                model,
                inputs["teacher_prompt_ids"], inputs["teacher_prompt_mask"],
                inputs["teacher_completion_ids"], inputs["teacher_completion_mask"],
                inputs["teacher_old_per_token_logps"], inputs["teacher_advantages"],
            )
            loss = loss + self._offline_weight * teacher_loss
            self._metrics[mode]["offline_loss"].append(teacher_loss.detach().item())

        return loss

    def _compute_offline_loss(self, model, prompt_ids, prompt_mask,
                              completion_ids, completion_mask,
                              old_per_token_logps, advantages):
        """Compute clipped GRPO loss for teacher completions (per-token clipping)."""
        mode = "train" if self.model.training else "eval"
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # Current policy logprobs on teacher completions
        per_token_logps, _ = self._get_per_token_logps_and_entropies(
            model, prompt_completion_ids, attention_mask, logits_to_keep,
        )

        # Per-token importance ratio (matching TRL's GRPO implementation)
        log_ratio = per_token_logps - old_per_token_logps  # (B, T)
        ratio = torch.exp(log_ratio)                        # per-token exp

        # Per-token clipped surrogate
        eps = 0.2
        clipped_ratio = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
        per_token_loss1 = ratio * advantages.unsqueeze(1)
        per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # Average over tokens, then over batch
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1.0)).mean()

        # ── Offline-specific diagnostics ──────────────────────────────
        with torch.no_grad():
            completion_token_count = completion_mask.sum().clamp(min=1.0)

            # IS ratio statistics
            valid_ratios = ratio[completion_mask.bool()]
            self._metrics[mode]["offline/is_ratio_mean"].append(valid_ratios.mean().item())
            self._metrics[mode]["offline/is_ratio_std"].append(valid_ratios.std().item())
            self._metrics[mode]["offline/is_ratio_median"].append(valid_ratios.median().item())

            # Clip fraction: how many tokens have ratio outside [1-eps, 1+eps]
            is_clipped = ((ratio < 1.0 - eps) | (ratio > 1.0 + eps)) & completion_mask.bool()
            clip_frac = is_clipped.float().sum() / completion_token_count
            self._metrics[mode]["offline/clip_fraction"].append(clip_frac.item())

            # Log-ratio stats (more numerically stable view)
            valid_log_ratios = log_ratio[completion_mask.bool()]
            self._metrics[mode]["offline/log_ratio_mean"].append(valid_log_ratios.mean().item())
            self._metrics[mode]["offline/log_ratio_std"].append(valid_log_ratios.std().item())

            # Ratio percentiles and clip direction breakdown
            self._metrics[mode]["offline/ratio_p5"].append(valid_ratios.quantile(0.05).item())
            self._metrics[mode]["offline/ratio_p95"].append(valid_ratios.quantile(0.95).item())
            clip_low = ((ratio < 1.0 - eps) & completion_mask.bool()).float().sum() / completion_token_count
            clip_high = ((ratio > 1.0 + eps) & completion_mask.bool()).float().sum() / completion_token_count
            self._metrics[mode]["offline/clip_fraction_low"].append(clip_low.item())
            self._metrics[mode]["offline/clip_fraction_high"].append(clip_high.item())

            # Per-completion clip fraction (mean across completions)
            per_comp_clipped = ((ratio < 1.0 - eps) | (ratio > 1.0 + eps)) & completion_mask.bool()
            per_comp_clip_frac = per_comp_clipped.float().sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1.0)
            self._metrics[mode]["offline/per_comp_clip_frac_mean"].append(per_comp_clip_frac.mean().item())
            self._metrics[mode]["offline/per_comp_clip_frac_max"].append(per_comp_clip_frac.max().item())

            # Teacher advantage statistics
            self._metrics[mode]["offline/advantage_mean"].append(advantages.mean().item())
            self._metrics[mode]["offline/advantage_std"].append(advantages.std().item())
            frac_zero_adv = (advantages.abs() < 1e-4).float().mean()
            self._metrics[mode]["offline/frac_zero_advantage"].append(frac_zero_adv.item())

            # Combined effective signal: (1 - H1) * (1 - H2)
            # H1 = frac_zero_advantage, H2 = clip_fraction
            effective = (1.0 - frac_zero_adv.item()) * (1.0 - clip_frac.item())
            self._metrics[mode]["offline/effective_signal"].append(effective)

        return loss

    # ------------------------------------------------------------------
    # Main override: generate student rollouts + prepare teacher batch
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
        # TRL's dataloader already duplicates each prompt num_generations times.
        # We generate 1 completion per row (matching TRL's design).
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
        unique_answers = answers[::num_gen]

        # Student rewards
        student_rewards = []
        for text, answer in zip(student_texts, answers):
            if self._dataset_type == "math":
                extracted = extract_boxed_answer(text)
                reward = compute_math_correctness(extracted, answer)
            else:
                extracted = extract_gsm8k_answer(text)
                reward = compute_gsm8k_correctness(extracted, answer)
            student_rewards.append(reward)

        # ---- 3. Compute ONLINE advantages (student only) ----------------
        online_advantages = []
        stu_idx = 0
        online_stats = []  # store (mu, std) per prompt for teacher advantage
        n_zero_std_groups = 0  # count groups where all student rewards are identical
        n_all_wrong_groups = 0
        n_all_correct_groups = 0
        for q in range(num_unique):
            group_rewards = []
            for g in range(num_gen):
                group_rewards.append(student_rewards[stu_idx + g])

            mean_r = sum(group_rewards) / len(group_rewards)
            std_r = (sum((r - mean_r) ** 2 for r in group_rewards) / len(group_rewards)) ** 0.5
            eps = 1e-4
            online_stats.append((mean_r, std_r))

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

        # Log reward diversity metrics (hypothesis 1)
        self._metrics[mode]["frac_reward_zero_std"].append(n_zero_std_groups / max(num_unique, 1))
        self._metrics[mode]["frac_all_wrong"].append(n_all_wrong_groups / max(num_unique, 1))
        self._metrics[mode]["frac_all_correct"].append(n_all_correct_groups / max(num_unique, 1))

        # ---- 4. Prepare teacher batch (OFFLINE) -------------------------
        # Teacher advantage uses student's online stats as baseline.
        # Build teacher tensors with same batch_size as student (num_gen per prompt)
        # so TRL can split them in sync.
        teacher_completion_id_lists = []
        teacher_logprob_lists = []
        teacher_advantages = []
        teacher_prompt_ids_list = []
        teacher_prompt_mask_list = []

        for q in range(num_unique):
            qid = unique_qids[q]
            tdata = self._teacher_data.get(qid)
            if tdata is None:
                continue

            mean_r, std_r = online_stats[q]
            eps = 1e-4

            # Use up to num_teacher rollouts per prompt
            n_tea = min(self._num_teacher, len(tdata["runs"]))
            for k in range(n_tea):
                run = tdata["runs"][k]
                tea_reward = run["reward"]
                tea_adv = (tea_reward - mean_r) / (std_r + eps) if std_r > eps else 0.0

                teacher_completion_id_lists.append(run["completion_ids"])
                teacher_logprob_lists.append(run["behavior_logprobs"])
                teacher_advantages.append(tea_adv)
                teacher_prompt_ids_list.append(prompt_ids[q * num_gen])
                teacher_prompt_mask_list.append(prompt_mask[q * num_gen])

        # ---- 5. Handle student completion padding -----------------------
        if self.max_completion_length is not None and student_completion_ids.size(1) > self.max_completion_length:
            student_completion_ids = student_completion_ids[:, :self.max_completion_length]
            student_comp_mask = student_comp_mask[:, :self.max_completion_length]
            student_old_logps = student_old_logps[:, :self.max_completion_length]

        # ---- 6. Compute ref logprobs ------------------------------------
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

        # ---- 7. Decode completions for logging --------------------------
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

        # ---- 8. Log metrics ---------------------------------------------
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
        if teacher_advantages:
            self._metrics[mode]["offline_advantage"].append(
                sum(teacher_advantages) / len(teacher_advantages)
            )

        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        self._logs["advantages"].extend(advantages.tolist())

        # ---- 9. Build output dict ---------------------------------------
        # Include both student (online) and teacher (offline) data.
        # Teacher tensors use "teacher_" prefix so TRL splits them in sync
        # with student data during gradient accumulation.
        completion_lengths = student_comp_mask.sum(dim=1)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
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

        # Add teacher data to output dict for auto-splitting
        if teacher_completion_id_lists:
            max_tea_len = max(len(c) for c in teacher_completion_id_lists)
            if self.max_completion_length is not None:
                max_tea_len = min(max_tea_len, self.max_completion_length)

            tea_cid_tensors = []
            tea_mask_tensors = []
            tea_logp_tensors = []
            for cids, blps in zip(teacher_completion_id_lists, teacher_logprob_lists):
                cids = cids[:max_tea_len]
                blps = blps[:max_tea_len]
                seq_len = len(cids)
                blps = [lp if lp is not None else 0.0 for lp in blps]

                cid_t = torch.tensor(cids, dtype=torch.long, device=device)
                mask_t = torch.ones(seq_len, dtype=torch.int, device=device)
                lp_t = torch.tensor(blps, dtype=torch.float32, device=device)

                pad_len = max_tea_len - seq_len
                if pad_len > 0:
                    cid_t = torch.cat([cid_t, torch.full((pad_len,), self.pad_token_id, dtype=torch.long, device=device)])
                    mask_t = torch.cat([mask_t, torch.zeros(pad_len, dtype=torch.int, device=device)])
                    lp_t = torch.cat([lp_t, torch.zeros(pad_len, dtype=torch.float32, device=device)])

                tea_cid_tensors.append(cid_t)
                tea_mask_tensors.append(mask_t)
                tea_logp_tensors.append(lp_t)

            output["teacher_prompt_ids"] = torch.stack(teacher_prompt_ids_list)
            output["teacher_prompt_mask"] = torch.stack(teacher_prompt_mask_list)
            output["teacher_completion_ids"] = torch.stack(tea_cid_tensors)
            output["teacher_completion_mask"] = torch.stack(tea_mask_tensors)
            output["teacher_old_per_token_logps"] = torch.stack(tea_logp_tensors)
            output["teacher_advantages"] = torch.tensor(teacher_advantages, dtype=torch.float32, device=device)

        return output
