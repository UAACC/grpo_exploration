"""Method A: Unified Mixture GRPO Trainer.

Extends TRL's GRPOTrainer to combine online student rollouts and offline
teacher rollouts into a single unified group for advantage computation.

All G+K completions (G student + K teacher) are pooled together:
- Shared advantage baseline (mixed mean)
- Single unified loss
- Each completion uses its own behavior policy for importance ratio
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


class UnifiedMixtureGRPOTrainer(GRPOTrainer):
    """GRPOTrainer that unifies online student rollouts with offline teacher rollouts."""

    def __init__(self, *args, teacher_data: dict, num_teacher_per_prompt: int = 1,
                 ref_sync_steps: int = 0, dataset_type: str = "gsm8k", **kwargs):
        """
        Args:
            teacher_data: dict keyed by question_id with teacher rollouts.
            num_teacher_per_prompt: number of teacher completions to include per prompt.
            ref_sync_steps: sync reference LoRA adapter every N steps. 0 = never.
            dataset_type: "gsm8k" or "math" — determines reward computation.
        """
        super().__init__(*args, **kwargs)
        self._teacher_data = teacher_data
        self._num_teacher = num_teacher_per_prompt
        self._ref_sync_steps = ref_sync_steps
        self._dataset_type = dataset_type
        self._ref_adapter_state = None
        self._steps_since_ref_sync = 0

        if self._ref_sync_steps > 0 and self.beta != 0.0:
            self._sync_ref_adapter()

    # ------------------------------------------------------------------
    # Reference adapter sync (same as offline GRPO)
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
    # Override _compute_loss: mask KL for teacher completions
    # ------------------------------------------------------------------
    def _compute_loss(self, model, inputs):
        """Compute GRPO loss with KL penalty only on student completions.

        This is a near-copy of TRL's GRPOTrainer._compute_loss, with one
        critical change: per_token_kl is multiplied by kl_mask so that
        teacher completions (kl_mask=0) contribute zero KL penalty.

        Why: KL(π_θ || π_ref) should be estimated from on-policy samples
        (student-generated). Including off-policy teacher completions in the
        KL estimate biases it toward regions the student rarely visits.
        """
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep, compute_entropy=True,
        )

        entropy_mask = None
        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, completion_mask, 1 - self.top_entropy_quantile)

        # KL divergence (k3 estimator), masked for teacher completions
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps) - 1
            )
            # kl_mask: (B,) → (B,1), zeros out teacher rows
            kl_mask = inputs["kl_mask"].unsqueeze(1)
            per_token_kl = per_token_kl * kl_mask

        # Clipped surrogate (identical to TRL)
        advantages = inputs["advantages"]
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(f"Unknown importance sampling level: {self.importance_sampling_level}")

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Metrics (same as TRL)
        mode = "train" if self.model.training else "eval"
        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        from trl.trainer.grpo_trainer import nanmin, nanmax
        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())
        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss

    # ------------------------------------------------------------------
    # Main override: generate student rollouts + merge with teacher
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

        # Extract completion part (strip prompt)
        prompt_len = prompt_ids.size(1)
        student_completion_ids = student_outputs[:, prompt_len:]  # (batch_size, C_stu)

        # Get student behavior logprobs
        student_full_ids = student_outputs
        student_full_mask = (student_full_ids != self.pad_token_id).int()
        student_comp_len = student_completion_ids.size(1)

        with torch.no_grad():
            student_old_logps, _ = self._get_per_token_logps_and_entropies(
                self.model, student_full_ids, student_full_mask, student_comp_len,
            )

        # Compute student rewards
        student_comp_mask = (student_completion_ids != self.pad_token_id).int()
        student_texts = self.processing_class.batch_decode(
            student_completion_ids, skip_special_tokens=True
        )

        # Get question_ids and ground truths
        question_ids = [x.get("question_id") for x in inputs]
        answers = [x.get("answer") for x in inputs]

        student_rewards = []
        for text, answer in zip(student_texts, answers):
            if self._dataset_type == "math":
                extracted = extract_boxed_answer(text)
                reward = compute_math_correctness(extracted, answer)
            else:
                extracted = extract_gsm8k_answer(text)
                reward = compute_gsm8k_correctness(extracted, answer)
            student_rewards.append(reward)

        # ---- 3. Retrieve teacher completions (OFFLINE) ------------------
        # For each unique prompt, get teacher completions
        teacher_completion_id_lists = []
        teacher_logprob_lists = []
        teacher_rewards = []

        unique_qids = question_ids[::num_gen]
        unique_answers = answers[::num_gen]

        for qid, answer in zip(unique_qids, unique_answers):
            tdata = self._teacher_data.get(qid)
            if tdata is None:
                raise KeyError(f"No teacher data for question_id={qid}")
            for k in range(min(self._num_teacher, len(tdata["runs"]))):
                run = tdata["runs"][k]
                teacher_completion_id_lists.append(run["completion_ids"])
                teacher_logprob_lists.append(run["behavior_logprobs"])
                teacher_rewards.append(run["reward"])

        # ---- 4. Compute unified advantages ------------------------------
        # Group by prompt: G student + K teacher completions
        all_advantages = []
        stu_idx = 0
        tea_idx = 0
        n_zero_std_groups = 0
        n_all_wrong_groups = 0
        n_all_correct_groups = 0
        for q in range(num_unique):
            # Gather rewards for this prompt
            group_rewards = []
            student_group_rewards = []
            # Student rewards for this prompt
            for g in range(num_gen):
                group_rewards.append(student_rewards[stu_idx + g])
                student_group_rewards.append(student_rewards[stu_idx + g])
            # Teacher rewards for this prompt
            n_tea = min(self._num_teacher, len(self._teacher_data.get(unique_qids[q], {}).get("runs", [])))
            for k in range(n_tea):
                group_rewards.append(teacher_rewards[tea_idx + k])

            # Normalize
            mean_r = sum(group_rewards) / len(group_rewards)
            std_r = (sum((r - mean_r) ** 2 for r in group_rewards) / len(group_rewards)) ** 0.5
            eps = 1e-4

            # Track student-only reward diversity
            stu_std = (sum((r - sum(student_group_rewards)/len(student_group_rewards)) ** 2
                          for r in student_group_rewards) / len(student_group_rewards)) ** 0.5
            if stu_std <= eps:
                n_zero_std_groups += 1
                if all(r == 0.0 for r in student_group_rewards):
                    n_all_wrong_groups += 1
                else:
                    n_all_correct_groups += 1

            # Student advantages
            for g in range(num_gen):
                adv = (group_rewards[g] - mean_r) / (std_r + eps) if std_r > eps else 0.0
                all_advantages.append(("student", adv))
            # Teacher advantages
            for k in range(n_tea):
                adv = (group_rewards[num_gen + k] - mean_r) / (std_r + eps) if std_r > eps else 0.0
                all_advantages.append(("teacher", adv))

            stu_idx += num_gen
            tea_idx += n_tea

        # ---- 5. Pad and merge all completions ---------------------------
        # Pad teacher completions to tensors
        tea_max_len = max((len(c) for c in teacher_completion_id_lists), default=0)
        stu_max_len = student_completion_ids.size(1) if student_completion_ids.numel() > 0 else 0
        max_comp_len = max(stu_max_len, tea_max_len)
        if self.max_completion_length is not None:
            max_comp_len = min(max_comp_len, self.max_completion_length)

        # Re-pad student completions to max_comp_len
        if stu_max_len < max_comp_len:
            pad_len = max_comp_len - stu_max_len
            student_completion_ids = torch.cat([
                student_completion_ids,
                torch.full((student_completion_ids.size(0), pad_len), self.pad_token_id, dtype=torch.long, device=device)
            ], dim=1)
            student_comp_mask = torch.cat([
                student_comp_mask,
                torch.zeros(student_comp_mask.size(0), pad_len, dtype=torch.int, device=device)
            ], dim=1)
            student_old_logps = torch.cat([
                student_old_logps,
                torch.zeros(student_old_logps.size(0), pad_len, dtype=torch.float32, device=device)
            ], dim=1)
        elif stu_max_len > max_comp_len:
            student_completion_ids = student_completion_ids[:, :max_comp_len]
            student_comp_mask = student_comp_mask[:, :max_comp_len]
            student_old_logps = student_old_logps[:, :max_comp_len]

        # Build teacher tensors
        teacher_cid_tensors = []
        teacher_mask_tensors = []
        teacher_logp_tensors = []
        for cids, blps in zip(teacher_completion_id_lists, teacher_logprob_lists):
            cids = cids[:max_comp_len]
            blps = blps[:max_comp_len]
            seq_len = len(cids)
            blps = [lp if lp is not None else 0.0 for lp in blps]

            cid_t = torch.tensor(cids, dtype=torch.long, device=device)
            mask_t = torch.ones(seq_len, dtype=torch.int, device=device)
            lp_t = torch.tensor(blps, dtype=torch.float32, device=device)

            pad_len = max_comp_len - seq_len
            if pad_len > 0:
                cid_t = torch.cat([cid_t, torch.full((pad_len,), self.pad_token_id, dtype=torch.long, device=device)])
                mask_t = torch.cat([mask_t, torch.zeros(pad_len, dtype=torch.int, device=device)])
                lp_t = torch.cat([lp_t, torch.zeros(pad_len, dtype=torch.float32, device=device)])

            teacher_cid_tensors.append(cid_t)
            teacher_mask_tensors.append(mask_t)
            teacher_logp_tensors.append(lp_t)

        # Interleave: for each prompt, student completions first, then teacher
        all_completion_ids = []
        all_completion_mask = []
        all_old_logps = []
        all_adv_values = []
        all_kl_mask = []  # 1.0 for student (KL applies), 0.0 for teacher (no KL)

        # Expand prompt_ids to match total completions (student + teacher per prompt)
        all_prompt_ids = []
        all_prompt_mask = []

        stu_idx = 0
        tea_idx = 0
        adv_idx = 0
        for q in range(num_unique):
            n_tea = min(self._num_teacher, len(self._teacher_data.get(unique_qids[q], {}).get("runs", [])))
            # Student completions
            for g in range(num_gen):
                all_completion_ids.append(student_completion_ids[stu_idx + g])
                all_completion_mask.append(student_comp_mask[stu_idx + g])
                all_old_logps.append(student_old_logps[stu_idx + g])
                _, adv = all_advantages[adv_idx]
                all_adv_values.append(adv)
                all_kl_mask.append(1.0)
                all_prompt_ids.append(prompt_ids[q * num_gen])
                all_prompt_mask.append(prompt_mask[q * num_gen])
                adv_idx += 1
            # Teacher completions
            for k in range(n_tea):
                all_completion_ids.append(teacher_cid_tensors[tea_idx + k])
                all_completion_mask.append(teacher_mask_tensors[tea_idx + k])
                all_old_logps.append(teacher_logp_tensors[tea_idx + k])
                _, adv = all_advantages[adv_idx]
                all_adv_values.append(adv)
                all_kl_mask.append(0.0)
                all_prompt_ids.append(prompt_ids[q * num_gen])
                all_prompt_mask.append(prompt_mask[q * num_gen])
                adv_idx += 1
            stu_idx += num_gen
            tea_idx += n_tea

        completion_ids = torch.stack(all_completion_ids)
        completion_mask = torch.stack(all_completion_mask)
        old_per_token_logps = torch.stack(all_old_logps)
        advantages = torch.tensor(all_adv_values, dtype=torch.float32, device=device)
        kl_mask = torch.tensor(all_kl_mask, dtype=torch.float32, device=device)  # (N,)
        prompt_ids = torch.stack(all_prompt_ids)
        prompt_mask = torch.stack(all_prompt_mask)

        # ---- 6. Compute ref logprobs ------------------------------------
        ref_per_token_logps = None
        if self.beta != 0.0:
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            logits_to_keep = completion_ids.size(1)

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
            completion_ids, skip_special_tokens=True
        )
        inputs_is_conversational = is_conversational(inputs[0])
        if inputs_is_conversational:
            completions = []
            for prompt, text in zip(prompts, completions_text[:len(prompts)]):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + text}])
        else:
            completions = completions_text

        # ---- 8. Log metrics ---------------------------------------------
        completion_lengths = completion_mask.sum(1)
        if mode == "train":
            full_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            self.state.num_input_tokens_seen += self.accelerator.gather(full_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())

        # Student reward logging
        stu_rewards_t = torch.tensor(student_rewards, device=device)
        self._metrics[mode]["reward"].append(stu_rewards_t.mean().item())
        self._metrics[mode]["reward_std"].append(stu_rewards_t.std().item())
        self._metrics[mode]["frac_reward_zero_std"].append(n_zero_std_groups / max(num_unique, 1))
        self._metrics[mode]["frac_all_wrong"].append(n_all_wrong_groups / max(num_unique, 1))
        self._metrics[mode]["frac_all_correct"].append(n_all_correct_groups / max(num_unique, 1))
        tea_rewards_t = torch.tensor(teacher_rewards, device=device)
        self._metrics[mode]["teacher_reward"].append(tea_rewards_t.mean().item())

        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text[:len(prompts_text)]))
        self._logs["advantages"].extend(advantages.tolist()[:len(prompts_text)])

        # ---- 9. Build output dict ---------------------------------------
        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "kl_mask": kl_mask,  # 1.0 for student, 0.0 for teacher
        }
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps

        return output
