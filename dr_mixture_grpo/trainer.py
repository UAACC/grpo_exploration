"""DrMixtureGRPOTrainer: offline-teacher / live-student policy gradient.

Loss = -A * log pi_student(token | context)

where A_i = r_teacher_i - r_mean_student(qid). The teacher reward
``r_teacher_i`` comes from the precomputed offline rollouts. The
per-problem student baseline ``r_mean_student(qid)`` is recomputed at every
training step under the CURRENT policy, by sampling K_s completions from
the LoRA-wrapped model in eval mode and scoring them with the canonical
Math_Verifier. This is the v2 ("live refresh") variant the algorithm
specifies; no static snapshot is used.

Pattern follows ``DG-offline/trainer.py``: inherit from ``trl.GRPOTrainer``,
override ``_generate_and_score_completions`` to load offline teacher
completions and set ``old_per_token_logps = current_per_token_logps`` so
TRL's stock loss reduces to ``-advantage * log pi`` (PPO clip becomes a
no-op).

See also Liu et al. (2025), arXiv:2503.20783 ("Dr. GRPO") for the
``/std`` removal rationale.
"""

from typing import Any, Union

import torch
from accelerate.utils import gather_object, is_peft_model
from trl import GRPOTrainer
from trl.data_utils import is_conversational, maybe_apply_chat_template


_NUMERIC_DATASETS = ("gsm8k", "svamp", "asdiv")


def _clean_gold(gold: str, dataset_name: str, extract_numeric):
    """For numeric datasets, strip "#### N" prefix so Math_Verifier compares
    against the clean number. For MATH-style, return as-is."""
    if dataset_name in _NUMERIC_DATASETS and gold is not None:
        cleaned = extract_numeric(gold)
        if cleaned is not None:
            return cleaned
        return gold.strip()
    return gold


class DrMixtureGRPOTrainer(GRPOTrainer):
    """Offline teacher data + live student baseline + clean -A log pi loss."""

    def __init__(
        self,
        *args,
        offline_data: dict,
        dataset_name: str,
        is_equiv_multi,
        extract_numeric,
        K_s: int = 5,
        baseline_temperature: float = 0.7,
        baseline_top_p: float = 1.0,
        baseline_max_new_tokens: int = 1024,
        ref_sync_steps: int = 0,
        **kwargs,
    ):
        """
        Args:
            offline_data: dict keyed by (question_id, run_id) with values
                containing completion_ids, reward, response, problem,
                ground_truth.
            dataset_name: one of {"gsm8k","math","svamp","asdiv"}; controls
                reward scale (numeric=1.0, MATH=2.0) and gold cleaning.
            is_equiv_multi: callable(problem, response, gold) -> bool, the
                canonical Math_Verifier scorer.
            extract_numeric: callable(text) -> str|None for "#### N" gold
                cleanup on numeric datasets (offline_grpo.configs.extract_gsm8k_answer).
            K_s: number of student samples per unique in-batch qid per step.
            baseline_temperature: sampling temperature for student samples
                (>0; greedy is meaningless under HF generate without seed-jitter).
            baseline_top_p: nucleus-sampling cutoff for student samples.
            baseline_max_new_tokens: per-sample completion budget.
            ref_sync_steps: snapshot reference LoRA every N steps for KL.
                0 = always use base model via disable_adapter().
        """
        super().__init__(*args, **kwargs)
        self._offline_data = offline_data
        self._dataset_name = dataset_name
        self._is_equiv_multi = is_equiv_multi
        self._extract_numeric = extract_numeric
        self._K_s = K_s
        self._baseline_temperature = baseline_temperature
        self._baseline_top_p = baseline_top_p
        self._baseline_max_new_tokens = baseline_max_new_tokens
        self._r_max = 1.0 if dataset_name in _NUMERIC_DATASETS else 2.0
        self._ref_sync_steps = ref_sync_steps
        self._ref_adapter_state = None
        self._steps_since_ref_sync = 0

        if self._ref_sync_steps > 0 and self.beta != 0.0:
            self._sync_ref_adapter()

    # ------------------------------------------------------------------
    # Reference-adapter machinery (verbatim from DG-offline / offline_grpo).
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
    # Live student baseline: generate K_s samples per unique qid under the
    # current LoRA-wrapped policy, score with Math_Verifier, return mean.
    # ------------------------------------------------------------------
    def _compute_live_student_baseline(self, unique_qids, unique_prompts_text,
                                       unique_problems, unique_golds):
        """Return {qid: r_mean_student(qid) under current policy}."""
        if not unique_qids:
            return {}

        tokenizer = self.processing_class
        device = self.accelerator.device

        # Left-pad prompts for batched generation.
        tokenizer.padding_side = "left"
        enc = tokenizer(
            unique_prompts_text,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        unwrapped = self.accelerator.unwrap_model(self.model)

        # Temporarily enable kv-cache + eval mode for generation.
        was_training = self.model.training
        prev_use_cache = getattr(unwrapped.config, "use_cache", False)
        unwrapped.config.use_cache = True
        self.model.eval()

        try:
            with torch.inference_mode():
                gen_out = unwrapped.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    max_new_tokens=self._baseline_max_new_tokens,
                    do_sample=True,
                    temperature=self._baseline_temperature,
                    top_p=self._baseline_top_p,
                    num_return_sequences=self._K_s,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=False,
                )
        finally:
            unwrapped.config.use_cache = prev_use_cache
            if was_training:
                self.model.train()

        # gen_out shape: (N * K_s, prompt_len + new_tokens). Slice off prompt.
        prompt_len = enc["input_ids"].shape[1]
        completion_ids = gen_out[:, prompt_len:]
        texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        # Score and aggregate per qid.
        N = len(unique_qids)
        baseline: dict[int, float] = {}
        for i, qid in enumerate(unique_qids):
            problem = unique_problems[i]
            cleaned_gold = _clean_gold(unique_golds[i], self._dataset_name,
                                       self._extract_numeric)
            r_sum = 0.0
            for k in range(self._K_s):
                text = texts[i * self._K_s + k]
                try:
                    correct = bool(self._is_equiv_multi(problem, text, cleaned_gold))
                except Exception:
                    correct = False
                if correct:
                    r_sum += self._r_max
            baseline[int(qid)] = r_sum / self._K_s
        return baseline

    # ------------------------------------------------------------------
    # Main override.
    # ------------------------------------------------------------------
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

        # ---- 2. Look up offline teacher completions ----------------------
        batch_size = len(inputs)
        num_gen = self.num_generations
        question_ids = [x.get("question_id") for x in inputs]
        run_ids = [i % num_gen for i in range(batch_size)]
        answers = [x.get("answer") for x in inputs]

        completion_id_lists = []
        teacher_rewards = []
        for qid, rid in zip(question_ids, run_ids):
            rec = self._offline_data.get((qid, rid))
            if rec is None:
                raise KeyError(
                    f"No offline data for (question_id={qid}, run_id={rid}). "
                    "Check that num_generations matches the rollout file."
                )
            completion_id_lists.append(rec["completion_ids"])
            teacher_rewards.append(float(rec["reward"]))

        # ---- 3. Live student baseline: generate K_s under CURRENT policy.
        # Dedupe in-batch qids to avoid regenerating the same prompt num_gen times.
        seen_qids: dict[int, tuple[str, str, str]] = {}
        for qid, p_text, x in zip(question_ids, prompts_text, inputs):
            qid_i = int(qid)
            if qid_i not in seen_qids:
                # Extract raw problem from the user message in the chat list.
                problem = ""
                chat = x.get("prompt")
                if isinstance(chat, list):
                    for msg in chat:
                        if msg.get("role") == "user":
                            problem = msg.get("content", "")
                            break
                seen_qids[qid_i] = (p_text, problem, x.get("answer", ""))

        unique_qids = list(seen_qids.keys())
        unique_prompts_text = [seen_qids[q][0] for q in unique_qids]
        unique_problems = [seen_qids[q][1] for q in unique_qids]
        unique_golds = [seen_qids[q][2] for q in unique_qids]

        live_baseline = self._compute_live_student_baseline(
            unique_qids, unique_prompts_text, unique_problems, unique_golds,
        )

        # ---- 4. Dr.Mixture advantage: A = r_teacher - r_mean_student(qid).
        advantages_list = [
            teacher_rewards[i] - live_baseline.get(int(question_ids[i]), 0.0)
            for i in range(batch_size)
        ]
        advantages = torch.tensor(advantages_list, dtype=torch.float32, device=device)

        # ---- 5. Pad completions & build masks -----------------------------
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

        # ---- 6. Current policy logprobs (on teacher completions) ----------
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        with torch.no_grad():
            current_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                self.model, prompt_completion_ids, attention_mask, logits_to_keep,
            )

        # ---- 7. Neutralize IS ratio: old_logps = current_logps ------------
        old_per_token_logps = current_per_token_logps.detach()

        # ---- 8. Reference logprobs for KL penalty -------------------------
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

        # ---- 9. Log metrics -----------------------------------------------
        completion_lengths = completion_mask.sum(1)
        if mode == "train":
            full_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            self.state.num_input_tokens_seen += self.accelerator.gather(full_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        rewards = torch.tensor(teacher_rewards, device=device)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(rewards.std().item())

        self._metrics[mode].setdefault("dr/advantage_mean", []).append(advantages.mean().item())
        self._metrics[mode].setdefault("dr/advantage_abs_mean", []).append(advantages.abs().mean().item())
        live_vals = torch.tensor(list(live_baseline.values()), dtype=torch.float32, device=device) \
                    if live_baseline else torch.tensor([0.0], device=device)
        self._metrics[mode].setdefault("dr/student_baseline_mean", []).append(live_vals.mean().item())
        self._metrics[mode].setdefault("dr/student_baseline_unique_qids", []).append(float(len(live_baseline)))

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
        self._logs["advantages"].extend(advantages.tolist())

        # ---- 10. Build output dict ----------------------------------------
        num_items_in_batch = agg_completion_lengths.sum()
        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "num_items_in_batch": num_items_in_batch,
            # Surface raw inputs for the DG variant to reuse.
            "dr_live_advantages": advantages,
            "dr_current_per_token_logps": current_per_token_logps,
            "dr_completion_mask": completion_mask,
        }
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        return output
