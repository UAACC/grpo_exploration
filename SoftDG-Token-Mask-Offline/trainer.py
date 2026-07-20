"""SoftDG-Token-Mask-Offline trainer: per-token PG masking.

Extends GRPOTrainer (offline mode) with:
  1. Per-token SoftDG gate: gate_t = sigmoid(training_signal * surprisal_t / eta)
  2. PG mask:  pg_loss_mask_t = completion_mask_t * [gate_t >= threshold]
  3. KL mask:  completion_mask (unchanged — all non-padding tokens)
  4. Token budget counter and early-stop callback

Key differences from SoftDGOfflineTrainer (completion-level):
  - Gating is per-token, not per-completion.
  - Skipped tokens do NOT receive PG gradient but still receive KL gradient.
  - The effective counter tracks tokens, not completions.
  - IS ratio is neutralized: old_per_token_logps = current_per_token_logps.detach().

Loss override (only grpo / dr_grpo supported):
  - PG term  uses pg_loss_mask.
  - KL term  uses completion_mask.
"""

from __future__ import annotations

from typing import Any, Union

import torch
from accelerate.utils import gather_object, is_peft_model
from transformers import TrainerCallback
from trl import GRPOTrainer
from trl.data_utils import is_conversational, maybe_apply_chat_template


# ---------------------------------------------------------------------------
# Effective token counter
# ---------------------------------------------------------------------------

class EffectiveTokenCounter:
    """Track per-token progress across DDP processes.

    All processes call accelerator.gather() before updating, so this object
    has the same values on every process.
    """

    def __init__(self, target: int):
        self.target = target
        self.effective = 0   # tokens with gate >= threshold AND |signal| > 1e-12
        self.scanned = 0     # total non-padding completion tokens seen
        self.kept_pg = 0     # tokens with gate >= threshold (before signal check)


# ---------------------------------------------------------------------------
# Early-stop callback
# ---------------------------------------------------------------------------

class EarlyStopOnTokenBudgetCallback(TrainerCallback):
    """Stop training when effective_tokens >= target."""

    def __init__(self, counter: EffectiveTokenCounter):
        self._counter = counter

    def on_step_end(self, args, state, control, **kwargs):
        if self._counter.effective >= self._counter.target:
            if getattr(args, "process_index", 0) == 0:
                print(
                    f"\n[SoftDG-TM] Token budget reached: "
                    f"{self._counter.effective} / {self._counter.target} "
                    f"effective tokens. Stopping."
                )
            control.should_training_stop = True
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        if getattr(args, "process_index", 0) == 0:
            c = self._counter
            keep_rate = c.effective / max(c.scanned, 1)
            print(
                f"[SoftDG-TM epoch end] effective={c.effective}/{c.target} "
                f"scanned={c.scanned} keep_rate={keep_rate:.4f} "
                f"kept_pg={c.kept_pg}"
            )
            if c.effective < c.target:
                print(
                    f"[SoftDG-TM epoch end] WARNING: token budget not yet reached "
                    f"after this epoch ({c.effective}/{c.target})."
                )
        return control


# ---------------------------------------------------------------------------
# SoftDG token-mask trainer
# ---------------------------------------------------------------------------

class SoftDGTokenMaskOfflineTrainer(GRPOTrainer):
    """GRPOTrainer with per-token SoftDG PG masking for offline rollouts.

    PG gradient flows only through tokens where gate_t >= softdg_gate_threshold.
    KL gradient flows through all non-padding completion tokens (completion_mask).
    """

    def __init__(
        self,
        *args,
        offline_data: dict,
        training_signal: str = "advantage",
        dg_temperature: float = 1.0,
        softdg_gate_threshold: float = 0.5,
        pg_weighting: bool = True,
        counter: EffectiveTokenCounter | None = None,
        ref_sync_steps: int = 0,
        **kwargs,
    ):
        """
        Args:
            offline_data: dict keyed by (question_id, run_id) with values
                {"completion_ids", "advantage", "reward", "response"}.
                reward_coding and advantage values are pre-processed in train.py.
            training_signal: "advantage" or "raw_reward" — per-completion scalar
                used as the gate signal and PG loss weight.
            dg_temperature: eta in sigmoid(delight/eta). Lower = sharper gate.
            softdg_gate_threshold: tokens with gate < threshold are excluded from PG.
            pg_weighting: if True, PG signal for selected tokens is scaled by DG(token_t)
                so PG_signal_t = training_signal_i * DG(token_t). DG weight is detached.
                If False, PG signal is uniform training_signal_i for all selected tokens
                (original binary-mask behaviour).
            counter: shared EffectiveTokenCounter for early stopping.
            ref_sync_steps: sync reference LoRA every N steps (0 = never).
        """
        super().__init__(*args, **kwargs)
        if training_signal not in ("advantage", "raw_reward"):
            raise ValueError(
                f"training_signal must be 'advantage' or 'raw_reward', got {training_signal!r}"
            )
        self._offline_data = offline_data
        self._training_signal = training_signal
        self._dg_temperature = dg_temperature
        self._softdg_gate_threshold = softdg_gate_threshold
        self._pg_weighting = pg_weighting
        self._counter = counter
        self._ref_sync_steps = ref_sync_steps
        self._ref_adapter_state = None
        self._steps_since_ref_sync = 0

        if self._ref_sync_steps > 0 and self.beta != 0.0:
            self._sync_ref_adapter()

    # ------------------------------------------------------------------
    # Reference model helpers
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
    # Core: generate + score + mask
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

        # ---- 2. Look up offline completions -----------------------------
        num_gen = self.num_generations
        question_ids = [x.get("question_id") for x in inputs]
        run_ids = [x.get("run_id", i % num_gen) for i, x in enumerate(inputs)]

        completion_id_lists = []
        signal_list = []

        for qid, rid in zip(question_ids, run_ids):
            rec = self._offline_data.get((qid, rid))
            if rec is None:
                raise KeyError(
                    f"No offline data for (question_id={qid}, run_id={rid}). "
                    "Check that num_generations matches the rollout file(s)."
                )
            completion_id_lists.append(rec["completion_ids"])
            if self._training_signal == "raw_reward":
                signal_list.append(rec["reward"])
            else:
                signal_list.append(rec["advantage"])

        # ---- 3. Pad completions -----------------------------------------
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
                cid_t = torch.cat([
                    cid_t,
                    torch.full((pad_len,), self.pad_token_id, dtype=torch.long, device=device),
                ])
                mask_t = torch.cat([mask_t, torch.zeros(pad_len, dtype=torch.int, device=device)])
            completion_ids_tensors.append(cid_t)
            completion_mask_tensors.append(mask_t)

        completion_ids = torch.stack(completion_ids_tensors)    # (B, C)
        completion_mask = torch.stack(completion_mask_tensors)  # (B, C) — full non-padding mask
        training_signal = torch.tensor(signal_list, dtype=torch.float32, device=device)  # (B,)

        # ---- 4. Current policy logprobs ---------------------------------
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask_full = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        with torch.no_grad():
            current_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                self.model, prompt_completion_ids, attention_mask_full, logits_to_keep,
            )

        # ---- 5. Per-token SoftDG gate -----------------------------------
        # surprisal_t = -log p_current(token_t)
        surprisal = -current_per_token_logps  # (B, C), non-negative

        # delight_t = training_signal * surprisal_t
        per_token_delight = training_signal.unsqueeze(1) * surprisal  # (B, C)

        # gate_t = sigmoid(delight_t / eta)
        per_token_gate = torch.sigmoid(per_token_delight / self._dg_temperature)  # (B, C)

        # ---- 6. Build PG mask -------------------------------------------
        # pg_loss_mask_t = completion_mask_t * [gate_t >= threshold]
        gate_passes = (per_token_gate >= self._softdg_gate_threshold)  # (B, C) bool
        pg_loss_mask = completion_mask * gate_passes.int()              # (B, C)

        # effective_token_mask: gate passes AND signal is nonzero
        has_signal = (training_signal.abs() > 1e-12)          # (B,) bool
        effective_token_mask = pg_loss_mask * has_signal.int().unsqueeze(1)  # (B, C)

        # ---- 6b. Per-token PG weights ------------------------------------
        # pg_weighting=True: PG_signal_t = training_signal_i * DG(token_t) for selected tokens
        # pg_weighting=False: PG_signal_t = training_signal_i (original binary-mask behaviour)
        if self._pg_weighting:
            pg_token_weights = (
                training_signal.unsqueeze(1) * per_token_gate.detach() * pg_loss_mask.float()
            )  # (B, C); 0 for unselected / padding tokens
        else:
            pg_token_weights = (
                training_signal.unsqueeze(1).expand_as(pg_loss_mask.float()) * pg_loss_mask.float()
            )  # (B, C); original behaviour

        # ---- 7. Track effective tokens ----------------------------------
        n_effective_local = int(effective_token_mask.sum().item())
        n_scanned_local = int(completion_mask.sum().item())
        n_kept_pg_local = int(pg_loss_mask.sum().item())

        if self._counter is not None and mode == "train":
            n_eff_t = torch.tensor([n_effective_local], dtype=torch.long, device=device)
            n_scan_t = torch.tensor([n_scanned_local], dtype=torch.long, device=device)
            n_kept_t = torch.tensor([n_kept_pg_local], dtype=torch.long, device=device)

            n_eff_global = int(self.accelerator.gather(n_eff_t).sum().item())
            n_scan_global = int(self.accelerator.gather(n_scan_t).sum().item())
            n_kept_global = int(self.accelerator.gather(n_kept_t).sum().item())

            self._counter.effective += n_eff_global
            self._counter.scanned += n_scan_global
            self._counter.kept_pg += n_kept_global

        # ---- 8. Neutralize IS ratio (old = current logps) ---------------
        old_per_token_logps = current_per_token_logps.detach()

        # ---- 9. Reference logprobs for KL (over full completion_mask) ---
        ref_per_token_logps = None
        if self.beta != 0.0:
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model, prompt_completion_ids, attention_mask_full, logits_to_keep,
                    )
                else:
                    ref_per_token_logps = self._get_ref_logprobs(
                        self.model, prompt_completion_ids, attention_mask_full, logits_to_keep,
                    )
            if self._ref_sync_steps > 0 and mode == "train":
                self._steps_since_ref_sync += 1
                if self._steps_since_ref_sync >= self._ref_sync_steps:
                    self._sync_ref_adapter()

        # ---- 10. Log metrics --------------------------------------------
        completion_lengths = completion_mask.sum(1)
        if mode == "train":
            full_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            self.state.num_input_tokens_seen += (
                self.accelerator.gather(full_mask.sum()).sum().item()
            )
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        agg_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(agg_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_lengths.float().max().item())

        rewards = torch.tensor(
            [self._offline_data[(qid, rid)]["reward"] for qid, rid in zip(question_ids, run_ids)],
            device=device,
        )
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(rewards.std().item())

        # Token-mask specific metrics (over non-padding tokens only)
        valid_gates = per_token_gate[completion_mask.bool()]  # flat vector
        if valid_gates.numel() > 0:
            self._metrics[mode]["softdg/gate_mean"].append(valid_gates.mean().item())
            self._metrics[mode]["softdg/gate_min"].append(valid_gates.min().item())
            self._metrics[mode]["softdg/gate_max"].append(valid_gates.max().item())
            self._metrics[mode]["softdg/gate_p10"].append(valid_gates.quantile(0.1).item())
            self._metrics[mode]["softdg/gate_p50"].append(valid_gates.quantile(0.5).item())
            self._metrics[mode]["softdg/gate_p90"].append(valid_gates.quantile(0.9).item())

        n_total_tokens = completion_mask.sum().float().clamp(min=1)
        self._metrics[mode]["softdg/token_keep_rate"].append(
            (pg_loss_mask.sum().float() / n_total_tokens).item()
        )
        self._metrics[mode]["softdg/effective_token_rate"].append(
            (effective_token_mask.sum().float() / n_total_tokens).item()
        )
        has_pg_token = (pg_loss_mask.sum(dim=1) > 0).float()
        self._metrics[mode]["softdg/completions_with_pg_token"].append(
            has_pg_token.mean().item()
        )
        if self._counter is not None:
            self._metrics[mode]["softdg/effective_tokens_total"].append(
                float(self._counter.effective)
            )

        # DG-weighted PG metrics (only when pg_weighting=True)
        if self._pg_weighting:
            selected_gates = per_token_gate[pg_loss_mask.bool()]
            if selected_gates.numel() > 0:
                self._metrics[mode]["softdg/mean_selected_gate"].append(
                    selected_gates.mean().item()
                )
            pg_mass = pg_token_weights.abs().sum().item()
            self._metrics[mode]["softdg/pg_mass_weighted"].append(pg_mass)

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
        self._logs["advantages"].extend(training_signal.tolist())

        # ---- 11. Build output dict --------------------------------------
        # completion_mask  = original non-padding mask (KL uses this)
        # pg_loss_mask     = token-level PG selection mask (1 = selected, 0 = dropped/padding)
        # pg_token_weights = per-token PG weights: training_signal*DG(t) if weighted, else
        #                    training_signal; always 0 for unselected / padding tokens
        num_items_in_batch = agg_lengths.sum()
        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,    # full non-padding mask (attention + KL)
            "pg_loss_mask": pg_loss_mask,           # per-token PG selection mask
            "pg_token_weights": pg_token_weights,  # (B, C) per-token PG signal weights
            "advantages": training_signal,          # (B,) per-completion signal (kept for compat)
            "old_per_token_logps": old_per_token_logps,
            "num_items_in_batch": num_items_in_batch,
        }
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        return output

    # ------------------------------------------------------------------
    # Loss override: PG uses pg_loss_mask, KL uses completion_mask
    # ------------------------------------------------------------------

    def _compute_loss(self, model, inputs):
        """Override parent to apply separate masks for PG and KL terms."""
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        pg_loss_mask = inputs["pg_loss_mask"]  # (B, C)

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep, compute_entropy=True,
        )

        # pg_token_weights: (B, C) — per-token PG signal weights (0 for unselected/padding).
        # For pg_weighting=True: training_signal_i * DG(token_t) for selected tokens.
        # For pg_weighting=False: training_signal_i for selected tokens (original behaviour).
        # Precomputed in _generate_and_score_completions with gate values detached.
        pg_token_weights = inputs["pg_token_weights"]  # (B, C)

        # IS ratio fully neutralized: force old = current.detach() here so ratio = 1
        # even across gradient accumulation steps (pre-computed old_per_token_logps
        # would drift after partial optimizer updates within the same accumulation window).
        old_per_token_logps = per_token_logps.detach()

        log_ratio = per_token_logps - old_per_token_logps
        coef_1 = torch.exp(log_ratio)
        epsilon_low = getattr(self, "epsilon_low", getattr(self.args, "epsilon", 0.2))
        epsilon_high = getattr(self, "epsilon_high", getattr(self.args, "epsilon_high", 0.2))
        coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)

        # PG term (B, C): clipped IS * per-token weight (already 0 for unselected tokens)
        per_token_pg = -torch.min(coef_1 * pg_token_weights, coef_2 * pg_token_weights)

        # KL term (B, C): uses full completion_mask
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps) - 1
            )

        mode = "train" if self.model.training else "eval"
        normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0

        if self.loss_type == "grpo":
            pg_norm = pg_loss_mask.sum(-1).clamp(min=1).float()
            pg_loss = ((per_token_pg * pg_loss_mask).sum(-1) / pg_norm).mean()
            if self.beta != 0.0:
                kl_norm = completion_mask.sum(-1).clamp(min=1).float()
                kl_loss = self.beta * (
                    (per_token_kl * completion_mask).sum(-1) / kl_norm
                ).mean()
                pg_loss = pg_loss + kl_loss
            loss = pg_loss / normalizer

        elif self.loss_type == "dr_grpo":
            if self.max_completion_length is None:
                raise ValueError("dr_grpo requires max_completion_length to be set (not None).")
            denom = float(per_token_pg.size(0) * self.max_completion_length)
            pg_loss = (per_token_pg * pg_loss_mask).sum() / denom
            if self.beta != 0.0:
                kl_loss = self.beta * (per_token_kl * completion_mask).sum() / denom
                pg_loss = pg_loss + kl_loss
            loss = pg_loss / normalizer

        else:
            raise ValueError(
                f"SoftDGTokenMaskOfflineTrainer only supports loss_type in "
                f"['grpo', 'dr_grpo'], got {self.loss_type!r}"
            )

        # Metrics
        n_completion_tokens = completion_mask.sum().clamp(min=1).float()
        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / n_completion_tokens
            self._metrics[mode]["kl"].append(
                self.accelerator.gather(mean_kl).nanmean().item()
            )

        mean_entropy = (entropies * completion_mask).sum() / n_completion_tokens
        self._metrics[mode]["entropy"].append(
            self.accelerator.gather(mean_entropy).nanmean().item()
        )

        return loss
