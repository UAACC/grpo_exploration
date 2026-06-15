"""SoftDG-Offline trainer with gate-threshold skipping.

Extends GRPOTrainer (offline mode) with:
  1. Selectable training signal: "advantage" or "raw_reward"
     (reward_coding is applied upstream in train.py before building the lookup)
  2. Hard gate threshold: skip completions where sigmoid(signal*surprisal/eta) < threshold
     by zeroing their advantage (zero gradient contribution to TRL loss)
  3. Effective completion counter: tracks completions with nonzero final signal
  4. Early stopping via EarlyStopOnTargetCallback when target_effective_completions is reached

Edge case (plan requirement):
  With reward_coding=zero_two + training_signal=raw_reward, wrong completions
  have reward=0.0 -> gate*0=0 -> nonzero gate still yields zero effective signal.
  These are correctly NOT counted as effective (is_effective = gated_signal != 0).

Multi-epoch behavior:
  Set num_train_epochs high (e.g. 20) in train.py and let the callback stop training.
  Each epoch re-processes the same offline completions with updated student logprobs.
"""

from __future__ import annotations

from typing import Any, Union

import torch
from accelerate.utils import gather_object, is_peft_model
from transformers import TrainerCallback
from trl import GRPOTrainer
from trl.data_utils import is_conversational, maybe_apply_chat_template


# ---------------------------------------------------------------------------
# Effective completion counter (shared across trainer and callback)
# ---------------------------------------------------------------------------

class EffectiveCompletionCounter:
    """Thread-local counter maintained consistently across all DDP processes.

    All processes call accelerator.gather() before updating, so this object
    has the same values on every process.
    """

    def __init__(self, target: int):
        self.target = target
        self.effective = 0        # completions with nonzero gated signal
        self.scanned = 0          # total completions seen
        self.skipped_low_gate = 0 # gate < threshold
        self.skipped_zero_signal = 0  # gate >= threshold but signal == 0


# ---------------------------------------------------------------------------
# Early-stop callback
# ---------------------------------------------------------------------------

class EarlyStopOnTargetCallback(TrainerCallback):
    """Stop training when effective_completions >= target."""

    def __init__(self, counter: EffectiveCompletionCounter):
        self._counter = counter

    def on_step_end(self, args, state, control, **kwargs):
        if self._counter.effective >= self._counter.target:
            if getattr(args, "process_index", 0) == 0:
                print(
                    f"\n[SoftDG] Target reached: {self._counter.effective} / "
                    f"{self._counter.target} effective completions. Stopping."
                )
            control.should_training_stop = True
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        """Log progress at epoch boundaries."""
        if getattr(args, "process_index", 0) == 0:
            c = self._counter
            keep_rate = c.effective / max(c.scanned, 1)
            print(
                f"[SoftDG epoch end] effective={c.effective}/{c.target} "
                f"scanned={c.scanned} keep_rate={keep_rate:.3f} "
                f"skipped_low_gate={c.skipped_low_gate} "
                f"skipped_zero_signal={c.skipped_zero_signal}"
            )
        return control


# ---------------------------------------------------------------------------
# SoftDG trainer
# ---------------------------------------------------------------------------

class SoftDGOfflineTrainer(GRPOTrainer):
    """GRPOTrainer with SoftDG gate-threshold skipping for offline rollouts."""

    def __init__(
        self,
        *args,
        offline_data: dict,
        training_signal: str = "advantage",
        dg_temperature: float = 1.0,
        dg_gating: str = "completion",
        softdg_gate_threshold: float = 0.2,
        counter: EffectiveCompletionCounter | None = None,
        ref_sync_steps: int = 0,
        **kwargs,
    ):
        """
        Args:
            offline_data: dict keyed by (question_id, run_id) with values
                {"completion_ids", "advantage", "reward", "response"}.
                reward_coding and advantage values are pre-processed in train.py.
            training_signal: "advantage" uses group-normalized advantage;
                "raw_reward" uses the raw reward value from the lookup.
            dg_temperature: eta in sigmoid(delight/eta). Lower = sharper gate.
            dg_gating: "completion" = mean per-token surprisal over completion (original);
                "token" = per-token gate averaged over non-padding completion tokens.
            softdg_gate_threshold: completions with gate < threshold get zeroed.
            counter: shared EffectiveCompletionCounter for early stopping.
            ref_sync_steps: sync reference LoRA every N steps (0 = never).
        """
        super().__init__(*args, **kwargs)
        if training_signal not in ("advantage", "raw_reward"):
            raise ValueError(f"training_signal must be 'advantage' or 'raw_reward', got {training_signal!r}")
        if dg_gating not in ("completion", "token"):
            raise ValueError(f"dg_gating must be 'completion' or 'token', got {dg_gating!r}")
        self._offline_data = offline_data
        self._training_signal = training_signal
        self._dg_temperature = dg_temperature
        self._dg_gating = dg_gating
        self._softdg_gate_threshold = softdg_gate_threshold
        self._counter = counter
        self._ref_sync_steps = ref_sync_steps
        self._ref_adapter_state = None
        self._steps_since_ref_sync = 0

        if self._ref_sync_steps > 0 and self.beta != 0.0:
            self._sync_ref_adapter()

    # ------------------------------------------------------------------
    # Reference model helpers (copied from DGOfflineTrainer)
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
    # Core: generate + score + threshold
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
            else:  # "advantage"
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

        completion_ids = torch.stack(completion_ids_tensors)
        completion_mask = torch.stack(completion_mask_tensors)
        training_signal = torch.tensor(signal_list, dtype=torch.float32, device=device)

        # ---- 4. Current policy logprobs ---------------------------------
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask_full = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        with torch.no_grad():
            current_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                self.model, prompt_completion_ids, attention_mask_full, logits_to_keep,
            )

        # ---- 5. Compute SoftDG gate -------------------------------------
        surprisal = -current_per_token_logps  # (B, C), non-negative
        lengths = completion_mask.sum(dim=1).clamp(min=1).float()
        mean_surprisal = (surprisal * completion_mask).sum(dim=1) / lengths  # (B,)

        if self._dg_gating == "completion":
            delight = training_signal * mean_surprisal
            gate = torch.sigmoid(delight / self._dg_temperature)  # (B,)
            mean_delight = delight
        else:  # "token": per-token gate averaged over non-padding completion tokens
            per_token_delight = training_signal.unsqueeze(1) * surprisal  # (B, C)
            per_token_gate = torch.sigmoid(per_token_delight / self._dg_temperature)  # (B, C)
            gate = (per_token_gate * completion_mask).sum(dim=1) / lengths  # (B,) mean gate
            mean_delight = (per_token_delight * completion_mask).sum(dim=1) / lengths  # (B,)

        # ---- 6. Apply threshold -----------------------------------------
        # Completions with gate < threshold are fully skipped: zero both advantage
        # (PG term) AND completion_mask (KL term) so beta != 0 doesn't leak gradients.
        # Empty completions (no tokens) are also excluded.
        has_tokens = completion_mask.sum(dim=1) > 0           # (B,) bool
        passes_gate = (gate >= self._softdg_gate_threshold) & has_tokens  # (B,) bool
        soft_signal = gate * training_signal                   # (B,) soft-gated values
        gated_signal = torch.where(passes_gate, soft_signal, torch.zeros_like(soft_signal))

        # Zero completion_mask for skipped rows so KL penalty is also suppressed.
        effective_completion_mask = completion_mask * passes_gate.int().unsqueeze(1)

        # ---- 7. Track effective completions (nonzero final signal) -------
        # Also handles zero_two+raw_reward: reward=0 -> gate*0=0 -> not effective.
        is_effective = gated_signal.abs() > 1e-12
        n_effective_local = int(is_effective.sum().item())
        n_scanned_local = len(inputs)
        n_skipped_gate_local = int((~passes_gate).sum().item())
        n_skipped_zero_local = int((passes_gate & ~is_effective).sum().item())

        if self._counter is not None and mode == "train":
            # Gather across all DDP processes so all processes maintain the same counter.
            n_eff_t = torch.tensor([n_effective_local], dtype=torch.long, device=device)
            n_scan_t = torch.tensor([n_scanned_local], dtype=torch.long, device=device)
            n_skip_gate_t = torch.tensor([n_skipped_gate_local], dtype=torch.long, device=device)
            n_skip_zero_t = torch.tensor([n_skipped_zero_local], dtype=torch.long, device=device)

            n_eff_global = int(self.accelerator.gather(n_eff_t).sum().item())
            n_scan_global = int(self.accelerator.gather(n_scan_t).sum().item())
            n_skip_gate_global = int(self.accelerator.gather(n_skip_gate_t).sum().item())
            n_skip_zero_global = int(self.accelerator.gather(n_skip_zero_t).sum().item())

            # All processes update counter identically -> consistent state for callback.
            self._counter.effective += n_eff_global
            self._counter.scanned += n_scan_global
            self._counter.skipped_low_gate += n_skip_gate_global
            self._counter.skipped_zero_signal += n_skip_zero_global

        # ---- 8. Neutralize IS ratio (old = current logps) ---------------
        old_per_token_logps = current_per_token_logps.detach()

        # ---- 9. Reference logprobs for KL penalty -----------------------
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
        # Use original completion_mask for length stats (total scanned, not just kept).
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

        # SoftDG-specific metrics
        gate_f = gate.float()
        self._metrics[mode]["softdg/gate_mean"].append(gate_f.mean().item())
        self._metrics[mode]["softdg/gate_min"].append(gate_f.min().item())
        self._metrics[mode]["softdg/gate_max"].append(gate_f.max().item())
        self._metrics[mode]["softdg/gate_p10"].append(gate_f.quantile(0.1).item())
        self._metrics[mode]["softdg/gate_p50"].append(gate_f.quantile(0.5).item())
        self._metrics[mode]["softdg/gate_p90"].append(gate_f.quantile(0.9).item())
        self._metrics[mode]["softdg/passes_gate_frac"].append(passes_gate.float().mean().item())
        self._metrics[mode]["softdg/effective_frac"].append(is_effective.float().mean().item())
        self._metrics[mode]["softdg/delight_mean"].append(mean_delight.mean().item())
        self._metrics[mode]["softdg/surprisal_mean"].append(mean_surprisal.mean().item())
        if self._counter is not None:
            self._metrics[mode]["softdg/effective_total"].append(float(self._counter.effective))

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
        self._logs["advantages"].extend(gated_signal.tolist())

        # ---- 11. Build output dict --------------------------------------
        # Return effective_completion_mask (skipped rows zeroed) so TRL's loss
        # computation zeros both the PG term (via advantages=0) and the KL term
        # (via mask=0) for skipped completions. This is required when beta != 0.
        num_items_in_batch = agg_lengths.sum()
        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": effective_completion_mask,
            "advantages": gated_signal,
            "old_per_token_logps": old_per_token_logps,
            "num_items_in_batch": num_items_in_batch,
        }
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps

        return output
