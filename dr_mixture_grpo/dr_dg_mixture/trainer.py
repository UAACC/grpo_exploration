"""DrDGMixtureTrainer: DG's sigmoid gate on top of Dr.Mixture's live advantage.

Combines:
  - Dr.Mixture (live):  A_i = r_teacher_i - r_mean_current_student(qid)
    with r_mean_current_student computed by sampling K_s from the current
    policy every step, scored with Math_Verifier.
  - DG gate:            weight = sigmoid(delight / eta), delight = A * surprisal,
    where surprisal = -log pi_current over the teacher completion.

Inherits everything from DrMixtureGRPOTrainer; just multiplies the final
``advantages`` by the DG gate before returning.

Note on eta scale. Plain DG-offline used eta values calibrated for within-
group-normalized advantages (A in roughly [-1, 1]). Under Dr.Mixture, A is on
the raw reward scale (~+/-2 for MATH, +/-1 for GSM8K). The sigmoid will
saturate at much smaller eta values; a new sweep is needed.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))


def _load_module_from_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_dr_mixture_trainer_mod = _load_module_from_path(
    "_dr_mixture_trainer",
    os.path.join(_PROJECT_ROOT, "dr_mixture_grpo", "trainer.py"),
)
DrMixtureGRPOTrainer = _dr_mixture_trainer_mod.DrMixtureGRPOTrainer


class DrDGMixtureTrainer(DrMixtureGRPOTrainer):
    """Dr.Mixture (live advantage) with the DG sigmoid gate applied."""

    def __init__(self, *args, dg_temperature: float = 1.0,
                 dg_gating: str = "completion", **kwargs):
        super().__init__(*args, **kwargs)
        self._dg_temperature = dg_temperature
        self._dg_gating = dg_gating
        if dg_gating not in ("completion", "token"):
            raise ValueError(f"Unknown dg_gating: {dg_gating}")

    def _generate_and_score_completions(self, inputs):
        out = super()._generate_and_score_completions(inputs)

        adv = out["dr_live_advantages"]                       # (B,)
        per_token_logps = out["dr_current_per_token_logps"]   # (B, C)
        completion_mask = out["dr_completion_mask"]           # (B, C)
        surprisal = -per_token_logps                           # (B, C)

        if self._dg_gating == "completion":
            lengths = completion_mask.sum(dim=1).clamp(min=1).float()
            completion_surprisal = (surprisal * completion_mask).sum(dim=1) / lengths
            delight = adv * completion_surprisal
            gate = torch.sigmoid(delight / self._dg_temperature)
            gated_adv = gate * adv
        else:  # token
            per_token_delight = adv.unsqueeze(1) * surprisal
            per_token_gate = torch.sigmoid(per_token_delight / self._dg_temperature)
            mean_gate = (per_token_gate * completion_mask).sum(dim=1) \
                        / completion_mask.sum(dim=1).clamp(min=1)
            gated_adv = mean_gate * adv

        out["advantages"] = gated_adv

        mode = "train" if self.model.training else "eval"
        if self._dg_gating == "completion":
            self._metrics[mode].setdefault("dr_dg/gate_mean", []).append(float(gate.mean().item()))
            self._metrics[mode].setdefault("dr_dg/gate_min", []).append(float(gate.min().item()))
            self._metrics[mode].setdefault("dr_dg/gate_max", []).append(float(gate.max().item()))
            self._metrics[mode].setdefault("dr_dg/delight_mean", []).append(float(delight.mean().item()))
            self._metrics[mode].setdefault("dr_dg/completion_surprisal_mean", []).append(
                float(completion_surprisal.mean().item())
            )

        return out
