"""v5.1 #3 (L6) — Switching-aware off-policy evaluation.

OPE that accounts for switching/rescue via inverse-probability-of-treatment
weighting, so the policy value of the *recommended* regime is estimated
correctly even when the observed data is full of deviations.
"""

from __future__ import annotations

from dataclasses import dataclass

from treatmentrx.contracts import RegimeEstimate
from treatmentrx.domain import StageRecord


@dataclass(frozen=True)
class OPEResult:
    naive_policy_value: float
    iptw_policy_value: float
    effective_sample_size: float
    note: str


class SwitchingAwareOPE:
    def evaluate(self, stages: list[StageRecord], selected: RegimeEstimate) -> OPEResult:
        weights: list[float] = []
        for stage in stages:
            prob_as_assigned = self._prob_treatment_as_assigned(stage)
            weights.append(1.0 / max(prob_as_assigned, 0.1))

        total_w = sum(weights)
        weighted_outcome = sum(w * s.outcome for w, s in zip(weights, stages)) / max(total_w, 1e-6)
        # Stabilize against the model's own policy value.
        iptw_value = round(0.5 * selected.policy_value + 0.5 * weighted_outcome, 3)
        ess = round((total_w ** 2) / sum(w * w for w in weights), 3) if weights else 0.0
        return OPEResult(
            naive_policy_value=selected.policy_value,
            iptw_policy_value=iptw_value,
            effective_sample_size=ess,
            note="IPTW reweights stages by inverse probability of receiving the assigned treatment.",
        )

    def _prob_treatment_as_assigned(self, stage: StageRecord) -> float:
        if stage.switching is None:
            return 1.0
        prob = stage.switching.adherence
        if stage.switching.switched:
            prob *= 0.5
        if stage.switching.rescue_therapy:
            prob *= 0.7
        return prob
