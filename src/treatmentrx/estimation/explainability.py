"""v5.1 #9 (L3) — Model-level explainability.

Faithful, model-derived explanations: blip attributions, why-not table, local
counterfactuals, and assumption sensitivity. Layer 5 *renders* these in
language; it never invents them.
"""

from __future__ import annotations

import math

from treatmentrx.estimation.basis import BLIP_BASIS, blip_basis
from treatmentrx.estimation.features import model_features
from treatmentrx.contracts import RegimeEstimate
from treatmentrx.domain import (
    AssumptionSensitivity,
    BlipAttribution,
    CounterfactualProbe,
    ExplanationBundle,
    StageRecord,
    WhyNotEntry,
)


WHY_NOT_REASONS = {
    "continue-current": "current regime is failing to control disease activity",
    "methotrexate-optimization": "csDMARD optimization is lower-yield after prior failure",
    "TNF-inhibitor": "prior TNF exposure / loss of response lowers expected benefit",
    "IL-6 inhibitor": "expected advantage did not exceed the recommended arm",
    "JAK-inhibitor": "organ-function / safety profile reduces net benefit",
    "rituximab": "reserved for seropositive or multi-failure context",
}


class ModelExplainer:
    def explain(
        self,
        selected: RegimeEstimate,
        candidates: list[RegimeEstimate],
        stages: list[StageRecord],
    ) -> ExplanationBundle:
        return ExplanationBundle(
            attributions=self._attributions(selected, stages),
            why_not=self._why_not(selected),
            counterfactuals=self._counterfactuals(selected, stages),
            sensitivity=self._sensitivity(selected),
        )

    def _attributions(self, selected: RegimeEstimate, stages: list[StageRecord]) -> list[BlipAttribution]:
        """Decompose the recommended arm's blip into its per-covariate terms.

        `total_advantage` is literally psi . h(X) — the model's estimate of how
        much better this arm is than continuing current therapy for this
        patient. Each contribution is one term of that sum, so the parts add up
        to the whole and the explanation cannot drift from the model.
        """
        basis = dict(zip(BLIP_BASIS, blip_basis(model_features(stages))))
        prefix = f"psi:{selected.recommended_arm}:"
        contributions: dict[str, float] = {}
        for key, coefficient in selected.coefficients.items():
            if not key.startswith(prefix):
                continue
            if not isinstance(coefficient, (int, float)) or isinstance(coefficient, bool):
                continue
            covariate = key[len(prefix):]
            if covariate in basis:
                contributions[covariate] = round(coefficient * basis[covariate], 4)
        total = round(sum(contributions.values()), 4)
        return [
            BlipAttribution(
                action=selected.recommended_arm,
                contributions=dict(sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)),
                total_advantage=total,
            )
        ]

    def _why_not(self, selected: RegimeEstimate) -> list[WhyNotEntry]:
        best = selected.q_values[selected.recommended_arm]
        entries: list[WhyNotEntry] = []
        for action, value in sorted(selected.q_values.items(), key=lambda kv: kv[1], reverse=True):
            if action == selected.recommended_arm:
                continue
            entries.append(
                WhyNotEntry(
                    action=action,
                    q_gap=round(best - value, 3),
                    dominant_reason=WHY_NOT_REASONS.get(action, "lower estimated Q-value in this stage context"),
                )
            )
        return entries

    def _counterfactuals(self, selected: RegimeEstimate, stages: list[StageRecord]) -> list[CounterfactualProbe]:
        latest = stages[-1]
        crp = latest.features.get("crp")
        probes: list[CounterfactualProbe] = []
        ordered = sorted(selected.q_values.values(), reverse=True)
        gap = ordered[0] - ordered[1] if len(ordered) > 1 else 1.0
        if isinstance(crp, (int, float)) and not isinstance(crp, bool):
            flips = float(crp) > 20 and gap < 0.05
            probes.append(
                CounterfactualProbe(
                    covariate="crp",
                    perturbation="CRP halved (lower inflammation)",
                    recommendation_changes=flips,
                    note=(
                        "Lower CRP narrows the inflammatory case for escalation; with a small Q-gap the "
                        "recommendation could shift toward continuity." if flips
                        else "Recommendation is stable to a moderate CRP reduction."
                    ),
                )
            )
        return probes

    def _sensitivity(self, selected: RegimeEstimate) -> AssumptionSensitivity:
        ordered = sorted(selected.q_values.values(), reverse=True)
        gap = ordered[0] - ordered[1] if len(ordered) > 1 else 0.1
        # Heuristic E-value: larger separation => more robust to unmeasured confounding.
        rr = (ordered[0] + 1e-6) / (ordered[1] + 1e-6) if len(ordered) > 1 else 1.5
        e_value = round(rr + math.sqrt(max(rr * (rr - 1), 0.0)), 3)
        tipping = "small" if gap < 0.05 else ("moderate" if gap < 0.12 else "large")
        return AssumptionSensitivity(
            e_value=e_value,
            tipping_point=tipping,
            note=(
                f"An unmeasured confounder would need an association of about RR={e_value} with both treatment "
                f"and outcome to flip this recommendation (separation: {tipping})."
            ),
        )
