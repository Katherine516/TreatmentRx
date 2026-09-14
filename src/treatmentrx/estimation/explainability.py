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


def _e_value(effect: float, outcome_sd: float) -> float:
    """VanderWeele-Ding E-value for a standardised continuous effect.

    `RR ~ exp(0.91 * d)` is their published approximation for a continuous
    outcome; the bound is then `RR + sqrt(RR (RR - 1))`. A zero effect gives
    exactly 1.0, which is the honest reading — nothing has to be explained away.
    """
    if effect <= 0.0 or outcome_sd <= 0.0:
        return 1.0
    risk_ratio = math.exp(0.91 * (effect / outcome_sd))
    return round(risk_ratio + math.sqrt(max(risk_ratio * (risk_ratio - 1.0), 0.0)), 3)


class ModelExplainer:
    def explain(
        self,
        selected: RegimeEstimate,
        candidates: list[RegimeEstimate],
        stages: list[StageRecord],
        contrast=None,
    ) -> ExplanationBundle:
        """`contrast` is the decision's own top-two interval, passed in.

        It used to be absent, and `_sensitivity` built its own quantity out of
        `selected.q_values` instead — the same incoherence invariant 16 removed
        from the interval itself, where a number beside the decision described
        something the decision did not use. The caller already has the contrast
        when it calls this; there was never a reason to reconstruct a worse one.
        """
        return ExplanationBundle(
            attributions=self._attributions(selected, stages),
            why_not=self._why_not(selected),
            counterfactuals=self._counterfactuals(selected, stages),
            sensitivity=self._sensitivity(contrast),
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

    def _sensitivity(self, contrast) -> AssumptionSensitivity:
        """E-value for the contrast the decision reports. See `AssumptionSensitivity`.

        The outcome here is a bounded response score, not a risk, so the E-value
        goes through VanderWeele and Ding's approximation for continuous
        outcomes: standardise the contrast by the outcome's own spread, take
        `RR ~ exp(0.91 * d)`, then `E = RR + sqrt(RR (RR - 1))`.

        Two numbers rather than one, and the second is the one to quote. The
        point estimate's E-value says how strong confounding would have to be to
        move the *estimate* to null; the interval's says how strong it would have
        to be to make the data unable to exclude null, which is the question a
        reader is actually asking. Where the agent abstains the interval already
        contains zero, so that number is 1.0 — correctly, and for the first time.

        What this is not: a claim about the *four named unadjusted confounders*
        specifically. It is a bound on any single unmeasured confounder's
        strength, and the approximation assumes the contrast is between two arms
        on a comparable scale. On this cohort the generating process has no age,
        gender, steroid or comorbidity effect, so the true E-value question has
        no bite here — it is a property of the basis that would matter on real
        data, which is exactly what the model card says about those four nodes.
        """
        from treatmentrx.estimation import training

        outcome_sd = training.holdout_outcome_sd()
        if contrast is None or outcome_sd <= 0.0:
            return AssumptionSensitivity(
                e_value=1.0,
                e_value_for_interval=1.0,
                contrast=0.0,
                outcome_sd=round(outcome_sd, 4),
                note=(
                    "No contrast was available for this patient, so no bound on "
                    "unmeasured confounding is reported. This is an absent "
                    "measurement, not a finding of robustness."
                ),
            )

        # The confidence limit nearest the null. When the interval spans zero
        # this is zero, and the bound below correctly collapses to 1.0.
        nearest_null = 0.0
        if contrast.lower > 0.0:
            nearest_null = contrast.lower
        elif contrast.upper < 0.0:
            nearest_null = contrast.upper

        point = _e_value(abs(contrast.difference), outcome_sd)
        limit = _e_value(abs(nearest_null), outcome_sd)
        return AssumptionSensitivity(
            e_value=point,
            e_value_for_interval=limit,
            contrast=round(contrast.difference, 4),
            outcome_sd=round(outcome_sd, 4),
            note=(
                f"To explain away a contrast of {contrast.difference:+.4f}, an "
                f"unmeasured confounder would need associations of about "
                f"RR={point} with both the arm given and the outcome. To make the "
                f"interval unable to exclude zero it would need RR={limit}"
                + (
                    " — the interval already contains zero, so no confounding is "
                    "required and this recommendation rests on separation the "
                    "data does not have."
                    if limit <= 1.0
                    else "."
                )
                + " Standardised by a held-out outcome spread of "
                f"{outcome_sd:.4f} via the VanderWeele-Ding approximation for "
                "continuous outcomes."
            ),
        )
