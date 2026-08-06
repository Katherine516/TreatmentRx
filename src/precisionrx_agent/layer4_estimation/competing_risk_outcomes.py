"""v5.1 #2 (L2) — Competing-risk endpoints.

Estimators optimize a clinically relevant cause (e.g. progression-free
survival), with death/dropout handled as competing events rather than folded
into a single survival time. This adjusts a fitted MethodResult so its policy
value reflects the competing-event landscape on the trajectory.
"""

from __future__ import annotations

from dataclasses import replace

from precisionrx_agent.shared.models import ClinicalEventType, MethodResult, StageRecord


# Causes that erode the value of the *recommended* regime if they dominate the
# observed trajectory (you cannot benefit a patient who died or dropped out).
COMPETING_PENALTY = {
    ClinicalEventType.DEATH.value: 0.5,
    ClinicalEventType.DROPOUT.value: 0.25,
    ClinicalEventType.SERIOUS_TOXICITY.value: 0.2,
}


class CompetingRiskEndpoint:
    """Re-weights policy value against the progression-free cause."""

    def adjust(self, result: MethodResult, stages: list[StageRecord], incidence: dict[str, float]) -> MethodResult:
        penalty = sum(
            COMPETING_PENALTY.get(cause, 0.0) * fraction
            for cause, fraction in incidence.items()
        )
        adjusted_value = round(max(result.policy_value * (1.0 - min(penalty, 0.9)), 0.0), 3)
        coefficients = result.coefficients | {
            "competing_risk_penalty": round(min(penalty, 0.9), 4),
            "endpoint": "progression_free",
        }
        return replace(result, policy_value=adjusted_value, coefficients=coefficients)
