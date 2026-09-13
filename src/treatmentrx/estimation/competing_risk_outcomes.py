"""v5.1 #2 (L2) — Competing-risk endpoints.

Estimators optimize a clinically relevant cause (e.g. progression-free
survival), with death/dropout handled as competing events rather than folded
into a single survival time. This annotates a fitted `RegimeEstimate` with how
much of *this patient's* trajectory was consumed by events that cap what any
treatment can deliver.

**It no longer multiplies the policy value.** It used to return
``policy_value * (1 - penalty)``, where `policy_value` is the estimator's
held-out IPW score over the whole cohort and `penalty` comes from one patient's
own competing-risk incidence. That is a population number scaled by an individual
one — the exact operation invariant 14 exists to forbid, and the resulting figure
answered no question: not "how good is this policy" (it had been discounted by a
patient it never saw) and not "how did this patient do" (it started from a
cohort-level score). It then travelled into the memory-guarded
`statistical_output`, the audit event, and `SwitchingAwareOPE.model_policy_value`,
where it was labelled as the estimator's held-out score.

The penalty is still computed and still reported. It is a patient-level
annotation, in `coefficients`, next to the other patient-level context.
"""

from __future__ import annotations

from dataclasses import replace

from treatmentrx.contracts import RegimeEstimate
from treatmentrx.domain import ClinicalEventType, StageRecord


# Causes that cap what any treatment can deliver for this patient, if they
# dominate the observed trajectory. Weights are clinical judgement, not fitted —
# which is another reason they must not touch a measured quantity.
COMPETING_PENALTY = {
    ClinicalEventType.DEATH.value: 0.5,
    ClinicalEventType.DROPOUT.value: 0.25,
    ClinicalEventType.SERIOUS_TOXICITY.value: 0.2,
}
PENALTY_CEILING = 0.9


class CompetingRiskEndpoint:
    """Annotates the estimate with this patient's competing-event burden."""

    def adjust(
        self,
        result: RegimeEstimate,
        stages: list[StageRecord],
        incidence: dict[str, float],
    ) -> RegimeEstimate:
        penalty = min(
            sum(
                COMPETING_PENALTY.get(cause, 0.0) * fraction
                for cause, fraction in incidence.items()
            ),
            PENALTY_CEILING,
        )
        return replace(
            result,
            coefficients=result.coefficients
            | {
                "competing_risk_penalty": round(penalty, 4),
                "endpoint": "progression_free",
            },
        )
