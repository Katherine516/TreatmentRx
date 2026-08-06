"""Hard clinical rules, enforced in code.

These are the checks that do not depend on the feasible-set filter: data-contract
failures, causal non-identifiability, an allergy naming the recommended arm, and
trajectory-level signals a single visit cannot show.

Severity is binary and load-bearing. "block" stops the recommendation and is
only ever emitted here; "warn" travels with the recommendation into the
narrative. Nothing downstream may promote or demote either one.
"""

from __future__ import annotations

from treatmentrx.contracts import Decision, PatientState
from treatmentrx.domain import SafetyFlag, StageRecord

# Organ-function and pregnancy limits for the drug classes that have them.
ALT_CEILING = 120.0
EGFR_FLOOR = 30.0

# Trajectory-level delayed-toxicity trigger.
HEPATOTOXIC_TOKENS = ("methotrexate", "jak", "tofacitinib", "baricitinib", "upadacitinib", "leflunomide")
RISING_ALT_FLOOR = 60.0
CUMULATIVE_EXPOSURE_STAGES = 2

# Out-of-distribution limits: beyond these the estimators are extrapolating.
DAS28_CEILING = 8.0
CRP_CEILING = 120.0
OOD_EGFR_FLOOR = 15.0


class SafetyRules:
    def evaluate(self, state: PatientState, decision: Decision) -> list[SafetyFlag]:
        flags: list[SafetyFlag] = []
        flags.extend(self._diagnostic_flags(state))
        flags.extend(self._allergy_flags(state, decision))
        delayed = self._delayed_toxicity(state.stages, decision.recommended_arm)
        if delayed:
            flags.append(delayed)
        if self._out_of_support(state.latest):
            flags.append(
                SafetyFlag(
                    "outside_training_support",
                    "warn",
                    "Patient covariates fall outside the training cohort's support; the estimates are extrapolations.",
                )
            )
        return flags

    def _diagnostic_flags(self, state: PatientState) -> list[SafetyFlag]:
        return [
            SafetyFlag(
                "identifiability_failed" if "causal" in diagnostic.name else "data_contract_error",
                "block",
                diagnostic.message,
            )
            for diagnostic in state.diagnostics
            if diagnostic.severity == "error" and not diagnostic.passed
        ]

    def _allergy_flags(self, state: PatientState, decision: Decision) -> list[SafetyFlag]:
        recommended = decision.recommended_arm.lower()
        return [
            SafetyFlag(
                "allergy_contraindication",
                "block",
                f"Recommended action conflicts with recorded allergy: {allergy}.",
                decision.recommended_arm,
            )
            for allergy in state.allergies
            if allergy.lower() in recommended
        ]

    def _delayed_toxicity(self, stages: list[StageRecord], recommended_arm: str) -> SafetyFlag | None:
        """A sequence safe at every visit can still accumulate toward harm.

        Per-visit checks cannot see this: each individual ALT may be acceptable
        while the trend across a hepatotoxic sequence is not.
        """
        exposure = sum(
            1 for stage in stages if any(token in stage.treatment.lower() for token in HEPATOTOXIC_TOKENS)
        )
        alts = [
            _numeric(stage.features.get("alt"))
            for stage in stages
            if _numeric(stage.features.get("alt")) is not None
        ]
        rising = len(alts) >= 2 and alts[-1] > alts[0] and alts[-1] > RISING_ALT_FLOOR
        escalating = any(token in recommended_arm.lower() for token in HEPATOTOXIC_TOKENS)
        if exposure >= CUMULATIVE_EXPOSURE_STAGES and rising and escalating:
            return SafetyFlag(
                "delayed_toxicity_accumulation",
                "warn",
                (
                    "Cumulative hepatotoxic exposure with a rising ALT trend across visits; "
                    "monitor for delayed toxicity within the assessment window before continuing."
                ),
                recommended_arm,
            )
        return None

    def _out_of_support(self, stage: StageRecord) -> bool:
        das28 = _numeric(stage.features.get("das28")) or 4.0
        crp = _numeric(stage.features.get("crp")) or 8.0
        egfr = _numeric(stage.features.get("egfr")) or 90.0
        return das28 > DAS28_CEILING or crp > CRP_CEILING or egfr < OOD_EGFR_FLOOR


def _numeric(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


__all__ = ["SafetyRules"]
