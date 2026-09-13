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
from treatmentrx.safety.feasible_set import allergy_matches

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
        """Block when a recorded allergy names the recommended arm.

        The token is stripped and checked for emptiness first. An allergy
        recorded as "" — a blank field in the source record, not a clinical fact
        — is a substring of every arm name, so without the guard one data-quality
        artefact blocked every patient it touched. `FeasibleSet` has always
        guarded this; the arm-level rule did not, and blocking is the fail-safe
        direction, which is exactly why it went unnoticed.
        """
        recommended = decision.recommended_arm.lower()
        return [
            SafetyFlag(
                "allergy_contraindication",
                "block",
                f"Recommended action conflicts with recorded allergy: {allergy}.",
                decision.recommended_arm,
            )
            for allergy in state.allergies
            if allergy_matches(allergy, recommended)
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
        """Are the covariates past the edge of the training cohort's support?

        `_numeric(...) or default` was wrong in the one direction that matters
        here: 0.0 is falsy, so an anuric patient's eGFR of 0 was read as the
        healthy default of 90 and the extrapolation warning never fired — on the
        most extreme patient the range admits. The composite filter still removed
        the renally-cleared arms, so no unsafe arm escaped; what was lost was the
        warning that the estimates for the arms that *remained* are
        extrapolations.
        """
        das28 = _numeric_or(stage, "das28", 4.0)
        crp = _numeric_or(stage, "crp", 8.0)
        egfr = _numeric_or(stage, "egfr", 90.0)
        return das28 > DAS28_CEILING or crp > CRP_CEILING or egfr < OOD_EGFR_FLOOR


def _numeric(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _numeric_or(stage: StageRecord, key: str, default: float) -> float:
    """The recorded value, or `default` only when there is genuinely no number.

    Distinguishing "absent" from "zero" has to be done on the `None`, never on
    truthiness — a measured 0 is a clinical fact and often the extreme one.
    """
    value = _numeric(stage.features.get(key))
    return default if value is None else value


__all__ = ["SafetyRules"]
