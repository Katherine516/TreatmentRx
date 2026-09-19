"""Hard clinical rules, enforced in code.

These are the checks that do not depend on the feasible-set filter: data-contract
failures, causal non-identifiability, an allergy naming the recommended arm, and
trajectory-level signals a single visit cannot show.

Severity is binary and load-bearing. "block" stops the recommendation and is
only ever emitted here; "warn" travels with the recommendation into the
narrative. Nothing downstream may promote or demote either one.
"""

from __future__ import annotations

from treatmentrx import formulary

from treatmentrx.contracts import Decision, PatientState
from treatmentrx.domain import RecommendationStatus, SafetyFlag, StageRecord
from treatmentrx.safety.feasible_set import allergy_matches

# Organ-function and pregnancy limits for the drug classes that have them.
ALT_CEILING = 120.0
EGFR_FLOOR = 30.0

# Trajectory-level delayed-toxicity trigger.
#: Arms carrying a hepatotoxic molecule, derived from the formulary.
#:
#: This was a tuple of *molecule* spellings, and both places that read it
#: substring-matched it against a **canonical arm name** — once against the arm
#: under consideration, once against `stage.treatment`, which Layer 1 maps
#: through `normalize_arm`. So four of its six tokens could never match in
#: either: no arm is called `tofacitinib`, `baricitinib`, `upadacitinib` or
#: `leflunomide`. Only `methotrexate` and `jak` did any work, so the list read as
#: though it broadened the rule and did not. An arm-level question wants
#: arm-level membership.
HEPATOTOXIC_ARMS = formulary.arms_with_hazard(formulary.HAZARD_HEPATOTOXIC)
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
        delayed = self._delayed_toxicity(state.stages, decision)
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

    def _delayed_toxicity(self, stages: list[StageRecord], decision: Decision) -> SafetyFlag | None:
        """A sequence safe at every visit can still accumulate toward harm.

        Per-visit checks cannot see this: each individual ALT may be acceptable
        while the trend across a hepatotoxic sequence is not.

        **Which arm this is about.** It used to read `decision.recommended_arm`
        unconditionally and phrase itself as "before continuing" — but on an
        equipoise decision that field is only the argmax, and Layer 3 has already
        said it cannot be separated from the rest of the candidate set. So the
        flag named an arm nobody had recommended and attributed an intent to
        continue it that did not exist. That is the defect invariant 2 fixed in
        `SafetyLayer._status`, here in a rule rather than a status.

        The trajectory evidence is the same either way, so the flag still fires;
        what changes is what it claims. When there is a recommendation it is
        about that arm. When there is not, it is about whichever arms are still
        under consideration, and it says so rather than picking one. Gating on
        the argmax also *suppressed* the warning for an undecided patient whose
        argmax happened to be non-hepatotoxic while the set still held
        hepatotoxic options — a warning lost for no reason.
        """
        # `stage.treatment` is a canonical arm name — Layer 1 maps free text
        # through `normalize_arm` — so this asks the same arm-level question the
        # branch below does, and the molecule spellings it used to match against
        # were unreachable here too.
        exposure = sum(1 for stage in stages if _is_hepatotoxic(stage.treatment))
        alts = [
            _numeric(stage.features.get("alt"))
            for stage in stages
            if _numeric(stage.features.get("alt")) is not None
        ]
        rising = len(alts) >= 2 and alts[-1] > alts[0] and alts[-1] > RISING_ALT_FLOOR
        if not (exposure >= CUMULATIVE_EXPOSURE_STAGES and rising):
            return None

        committed = decision.status is RecommendationStatus.RECOMMEND
        under_consideration = (
            [decision.recommended_arm]
            if committed
            else list(decision.candidate_arms) or [decision.recommended_arm]
        )
        hepatotoxic = [arm for arm in under_consideration if _is_hepatotoxic(arm)]
        if not hepatotoxic:
            return None

        if committed:
            message = (
                "Cumulative hepatotoxic exposure with a rising ALT trend across visits; "
                "monitor for delayed toxicity within the assessment window before continuing."
            )
            subject = decision.recommended_arm
        else:
            # No arm was recommended, so nothing is being "continued". The
            # observation is about the trajectory and the options still open.
            message = (
                "Cumulative hepatotoxic exposure with a rising ALT trend across visits. "
                "No arm has been recommended; "
                f"{len(hepatotoxic)} of {len(under_consideration)} arms still under "
                "consideration are hepatotoxic, and delayed toxicity is a "
                "consideration for those."
            )
            subject = None
        return SafetyFlag("delayed_toxicity_accumulation", "warn", message, subject)

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


def _is_hepatotoxic(arm: str) -> bool:
    return arm in HEPATOTOXIC_ARMS


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
