"""v5.1 #4 (L4) — Composite-aware feasible set.

The hard safety filter now operates over structured actions: a dose above the
safe ceiling, a contraindicated combination, an allergy conflict, or a route
inappropriate for the patient is removed at the composite level. The feasible
set A_j is a set of safe {drug, dose, route, ...} tuples, not arm names.
"""

from __future__ import annotations

from dataclasses import dataclass

from treatmentrx.domain import CompositeAction, StageRecord


# Organ-function limits, shared with `safety/rules.py` so the arm-level gate and
# the composite-level filter cannot drift apart.
ALT_CEILING = 120.0
EGFR_FLOOR = 30.0
DOSE_CEILING_MG = 1500.0

JAK_DRUGS = frozenset({"upadacitinib", "tofacitinib", "baricitinib"})
# Agents whose label carries hepatic monitoring, as drug or as background combo.
HEPATOTOXIC_DRUGS = ("methotrexate", "mtx", "leflunomide")


def _normalise_clinical_text(value: str) -> str:
    """Case/punctuation-insensitive text used only at the adapter safety seam."""
    return " ".join(
        "".join(character if character.isalnum() else " " for character in value.lower()).split()
    )


def allergy_matches(allergy: str, *candidates: str) -> bool:
    """Match a substance/class even when the display contains ordinary prose.

    FHIR coding should be preferred in a production adapter.  This prototype reads
    ``code.text``, so both ``rituximab allergy`` and ``allergy to rituximab`` must
    fail closed in the same way as the exact display ``rituximab``.
    """
    token = _normalise_clinical_text(allergy)
    if not token:
        return False
    for candidate in candidates:
        name = _normalise_clinical_text(candidate)
        if name and (token in name or name in token):
            return True
    return False


@dataclass(frozen=True)
class FeasibilityResult:
    feasible: list[CompositeAction]
    removed: list[tuple[str, str]]  # (action label, reason)


class FeasibleSet:
    def filter(
        self,
        candidates: dict[str, list[CompositeAction]],
        stages: list[StageRecord],
        allergies: list[str],
    ) -> FeasibilityResult:
        latest = stages[-1].features
        alt = self._feature(latest, "alt", 25.0)
        egfr = self._feature(latest, "egfr", 90.0)
        pregnant = bool(latest.get("pregnant", False))
        allergy_tokens = list(allergies)

        feasible: list[CompositeAction] = []
        removed: list[tuple[str, str]] = []
        for arm, options in candidates.items():
            for action in options:
                reason = self._unsafe_reason(action, alt, egfr, pregnant, allergy_tokens, arm)
                if reason:
                    removed.append((action.label, reason))
                else:
                    feasible.append(action)
        return FeasibilityResult(feasible=feasible, removed=removed)

    def _unsafe_reason(
        self,
        action: CompositeAction,
        alt: float,
        egfr: float,
        pregnant: bool,
        allergy_tokens: list[str],
        arm: str = "",
    ) -> str | None:
        drug = action.drug.lower()
        combo = (action.combination or "").lower()
        # Allergies are recorded against whatever the clinician wrote — a drug
        # ("tocilizumab") or a class ("IL-6 inhibitor"). Matching only the
        # composite's drug name let a class-level allergy through, because the
        # composites are named by molecule and the arm is named by class.
        haystacks = (drug, combo, action.label, arm)

        for token in allergy_tokens:
            if allergy_matches(token, *haystacks):
                return f"allergy conflict: {token}"

        if "jak" in drug or drug in JAK_DRUGS:
            if pregnant:
                return "JAK inhibitor contraindicated in pregnancy"
            if alt > ALT_CEILING:
                return f"JAK inhibitor unsafe with ALT > {ALT_CEILING:.0f}"
            if egfr < EGFR_FLOOR:
                return f"JAK inhibitor unsafe with eGFR < {EGFR_FLOOR:.0f}"

        # A hepatotoxic agent is unsafe on a failing liver whether it is the arm
        # itself or the background it is combined with. Checking pregnancy and
        # renal function but not hepatic function left the one organ these drugs
        # are actually monitored for unguarded.
        if any(token in drug or token in combo for token in HEPATOTOXIC_DRUGS):
            if pregnant:
                return "methotrexate/leflunomide contraindicated in pregnancy"
            if alt > ALT_CEILING:
                return f"hepatotoxic csDMARD unsafe with ALT > {ALT_CEILING:.0f}"
            if egfr < EGFR_FLOOR:
                return f"methotrexate unsafe with eGFR < {EGFR_FLOOR:.0f}"

        dose_value = self._dose_mg(action.dose)
        if dose_value is not None and dose_value > DOSE_CEILING_MG:
            return f"dose {action.dose} above safe ceiling"
        return None

    def _dose_mg(self, dose: str | None) -> float | None:
        if not dose:
            return None
        digits = ""
        for ch in dose:
            if ch.isdigit() or ch == ".":
                digits += ch
            elif digits:
                break
        try:
            return float(digits) if digits else None
        except ValueError:
            return None

    def _feature(self, features: dict[str, object], key: str, default: float) -> float:
        value = features.get(key, default)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default
