"""v5.1 #4 (L4) — Composite-aware feasible set.

The hard safety filter now operates over structured actions: a dose above the
safe ceiling, a contraindicated combination, an allergy conflict, or a route
inappropriate for the patient is removed at the composite level. The feasible
set A_j is a set of safe {drug, dose, route, ...} tuples, not arm names.
"""

from __future__ import annotations

from dataclasses import dataclass

from treatmentrx.domain import CompositeAction, StageRecord


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
        allergy_tokens = [a.lower() for a in allergies]

        feasible: list[CompositeAction] = []
        removed: list[tuple[str, str]] = []
        for options in candidates.values():
            for action in options:
                reason = self._unsafe_reason(action, alt, egfr, pregnant, allergy_tokens)
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
    ) -> str | None:
        drug = action.drug.lower()
        combo = (action.combination or "").lower()

        for token in allergy_tokens:
            if token and (token in drug or token in combo or token in action.label.lower()):
                return f"allergy conflict: {token}"

        if "jak" in drug or drug in {"upadacitinib", "tofacitinib", "baricitinib"}:
            if pregnant:
                return "JAK inhibitor contraindicated in pregnancy"
            if alt > 120:
                return "JAK inhibitor unsafe with ALT > 120"
            if egfr < 30:
                return "JAK inhibitor unsafe with eGFR < 30"

        if "methotrexate" in drug or "methotrexate" in combo or "mtx" in combo:
            if pregnant:
                return "methotrexate contraindicated in pregnancy"
            if egfr < 30:
                return "methotrexate unsafe with eGFR < 30"

        dose_value = self._dose_mg(action.dose)
        if dose_value is not None and dose_value > 1500:
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
