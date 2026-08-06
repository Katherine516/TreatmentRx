"""v5.1 #4 — Multi-action / combination representation.

The action was an arm label; real decisions are structured {drug, dose, route,
timing, combination, stop/continue}. Combinatorial blow-up is controlled by a
clinically-curated candidate set per stage — not a free product space.
"""

from __future__ import annotations

from precisionrx_agent.shared.models import CompositeAction, StageRecord


# Curated candidate composites per RA arm. The decision layer searches over
# these, not over a raw product of every dimension.
ARM_CANDIDATES: dict[str, list[CompositeAction]] = {
    "continue-current": [CompositeAction(drug="continue-current", stop_continue="continue")],
    "methotrexate-optimization": [
        CompositeAction(drug="methotrexate", dose="25mg", route="PO", timing="weekly"),
        CompositeAction(drug="methotrexate", dose="25mg", route="SC", timing="weekly"),
    ],
    "TNF-inhibitor": [
        CompositeAction(drug="adalimumab", dose="40mg", route="SC", timing="q2wk", combination="MTX"),
        CompositeAction(drug="etanercept", dose="50mg", route="SC", timing="weekly", combination="MTX"),
    ],
    "IL-6 inhibitor": [
        CompositeAction(drug="tocilizumab", dose="8mg/kg", route="IV", timing="q4wk", combination="MTX"),
        CompositeAction(drug="tocilizumab", dose="162mg", route="SC", timing="weekly"),
    ],
    "JAK-inhibitor": [
        CompositeAction(drug="upadacitinib", dose="15mg", route="PO", timing="daily"),
    ],
    "rituximab": [
        CompositeAction(drug="rituximab", dose="1000mg", route="IV", timing="x2 q2wk", combination="MTX"),
    ],
}


class CompositeActionSpace:
    """Expands arm-level menus to clinically-curated composite candidates."""

    def candidates(self, arms: list[str], stages: list[StageRecord]) -> dict[str, list[CompositeAction]]:
        space: dict[str, list[CompositeAction]] = {}
        prefer_oral = self._injection_fatigue(stages)
        for arm in arms:
            options = ARM_CANDIDATES.get(arm, [CompositeAction(drug=arm)])
            if prefer_oral:
                options = sorted(options, key=lambda a: 0 if (a.route or "").upper() == "PO" else 1)
            space[arm] = options
        return space

    def default_for(self, arm: str) -> CompositeAction:
        options = ARM_CANDIDATES.get(arm)
        return options[0] if options else CompositeAction(drug=arm)

    def _injection_fatigue(self, stages: list[StageRecord]) -> bool:
        return any(bool(stage.features.get("injection_fatigue")) for stage in stages)
