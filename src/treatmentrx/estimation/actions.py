"""v5.1 #4 — Multi-action / combination representation.

The action was an arm label; real decisions are structured {drug, dose, route,
timing, combination, stop/continue}. Combinatorial blow-up is controlled by a
clinically-curated candidate set per stage — not a free product space.
"""

from __future__ import annotations

from treatmentrx import formulary
from treatmentrx.domain import CompositeAction, StageRecord


#: The menu, derived from `treatmentrx.formulary` rather than declared here.
#: It used to be a literal, and that literal was the one place the molecule
#: vocabulary had fallen behind: `arms.py` recognises three JAK molecules and
#: `safety/feasible_set.py` hazard-classes the same three, while this offered
#: **one**. Nothing compared them, so nothing caught it.
#:
#: What is left in this module is expansion logic — route preference, defaults —
#: with no clinical content. Which molecules exist and what they are dangerous
#: for is a formulary question and lives in one file.
ARM_CANDIDATES: dict[str, list[CompositeAction]] = formulary.arm_candidates()


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
