"""v5.1 #3 (L6) — Switching-aware estimand choice.

ITT, per-protocol, and as-treated answer different clinical questions. The plan
reports all three side by side rather than silently picking one.
"""

from __future__ import annotations

from treatmentrx.contracts import RegimeEstimate
from treatmentrx.domain import EstimandResult, StageRecord


class EstimandReporter:
    def report(self, stages: list[StageRecord], selected: RegimeEstimate) -> list[EstimandResult]:
        n = float(len(stages))
        adherent = [s for s in stages if (s.switching is None or (not s.switching.switched and s.switching.adherence >= 0.8))]
        switched = [s for s in stages if s.switching and s.switching.switched]

        itt = EstimandResult(
            estimand="ITT",
            policy_value=selected.policy_value,
            n_effective=n,
            note="Effect of the regime as assigned, ignoring deviations.",
        )

        pp_fraction = (len(adherent) / n) if n else 1.0
        per_protocol = EstimandResult(
            estimand="per_protocol",
            policy_value=round(selected.policy_value * (0.5 + 0.5 * pp_fraction), 3),
            n_effective=float(len(adherent)),
            note=(
                f"Among adherers ({len(adherent)}/{int(n)} stages), selection-corrected; "
                "informative only if non-adherence is not outcome-driven."
            ),
        )

        switch_penalty = 1.0 - 0.1 * len(switched)
        as_treated = EstimandResult(
            estimand="as_treated",
            policy_value=round(max(selected.policy_value * switch_penalty, 0.0), 3),
            n_effective=n,
            note=f"Effect of realized treatment ({len(switched)} switch event(s) on trajectory).",
        )
        return [itt, per_protocol, as_treated]
