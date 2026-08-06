"""ITT, per-protocol and as-treated, reported side by side.

They answer different clinical questions and routinely disagree, so the system
reports all three rather than silently picking the flattering one.

All three are **model-level** quantities measured on held-out patients where the
deviations are observed. An earlier build computed them by scaling the model's
policy value by the current patient's adherence fraction, which is neither the
effect of the regime nor the effect on that patient — it is a population number
multiplied by an individual one. The patient's own deviation profile is still
reported, as context alongside the estimands rather than mixed into them.
"""

from __future__ import annotations

from treatmentrx.domain import EstimandResult, StageRecord

_NOTES = {
    "ITT": "Effect of the regime as assigned, deviations included.",
    "per_protocol": (
        "Restricted to decision points with no switch away from the previous arm. "
        "Informative only if switching is not outcome-driven — on this cohort it is, "
        "so read it as an upper bound, not an effect."
    ),
    "as_treated": "Value realised under whatever treatment was actually given.",
}


class EstimandReporter:
    def report(self, stages: list[StageRecord], selected=None) -> list[EstimandResult]:
        """Model-level estimands from the held-out cohort.

        `stages` is the patient's own history; it does not enter the estimand
        values and is used only to describe their deviation profile.
        """
        from treatmentrx.estimation import training

        values = training.holdout_estimands()
        deviations = self.deviation_profile(stages)
        return [
            EstimandResult(
                estimand=name,
                policy_value=value,
                n_effective=n_effective,
                note=f"{_NOTES[name]} Patient profile: {deviations}.",
            )
            for name, (value, n_effective) in values.items()
        ]

    def deviation_profile(self, stages: list[StageRecord]) -> str:
        """How far this patient's own trajectory departed from its assignments."""
        switched = sum(1 for stage in stages if stage.switching and stage.switching.switched)
        adherent = sum(
            1
            for stage in stages
            if stage.switching is None
            or (not stage.switching.switched and stage.switching.adherence >= 0.8)
        )
        return f"{adherent}/{len(stages)} adherent stages, {switched} switch event(s)"


__all__ = ["EstimandReporter"]
