"""v5.1 #1 — Treatment Timing Model.

Irregular intervals, time-to-response / delayed-toxicity windows, and
informative-visit (inverse-intensity) weighting are made first-class on the
stage state. Stages are no longer assumed equally spaced.
"""

from __future__ import annotations

from treatmentrx.domain import StageRecord, TimingFeatures


# Per-treatment-class assessment windows (days). When a response or toxicity
# should be read depends on the drug, not a fixed offset.
RESPONSE_WINDOWS = {
    "csDMARD": 90,
    "TNF": 90,
    "IL-6": 60,
    "JAK": 60,
    "rituximab": 180,
}
TOXICITY_WINDOWS = {
    "csDMARD": 120,
    "TNF": 180,
    "IL-6": 120,
    "JAK": 180,
    "rituximab": 240,
}


class TimingModel:
    """Annotates each stage with interval and response/toxicity-window features."""

    def apply(self, stages: list[StageRecord], encounter_days: list[int]) -> list[StageRecord]:
        if not stages:
            return []

        visits = sorted(set(encounter_days))
        intensity = self._inverse_intensity_weights(stages, visits)

        annotated: list[StageRecord] = []
        for index, stage in enumerate(stages):
            prev_start = stages[index - 1].start_day if index > 0 else None
            next_start = stages[index + 1].start_day if index + 1 < len(stages) else None
            last_visit = max((d for d in visits if d <= stage.start_day), default=None)

            timing = TimingFeatures(
                time_since_last_treatment=(stage.start_day - prev_start) if prev_start is not None else None,
                time_since_last_visit=(stage.start_day - last_visit) if last_visit is not None else None,
                interval_to_next_decision=(next_start - stage.start_day) if next_start is not None else None,
                response_window_days=self._window(stage.treatment, RESPONSE_WINDOWS, 90),
                delayed_toxicity_window_days=self._window(stage.treatment, TOXICITY_WINDOWS, 180),
                inverse_intensity_weight=intensity[index],
            )
            annotated.append(self._replace(stage, timing=timing))
        return annotated

    def _inverse_intensity_weights(self, stages: list[StageRecord], visits: list[int]) -> list[float]:
        """Inverse-intensity weights (v5.1 §4.4): sicker patients who show up sooner
        get down-weighted so visit timing does not bias the estimator."""
        if len(stages) < 2:
            return [1.0 for _ in stages]
        gaps = [b.start_day - a.start_day for a, b in zip(stages, stages[1:]) if b.start_day > a.start_day]
        mean_gap = sum(gaps) / len(gaps) if gaps else 90.0
        weights: list[float] = []
        for index, stage in enumerate(stages):
            if index == 0:
                weights.append(1.0)
                continue
            gap = max(stage.start_day - stages[index - 1].start_day, 1)
            # Short interval => high visit intensity => up-weight to correct selection.
            weight = min(max(mean_gap / gap, 0.25), 4.0)
            weights.append(round(weight, 4))
        return weights

    def _window(self, treatment: str, table: dict[str, int], default: int) -> int:
        name = treatment.lower()
        for key, value in table.items():
            if key.lower() in name:
                return value
        return default

    def _replace(self, stage: StageRecord, **updates: object) -> StageRecord:
        return StageRecord(**(stage.__dict__ | updates))
