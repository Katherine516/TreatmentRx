"""Inverse-intensity weights for irregular observation.

Patients are not observed on a schedule. Sicker patients come back sooner, so
they contribute more decision points, so the fitted model is pulled toward the
covariate values that bring people in. That is selection by observation
frequency, and it is a distinct problem from dropout: censoring decides *whether*
a patient is seen again, intensity decides *how often*.

Layer 1 has always attached a `visit_weight` to each stage, computed from the
local visit rate against the patient's own mean gap. Two things were wrong with
it. It was a heuristic with no model behind it — the cohort now generates
severity-driven spacing, so there is a relationship to fit rather than guess. And
nothing statistical consumed it: `StageRecord.visit_weight` reached the GRU
encoder's summary and stopped, so the estimators never saw it at all.

This module fits the intensity from the data and puts the weights where they act.
Weights are stabilised against the marginal mean interval and clipped, for the
same reason the censoring weights are: an unstabilised inverse on a short
interval hands one patient the weight of several.
"""

from __future__ import annotations

from treatmentrx.estimation import linalg
from treatmentrx.estimation.basis import treatment_free_basis
from treatmentrx.simulation.ra_cohort import CohortTrajectory, clamp

# A predicted interval outside these is the model extrapolating, not a patient
# with an unusual schedule.
MIN_INTERVAL_DAYS = 14.0
MAX_INTERVAL_DAYS = 400.0
WEIGHT_CEILING = 4.0
_RIDGE = 1e-4


class VisitIntensityModel:
    """Expected days to the next decision point, as a function of covariates."""

    def __init__(self, cohort: list[CohortTrajectory]) -> None:
        self._coefficients: list[float] = []
        self.mean_interval = 0.0
        self._fit(cohort)

    def _fit(self, cohort: list[CohortTrajectory]) -> None:
        design: list[list[float]] = []
        targets: list[float] = []
        for trajectory in cohort:
            for stage in trajectory.stages:
                if stage.interval_days is None:
                    continue
                design.append(treatment_free_basis(stage.features))
                targets.append(float(stage.interval_days))

        if not targets:
            return
        self.mean_interval = sum(targets) / len(targets)
        if len(design) > len(design[0]):
            self._coefficients = linalg.weighted_least_squares(
                design, targets, [1.0] * len(design), ridge=_RIDGE
            )

    @property
    def fitted(self) -> bool:
        return bool(self._coefficients)

    def expected_interval(self, features: dict[str, float]) -> float:
        """Predicted days until this patient is next seen."""
        if not self._coefficients:
            return self.mean_interval or 90.0
        predicted = linalg.dot(treatment_free_basis(features), self._coefficients)
        return clamp(predicted, MIN_INTERVAL_DAYS, MAX_INTERVAL_DAYS)

    def intensity(self, features: dict[str, float]) -> float:
        """Visits per day — the reciprocal of the expected interval."""
        return 1.0 / self.expected_interval(features)

    def weight(self, features: dict[str, float]) -> float:
        """Stabilised inverse-intensity weight.

        A patient seen twice as often as average carries half the weight per
        visit, so the total contribution of a patient does not depend on how
        frequently their clinician happened to book them.
        """
        if not self.mean_interval:
            return 1.0
        return clamp(
            self.expected_interval(features) / self.mean_interval,
            1.0 / WEIGHT_CEILING,
            WEIGHT_CEILING,
        )

    def severity_slope(self) -> float:
        """Days of interval per standard deviation of disease activity.

        Negative by construction in the cohort: more active disease brings the
        next visit forward. Exposed so the fit can be checked against the
        generating process rather than assumed.
        """
        if not self._coefficients:
            return 0.0
        return self._coefficients[1]  # das28_std is the second basis term


__all__ = ["MAX_INTERVAL_DAYS", "MIN_INTERVAL_DAYS", "WEIGHT_CEILING", "VisitIntensityModel"]
