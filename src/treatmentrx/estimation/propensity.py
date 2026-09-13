"""P(arm | covariates) under the policy that generated the data — estimated.

`CohortStage.propensity` is the probability the *simulator* used when it drew the
arm. No real deployment observes that number, so every quantity computed from it
inherits an advantage it will not have: the held-out IPW policy value, its
bootstrap interval, and the improvement over the behaviour policy that the
validation ladder gates on all read `stage.propensity` directly.

This fits the same quantity from the data an analyst would actually have. It is a
multinomial logit over the blip basis — the covariates the assignment policy in
`simulation/ra_cohort` really uses — fit by iteratively reweighted least squares
on the dependency-free solver in `linalg.py`.

**What it is not.** A propensity model is where unmeasured confounding enters,
and fitting one on the covariates that generated the assignment is the easy case
by construction. On real data the assignment depends on things the record does
not hold — a clinician's read of frailty, what the insurer approved — and no
amount of fitting recovers those. What this establishes is the weaker and still
necessary claim: that the evaluation does not *require* oracle knowledge, and
that swapping the true propensity for an estimated one moves the headline
numbers by a stated amount rather than an unknown one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from treatmentrx.estimation import linalg
from treatmentrx.estimation.basis import blip_basis
from treatmentrx.simulation.ra_cohort import TREATMENT_ARMS, CohortTrajectory, clamp

# Matches the clipping used when fitting and when weighting, so a near-
# deterministic preference cannot produce an unbounded weight.
PROPENSITY_FLOOR = 0.02
PROPENSITY_CEILING = 0.98

_RIDGE = 1e-3
# The block-diagonal step below is a bound optimisation: monotone and slow. The
# tolerance is set where the *estimate* stops moving rather than where the
# coefficients do, because the propensity is used as `1/p` with `p` clipped at
# 0.02 and precision past that buys nothing. Measured on the deployed split:
#
#   tolerance   passes   held-out MAE   policy value
#     1e-4         53       0.02113        0.7395
#     1e-6         95       0.02114        0.7395
#     1e-8        136       0.02114        0.7395
#
# Three decimal places of coefficient precision cost 83 extra passes and change
# the number this feeds by nothing at four decimal places.
_MAX_ITERATIONS = 200
_TOLERANCE = 1e-4


@dataclass(frozen=True)
class PropensityFit:
    """Per-arm coefficient vectors over the blip basis, plus fit diagnostics."""

    coefficients: dict[str, list[float]]
    reference: str
    n_rows: int
    iterations: int
    converged: bool


class PropensityModel:
    """Multinomial logit for the behaviour policy, fit by IRLS.

    One coefficient vector per non-reference arm; the reference arm's is zero by
    construction, which is what makes the others identified.
    """

    def __init__(
        self,
        cohort: list[CohortTrajectory],
        arms: tuple[str, ...] = TREATMENT_ARMS,
        ridge: float = _RIDGE,
    ) -> None:
        if not cohort:
            raise ValueError("Cannot fit a propensity model on an empty cohort")
        self.arms = arms
        self.reference = arms[0]
        self.ridge = ridge
        self._rows = [
            (blip_basis(stage.features), stage.arm)
            for trajectory in cohort
            for stage in trajectory.stages
        ]
        self._width = len(self._rows[0][0])
        self.fit = self._fit()

    def _fit(self) -> PropensityFit:
        """Iteratively reweighted least squares, one binary fit per arm per pass.

        A full multinomial Newton step would need the cross-arm blocks of the
        Hessian; this uses the standard block-diagonal approximation, which is
        what makes each step a weighted least squares the existing solver can do.
        It converges more slowly and to the same place.
        """
        active = [arm for arm in self.arms if arm != self.reference]
        coefficients = {arm: [0.0] * self._width for arm in active}
        iterations = 0
        converged = False
        # The design never changes — only the working response and the weights
        # do. Rebuilding it inside the arm loop meant constructing this list
        # once per arm per pass, several hundred times over.
        design = [row for row, _ in self._rows]
        observed_arms = [arm for _, arm in self._rows]

        for iterations in range(1, _MAX_ITERATIONS + 1):
            shift = 0.0
            probabilities = [self._probabilities(row, coefficients) for row in design]
            for arm in active:
                beta = coefficients[arm]
                targets: list[float] = []
                weights: list[float] = []
                for row, observed, probability in zip(design, observed_arms, probabilities):
                    p = clamp(probability[arm], 1e-6, 1 - 1e-6)
                    weight = p * (1.0 - p)
                    eta = linalg.dot(row, beta)
                    # Working response: the current linear predictor plus the
                    # score, which is what makes the weighted LS a Newton step.
                    targets.append(eta + ((1.0 if observed == arm else 0.0) - p) / weight)
                    weights.append(weight)
                updated = linalg.weighted_least_squares(design, targets, weights, ridge=self.ridge)
                shift = max(shift, max(abs(a - b) for a, b in zip(updated, beta)))
                coefficients[arm] = updated
            if shift < _TOLERANCE:
                converged = True
                break

        return PropensityFit(
            coefficients=coefficients,
            reference=self.reference,
            n_rows=len(self._rows),
            iterations=iterations,
            converged=converged,
        )

    def _probabilities(self, row: list[float], coefficients) -> dict[str, float]:
        scores = {self.reference: 0.0}
        for arm, beta in coefficients.items():
            scores[arm] = linalg.dot(row, beta)
        largest = max(scores.values())
        exponentiated = {arm: math.exp(score - largest) for arm, score in scores.items()}
        total = sum(exponentiated.values())
        return {arm: value / total for arm, value in exponentiated.items()}

    def probabilities(self, features: dict[str, float]) -> dict[str, float]:
        """Estimated P(arm | X) for every arm on the menu."""
        return self._probabilities(blip_basis(features), self.fit.coefficients)

    def propensity(self, features: dict[str, float], arm: str) -> float:
        """Estimated probability that this patient received `arm`, clipped."""
        return clamp(
            self.probabilities(features).get(arm, PROPENSITY_FLOOR),
            PROPENSITY_FLOOR,
            PROPENSITY_CEILING,
        )

    def calibration(self, cohort: list[CohortTrajectory]) -> dict[str, float]:
        """How close the fitted propensities are to the ones that generated the data.

        Available only in simulation — `CohortStage.propensity` is the truth here
        and does not exist on real data. It is the check that says whether the
        estimated weights are standing in for the true ones or replacing them
        with something else.
        """
        errors: list[float] = []
        ratios: list[float] = []
        for trajectory in cohort:
            for stage in trajectory.stages:
                estimated = self.propensity(stage.features, stage.arm)
                truth = max(stage.propensity, PROPENSITY_FLOOR)
                errors.append(abs(estimated - truth))
                ratios.append(estimated / truth)
        n = len(errors) or 1
        return {
            "mean_absolute_error": round(sum(errors) / n, 5),
            "max_absolute_error": round(max(errors), 5) if errors else 0.0,
            "mean_ratio_to_truth": round(sum(ratios) / n, 4),
        }


__all__ = ["PROPENSITY_CEILING", "PROPENSITY_FLOOR", "PropensityFit", "PropensityModel"]
