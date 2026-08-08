"""Inverse-probability-of-censoring weights for informative dropout.

Patients leave the cohort for reasons the treatment caused — toxicity and poor
response — so the patients still under observation at stage 3 are a healthier,
better-responding sample than the ones who started. Fitting on them as if they
were a random sample biases every blip toward whatever the survivors experienced.

The fix is standard: weight each observed decision point by the inverse
probability that the patient was still under observation to reach it. A patient
who was unlikely to still be there stands in for the similar patients who were
not.

**The probabilities are estimated, not read off the generator.** The simulation
knows the true dropout hazard, but using it would be cheating — in real data it
is unobservable. `CensoringModel` fits a linear-probability dropout model on the
covariates and the observed response, which is what an analyst would actually
have. `ra_cohort.CohortStage.uncensored_probability` carries the truth, and
`tests/test_censoring.py` uses it only to check that the estimate is close.

Weights are **stabilised** (the marginal survival probability over the estimated
conditional one) and clipped. Unstabilised weights on a strongly informative
hazard produce a handful of enormous weights that a few patients then dominate.

**How much this actually buys, measured.** On the current generating process the
weights are a small correction — roughly 5% of the total blip error. Almost all
of the censoring bias comes from something else entirely: treating a censored
patient's last observed stage as a *terminal* decision, which tells the model
that continuing is worth nothing after a dropout. Excluding those rows (see
`q_learning.QLearningModel._fit`) removes about 0.25 of blip error; the weights
then remove about 0.004 more.

That is the honest result and it is worth stating rather than tuning the
simulation until the weights look indispensable. They matter here because the
outcome model is correctly specified and conditions on the covariates that drive
both dropout and response. On real data neither of those holds, which is exactly
when this machinery earns its place — so it stays, with its effect size
measured rather than assumed.
"""

from __future__ import annotations

from treatmentrx.arms import REFERENCE_ARM, TREATMENT_ARMS
from treatmentrx.estimation import linalg
from treatmentrx.estimation.basis import treatment_free_basis
from treatmentrx.simulation.ra_cohort import CohortTrajectory, clamp

# Arms other than the reference get an indicator and a response interaction:
# retention depends on the arm, and on how well *that* arm is working.
_ARM_ORDER = tuple(arm for arm in TREATMENT_ARMS if arm != REFERENCE_ARM)

# A patient with an estimated survival probability below this is treated as if
# it were this — beyond it the weight says more about the model than the patient.
PROBABILITY_FLOOR = 0.05
# Even stabilised, a weight this large is a single patient carrying a stage.
WEIGHT_CEILING = 10.0
_RIDGE = 1e-4


class CensoringModel:
    """Estimated probability of remaining under observation at each stage."""

    def __init__(self, cohort: list[CohortTrajectory]) -> None:
        self.n_stages = max((traj.n_observed for traj in cohort), default=1)
        self._coefficients: list[float] = []
        self._marginal: list[float] = [1.0] * (self.n_stages + 1)
        # Keyed by identity: trajectories are frozen and the cache lives exactly
        # as long as the model that built it.
        self._survival_cache: dict[int, list[float]] = {}
        self._fit(cohort)

    def _fit(self, cohort: list[CohortTrajectory]) -> None:
        # One row per observed transition: did this patient come back?
        design: list[list[float]] = []
        targets: list[float] = []
        for trajectory in cohort:
            for position, stage in enumerate(trajectory.stages):
                at_risk = position + 1 < self.n_stages
                if not at_risk:
                    continue
                returned = position + 1 < trajectory.n_observed
                design.append(self._row(stage.features, stage.outcome, stage.arm))
                targets.append(1.0 if returned else 0.0)

        if design and len(design) > len(design[0]):
            self._coefficients = linalg.weighted_least_squares(
                design, targets, [1.0] * len(design), ridge=_RIDGE
            )

        # Marginal survival by stage, for the numerator of the stabilised weight.
        total = len(cohort) or 1
        for stage_index in range(self.n_stages + 1):
            reached = sum(1 for t in cohort if t.n_observed > stage_index)
            self._marginal[stage_index] = max(reached / total, PROBABILITY_FLOOR)

    def _row(self, features: dict[str, float], outcome: float, arm: str) -> list[float]:
        """Toxicity, response, the arm, and how well the arm is working.

        The interaction terms are the ones that matter: retention on a
        burdensome arm falls away much faster with a poor response than
        retention on an oral one, and a model without that interaction cannot
        remove the differential selection it causes.
        """
        indicators = [1.0 if arm == candidate else 0.0 for candidate in _ARM_ORDER]
        interactions = [indicator * (outcome - 0.5) for indicator in indicators]
        return treatment_free_basis(features) + [outcome] + indicators + interactions

    def continuation_probability(
        self, features: dict[str, float], outcome: float, arm: str = REFERENCE_ARM
    ) -> float:
        """Estimated P(patient returns for the next decision point)."""
        if not self._coefficients:
            return 1.0
        row = self._row(features, outcome, arm)
        return clamp(linalg.dot(row, self._coefficients), PROBABILITY_FLOOR, 1.0)

    def survival_probabilities(self, trajectory: CohortTrajectory) -> list[float]:
        """P(still observed) at each of this trajectory's stages, cumulatively.

        Memoised: the fit asks for a weight once per row, and each ask used to
        rebuild the patient's whole survival chain from the start.
        """
        cached = self._survival_cache.get(id(trajectory))
        if cached is not None:
            return cached
        probabilities = [1.0]
        for stage in trajectory.stages[:-1]:
            probabilities.append(
                probabilities[-1]
                * self.continuation_probability(stage.features, stage.outcome, stage.arm)
            )
        self._survival_cache[id(trajectory)] = probabilities
        return probabilities

    def weights(self, trajectory: CohortTrajectory) -> list[float]:
        """Stabilised IPCW weight for each observed stage of this trajectory."""
        survival = self.survival_probabilities(trajectory)
        return [
            self._stabilise(position, probability)
            for position, probability in enumerate(survival)
        ]

    def row_weight(self, trajectory: CohortTrajectory, position: int, needs_next: bool) -> float:
        """Weight for one regression row.

        A row that carries a pseudo-outcome needs the patient to have been
        observed at the *next* decision point too, so it is weighted by the
        probability of surviving that far — not merely to the current stage.
        Weighting it by the current stage's probability would leave the
        selection that the pseudo-outcome depends on uncorrected.
        """
        survival = self.survival_probabilities(trajectory)
        index = position + 1 if needs_next else position
        if index < len(survival):
            probability = survival[index]
        elif survival:
            # One step past the observed trajectory: extend by the fitted hazard.
            last = trajectory.stages[len(survival) - 1]
            probability = survival[-1] * self.continuation_probability(
                last.features, last.outcome, last.arm
            )
        else:  # pragma: no cover - a trajectory with no stages
            probability = 1.0
        return self._stabilise(index, probability)

    def _stabilise(self, index: int, probability: float) -> float:
        marginal = self._marginal[min(index, len(self._marginal) - 1)]
        return clamp(
            marginal / max(probability, PROBABILITY_FLOOR),
            1.0 / WEIGHT_CEILING,
            WEIGHT_CEILING,
        )


def censoring_weights(cohort: list[CohortTrajectory], model: CensoringModel | None = None) -> list[list[float]]:
    """Per-trajectory, per-stage stabilised weights for a whole cohort."""
    model = model or CensoringModel(cohort)
    return [model.weights(trajectory) for trajectory in cohort]


__all__ = ["PROBABILITY_FLOOR", "WEIGHT_CEILING", "CensoringModel", "censoring_weights"]
