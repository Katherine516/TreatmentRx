"""Penalized Q-learning with backward induction over multi-stage treatment regimes.

This replaces the hand-tuned scorer that used to stand in for Q-learning. The
model is

    Q_j(X, a) = beta_j . f(X) + psi_a . h(X)          (shared blip)
    Q_j(X, a) = beta_j . f(X) + psi_{j,a} . h(X)      (stage-specific blip)

fit by penalized least squares. `f` is the treatment-free basis, `h` the blip
basis (the tailoring variables the treatment effect may vary over), and the
reference arm carries a zero blip so every other arm's psi is its causal
advantage over "continue-current".

**Backward induction.** The target at the terminal stage is the observed
outcome; at earlier stages it is the pseudo-outcome

    Y_j + max_a Q_{j+1}(X_{j+1}, a)

so a stage-1 blip absorbs the *delayed* consequences of the choice — which is
the entire reason a sequential method is used instead of a per-visit predictor.
Because the shared blip couples the stages, the pseudo-outcomes and the fit are
iterated to a fixed point (a handful of passes; stage-specific blips converge on
the first).

**Penalization.** Ridge is applied to the blip block only. Treatment-free
nuisance terms are left unpenalized, and the penalty is what makes the
stage-specific variant (five times as many blip parameters, fit on the same
rows) usable at all — which is exactly the shared-vs-stage-specific trade-off
the regime selector is choosing between.
"""

from __future__ import annotations

import math

from treatmentrx.estimation import linalg
from treatmentrx.estimation.inference import (
    DEFAULT_ALPHA,
    ContrastTest,
    contrast_test,
    sandwich_covariance,
)
from treatmentrx.estimation.basis import (
    BLIP_BASIS,
    TREATMENT_FREE_BASIS,
    blip_basis,
    treatment_free_basis,
)
from treatmentrx.simulation.ra_cohort import (
    REFERENCE_ARM,
    TREATMENT_ARMS,
    CohortTrajectory,
    clamp,
)

# Estimator identities live with the models they name, so nothing has to import
# `training` at class-definition time just to know what it is called.
Q_SHARED_METHOD = "Q-Shared + Penalized"
STAGE_SPECIFIC_METHOD = "Stage-Specific Q-learning"

DEFAULT_BLIP_RIDGE = 1.0
_NUISANCE_RIDGE = 1e-6
_MAX_ITERATIONS = 25
_TOLERANCE = 1e-9

# Q-values are reported as probabilities of response; keep them off the boundary
# so downstream ratio-based diagnostics stay finite.
Q_FLOOR = 0.01
Q_CEILING = 0.99


class QLearningModel:
    """Fitted multi-stage Q-function over the RA arm menu."""

    def __init__(
        self,
        cohort: list[CohortTrajectory],
        share_blip: bool = True,
        blip_ridge: float = DEFAULT_BLIP_RIDGE,
        arms: tuple[str, ...] = TREATMENT_ARMS,
        backward_induction: bool = True,
    ) -> None:
        if not cohort:
            raise ValueError("Cannot fit a Q-learning model on an empty cohort")
        self.share_blip = share_blip
        self.blip_ridge = blip_ridge
        self.backward_induction = backward_induction
        self.arms = arms
        self.blip_arms = tuple(arm for arm in arms if arm != REFERENCE_ARM)
        self.n_stages = max(len(trajectory.stages) for trajectory in cohort)
        self.n_train = len(cohort)
        self._n_free = len(TREATMENT_FREE_BASIS)
        self._n_blip = len(BLIP_BASIS)
        self._blip_offset = self.n_stages * self._n_free
        self._blip_span = self._n_blip * len(self.blip_arms)
        self.n_features = self._blip_offset + (
            self._blip_span * (1 if share_blip else self.n_stages)
        )
        self.iterations = 0
        self._beta: list[float] = [0.0] * self.n_features
        self._covariance: list[list[float]] = []
        self._fit(cohort)

    # ---------------------------------------------------------------- fitting

    def _blip_columns(self, stage_index: int, arm: str) -> list[int] | None:
        if arm == REFERENCE_ARM or arm not in self.blip_arms:
            return None
        arm_index = self.blip_arms.index(arm)
        block = 0 if self.share_blip else stage_index
        start = self._blip_offset + block * self._blip_span + arm_index * self._n_blip
        return list(range(start, start + self._n_blip))

    def _row(self, stage_index: int, features: dict[str, float], arm: str) -> list[tuple[int, float]]:
        offset = stage_index * self._n_free
        row = [
            (offset + i, value)
            for i, value in enumerate(treatment_free_basis(features))
            if value != 0.0
        ]
        columns = self._blip_columns(stage_index, arm)
        if columns is not None:
            row.extend(
                (column, value)
                for column, value in zip(columns, blip_basis(features))
                if value != 0.0
            )
        return row

    def _penalties(self) -> list[float]:
        penalties = [_NUISANCE_RIDGE] * self.n_features
        for index in range(self._blip_offset, self.n_features):
            penalties[index] = self.blip_ridge
        return penalties

    def _fit(self, cohort: list[CohortTrajectory]) -> None:
        rows: list[list[tuple[int, float]]] = []
        stage_of_row: list[int] = []
        cluster_of_row: list[int] = []
        observed: list[float] = []
        next_features: list[dict[str, float] | None] = []

        for cluster, trajectory in enumerate(cohort):
            stages = trajectory.stages
            for position, stage in enumerate(stages):
                rows.append(self._row(position, stage.features, stage.arm))
                stage_of_row.append(position)
                cluster_of_row.append(cluster)
                observed.append(stage.outcome)
                following = stages[position + 1] if position + 1 < len(stages) else None
                next_features.append(dict(following.features) if following else None)

        weights = [1.0] * len(rows)
        penalties = self._penalties()
        targets = list(observed)

        for iteration in range(1, _MAX_ITERATIONS + 1):
            beta = linalg.sparse_weighted_least_squares(
                rows, targets, weights, self.n_features, penalties
            )
            shift = max(abs(a - b) for a, b in zip(beta, self._beta))
            self._beta = beta
            self.iterations = iteration
            if not self.backward_induction:
                break
            # Recompute pseudo-outcomes under the updated Q-function.
            updated = [
                value
                if features is None
                else value + self._optimal_value(features, min(position + 1, self.n_stages - 1))
                for value, features, position in zip(observed, next_features, stage_of_row)
            ]
            converged = shift < _TOLERANCE and all(
                abs(a - b) < _TOLERANCE for a, b in zip(updated, targets)
            )
            targets = updated
            if converged:
                break

        self._covariance = self._sandwich(rows, targets, weights, cluster_of_row, penalties)

    def _sandwich(self, rows, targets, weights, clusters, penalties) -> list[list[float]]:
        """Cluster-robust covariance of the fitted parameters, clustered by patient."""
        residuals = [
            target - sum(self._beta[index] * value for index, value in row)
            for row, target in zip(rows, targets)
        ]
        normal_matrix = linalg.sparse_normal_matrix(rows, weights, self.n_features, penalties)
        return sandwich_covariance(
            rows, residuals, weights, clusters, normal_matrix, self.n_features
        )

    # ------------------------------------------------------------- prediction

    def treatment_free(self, features: dict[str, float], stage_index: int) -> float:
        offset = self._clamp_stage(stage_index) * self._n_free
        basis = treatment_free_basis(features)
        return sum(self._beta[offset + i] * value for i, value in enumerate(basis))

    def blip(self, arm: str, features: dict[str, float], stage_index: int) -> float:
        """Estimated causal advantage of `arm` over the reference arm."""
        columns = self._blip_columns(self._clamp_stage(stage_index), arm)
        if columns is None:
            return 0.0
        basis = blip_basis(features)
        return sum(self._beta[column] * value for column, value in zip(columns, basis))

    def raw_q(self, features: dict[str, float], arm: str, stage_index: int) -> float:
        """Value-to-go: this stage's outcome plus the optimal remaining horizon.

        This is the quantity the policy maximises. It is *not* on the
        single-stage outcome scale — see `q_values` and `predict_outcome`.
        """
        return self.treatment_free(features, stage_index) + self.blip(arm, features, stage_index)

    def remaining_stages(self, stage_index: int) -> int:
        return self.n_stages - self._clamp_stage(stage_index)

    def predict_outcome(self, features: dict[str, float], arm: str, stage_index: int = 0) -> float:
        """Predicted **single-stage** outcome — what calibration is measured on.

        The terminal-stage parameters are the single-stage outcome model: at the
        last decision there is no future to carry, so `Q_J` is fit directly
        against observed outcomes. Earlier-stage parameters are values-to-go and
        would be badly miscalibrated against a single visit's outcome.
        """
        return clamp(self.raw_q(features, arm, self.n_stages - 1), 0.0, 1.0)

    def q_values(
        self,
        features: dict[str, float],
        stage_index: int,
        menu: tuple[str, ...] | None = None,
    ) -> dict[str, float]:
        """Q-values rescaled to expected response *per remaining visit*.

        Dividing the value-to-go by the number of remaining stages is a positive
        constant, so it leaves the ranking (and therefore the recommendation)
        untouched, while putting every estimator's numbers on one interpretable
        scale. Without this, model averaging would mix a two-visit value-to-go
        from Q-learning with a single-visit outcome from dWOLS.
        """
        arms = menu or self.arms
        horizon = self.remaining_stages(stage_index)
        return {
            arm: round(clamp(self.raw_q(features, arm, stage_index) / horizon, Q_FLOOR, Q_CEILING), 3)
            for arm in arms
        }

    def recommend(self, features: dict[str, float], stage_index: int, menu: tuple[str, ...] | None = None) -> str:
        arms = menu or self.arms
        return max(arms, key=lambda arm: self.raw_q(features, arm, stage_index))

    def greedy_policy(self):
        """A `policy(features, stage_index) -> arm` callable for policy evaluation."""

        def policy(features: dict[str, float], stage_index: int) -> str:
            return self.recommend(features, stage_index)

        return policy

    def contrast(
        self,
        arm: str,
        comparator: str,
        features: dict[str, float],
        stage_index: int,
        alpha: float = DEFAULT_ALPHA,
    ) -> ContrastTest:
        """Is `arm` separable from `comparator` for this patient, and by how much?

        The treatment-free term is common to both arms and cancels, so the
        contrast is a difference of blips — which is exactly the quantity the
        blip parameters were fit to estimate, and the one that carries a usable
        standard error.
        """
        index = self._clamp_stage(stage_index)
        loading = [0.0] * self.n_features
        basis = blip_basis(features)
        for sign, candidate in ((1.0, arm), (-1.0, comparator)):
            columns = self._blip_columns(index, candidate)
            if columns is None:
                continue
            for column, value in zip(columns, basis):
                loading[column] += sign * value

        difference = self.blip(arm, features, index) - self.blip(comparator, features, index)
        horizon = self.remaining_stages(index)
        caveat = (
            ""
            if index == self.n_stages - 1
            else (
                "Non-terminal stage: the interval treats the pseudo-outcomes as fixed and "
                "therefore understates uncertainty."
            )
        )
        test = contrast_test(
            self._covariance, loading, difference, arm, comparator, alpha=alpha, caveat=caveat
        )
        # Report on the same per-remaining-visit scale as `q_values`.
        return ContrastTest(
            arm=test.arm,
            comparator=test.comparator,
            difference=test.difference / horizon,
            standard_error=test.standard_error / horizon,
            lower=test.lower / horizon,
            upper=test.upper / horizon,
            alpha=test.alpha,
            caveat=test.caveat,
        )

    def blip_standard_error(self, arm: str, features: dict[str, float], stage_index: int) -> float:
        """Standard error of a single arm's blip, on the per-remaining-visit scale."""
        index = self._clamp_stage(stage_index)
        columns = self._blip_columns(index, arm)
        if columns is None:
            return 0.0
        loading = [0.0] * self.n_features
        for column, value in zip(columns, blip_basis(features)):
            loading[column] = value
        variance = max(linalg.quadratic_form(loading, self._covariance), 0.0)
        return math.sqrt(variance) / self.remaining_stages(index)

    def blip_parameters(self, arm: str, stage_index: int = 0) -> dict[str, float]:
        columns = self._blip_columns(self._clamp_stage(stage_index), arm)
        if columns is None:
            return {name: 0.0 for name in BLIP_BASIS}
        return {name: self._beta[column] for name, column in zip(BLIP_BASIS, columns)}

    def coefficient_summary(self, arm: str, stage_index: int = 0) -> dict[str, float]:
        """Flat, audit-friendly view of every parameter behind the decision.

        Blips for *all* arms are reported, not just the recommended one: model
        averaging can select an arm the dominant estimator did not, and the
        explanation layer must still be able to decompose it.
        """
        offset = self._clamp_stage(stage_index) * self._n_free
        summary = {
            f"beta:{name}": round(self._beta[offset + i], 4)
            for i, name in enumerate(TREATMENT_FREE_BASIS)
        }
        for candidate in self.arms:
            if candidate == REFERENCE_ARM:
                continue
            summary.update(
                {
                    f"psi:{candidate}:{name}": round(value, 4)
                    for name, value in self.blip_parameters(candidate, stage_index).items()
                }
            )
        return summary

    def _optimal_value(self, features: dict[str, float], stage_index: int) -> float:
        return max(self.raw_q(features, arm, stage_index) for arm in self.arms)

    def _clamp_stage(self, stage_index: int) -> int:
        return max(0, min(stage_index, self.n_stages - 1))


def myopic_model(cohort: list[CohortTrajectory], **kwargs) -> QLearningModel:
    """Fit without backward induction — every stage target is its own outcome.

    An explicit comparator so `tests/test_estimators.py` can show that backward
    induction actually buys something on the delayed-toxicity DGP.
    """
    return QLearningModel(cohort, backward_induction=False, **kwargs)
