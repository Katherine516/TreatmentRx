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
    DEFAULT_BOOTSTRAP_ALPHA,
    DEFAULT_REPLICATES,
    ContrastTest,
    contrast_test,
    m_out_of_n_bootstrap,
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

# Chosen by measurement, not by feel. `treatmentrx.cli coverage` sweeps it: at
# 1.0 the penalty shrinks contrasts enough to cost ~0.014 of bias on an effect of
# 0.088 and drop interval coverage from 88% to 85%, while buying only ~6% of
# variance. At 0 the stage-specific fit (78 parameters) becomes unstable at the
# sample sizes the bootstrap resamples to. 0.25 is the best worst-case parameter
# error at n=88, n=120 and n=250 alike.
DEFAULT_BLIP_RIDGE = 0.25
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
        use_ipcw: bool = True,
        compute_covariance: bool = True,
        treat_censored_as_terminal: bool = False,
    ) -> None:
        if not cohort:
            raise ValueError("Cannot fit a Q-learning model on an empty cohort")
        self.share_blip = share_blip
        self.blip_ridge = blip_ridge
        self.backward_induction = backward_induction
        self.use_ipcw = use_ipcw
        self.compute_covariance = compute_covariance
        # Only ever True as an explicit comparator — see `_fit` and
        # `tests/test_censoring.py`. It reproduces a defect, not an option.
        self.treat_censored_as_terminal = treat_censored_as_terminal
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
        self.censoring = None
        self.bootstrap = None
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
        censoring = self._censoring_model(cohort)
        rows: list[list[tuple[int, float]]] = []
        stage_of_row: list[int] = []
        cluster_of_row: list[int] = []
        observed: list[float] = []
        next_features: list[dict[str, float] | None] = []
        weights: list[float] = []

        for cluster, trajectory in enumerate(cohort):
            stages = trajectory.stages
            for position, stage in enumerate(stages):
                terminal = position == self.n_stages - 1
                following = stages[position + 1] if position + 1 < len(stages) else None
                # A censored patient's last observed stage is *not* a terminal
                # decision — their future is unobserved, not absent. Using the
                # observed outcome alone as the target would tell the model that
                # continuing is worth nothing after a dropout. Those rows are
                # dropped and the patients who did return carry their weight.
                if not terminal and following is None:
                    if not self.treat_censored_as_terminal:
                        continue
                    # The defect: count the row as a terminal decision anyway.
                rows.append(self._row(position, stage.features, stage.arm))
                stage_of_row.append(position)
                cluster_of_row.append(cluster)
                observed.append(stage.outcome)
                next_features.append(dict(following.features) if following else None)
                weights.append(
                    censoring.row_weight(trajectory, position, needs_next=not terminal)
                    if censoring
                    else 1.0
                )

        if not rows:
            raise ValueError("No usable rows: every trajectory was censored before a decision")

        penalties = self._penalties()
        targets = list(observed)

        # X'WX does not change across the fixed point — only the pseudo-outcomes
        # do. Factor it once and each iteration is a cheap X'Wy plus a multiply,
        # which is what makes the bootstrap's hundreds of refits affordable.
        normal_matrix = linalg.sparse_normal_matrix(rows, weights, self.n_features, penalties)
        normal_inverse = linalg.inverse(normal_matrix)

        # The covariate bases of the *next* stage never change across the fixed
        # point either; only the coefficients they multiply do. Precomputing
        # them turns each pseudo-outcome update into arithmetic.
        futures = [
            None
            if features is None
            else self._future_terms(features, min(position + 1, self.n_stages - 1))
            for features, position in zip(next_features, stage_of_row)
        ]

        for iteration in range(1, _MAX_ITERATIONS + 1):
            beta = linalg.solve_precomputed(normal_inverse, rows, targets, weights, self.n_features)
            shift = max(abs(a - b) for a, b in zip(beta, self._beta))
            self._beta = beta
            self.iterations = iteration
            if not self.backward_induction:
                break
            # Recompute pseudo-outcomes under the updated Q-function.
            updated = [
                value if terms is None else value + self._optimal_value_from(terms, beta)
                for value, terms in zip(observed, futures)
            ]
            converged = shift < _TOLERANCE and all(
                abs(a - b) < _TOLERANCE for a, b in zip(updated, targets)
            )
            targets = updated
            if converged:
                break

        # Bootstrap replicates only need the point estimate; the covariance is
        # the expensive part and nothing asks a replicate for its own interval.
        if self.compute_covariance:
            self._covariance = self._sandwich(
                rows, targets, weights, cluster_of_row, normal_matrix
            )

    def _future_terms(self, features: dict[str, float], stage_index: int):
        """Column/value pairs for `max_a Q(features, a)` at a fixed stage.

        Returns the treatment-free pairs once, plus one list of blip pairs per
        arm — everything the optimal-value calculation needs that does not
        depend on the current coefficients.
        """
        offset = stage_index * self._n_free
        free = [
            (offset + i, value)
            for i, value in enumerate(treatment_free_basis(features))
            if value != 0.0
        ]
        basis = blip_basis(features)
        per_arm = []
        for arm in self.arms:
            columns = self._blip_columns(stage_index, arm)
            if columns is None:
                per_arm.append(())
            else:
                per_arm.append(
                    tuple((column, value) for column, value in zip(columns, basis) if value != 0.0)
                )
        return free, per_arm

    def _optimal_value_from(self, terms, beta: list[float]) -> float:
        free, per_arm = terms
        base = sum(beta[index] * value for index, value in free)
        return base + max(
            sum(beta[index] * value for index, value in pairs) for pairs in per_arm
        )

    def _censoring_model(self, cohort: list[CohortTrajectory]):
        """Fit the dropout model that supplies the row weights.

        Dropout is informative — patients leave because of toxicity and poor
        response, both consequences of the arm they were given — so the patients
        still under observation at the last stage are a healthier sample than the
        ones who started. Without these weights the blips are biased toward
        whatever the survivors experienced.
        """
        if not self.use_ipcw:
            self.censoring = None
            return None
        from treatmentrx.estimation.censoring import CensoringModel

        self.censoring = CensoringModel(cohort)
        return self.censoring

    def _sandwich(self, rows, targets, weights, clusters, normal_matrix) -> list[list[float]]:
        """Cluster-robust covariance of the fitted parameters, clustered by patient."""
        residuals = [
            target - sum(self._beta[index] * value for index, value in row)
            for row, target in zip(rows, targets)
        ]
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
        if self.bootstrap is not None and not self._sandwich_is_exact(index):
            return self.bootstrap_contrast(arm, comparator, features, index, alpha)
        return self.sandwich_contrast(arm, comparator, features, index, alpha)

    def _sandwich_is_exact(self, index: int) -> bool:
        """Is the sandwich interval honest at this stage?

        Only at a terminal stage *and* only when the blip is stage-specific.
        With a shared blip there is no fully regular stage: the one parameter
        vector is estimated jointly from every stage's rows, and the
        earlier-stage rows carry pseudo-outcomes. Its terminal-stage interval
        inherits that, so the sandwich understates it too.
        """
        return index == self.n_stages - 1 and not self.share_blip

    def sandwich_contrast(
        self,
        arm: str,
        comparator: str,
        features: dict[str, float],
        stage_index: int,
        alpha: float = DEFAULT_ALPHA,
    ) -> ContrastTest:
        """Cluster-robust interval, treating the pseudo-outcomes as fixed."""
        index = self._clamp_stage(stage_index)
        loading = self._contrast_loading(arm, comparator, features, index)
        difference = self.blip(arm, features, index) - self.blip(comparator, features, index)
        horizon = self.remaining_stages(index)
        caveat = (
            ""
            if self._sandwich_is_exact(index)
            else (
                "The interval treats the pseudo-outcomes as fixed and therefore "
                "understates uncertainty; run `treatmentrx inference` for a "
                "resampled interval that does not."
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

    # ------------------------------------------------------- bootstrap support

    def refit(self, cohort: list[CohortTrajectory]) -> list[float]:
        """Re-run the whole procedure on a resample and return the parameters."""
        replica = QLearningModel(
            cohort,
            share_blip=self.share_blip,
            blip_ridge=self.blip_ridge,
            arms=self.arms,
            backward_induction=self.backward_induction,
            use_ipcw=self.use_ipcw,
            compute_covariance=False,
        )
        if replica.n_features != self.n_features:
            # A resample that lost a whole stage cannot be aligned with the fit.
            raise ValueError("resample produced a different design")
        return replica._beta

    def non_regularity(self, cohort: list[CohortTrajectory], nu: float = 0.05) -> float:
        """Proportion of patients whose next-stage optimal arm is ambiguous.

        The non-smoothness that breaks the ordinary bootstrap is the `max` in the
        pseudo-outcome: where two arms are tied, the argmax jumps. This measures
        how much of the cohort sits at or near such a tie, using the
        terminal-stage sandwich, which is valid there.
        """
        terminal = self.n_stages - 1
        ambiguous = 0
        total = 0
        for trajectory in cohort:
            features = trajectory.stages[-1].features
            ordered = sorted(self.arms, key=lambda arm: self.raw_q(features, arm, terminal), reverse=True)
            if len(ordered) < 2:
                continue
            total += 1
            # Always the sandwich: this is a plug-in measure that *decides* the
            # resample size, so it must not depend on a bootstrap that may
            # already be attached, or a refit would not be reproducible.
            if not self.sandwich_contrast(
                ordered[0], ordered[1], features, terminal, alpha=nu
            ).distinguishable:
                ambiguous += 1
        return ambiguous / total if total else 0.0

    def fit_bootstrap(
        self,
        cohort: list[CohortTrajectory],
        replicates: int = DEFAULT_REPLICATES,
        alpha: float = DEFAULT_BOOTSTRAP_ALPHA,
        seed: int = 17,
    ):
        """Run the m-out-of-n bootstrap and attach it. One refit per replicate."""
        distribution = m_out_of_n_bootstrap(
            self.refit,
            cohort,
            self._beta,
            self.non_regularity(cohort),
            replicates=replicates,
            alpha=alpha,
            seed=seed,
        )
        self.attach_bootstrap(distribution)
        return distribution

    def attach_bootstrap(self, distribution) -> None:
        """Use these draws for non-terminal contrasts from now on."""
        self.bootstrap = distribution

    def bootstrap_contrast(
        self,
        arm: str,
        comparator: str,
        features: dict[str, float],
        stage_index: int,
        alpha: float = DEFAULT_ALPHA,
    ) -> ContrastTest:
        if self.bootstrap is None:
            raise ValueError("No bootstrap draws attached; call attach_bootstrap first")
        index = self._clamp_stage(stage_index)
        loading = self._contrast_loading(arm, comparator, features, index)
        horizon = self.remaining_stages(index)
        difference = (self.blip(arm, features, index) - self.blip(comparator, features, index)) / horizon
        lower, upper = self.bootstrap.interval(loading, alpha)
        return ContrastTest(
            arm=arm,
            comparator=comparator,
            difference=difference,
            standard_error=self.bootstrap.standard_error(loading) / horizon,
            lower=lower / horizon,
            upper=upper / horizon,
            alpha=alpha,
            caveat=(
                f"m-out-of-n bootstrap (m={self.bootstrap.m} of n={self.bootstrap.n}, "
                f"non-regularity {self.bootstrap.non_regularity:.2f}); accounts for the "
                "pseudo-outcome step."
            ),
        )

    def _contrast_loading(
        self, arm: str, comparator: str, features: dict[str, float], index: int
    ) -> list[float]:
        loading = [0.0] * self.n_features
        basis = blip_basis(features)
        for sign, candidate in ((1.0, arm), (-1.0, comparator)):
            columns = self._blip_columns(index, candidate)
            if columns is None:
                continue
            for column, value in zip(columns, basis):
                loading[column] += sign * value
        return loading

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
