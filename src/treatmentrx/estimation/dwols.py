"""dWOLS-Shared: doubly-robust blip estimation (Wallace & Moodie weighted OLS).

The blip parameters psi are doubly robust — consistent if *either* the
propensity model or the treatment-free model is correctly specified. That is a
genuinely different failure mode from the outcome-model-only Q-learning fit in
`q_learning.py`, which is what makes averaging the two informative rather than
decorative.

**Multi-arm.** dWOLS is a two-arm method, so each non-reference arm is fit
one-vs-reference: restrict to patients who received either that arm or
"continue-current", model the propensity of the active arm, weight each row by
``w = |A - pi(X)|``, and regress the outcome on the treatment-free basis plus
the ``A * h(X)`` blip block. Because the reference arm carries a zero blip by
construction, the per-arm psi vectors are directly comparable and can be
assembled into a full Q-function over the menu.

Pure Python on purpose — the designs are small, so the dependency-free linalg in
`linalg.py` is sufficient.
"""

from __future__ import annotations

import math

from treatmentrx.estimation import linalg
from treatmentrx.estimation.basis import (
    BLIP_BASIS,
    TREATMENT_FREE_BASIS,
    blip_basis,
    treatment_free_basis,
)
from treatmentrx.estimation.features import model_features, top_tailoring_variables
from treatmentrx.estimation.inference import (
    DEFAULT_ALPHA,
    Z_QUANTILE,
    ContrastTest,
    sandwich_covariance,
)
from treatmentrx.estimation.q_learning import Q_CEILING, Q_FLOOR
from treatmentrx.simulation.ra_cohort import (
    REFERENCE_ARM,
    TREATMENT_ARMS,
    CohortTrajectory,
    clamp,
)
from treatmentrx.contracts import RegimeEstimate
from treatmentrx.domain import RegimeType, StageRecord

DEFAULT_TREATMENT_MENU = TREATMENT_ARMS
DWOLS_METHOD = "dWOLS-Shared"

# Propensities are clipped away from 0/1 so a near-deterministic clinician
# preference cannot produce an unbounded weight.
_PROPENSITY_FLOOR = 0.02
_PROPENSITY_CEILING = 0.98
_RIDGE = 1e-4


class ArmFit:
    """One arm's doubly-robust fit against the reference arm."""

    def __init__(
        self,
        arm: str,
        rows: list[tuple[dict[str, float], float, float]],
        clusters: list[int] | None = None,
    ) -> None:
        self.arm = arm
        self.n = len(rows)
        self.clusters = clusters or list(range(len(rows)))
        self.treatment_free: list[float] = [0.0] * len(TREATMENT_FREE_BASIS)
        self.psi: list[float] = [0.0] * len(BLIP_BASIS)
        self.covariance: list[list[float]] = []
        if self.n >= len(TREATMENT_FREE_BASIS) + len(BLIP_BASIS):
            self._fit(rows)

    def _propensities(self, rows: list[tuple[dict[str, float], float, float]]) -> list[float]:
        # Linear probability model of treatment on the blip covariates, fit by
        # OLS then clipped. Only used to form weights, so a simple model is
        # adequate — and misspecifying it is exactly what double robustness is
        # supposed to survive.
        design = [blip_basis(features) for features, _, _ in rows]
        treated = [assignment for _, assignment, _ in rows]
        gamma = linalg.weighted_least_squares(design, treated, [1.0] * len(rows), ridge=_RIDGE)
        return [
            clamp(linalg.dot(row, gamma), _PROPENSITY_FLOOR, _PROPENSITY_CEILING)
            for row in design
        ]

    def _fit(self, rows: list[tuple[dict[str, float], float, float]]) -> None:
        propensities = self._propensities(rows)
        n_free = len(TREATMENT_FREE_BASIS)
        design: list[list[float]] = []
        targets: list[float] = []
        weights: list[float] = []
        for (features, assignment, outcome), propensity in zip(rows, propensities):
            free = treatment_free_basis(features)
            blip = [assignment * value for value in blip_basis(features)]
            design.append(free + blip)
            targets.append(outcome)
            weights.append(abs(assignment - propensity))
        beta = linalg.weighted_least_squares(design, targets, weights, ridge=_RIDGE)
        self.treatment_free = beta[:n_free]
        self.psi = beta[n_free:]

        n_features = len(beta)
        sparse = [[(i, value) for i, value in enumerate(row) if value != 0.0] for row in design]
        residuals = [y - linalg.dot(row, beta) for row, y in zip(design, targets)]
        normal = linalg.sparse_normal_matrix(sparse, weights, n_features, _RIDGE)
        self.covariance = sandwich_covariance(
            sparse, residuals, weights, self.clusters, normal, n_features
        )

        # Kept for the *cross*-arm covariance. Each arm is fit one-vs-reference,
        # so two arms share every reference-arm row and their estimates are
        # correlated — measured at +0.20 to +0.51 on this cohort. A contrast that
        # adds their variances as if independent is therefore too wide by 15-29%,
        # which is pure lost precision: it buys no validity and it feeds straight
        # into the abstention rate. `cross_covariance` needs the bread and the
        # per-cluster scores, and recomputing them later would mean refitting.
        self._bread = linalg.inverse(normal)
        self._scores: dict[int, list[float]] = {}
        distinct = len(set(self.clusters))
        self._cluster_scale = distinct / max(distinct - 1, 1)
        for row, residual, weight, cluster in zip(sparse, residuals, weights, self.clusters):
            score = self._scores.setdefault(cluster, [0.0] * n_features)
            for index, value in row:
                score[index] += weight * value * residual

    def blip_standard_error(self, features: dict[str, float]) -> float:
        if not self.covariance:
            return 0.0
        n_free = len(TREATMENT_FREE_BASIS)
        loading = [0.0] * len(self.covariance)
        for offset, value in enumerate(blip_basis(features)):
            loading[n_free + offset] = value
        return math.sqrt(max(linalg.quadratic_form(loading, self.covariance), 0.0))

    def cross_covariance(self, other: "ArmFit") -> list[list[float]]:
        """Cov(beta_self, beta_other) for two one-vs-reference fits.

        The usual M-estimator sandwich with a cross meat term:

            Cov = A_self^-1 . (sum_i s_self,i s_other,i') . A_other^-1'

        summed over clusters present in *both* fits. A patient who received only
        one of the two arms has a zero score in the other fit and contributes
        nothing, so the whole cross term comes from the reference-arm rows the
        two fits share — which is exactly the mechanism that correlates them.
        """
        if not self._scores or not other._scores:
            return []
        size = len(self._bread)
        shared = self._scores.keys() & other._scores.keys()
        if not shared:
            return [[0.0] * size for _ in range(size)]

        meat = [[0.0] * size for _ in range(size)]
        for cluster in shared:
            mine = self._scores[cluster]
            theirs = other._scores[cluster]
            active_mine = [(i, v) for i, v in enumerate(mine) if v != 0.0]
            active_theirs = [(j, v) for j, v in enumerate(theirs) if v != 0.0]
            for i, vi in active_mine:
                row = meat[i]
                for j, vj in active_theirs:
                    row[j] += vi * vj

        # `sandwich_covariance` scales by G/(G-1) — without that small-cluster
        # correction the sandwich is anti-conservative — so the cross term has to
        # carry a matching one or it will not reduce to the variance when the two
        # fits are the same. The geometric mean of the two corrections is the
        # symmetric rule that does: it equals G/(G-1) exactly when `other is
        # self`, which is what makes Var(x - x) come out at zero rather than at a
        # small positive residual.
        scale = math.sqrt(self._cluster_scale * other._cluster_scale)

        # A_self^-1 . meat . A_other^-1', with A symmetric so the transpose is free.
        left = linalg.matmul(self._bread, meat)
        product = linalg.matmul(left, other._bread)
        return [[value * scale for value in row] for row in product]

    def treatment_free_value(self, features: dict[str, float]) -> float:
        return linalg.dot(self.treatment_free, treatment_free_basis(features))

    def blip(self, features: dict[str, float]) -> float:
        return linalg.dot(self.psi, blip_basis(features))

    def parameters(self) -> dict[str, float]:
        return {name: value for name, value in zip(BLIP_BASIS, self.psi)}


class DWOLSModel:
    """Full-menu Q-function assembled from per-arm doubly-robust blip fits."""

    def __init__(
        self,
        cohort: list[CohortTrajectory],
        arms: tuple[str, ...] = TREATMENT_ARMS,
        reference: str = REFERENCE_ARM,
    ) -> None:
        if not cohort:
            raise ValueError("Cannot fit dWOLS on an empty cohort")
        self.arms = arms
        self.reference = reference
        self.n_stages = max(len(trajectory.stages) for trajectory in cohort)
        self.n_train = len(cohort)
        # Clustered by patient: two stages from one trajectory are correlated.
        observations = [
            (dict(stage.features), stage.arm, stage.outcome, cluster)
            for cluster, trajectory in enumerate(cohort)
            for stage in trajectory.stages
        ]
        self.fits: dict[str, ArmFit] = {}
        for arm in arms:
            if arm == reference:
                continue
            selected = [
                (features, 1.0 if observed == arm else 0.0, outcome, cluster)
                for features, observed, outcome, cluster in observations
                if observed in (arm, reference)
            ]
            self.fits[arm] = ArmFit(
                arm,
                [(f, a, y) for f, a, y, _ in selected],
                [cluster for *_, cluster in selected],
            )
        self._reference_fit = self._fit_reference(observations)
        # Cross-arm covariance, memoised per ordered pair.
        #
        # `ArmFit.cross_covariance` takes no features: it walks the clusters the
        # two fits share and returns a matrix that is a property of the *fit*,
        # not of the patient. The patient enters afterwards, in the quadratic
        # form. So it was being rebuilt identically on every request — measured
        # at 1.31ms a call and four calls a request, 56% of a warm request,
        # producing the same 10x10 matrix every time.
        #
        # This is not the cache invariant 7 forbids, and the distinction is the
        # whole point: that one was a second *model*, fit from its own cohort and
        # diverging from the one the studies measured. Nothing here caches a fit.
        # The model owns both `ArmFit`s, they are written once in `__init__` and
        # never mutated, and `refit` builds a whole new `DWOLSModel` — so a
        # bootstrap replicate gets its own empty cache and cannot read this one.
        self._cross_covariance: dict[tuple[str, str], list[list[float]]] = {}

    def _fit_reference(self, observations) -> list[float]:
        """Treatment-free model for the reference arm, from its own rows."""
        rows = [(features, outcome) for features, arm, outcome, _ in observations if arm == self.reference]
        if len(rows) < len(TREATMENT_FREE_BASIS):
            return [0.5] + [0.0] * (len(TREATMENT_FREE_BASIS) - 1)
        design = [treatment_free_basis(features) for features, _ in rows]
        targets = [outcome for _, outcome in rows]
        return linalg.weighted_least_squares(design, targets, [1.0] * len(rows), ridge=_RIDGE)

    def treatment_free_value(self, features: dict[str, float]) -> float:
        return linalg.dot(self._reference_fit, treatment_free_basis(features))

    def blip(self, arm: str, features: dict[str, float]) -> float:
        fit = self.fits.get(arm)
        return fit.blip(features) if fit else 0.0

    def blip_parameters(self, arm: str) -> dict[str, float]:
        fit = self.fits.get(arm)
        return fit.parameters() if fit else {name: 0.0 for name in BLIP_BASIS}

    def blip_standard_error(self, arm: str, features: dict[str, float]) -> float:
        fit = self.fits.get(arm)
        return fit.blip_standard_error(features) if fit else 0.0

    def contrast_standard_error(
        self, arm: str, comparator: str, features: dict[str, float]
    ) -> float:
        """SE of `blip(arm) - blip(comparator)`, with the covariance kept.

            Var(a - b) = Var(a) + Var(b) - 2 Cov(a, b)

        The third term is not optional here. Both arms are fit against the same
        reference, so they share its rows and move together: measured over 60
        refits the correlation runs +0.20 to +0.51, and dropping it left the
        contrast 15-29% too wide. Widening an interval for a covariance that is
        positive and known is the same class of error as widening one for a
        correction already paid (invariant 21) — it costs decisiveness and buys
        nothing.

        The reference arm is its own comparator: it has no `ArmFit`, its blip is
        identically zero, and the contrast reduces to the other arm's own SE.
        """
        mine, theirs = self.fits.get(arm), self.fits.get(comparator)
        if mine is None or theirs is None:
            present = mine or theirs
            return present.blip_standard_error(features) if present else 0.0

        variance = (
            mine.blip_standard_error(features) ** 2
            + theirs.blip_standard_error(features) ** 2
        )
        cross = self._cross(arm, comparator, mine, theirs)
        if cross:
            n_free = len(TREATMENT_FREE_BASIS)
            loading = [0.0] * len(cross)
            for offset, value in enumerate(blip_basis(features)):
                loading[n_free + offset] = value
            covariance = sum(
                loading[i] * cross[i][j] * loading[j]
                for i in range(len(loading))
                if loading[i]
                for j in range(len(loading))
                if loading[j]
            )
            variance -= 2.0 * covariance
        return math.sqrt(max(variance, 0.0))

    def _cross(
        self, arm: str, comparator: str, mine: ArmFit, theirs: ArmFit
    ) -> list[list[float]]:
        """`mine.cross_covariance(theirs)`, computed once per ordered pair.

        Keyed on the ordered pair rather than the unordered one: swapping the
        arms transposes the matrix, and while the quadratic form beside it is
        symmetric and would not notice, storing the transpose under the same key
        would make the cache return something the uncached call did not.
        At most 30 entries for six arms.

        `cross_covariance` legitimately returns `[]` when either fit has no
        scores, so membership is tested rather than truthiness — `or` here would
        recompute the empty case on every request, which is the one where the
        answer is cheapest to get wrong and hardest to notice.
        """
        key = (arm, comparator)
        if key not in self._cross_covariance:
            self._cross_covariance[key] = mine.cross_covariance(theirs)
        return self._cross_covariance[key]

    # --------------------------------------------------- joint-bootstrap support
    #
    # dWOLS stores its parameters per arm, one small fit each. A joint resampling
    # scheme needs them as one flat vector so a replicate can be recorded and a
    # contrast recomputed from it later, the same way the Q-learning models are
    # handled.

    @property
    def blip_arms(self) -> tuple[str, ...]:
        return tuple(arm for arm in self.arms if arm != self.reference)

    def flat_parameters(self) -> list[float]:
        """Every non-reference arm's psi, concatenated in `blip_arms` order."""
        return [value for arm in self.blip_arms for value in self.fits[arm].psi]

    def contrast_loading(
        self, arm: str, comparator: str, features: dict[str, float]
    ) -> list[float]:
        """Loading vector such that `dot(loading, flat_parameters()) == contrast`."""
        width = len(BLIP_BASIS)
        loading = [0.0] * (len(self.blip_arms) * width)
        basis = blip_basis(features)
        for sign, candidate in ((1.0, arm), (-1.0, comparator)):
            if candidate not in self.blip_arms:
                continue
            start = self.blip_arms.index(candidate) * width
            for offset, value in enumerate(basis):
                loading[start + offset] += sign * value
        return loading

    def refit(self, cohort: list[CohortTrajectory]) -> list[float]:
        """Re-run the whole fit on a resample and return the flat parameters."""
        replica = DWOLSModel(cohort, arms=self.arms, reference=self.reference)
        if len(replica.flat_parameters()) != len(self.flat_parameters()):
            raise ValueError("resample produced a different design")
        return replica.flat_parameters()

    def raw_q(self, features: dict[str, float], arm: str) -> float:
        return self.treatment_free_value(features) + self.blip(arm, features)

    def predict_outcome(self, features: dict[str, float], arm: str, stage_index: int = 0) -> float:
        return clamp(self.raw_q(features, arm), 0.0, 1.0)

    def q_values(self, features: dict[str, float], menu: tuple[str, ...] | None = None) -> dict[str, float]:
        return {
            arm: round(clamp(self.raw_q(features, arm), Q_FLOOR, Q_CEILING), 3)
            for arm in (menu or self.arms)
        }

    def recommend(self, features: dict[str, float], menu: tuple[str, ...] | None = None) -> str:
        arms = menu or self.arms
        return max(arms, key=lambda arm: self.raw_q(features, arm))

    def greedy_policy(self):
        def policy(features: dict[str, float], stage_index: int) -> str:
            return self.recommend(features)

        return policy

    def coefficient_summary(self, arm: str) -> dict[str, float]:
        """Every arm's blip, so the explanation layer can decompose any of them."""
        summary = {
            f"beta:{name}": round(value, 4)
            for name, value in zip(TREATMENT_FREE_BASIS, self._reference_fit)
        }
        for candidate in self.fits:
            summary.update(
                {
                    f"psi:{candidate}:{name}": round(value, 4)
                    for name, value in self.blip_parameters(candidate).items()
                }
            )
        return summary


def fitted_model() -> DWOLSModel:
    """The fitted dWOLS model — owned by `training`, not cached here.

    This used to keep its own `_MODEL` global, fit from `training_cohort()`. The
    result was two dWOLS objects: `training.fitted().dwols`, which is scored,
    joint-bootstrapped and measured by every study in `feedback/`, and this one,
    which is what actually served patients. They agreed only because both were
    fit on the same default cohort.

    `training.reset()` cleared the first and not the second, so after changing a
    training constant the serving estimator silently kept the *old* fit. It cost
    a duplicated fit on every cold start, and it hid a sample-size effect: with
    the Q-learning half refitting and the dWOLS half frozen, the averaged
    contrast's standard error appeared to shrink at n^-0.15 instead of n^-0.39.

    Invariant 7 says every estimator is fit through `training.py`. This is that,
    enforced by having nowhere else to keep a model.
    """
    from treatmentrx.estimation.training import fitted

    return fitted().dwols


class DWOLSSharedEstimator:
    """dWOLS-Shared: doubly-robust weighted blip-function estimator."""

    method_name = DWOLS_METHOD

    def fit_predict(
        self,
        stages: list[StageRecord],
        treatment_menu: tuple[str, ...] = DEFAULT_TREATMENT_MENU,
    ) -> RegimeEstimate:
        from treatmentrx.estimation.training import policy_value_for

        model = fitted_model()
        features = model_features(stages)
        q_values = model.q_values(features, treatment_menu)
        # See `_QLearningEstimator.fit_predict`: `q_values` is a display
        # quantity and ranking on it loses the order at the clamp.
        recommended = model.recommend(features, treatment_menu)
        best = q_values[recommended]
        # The band widens with the standard error the blip fit is entitled to:
        # arms with few one-vs-reference rows are reported less confidently.
        half_width = Z_QUANTILE[DEFAULT_ALPHA] * model.blip_standard_error(recommended, features)
        return RegimeEstimate(
            estimator=self.method_name,
            regime_type=RegimeType.SPTR,
            recommended_arm=recommended,
            q_values=q_values,
            policy_value=policy_value_for(self.method_name),
            confidence_band=(round(max(best - half_width, 0.0), 3), round(min(best + half_width, 1.0), 3)),
            coefficients=model.coefficient_summary(recommended),
            top_tailoring_variables=top_tailoring_variables(
                model.blip_parameters(recommended), features
            ),
        )

    def contrast(
        self,
        stages: list[StageRecord],
        arm: str,
        comparator: str,
        alpha: float = DEFAULT_ALPHA,
    ) -> ContrastTest:
        """Difference of two one-vs-reference blips.

        The arms are fit on separate but *overlapping* subsets — both against the
        same reference — so their covariance is real and is now computed rather
        than dropped. Adding the variances as if independent left this 24% wider
        than the estimator's actual spread across refits; with the cross term it
        sits at 0.95 of it. Conservatism bought by ignoring a known positive
        covariance is not a safety margin, it is lost precision, and here it fed
        straight into the abstention rate.
        """
        model = fitted_model()
        features = model_features(stages)
        difference = model.blip(arm, features) - model.blip(comparator, features)
        standard_error = model.contrast_standard_error(arm, comparator, features)
        margin = Z_QUANTILE.get(alpha, Z_QUANTILE[DEFAULT_ALPHA]) * standard_error
        return ContrastTest(
            arm=arm,
            comparator=comparator,
            difference=difference,
            standard_error=standard_error,
            lower=difference - margin,
            upper=difference + margin,
            alpha=alpha,
            caveat="",
        )


__all__ = [
    "ArmFit",
    "DEFAULT_TREATMENT_MENU",
    "DWOLSModel",
    "DWOLSSharedEstimator",
    "fitted_model",
]
