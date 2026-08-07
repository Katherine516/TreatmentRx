"""Standard errors and contrast tests for the blip parameters.

The confidence bands used to be a support-size heuristic — a number that got
narrower with more data but was never a standard error of anything. This module
replaces that with a cluster-robust (sandwich) covariance matrix for the fitted
parameters, and with it the quantity a clinician actually needs: **is the
top-scored arm distinguishable from the runner-up for this patient?**

    Var(beta) = A B A,   A = (X'WX + ridge)^-1,   B = sum_c s_c s_c'

`s_c` is the score contribution of one trajectory, so the clustering is by
patient rather than by visit. Two stages from the same patient are correlated —
the second stage's covariates are a function of the first stage's outcome — and
treating them as independent would understate every standard error by roughly
sqrt(2).

**What this does not cover.** At non-terminal stages the regression target is a
pseudo-outcome built from the fitted stage-2 model, and this treats those
pseudo-outcomes as fixed data. That understates uncertainty at earlier stages,
because it ignores the sampling variability propagated from the downstream fit.
The terminal-stage standard errors are the honest ones; earlier stages are
optimistic and labelled as such in `ContrastTest.caveat`. A full accounting needs
the m-out-of-n bootstrap that non-regular DTR inference calls for, which is
`bootstrap_contrast` below and is deliberately not on the inference path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from treatmentrx.estimation import linalg

# Two-sided normal quantiles. Kept explicit rather than pulling in scipy.
Z_QUANTILE = {0.10: 1.6449, 0.05: 1.9600, 0.01: 2.5758}
DEFAULT_ALPHA = 0.05

# How much wider an honest interval is than the sandwich's, measured rather than
# assumed: `cli coverage` puts the sandwich at 0.88 of the estimator's actual
# spread, so a correct interval is about 1/0.88 = 1.14x wider, and the bootstrap
# runs a little more conservative still at ~1.20. A separation verdict that flips
# anywhere inside that range is not a verdict, so the band is set at the
# conservative end.
SANDWICH_INFLATION = 1.25


@dataclass(frozen=True)
class ContrastTest:
    """Whether two arms are separable for this patient, and by how much."""

    arm: str
    comparator: str
    difference: float
    standard_error: float
    lower: float
    upper: float
    alpha: float
    caveat: str = ""
    # True when the interval is already at least as wide as an honest one. Set
    # explicitly rather than inferred from the caveat text, because whether an
    # interval has already paid its correction is a property of how it was
    # built, not of how it was described.
    conservative: bool = False

    @property
    def distinguishable(self) -> bool:
        """True when the interval for the difference excludes zero."""
        return self.lower > 0.0

    @property
    def exact(self) -> bool:
        """Is this interval already honest, needing no further widening?

        Inflating an interval that has already accounted for the shortfall would
        charge for the same correction twice and push borderline decisions into
        equipoise for no statistical reason.
        """
        return self.conservative or not self.caveat or self.caveat.startswith("m-out-of-n")

    def survives_inflation(self, factor: float = SANDWICH_INFLATION) -> bool:
        """Would the verdict hold if the interval were `factor` times wider?"""
        half_width = (self.upper - self.lower) / 2.0
        return abs(self.difference) > half_width * factor

    @property
    def robustly_distinguishable(self) -> bool:
        """Separation that does not depend on which interval method was used.

        The sandwich is measurably too narrow, and it is what drives the
        equipoise decision — a clinical output. Rather than pay for a bootstrap
        on every patient, a verdict is only accepted when it survives the
        interval being as wide as the honest one would be. Near the boundary,
        where the two methods disagree, the system declines to claim separation.
        """
        if not self.distinguishable:
            return False
        return self.exact or self.survives_inflation()

    @property
    def z(self) -> float:
        if self.standard_error <= 0.0:
            return 0.0
        return self.difference / self.standard_error

    def as_dict(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "comparator": self.comparator,
            "difference": round(self.difference, 4),
            "standard_error": round(self.standard_error, 4),
            "interval": [round(self.lower, 4), round(self.upper, 4)],
            "distinguishable": self.distinguishable,
            "robustly_distinguishable": self.robustly_distinguishable,
            "conservative": self.conservative,
            "alpha": self.alpha,
            "caveat": self.caveat,
        }


def sandwich_covariance(
    rows: list[list[tuple[int, float]]],
    residuals: list[float],
    weights: list[float],
    clusters: list[int],
    normal_matrix: list[list[float]],
    n_features: int,
) -> list[list[float]]:
    """Cluster-robust covariance of a weighted least-squares fit.

    `normal_matrix` is the X'WX (plus ridge) already accumulated by the fit;
    `clusters` assigns each row to a patient.
    """
    bread = linalg.inverse(normal_matrix)

    scores: dict[int, list[float]] = {}
    for row, residual, weight, cluster in zip(rows, residuals, weights, clusters):
        score = scores.setdefault(cluster, [0.0] * n_features)
        for index, value in row:
            score[index] += weight * value * residual

    meat = [[0.0] * n_features for _ in range(n_features)]
    for score in scores.values():
        active = [(i, v) for i, v in enumerate(score) if v != 0.0]
        for i, vi in active:
            row_i = meat[i]
            for j, vj in active:
                row_i[j] += vi * vj

    # Small-cluster correction: without it the sandwich is anti-conservative.
    n_clusters = len(scores)
    scale = n_clusters / max(n_clusters - 1, 1)
    covariance = linalg.matmul(linalg.matmul(bread, meat), bread)
    return [[value * scale for value in row] for row in covariance]


def contrast_test(
    covariance: list[list[float]],
    loading: list[float],
    difference: float,
    arm: str,
    comparator: str,
    alpha: float = DEFAULT_ALPHA,
    caveat: str = "",
) -> ContrastTest:
    """Interval for a linear contrast `loading' beta` of the fitted parameters."""
    variance = max(linalg.quadratic_form(loading, covariance), 0.0)
    standard_error = math.sqrt(variance)
    z = Z_QUANTILE.get(alpha, Z_QUANTILE[DEFAULT_ALPHA])
    margin = z * standard_error
    return ContrastTest(
        arm=arm,
        comparator=comparator,
        difference=difference,
        standard_error=standard_error,
        lower=difference - margin,
        upper=difference + margin,
        alpha=alpha,
        caveat=caveat,
    )


# --------------------------------------------------------------------------
# m-out-of-n bootstrap for the non-regular, non-terminal stages
# --------------------------------------------------------------------------

# Tuning constant for the adaptive resample size. Larger values shrink the
# resample harder as non-regularity rises. The literature chooses this by a
# double bootstrap; this is a documented default, not a tuned one.
DEFAULT_BOOTSTRAP_ALPHA = 0.5
DEFAULT_REPLICATES = 200
MIN_RESAMPLE = 25


def adaptive_resample_size(n: int, non_regularity: float, alpha: float = DEFAULT_BOOTSTRAP_ALPHA) -> int:
    """Resample size m for the m-out-of-n bootstrap.

        m = n ** ((1 + alpha * (1 - p)) / (1 + alpha))

    With p = 0 (no patient sits near a decision boundary) this returns n and the
    procedure degenerates to the ordinary bootstrap, which is correct: the
    estimator is regular there. As p rises toward 1 the resample shrinks, which
    is what restores consistency — the ordinary bootstrap is *inconsistent* at a
    non-smooth point, and taking m < n with m/n → 0 is the standard repair.
    """
    non_regularity = max(0.0, min(1.0, non_regularity))
    exponent = (1.0 + alpha * (1.0 - non_regularity)) / (1.0 + alpha)
    return max(MIN_RESAMPLE, min(n, int(round(n ** exponent))))


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    position = q * (len(sorted_values) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


@dataclass(frozen=True)
class BootstrapDistribution:
    """Parameter draws from an m-out-of-n resampling of whole trajectories.

    Storing the draws rather than a single standard error is what makes this
    affordable: the refits are paid once, and the interval for *any* linear
    contrast afterwards is a dot product per draw.
    """

    draws: list[list[float]]
    point: list[float]
    n: int
    m: int
    non_regularity: float
    alpha: float
    replicates: int

    @property
    def scale(self) -> float:
        """sqrt(m/n) — converts the resample's spread to the full sample's."""
        return math.sqrt(self.m / self.n) if self.n else 1.0

    def standard_error(self, loading: list[float]) -> float:
        values = [_dot_sparse(loading, draw) for draw in self.draws]
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        return math.sqrt(variance) * self.scale

    def interval(self, loading: list[float], alpha: float = DEFAULT_ALPHA) -> tuple[float, float]:
        """Percentile interval from the centred, scaled bootstrap statistics.

        Percentile rather than normal-approximation on purpose: non-regularity
        produces an asymmetric sampling distribution, and capturing that
        asymmetry is the whole reason for using a bootstrap here.
        """
        estimate = _dot_sparse(loading, self.point)
        root_m = math.sqrt(self.m)
        statistics = sorted(root_m * (_dot_sparse(loading, draw) - estimate) for draw in self.draws)
        root_n = math.sqrt(self.n) if self.n else 1.0
        upper_statistic = _quantile(statistics, 1.0 - alpha / 2.0)
        lower_statistic = _quantile(statistics, alpha / 2.0)
        return (estimate - upper_statistic / root_n, estimate - lower_statistic / root_n)


def _dot_sparse(loading: list[float], values: list[float]) -> float:
    return sum(l * v for l, v in zip(loading, values) if l != 0.0)


def m_out_of_n_bootstrap(
    refit,
    cohort: list,
    point: list[float],
    non_regularity: float,
    replicates: int = DEFAULT_REPLICATES,
    alpha: float = DEFAULT_BOOTSTRAP_ALPHA,
    seed: int = 17,
) -> BootstrapDistribution:
    """Resample m whole trajectories with replacement and re-run the full fit.

    `refit(sample) -> list[float]` must repeat the *entire* estimation procedure,
    pseudo-outcome construction included. That is what the sandwich cannot do and
    why this exists: at a non-terminal stage the regression target is built from
    the fitted downstream model, and treating it as fixed data understates the
    uncertainty.

    Clusters are trajectories, never rows — resampling visits independently would
    break the within-patient correlation the whole design depends on.
    """
    import random

    n = len(cohort)
    m = adaptive_resample_size(n, non_regularity, alpha)
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        sample = [cohort[rng.randrange(n)] for _ in range(m)]
        try:
            draws.append(refit(sample))
        except (ValueError, ZeroDivisionError):
            # A resample can omit an arm entirely; skip it rather than let one
            # degenerate draw define the interval.
            continue
    return BootstrapDistribution(
        draws=draws,
        point=list(point),
        n=n,
        m=m,
        non_regularity=non_regularity,
        alpha=alpha,
        replicates=len(draws),
    )


@dataclass(frozen=True)
class JointBootstrap:
    """Every estimator refit on the *same* resample, replicate by replicate.

    The decision is made on a weighted average of the three estimators, so its
    standard error depends on how they co-vary. Fitting them separately leaves
    that covariance unmeasured, and the only defensible fallback is the
    perfect-correlation upper bound — which measures 1.29x the averaged
    estimator's actual spread, so every interval is about a quarter wider than it
    needs to be and borderline decisions are pushed into equipoise for no
    statistical reason.

    Refitting all three on one resample recovers the joint distribution directly.
    The averaged contrast is then evaluated per replicate and its spread read off,
    with no assumption about correlation at all.
    """

    draws: list[dict[str, list[float]]]
    point: dict[str, list[float]]
    n: int
    m: int
    non_regularity: float

    @property
    def replicates(self) -> int:
        return len(self.draws)

    @property
    def scale(self) -> float:
        return math.sqrt(self.m / self.n) if self.n else 1.0

    def _averaged(self, draw, loadings, weights) -> float:
        total = sum(weights.get(name, 0.0) for name in loadings)
        if total <= 0.0:
            return 0.0
        return (
            sum(
                weights.get(name, 0.0) * _dot_sparse(loading, draw[name])
                for name, loading in loadings.items()
                if name in draw
            )
            / total
        )

    def contrast(
        self,
        loadings: dict[str, list[float]],
        weights: dict[str, float],
        arm: str,
        comparator: str,
        alpha: float = DEFAULT_ALPHA,
        scale_by: float = 1.0,
    ) -> ContrastTest:
        """Percentile interval for the model-averaged contrast."""
        estimate = self._averaged(self.point, loadings, weights)
        values = [self._averaged(draw, loadings, weights) for draw in self.draws]
        if len(values) < 2:
            raise ValueError("joint bootstrap needs at least two usable replicates")

        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        standard_error = math.sqrt(variance) * self.scale

        root_m, root_n = math.sqrt(self.m), math.sqrt(self.n) if self.n else 1.0
        statistics = sorted(root_m * (value - estimate) for value in values)
        lower = estimate - _quantile(statistics, 1.0 - alpha / 2.0) / root_n
        upper = estimate - _quantile(statistics, alpha / 2.0) / root_n

        return ContrastTest(
            arm=arm,
            comparator=comparator,
            difference=estimate / scale_by,
            standard_error=standard_error / scale_by,
            lower=lower / scale_by,
            upper=upper / scale_by,
            alpha=alpha,
            conservative=True,
            caveat=(
                f"joint m-out-of-n bootstrap over {len(loadings)} estimators "
                f"(m={self.m} of n={self.n}, {self.replicates} replicates); the "
                "components' covariance is measured rather than bounded."
            ),
        )


def joint_bootstrap(
    refit_all,
    cohort: list,
    point: dict[str, list[float]],
    non_regularity: float,
    replicates: int = DEFAULT_REPLICATES,
    alpha: float = DEFAULT_BOOTSTRAP_ALPHA,
    seed: int = 17,
) -> JointBootstrap:
    """Resample once per replicate and refit *every* estimator on that resample.

    Sharing the resample is the whole point: fitting them on independent
    resamples would destroy exactly the correlation this exists to measure.
    """
    import random

    n = len(cohort)
    m = adaptive_resample_size(n, non_regularity, alpha)
    rng = random.Random(seed)
    draws: list[dict[str, list[float]]] = []
    for _ in range(replicates):
        sample = [cohort[rng.randrange(n)] for _ in range(m)]
        try:
            draws.append(refit_all(sample))
        except (ValueError, ZeroDivisionError):
            continue
    return JointBootstrap(
        draws=draws, point=dict(point), n=n, m=m, non_regularity=non_regularity
    )


__all__ = [
    "ContrastTest",
    "JointBootstrap",
    "joint_bootstrap",
    "DEFAULT_ALPHA",
    "bootstrap_contrast",
    "contrast_test",
    "sandwich_covariance",
]
