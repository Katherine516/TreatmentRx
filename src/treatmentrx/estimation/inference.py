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

    @property
    def distinguishable(self) -> bool:
        """True when the interval for the difference excludes zero."""
        return self.lower > 0.0

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


def bootstrap_contrast(
    refit,
    cohort: list,
    loading_fn,
    replicates: int = 200,
    seed: int = 17,
) -> float:
    """Nonparametric cluster bootstrap standard error for a contrast.

    Resamples whole trajectories and re-runs the *entire* fitting procedure, so
    unlike the sandwich it does account for the pseudo-outcome step. It costs one
    full refit per replicate, which is why it is a validation tool rather than
    something the inference path calls.
    """
    import random

    rng = random.Random(seed)
    values = []
    for _ in range(replicates):
        sample = [cohort[rng.randrange(len(cohort))] for _ in range(len(cohort))]
        values.append(loading_fn(refit(sample)))
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
    return math.sqrt(variance)


__all__ = [
    "ContrastTest",
    "DEFAULT_ALPHA",
    "bootstrap_contrast",
    "contrast_test",
    "sandwich_covariance",
]
