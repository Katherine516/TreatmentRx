"""Is a covariate the blip basis leaves out actually an effect modifier?

`cli misspecification --omitted-modifier` establishes that an omitted effect
modifier is the one failure nothing in this system survives: coverage falls to
29% and the reported standard error *narrows* as it does, because a standard
error computed under the wrong basis has no way to see a bias in the estimand.
That study can only exist in simulation, though — it compares against
`true_blip`, and on real data there is no such thing.

This is the part that transfers. Augment the blip block with a candidate
covariate, fit the one-vs-reference regression with it in, and test whether its
coefficient is zero using the same cluster-robust covariance the rest of the
system uses. It answers a question an analyst can actually ask of their own data:
*does adding CRP as an effect modifier change the treatment effect?*

**What it is and is not.** It is a falsification test. Rejecting says the named
covariate belongs in the basis. Failing to reject says *this* covariate, at *this*
sample size, did not show — never that the basis is right, because the modifier
you did not think to test is exactly the one that will hurt you. Read a clean
result as "the candidates I could name are not the problem", which is weaker than
it sounds and still worth having.

The test is deliberately per-arm. A modifier can matter for one arm and not
another — CRP plausibly modifies an IL-6 blockade more than a csDMARD — and a
single pooled statistic would average that away.

**Candidates must be pre-treatment.** A covariate caused by an earlier decision
is a mediator, and interacting it with the current arm measures the mediation
rather than effect modification. `EXCLUDED_CANDIDATES` records the one this
cohort contains, along with the numbers that identified it — it was found by
pointing the test at ALT and watching it reject on a cohort with no ALT effect
modification in it at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from treatmentrx.estimation import linalg
from treatmentrx.estimation.basis import BLIP_BASIS, blip_basis, treatment_free_basis
from treatmentrx.estimation.inference import sandwich_covariance
from treatmentrx.simulation.ra_cohort import (
    REFERENCE_ARM,
    TREATMENT_ARMS,
    CohortTrajectory,
    clamp,
    das28_std,
)

_PROPENSITY_FLOOR = 0.02
_PROPENSITY_CEILING = 0.98
_RIDGE = 1e-4

# Candidate effect modifiers a real analyst would think to try. Each is a
# function of the same six covariates the estimators already condition on — the
# question is not whether the record carries them but whether the *treatment
# effect* varies over them, which is a different claim and the one the blip basis
# makes.
CANDIDATE_MODIFIERS = {
    "crp_std": lambda f: (f["crp"] - 30.0) / 25.0,
    "egfr_std": lambda f: (f["egfr"] - 90.0) / 30.0,
    "das28_squared": lambda f: das28_std(f["das28"]) ** 2,
}

# Covariates that must **not** be tested as effect modifiers, and why. A
# candidate has to be pre-treatment with respect to the decision being modelled;
# one that is caused by an earlier decision is a mediator, and interacting it
# with the current arm asks a question the design cannot answer.
#
# `alt_excess` is the worked example and it was found by running the test rather
# than by reasoning about it. It rejects at z = 4.35 on a cohort with *no* ALT
# effect modification at all, and the statistic does not respond to the modifier
# that is actually present (4.35, 4.08, 4.52, 3.03 as the CRP modifier goes 0 to
# 0.20 — noise, not signal). The reason is in the generator: hepatotoxic arms
# raise ALT, so mean ALT after one is 62.2 against 27.9 after any other arm.
# ALT_{j+1} is a direct function of A_j. CRP, by contrast, sits at 25.8 against
# 25.3 — it moves only through response, which is what a legitimate
# time-varying covariate looks like.
#
# Roughly half the spurious statistic is differential censoring (ALT drives
# dropout; turning dropout off takes 4.15 to 2.12) and the rest is the mediation
# itself. Weighting fixes the first and nothing fixes the second.
EXCLUDED_CANDIDATES = {
    "alt_excess": (
        "post-treatment: hepatotoxic arms raise ALT, so it mediates the previous "
        "decision's delayed effect and also drives dropout"
    ),
}

# Two-sided alpha before multiplicity and before the sandwich correction.
DEFAULT_ALPHA = 0.05

# The threshold is built from `DEFAULT_ALPHA` in two steps, both measured rather
# than assumed, because the naive version is badly wrong: correcting over arms
# alone and taking the sandwich at face value gives a **33% false-positive rate
# against a nominal 5%**, measured over 30 null cohorts. A specification test
# that cries wolf a third of the time sends an analyst chasing a modifier that
# is not there, which is worse than not having one.
#
# Step one: Bonferroni over *every* test performed, covariates times arms, not
# arms alone. Three covariates and five arms is fifteen questions.
#
# Step two: widen by `SANDWICH_INFLATION`. The sandwich is measured at 0.86-0.90
# of the estimator's actual spread everywhere else in this repo (`cli coverage`),
# and a z built on a standard error that is 12% too small is a z that is 12% too
# large. This is the same correction `ContrastTest.robustly_distinguishable`
# applies for the same reason.
#
# Measured false-positive rate after both, over the same 30 null cohorts: see
# `tests/test_robustness.py::SpecificationTestCalibrationTests`.


@dataclass(frozen=True)
class ModifierTest:
    """One arm's answer for one candidate covariate."""

    arm: str
    covariate: str
    coefficient: float
    standard_error: float
    n_rows: int

    @property
    def z(self) -> float:
        if self.standard_error <= 0.0:
            return 0.0
        return self.coefficient / self.standard_error

    def rejects(self, threshold: float) -> bool:
        return abs(self.z) > threshold

    def as_dict(self, threshold: float) -> dict[str, object]:
        return {
            "arm": self.arm,
            "covariate": self.covariate,
            "coefficient": round(self.coefficient, 5),
            "standard_error": round(self.standard_error, 5),
            "z": round(self.z, 3),
            "rejects": self.rejects(threshold),
            "n_rows": self.n_rows,
        }


def _augmented_fit(
    rows: list[tuple[dict[str, float], float, float, int, float]],
    candidate,
) -> tuple[float, float, int]:
    """One-vs-reference blip fit with one extra interaction, and its standard error.

    The design is the dWOLS one with a single column added: the treatment-free
    basis, the `A * h(X)` blip block, and `A * candidate(X)`. Weighting and
    clustering match `dwols.ArmFit`, with one addition — the rows carry an
    inverse-probability-of-censoring weight.

    **That weight is not optional here, and finding out why was the point.**
    Without it the test rejects `alt_excess` on a cohort with no ALT effect
    modification at all (z = 4.15 against a 2.58 threshold). ALT drives dropout
    in the generating process and hepatotoxic arms raise it, so among the rows
    that survive, the ALT distribution differs by arm in a way that correlates
    with outcome. The test was reading differential censoring as effect
    modification. Turning dropout off collapses the statistic to 2.12, which is
    what identified it.

    A specification test on observational data has to carry every correction the
    estimator it is testing carries, or it finds the corrections' absence instead
    of the thing it was pointed at.
    """
    if len(rows) < len(BLIP_BASIS) + 8:
        return 0.0, 0.0, len(rows)

    design_propensity = [blip_basis(features) for features, _, _, _, _ in rows]
    treated = [assignment for _, assignment, _, _, _ in rows]
    gamma = linalg.weighted_least_squares(
        design_propensity, treated, [1.0] * len(rows), ridge=_RIDGE
    )
    propensities = [
        clamp(linalg.dot(row, gamma), _PROPENSITY_FLOOR, _PROPENSITY_CEILING)
        for row in design_propensity
    ]

    design: list[list[float]] = []
    targets: list[float] = []
    weights: list[float] = []
    clusters: list[int] = []
    for (features, assignment, outcome, cluster, censoring), propensity in zip(
        rows, propensities
    ):
        free = treatment_free_basis(features)
        blip = [assignment * value for value in blip_basis(features)]
        design.append(free + blip + [assignment * candidate(features)])
        targets.append(outcome)
        weights.append(abs(assignment - propensity) * censoring)
        clusters.append(cluster)

    beta = linalg.weighted_least_squares(design, targets, weights, ridge=_RIDGE)
    n_features = len(beta)
    sparse = [[(i, v) for i, v in enumerate(row) if v != 0.0] for row in design]
    residuals = [y - linalg.dot(row, beta) for row, y in zip(design, targets)]
    covariance = sandwich_covariance(
        sparse,
        residuals,
        weights,
        clusters,
        linalg.sparse_normal_matrix(sparse, weights, n_features, _RIDGE),
        n_features,
    )
    index = n_features - 1
    variance = max(covariance[index][index], 0.0)
    return beta[index], math.sqrt(variance), len(rows)


def test_modifier(
    cohort: list[CohortTrajectory],
    covariate: str,
    arms: tuple[str, ...] = TREATMENT_ARMS,
    reference: str = REFERENCE_ARM,
) -> list[ModifierTest]:
    """Per-arm tests for one candidate covariate."""
    candidate = CANDIDATE_MODIFIERS.get(covariate)
    if candidate is None:
        raise ValueError(
            f"unknown candidate modifier {covariate!r}; "
            f"known: {sorted(CANDIDATE_MODIFIERS)}"
        )

    from treatmentrx.estimation.censoring import CensoringModel

    censoring = CensoringModel(cohort)
    observations = [
        (
            dict(stage.features),
            stage.arm,
            stage.outcome,
            cluster,
            censoring.row_weight(trajectory, position, needs_next=False),
        )
        for cluster, trajectory in enumerate(cohort)
        for position, stage in enumerate(trajectory.stages)
    ]
    results = []
    for arm in arms:
        if arm == reference:
            continue
        selected = [
            (features, 1.0 if observed == arm else 0.0, outcome, cluster, weight)
            for features, observed, outcome, cluster, weight in observations
            if observed in (arm, reference)
        ]
        coefficient, error, n_rows = _augmented_fit(selected, candidate)
        results.append(ModifierTest(arm, covariate, coefficient, error, n_rows))
    return results


def specification_report(
    cohort: list[CohortTrajectory] | None = None,
    covariates: tuple[str, ...] = tuple(CANDIDATE_MODIFIERS),
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, object]:
    """Test every candidate modifier against the training cohort.

    Uses the training split by default, for the same reason everything else does:
    a specification test run on the holdout spends the data the policy value is
    measured on.
    """
    from treatmentrx.estimation import training

    cohort = cohort if cohort is not None else training.fitted().train
    n_arms = len([arm for arm in TREATMENT_ARMS if arm != REFERENCE_ARM])
    threshold = rejection_threshold(alpha, len(covariates), n_arms)

    findings: dict[str, object] = {}
    flagged: list[str] = []
    for covariate in covariates:
        tests = test_modifier(cohort, covariate)
        rejecting = [test for test in tests if test.rejects(threshold)]
        findings[covariate] = {
            "per_arm": [test.as_dict(threshold) for test in tests],
            "arms_rejecting": [test.arm for test in rejecting],
            "max_abs_z": round(max((abs(test.z) for test in tests), default=0.0), 3),
        }
        if rejecting:
            flagged.append(covariate)

    return {
        "cohort_trajectories": len(cohort),
        "alpha": alpha,
        "arms_tested": n_arms,
        "bonferroni_z_threshold": round(threshold, 3),
        "candidates": findings,
        "flagged": flagged,
        "verdict": _verdict(flagged, findings),
    }


def rejection_threshold(alpha: float, n_covariates: int, n_arms: int) -> float:
    """The |z| a coefficient must clear, corrected for multiplicity and the sandwich.

    Separated out and exported so the calibration test can assert against the
    same number the report uses, rather than a copy of it.
    """
    from treatmentrx.estimation.inference import SANDWICH_INFLATION

    tests = max(n_covariates * n_arms, 1)
    return _normal_quantile(alpha / tests) * SANDWICH_INFLATION


def _normal_quantile(p: float) -> float:
    """Two-sided normal quantile, by bisection on the error function.

    `Z_QUANTILE` only carries the three alphas the rest of the system uses, and a
    Bonferroni correction produces others. Bisection rather than a rational
    approximation because it is a handful of iterations at startup and exactly
    right, and this file is not on the hot path.
    """
    target = 1.0 - p / 2.0
    low, high = 0.0, 10.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if 0.5 * (1.0 + math.erf(middle / math.sqrt(2.0))) < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def _verdict(flagged: list[str], findings: dict[str, object]) -> str:
    if flagged:
        detail = "; ".join(
            f"{name} for {', '.join(findings[name]['arms_rejecting'])}" for name in flagged
        )
        return (
            f"The blip basis is missing an effect modifier: {detail}. The reported "
            "contrasts are the covariate-averaged ones, not this patient's, and the "
            "interval will not show it — add the covariate to BLIP_BASIS and refit."
        )
    return (
        "No candidate modifier rejected at the Bonferroni-corrected threshold. That "
        "is weaker than it sounds: it says the covariates named in "
        "CANDIDATE_MODIFIERS did not show at this sample size, not that the basis is "
        "correct. The modifier nobody thought to test is the one that hurts."
    )


__all__ = [
    "CANDIDATE_MODIFIERS",
    "DEFAULT_ALPHA",
    "ModifierTest",
    "specification_report",
    "test_modifier",
]
