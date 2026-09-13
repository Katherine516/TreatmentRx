"""Is the estimator ranking real, or is it one split's worth of noise?

`training.py` fits on a single train/holdout split. One split cannot tell you
whether "dWOLS scored 0.7335 and Q-Shared scored 0.7333" means anything, and
reporting an ordering built on that difference would be false precision.

This module answers the question directly, two ways:

* **k-fold cross-validation** — refit on every fold, so each trajectory is scored
  by a model that never saw it. Gives a spread, not a point.
* **Seed sweep** — regenerate the whole cohort under different seeds. Catches the
  case where a result is an artefact of one particular synthetic draw rather than
  of the estimator.

The verdict is the point of the module: `resolved` is True only when the best
estimator's advantage survives the spread across folds and seeds. When it is
False, the honest report is that the estimators are indistinguishable at this
sample size — which is currently the case, and is why Bayesian model averaging
weights them near-uniformly instead of picking a winner.

Both routines refit from scratch, so this is a command you run
(`treatmentrx stability`), not something on the inference path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from treatmentrx.estimation.dwols import DWOLS_METHOD, DWOLSModel
from treatmentrx.estimation.q_learning import (
    DEFAULT_POOLING_RIDGE,
    Q_POOLED_METHOD,
    Q_SHARED_METHOD,
    STAGE_SPECIFIC_METHOD,
    QLearningModel,
)
from treatmentrx.feedback.offline_evaluation import evaluate_policy
from treatmentrx.simulation.ra_cohort import CohortTrajectory, generate_ra_cohort

DEFAULT_FOLDS = 5
DEFAULT_SEEDS = (7, 23, 101, 202)
DEFAULT_COHORT_SIZE = 400


@dataclass(frozen=True)
class Spread:
    """A metric summarised across resamples."""

    mean: float
    standard_deviation: float
    minimum: float
    maximum: float
    n: int

    @property
    def standard_error(self) -> float:
        return self.standard_deviation / math.sqrt(self.n) if self.n else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "mean": round(self.mean, 4),
            "sd": round(self.standard_deviation, 4),
            "se": round(self.standard_error, 4),
            "min": round(self.minimum, 4),
            "max": round(self.maximum, 4),
            "n": self.n,
        }


@dataclass(frozen=True)
class StabilityResult:
    estimator: str
    policy_value: Spread
    optimal_arm_rate: Spread
    first_place_rate: float
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "estimator": self.estimator,
            "ipw_policy_value": self.policy_value.as_dict(),
            "optimal_arm_rate": self.optimal_arm_rate.as_dict(),
            "first_place_rate": round(self.first_place_rate, 3),
        }


def _spread(values: list[float]) -> Spread:
    if not values:
        return Spread(0.0, 0.0, 0.0, 0.0, 0)
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)
    return Spread(mean, math.sqrt(variance), min(values), max(values), len(values))


def _fit_and_score(train: list[CohortTrajectory], holdout: list[CohortTrajectory]) -> dict[str, dict]:
    q_shared = QLearningModel(train, share_blip=True)
    stage_specific = QLearningModel(train, share_blip=False)
    # The serving Q-learning model has to be in the sweep, or the verdict about
    # whether the ranking is resolved is about estimators that do not serve.
    pooled = QLearningModel(train, share_blip=True, pooling_ridge=DEFAULT_POOLING_RIDGE)
    dwols = DWOLSModel(train)
    scored = {}
    for name, model in (
        (Q_SHARED_METHOD, q_shared),
        (STAGE_SPECIFIC_METHOD, stage_specific),
        (Q_POOLED_METHOD, pooled),
        (DWOLS_METHOD, dwols),
    ):
        # No intervals: this reads the point value only, and the sweep pays this
        # once per estimator per fold per seed.
        score = evaluate_policy(
            name, model.greedy_policy(), model.predict_outcome, holdout, with_intervals=False
        )
        scored[name] = {
            "policy_value": score.ipw_policy_value,
            "optimal_arm_rate": score.optimal_arm_rate,
        }
    return scored


def kfold_scores(
    cohort: list[CohortTrajectory] | None = None,
    folds: int = DEFAULT_FOLDS,
) -> list[dict[str, dict]]:
    """Refit on each fold's complement and score on the fold itself."""
    cohort = cohort if cohort is not None else generate_ra_cohort(DEFAULT_COHORT_SIZE)
    if folds < 2:
        raise ValueError("cross-validation needs at least two folds")
    size = len(cohort) // folds
    results = []
    for index in range(folds):
        start = index * size
        end = start + size if index < folds - 1 else len(cohort)
        holdout = cohort[start:end]
        train = cohort[:start] + cohort[end:]
        results.append(_fit_and_score(train, holdout))
    return results


def seed_sweep(
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    size: int = DEFAULT_COHORT_SIZE,
) -> list[dict[str, dict]]:
    """Regenerate the cohort per seed, refit, and score on that seed's holdout."""
    results = []
    for seed in seeds:
        cohort = generate_ra_cohort(size, seed)
        cut = int(len(cohort) * 0.7)
        results.append(_fit_and_score(cohort[:cut], cohort[cut:]))
    return results


def summarise(runs: list[dict[str, dict]]) -> list[StabilityResult]:
    estimators = sorted({name for run in runs for name in run})
    winners = [max(run, key=lambda name: run[name]["policy_value"]) for run in runs]
    return [
        StabilityResult(
            estimator=name,
            policy_value=_spread([run[name]["policy_value"] for run in runs if name in run]),
            optimal_arm_rate=_spread([run[name]["optimal_arm_rate"] for run in runs if name in run]),
            first_place_rate=winners.count(name) / len(runs) if runs else 0.0,
        )
        for name in estimators
    ]


def is_ranking_resolved(results: list[StabilityResult]) -> tuple[bool, str]:
    """Does the leader's advantage exceed the noise across resamples?

    The test is deliberately blunt: the best estimator's mean policy value must
    beat the runner-up's by more than the combined standard error, *and* it must
    actually win most of the individual resamples. Anything less is a coin flip
    dressed up as a ranking.
    """
    if len(results) < 2:
        return False, "fewer than two estimators to compare"
    ordered = sorted(results, key=lambda r: r.policy_value.mean, reverse=True)
    best, runner_up = ordered[0], ordered[1]
    margin = best.policy_value.mean - runner_up.policy_value.mean
    noise = math.sqrt(best.policy_value.standard_error ** 2 + runner_up.policy_value.standard_error ** 2)
    consistent = best.first_place_rate > 0.5
    if margin > noise and consistent:
        return True, (
            f"{best.estimator} leads {runner_up.estimator} by {margin:.4f}, above the "
            f"{noise:.4f} combined standard error, and wins "
            f"{best.first_place_rate:.0%} of resamples."
        )
    return False, (
        f"{best.estimator} leads {runner_up.estimator} by only {margin:.4f} against a "
        f"{noise:.4f} combined standard error (winning {best.first_place_rate:.0%} of "
        "resamples). The estimators are not distinguishable at this sample size; "
        "treat the scorecard ordering as arbitrary and keep averaging them."
    )


def stability_report(
    folds: int = DEFAULT_FOLDS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    size: int = DEFAULT_COHORT_SIZE,
) -> dict[str, object]:
    cv = summarise(kfold_scores(generate_ra_cohort(size), folds))
    sweep = summarise(seed_sweep(seeds, size))
    cv_resolved, cv_verdict = is_ranking_resolved(cv)
    sweep_resolved, sweep_verdict = is_ranking_resolved(sweep)
    return {
        "cross_validation": {
            "folds": folds,
            "cohort_size": size,
            "estimators": [result.as_dict() for result in cv],
            "ranking_resolved": cv_resolved,
            "verdict": cv_verdict,
        },
        "seed_sweep": {
            "seeds": list(seeds),
            "cohort_size": size,
            "estimators": [result.as_dict() for result in sweep],
            "ranking_resolved": sweep_resolved,
            "verdict": sweep_verdict,
        },
    }


__all__ = [
    "Spread",
    "StabilityResult",
    "is_ranking_resolved",
    "kfold_scores",
    "seed_sweep",
    "stability_report",
    "summarise",
]
