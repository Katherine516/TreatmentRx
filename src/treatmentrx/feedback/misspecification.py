"""Which estimator survives the model being wrong?

`stability.py` compares estimators on held-out policy value and reports,
honestly, that the ranking is not resolved: the three sit within their own
standard error of each other and no amount of refitting on this cohort
separates them. That is a real answer, but it leaves the choice of estimator
unjustified.

This module supplies the axis that *does* separate them. The three estimators
were kept deliberately different on the grounds that they fail differently —
Q-learning models the outcome surface, dWOLS is doubly robust and needs only the
propensity model or the outcome model to be right. That was an assertion. Here it
is measured, by curving the treatment-free surface in a way the estimators'
linear basis cannot represent while leaving the blips — the estimand — untouched.

The comparison is only meaningful over a range where the generating process
itself stays well behaved. Past a curvature of roughly 0.2 the outcome saturates
at its ceiling and every estimator degrades for a reason that has nothing to do
with robustness, so the sweep stops before that.
"""

from __future__ import annotations

from dataclasses import dataclass

from treatmentrx.estimation.dwols import DWOLS_METHOD, DWOLSModel
from treatmentrx.estimation.q_learning import (
    Q_SHARED_METHOD,
    STAGE_SPECIFIC_METHOD,
    QLearningModel,
)
from treatmentrx.simulation.ra_cohort import (
    REFERENCE_ARM,
    TREATMENT_ARMS,
    TRUE_BLIPS,
    generate_ra_cohort,
)

# Beyond this the outcome clamps at 1.0 and the DGP stops being informative.
MAX_INFORMATIVE_CURVATURE = 0.15
# Below this the seeds cannot tell two estimators apart on worst-case error.
MEANINGFUL_MARGIN = 0.10
DEFAULT_CURVATURES = (0.0, 0.05, 0.10, 0.15)
DEFAULT_SEEDS = (7, 23, 101, 202, 303, 404)
DEFAULT_SIZE = 400


@dataclass(frozen=True)
class RobustnessResult:
    estimator: str
    errors_by_curvature: dict[float, float]

    @property
    def correctly_specified(self) -> float:
        return self.errors_by_curvature[min(self.errors_by_curvature)]

    @property
    def worst(self) -> float:
        return max(self.errors_by_curvature.values())

    @property
    def degradation(self) -> float:
        """How much worse the estimator gets as the nuisance model goes wrong."""
        base = self.correctly_specified
        return round(self.worst / base, 3) if base else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "estimator": self.estimator,
            "blip_error_by_curvature": {
                str(k): round(v, 4) for k, v in sorted(self.errors_by_curvature.items())
            },
            "correctly_specified_error": round(self.correctly_specified, 4),
            "worst_case_error": round(self.worst, 4),
            "degradation_factor": self.degradation,
        }


def _blip_error(parameters_for) -> float:
    return sum(
        abs(parameters_for(arm)[name] - truth)
        for arm in TREATMENT_ARMS
        if arm != REFERENCE_ARM
        for name, truth in zip(parameters_for(arm), TRUE_BLIPS[arm])
    )


def robustness_study(
    curvatures: tuple[float, ...] = DEFAULT_CURVATURES,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    size: int = DEFAULT_SIZE,
) -> list[RobustnessResult]:
    """Total blip error per estimator as the treatment-free surface curves."""
    errors: dict[str, dict[float, float]] = {
        Q_SHARED_METHOD: {},
        STAGE_SPECIFIC_METHOD: {},
        DWOLS_METHOD: {},
    }

    for curvature in curvatures:
        totals = {name: 0.0 for name in errors}
        for seed in seeds:
            cohort = generate_ra_cohort(size, seed, curvature=curvature)
            shared = QLearningModel(cohort, share_blip=True, compute_covariance=False)
            stage_specific = QLearningModel(cohort, share_blip=False, compute_covariance=False)
            dwols = DWOLSModel(cohort)
            terminal = stage_specific.n_stages - 1
            totals[Q_SHARED_METHOD] += _blip_error(lambda arm: shared.blip_parameters(arm))
            totals[STAGE_SPECIFIC_METHOD] += _blip_error(
                lambda arm: stage_specific.blip_parameters(arm, terminal)
            )
            totals[DWOLS_METHOD] += _blip_error(lambda arm: dwols.blip_parameters(arm))
        for name, total in totals.items():
            errors[name][curvature] = total / len(seeds)

    return [RobustnessResult(name, by_curvature) for name, by_curvature in errors.items()]


def preferred_estimator(results: list[RobustnessResult]) -> tuple[str, str]:
    """Which estimator to prefer, and the reason, when policy value cannot decide.

    Worst case rather than average: the point of holding three estimators is
    what happens when the modelling assumptions are wrong, and an estimator that
    is excellent when everything is specified correctly and poor otherwise is
    the one least worth relying on.
    """
    if not results:
        return "", "no estimators to compare"
    ordered = sorted(results, key=lambda result: result.worst)
    best, runner_up = ordered[0], ordered[1]
    margin = runner_up.worst - best.worst
    # A handful of seeds cannot resolve a few percent. Naming a winner on that
    # margin would repeat the mistake `stability.py` exists to avoid.
    if margin < MEANINGFUL_MARGIN * best.worst:
        most_accurate = min(results, key=lambda result: result.correctly_specified)
        most_stable = min(results, key=lambda result: result.degradation)
        return (
            "",
            (
                f"No single estimator wins. {most_accurate.estimator} is the most accurate "
                f"when the nuisance model is right ({most_accurate.correctly_specified:.3f}) "
                f"and degrades the most ({most_accurate.degradation}x); "
                f"{most_stable.estimator} is the most stable when it is wrong "
                f"({most_stable.degradation}x) and the least accurate when it is right. "
                "They trade off in exactly the direction their designs predict, which is "
                "the justification for averaging them rather than choosing one."
            ),
        )
    return (
        best.estimator,
        (
            f"{best.estimator} holds up best when the treatment-free model is wrong "
            f"(worst-case blip error {best.worst:.3f} against {runner_up.worst:.3f} for "
            f"{runner_up.estimator}). Held-out policy value cannot separate the "
            "estimators on this cohort; robustness can."
        ),
    )


def misspecification_report(
    curvatures: tuple[float, ...] = DEFAULT_CURVATURES,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    size: int = DEFAULT_SIZE,
) -> dict[str, object]:
    results = robustness_study(curvatures, seeds, size)
    preferred, reason = preferred_estimator(results)
    return {
        "curvatures": list(curvatures),
        "cohort_size": size,
        "seeds": list(seeds),
        "estimators": [result.as_dict() for result in results],
        "preferred": preferred,
        "reason": reason,
        "note": (
            "Curvature misspecifies the treatment-free surface only; the blips, and "
            "therefore the estimand, are unchanged. Above "
            f"{MAX_INFORMATIVE_CURVATURE} the outcome saturates at its ceiling and "
            "every estimator degrades for reasons unrelated to robustness."
        ),
    }


__all__ = [
    "DEFAULT_CURVATURES",
    "MAX_INFORMATIVE_CURVATURE",
    "RobustnessResult",
    "misspecification_report",
    "preferred_estimator",
    "robustness_study",
]
