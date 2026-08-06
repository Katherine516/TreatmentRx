"""Single owner of the fitted estimators, their training split, and their scores.

Every estimator in Layer 4 is fit **once per process** on the same training
split, and scored on the same held-out split. Centralising that here buys three
things the prototype previously lacked:

* Estimators cannot silently be fit on different data, so their blip estimates
  and their model-averaging weights are comparable.
* `MethodResult.policy_value` becomes a real, measured quantity — the held-out
  IPW policy value of that estimator's greedy policy — instead of a rescaling of
  the estimator's own score. Bayesian model averaging then weights estimators by
  out-of-sample policy performance rather than by self-report.
* Calibration is a model-level property measured on held-out rows, which is what
  the validation ladder needs to gate on.

Fitting is lazy and cached: the first call pays it, everything after is a dict
lookup. The cohort is seeded, so a given commit always produces the same models.
"""

from __future__ import annotations

from dataclasses import dataclass

from treatmentrx.estimation.dwols import DWOLS_METHOD, DWOLSModel
from treatmentrx.estimation.q_learning import (
    Q_SHARED_METHOD,
    STAGE_SPECIFIC_METHOD,
    QLearningModel,
)
from treatmentrx.feedback.offline_evaluation import PolicyScore, evaluate_policy
from treatmentrx.domain import CalibrationReport
from treatmentrx.simulation.ra_cohort import (
    CohortTrajectory,
    generate_ra_cohort,
    rollout_value,
    train_test_split,
)

COHORT_SIZE = 400
COHORT_SEED = 7
HOLDOUT_FRACTION = 0.3
ROLLOUT_SAMPLES = 1500

Q_SHARED = Q_SHARED_METHOD
DWOLS_SHARED = DWOLS_METHOD
STAGE_SPECIFIC = STAGE_SPECIFIC_METHOD


@dataclass(frozen=True)
class FittedEstimators:
    cohort: list[CohortTrajectory]
    train: list[CohortTrajectory]
    holdout: list[CohortTrajectory]
    q_shared: QLearningModel
    stage_specific: QLearningModel
    dwols: DWOLSModel
    scores: dict[str, PolicyScore]

    @property
    def calibration(self) -> CalibrationReport:
        """Calibration of the best-scoring estimator on held-out data."""
        best = max(self.scores.values(), key=lambda score: score.ipw_policy_value)
        return best.calibration


_FITTED: FittedEstimators | None = None


def training_cohort() -> list[CohortTrajectory]:
    """The training split — the only data any estimator is allowed to be fit on."""
    train, _ = train_test_split(generate_ra_cohort(COHORT_SIZE, COHORT_SEED), HOLDOUT_FRACTION)
    return train


def fitted() -> FittedEstimators:
    global _FITTED
    if _FITTED is None:
        _FITTED = _fit_all()
    return _FITTED


def reset() -> None:
    """Drop the cached fit, forcing a refit on the next call.

    Needed only after changing a training constant (cohort size, seed, holdout
    fraction) inside a live process; the models are otherwise immutable.
    """
    global _FITTED
    _FITTED = None


def policy_value_for(method_name: str) -> float:
    """Held-out IPW policy value for an estimator, used as its `policy_value`."""
    score = fitted().scores.get(method_name)
    return score.ipw_policy_value if score else 0.0


def score_for(method_name: str) -> PolicyScore | None:
    return fitted().scores.get(method_name)


def holdout_calibration() -> CalibrationReport:
    return fitted().calibration


def scorecard() -> list[dict[str, object]]:
    """Ordered, serialisable comparison of every fitted estimator."""
    scores = fitted().scores.values()
    return [
        score.as_dict()
        for score in sorted(scores, key=lambda score: score.ipw_policy_value, reverse=True)
    ]


def _fit_all() -> FittedEstimators:
    cohort = generate_ra_cohort(COHORT_SIZE, COHORT_SEED)
    train, holdout = train_test_split(cohort, HOLDOUT_FRACTION)

    q_shared = QLearningModel(train, share_blip=True)
    stage_specific = QLearningModel(train, share_blip=False)
    dwols = DWOLSModel(train)

    scores = {
        Q_SHARED: evaluate_policy(
            Q_SHARED,
            q_shared.greedy_policy(),
            q_shared.predict_outcome,
            holdout,
            oracle_rollout_value=rollout_value(q_shared.greedy_policy(), n=ROLLOUT_SAMPLES),
        ),
        STAGE_SPECIFIC: evaluate_policy(
            STAGE_SPECIFIC,
            stage_specific.greedy_policy(),
            stage_specific.predict_outcome,
            holdout,
            oracle_rollout_value=rollout_value(stage_specific.greedy_policy(), n=ROLLOUT_SAMPLES),
        ),
        DWOLS_SHARED: evaluate_policy(
            DWOLS_SHARED,
            dwols.greedy_policy(),
            dwols.predict_outcome,
            holdout,
            oracle_rollout_value=rollout_value(dwols.greedy_policy(), n=ROLLOUT_SAMPLES),
        ),
    }

    return FittedEstimators(
        cohort=cohort,
        train=train,
        holdout=holdout,
        q_shared=q_shared,
        stage_specific=stage_specific,
        dwols=dwols,
        scores=scores,
    )
