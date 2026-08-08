"""Single owner of the fitted estimators, their training split, and their scores.

Every estimator in Layer 4 is fit **once per process** on the same training
split, and scored on the same held-out split. Centralising that here buys three
things the prototype previously lacked:

* Estimators cannot silently be fit on different data, so their blip estimates
  and their model-averaging weights are comparable.
* `RegimeEstimate.policy_value` becomes a real, measured quantity — the held-out
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
from treatmentrx.estimation.inference import joint_bootstrap
from treatmentrx.feedback.offline_evaluation import PolicyScore, estimand_values, evaluate_policy
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
_ORACLE_CACHE: dict[str, float | None] = {}
_ESTIMAND_CACHE: dict[str, tuple[float, float]] | None = None
_JOINT_BOOTSTRAP = None


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
    """Drop the cached fit and rollouts, forcing a refit on the next call.

    Needed only after changing a training constant (cohort size, seed, holdout
    fraction) inside a live process; the models are otherwise immutable.
    """
    global _FITTED, _ESTIMAND_CACHE, _JOINT_BOOTSTRAP
    _FITTED = None
    _ESTIMAND_CACHE = None
    _JOINT_BOOTSTRAP = None
    _ORACLE_CACHE.clear()


def enable_bootstrap_inference(
    replicates: int = 200,
    alpha: float = 0.5,
    seed: int = 17,
) -> dict[str, object]:
    """Attach m-out-of-n bootstrap draws to the Q-learning models.

    Off by default: it costs one full refit per replicate, and the sandwich is
    exact at the terminal stage where most decisions are made. Turn it on when a
    non-terminal interval has to be defensible — after this, `contrast()` on a
    non-terminal stage returns the bootstrap interval instead of the sandwich's
    optimistic one.
    """
    fit = fitted()
    report = {}
    for name, model in ((Q_SHARED, fit.q_shared), (STAGE_SPECIFIC, fit.stage_specific)):
        distribution = model.fit_bootstrap(
            fit.train, replicates=replicates, alpha=alpha, seed=seed
        )
        report[name] = {
            "n": distribution.n,
            "m": distribution.m,
            "non_regularity": round(distribution.non_regularity, 4),
            "replicates": distribution.replicates,
        }
    return report


def enable_joint_inference(
    replicates: int = 120,
    alpha: float = 0.5,
    seed: int = 17,
):
    """Refit all three estimators on shared resamples and cache the draws.

    The decision layer averages the estimators, so its standard error depends on
    their covariance. Without this it has to fall back on the perfect-correlation
    upper bound, which measures about 1.29x the averaged estimator's actual
    spread — every interval a quarter wider than necessary. This measures the
    covariance instead.

    It costs one refit of each estimator per replicate, so it is opt-in. Once
    enabled the draws are model-level: any patient's interval afterwards is a
    handful of dot products.
    """
    global _JOINT_BOOTSTRAP
    fit = fitted()

    def refit_all(sample):
        return {
            Q_SHARED: fit.q_shared.refit(sample),
            STAGE_SPECIFIC: fit.stage_specific.refit(sample),
            DWOLS_SHARED: fit.dwols.refit(sample),
        }

    _JOINT_BOOTSTRAP = joint_bootstrap(
        refit_all,
        fit.train,
        {
            Q_SHARED: fit.q_shared.flat_parameters(),
            STAGE_SPECIFIC: fit.stage_specific.flat_parameters(),
            DWOLS_SHARED: fit.dwols.flat_parameters(),
        },
        fit.q_shared.non_regularity(fit.train),
        replicates=replicates,
        alpha=alpha,
        seed=seed,
    )
    return _JOINT_BOOTSTRAP


def joint_inference():
    """The cached joint bootstrap, or None if it was never enabled."""
    return _JOINT_BOOTSTRAP


def disable_joint_inference() -> None:
    """Turn off joint inference without discarding the fits it was built from.

    Distinct from `reset()`: switching an inference mode off is not a reason to
    throw away models that cost a second to fit and have not changed.
    """
    global _JOINT_BOOTSTRAP
    _JOINT_BOOTSTRAP = None


def disable_bootstrap_inference() -> None:
    """Detach the per-estimator bootstrap draws, keeping the fits."""
    fit = fitted()
    fit.q_shared.attach_bootstrap(None)
    fit.stage_specific.attach_bootstrap(None)


def policy_value_for(method_name: str) -> float:
    """Held-out IPW policy value for an estimator, used as its `policy_value`."""
    score = fitted().scores.get(method_name)
    return score.ipw_policy_value if score else 0.0


def score_for(method_name: str) -> PolicyScore | None:
    return fitted().scores.get(method_name)


def holdout_calibration() -> CalibrationReport:
    return fitted().calibration


def holdout_estimands() -> dict[str, tuple[float, float]]:
    """ITT / per-protocol / as-treated for the best-scoring estimator's policy.

    Model-level and measured on held-out patients, cached because the estimands
    describe the policy rather than the patient in front of you.
    """
    global _ESTIMAND_CACHE
    if _ESTIMAND_CACHE is None:
        fit = fitted()
        best = max(fit.scores.values(), key=lambda score: score.ipw_policy_value)
        model = {
            Q_SHARED: fit.q_shared,
            STAGE_SPECIFIC: fit.stage_specific,
            DWOLS_SHARED: fit.dwols,
        }[best.estimator]
        _ESTIMAND_CACHE = estimand_values(model.greedy_policy(), fit.holdout)
    return _ESTIMAND_CACHE


def scorecard(include_oracle: bool = False) -> list[dict[str, object]]:
    """Ordered, serialisable comparison of every fitted estimator.

    `include_oracle` runs the simulation-only rollout benchmark, which costs a
    few hundred milliseconds of Monte Carlo per estimator. The audit event does
    not ask for it; `treatmentrx.cli evaluate` does.
    """
    scores = sorted(fitted().scores.values(), key=lambda score: score.ipw_policy_value, reverse=True)
    if not include_oracle:
        return [score.as_dict() for score in scores]
    return [score.as_dict() | {"oracle_rollout_value": oracle_rollout_value(score.estimator)} for score in scores]


def oracle_rollout_value(method_name: str) -> float | None:
    """Expected reward of an estimator's policy under the generating process.

    Cached per estimator: only available in simulation, and only used to check
    that the observational IPW estimate is not disagreeing with a known answer.
    """
    if method_name in _ORACLE_CACHE:
        return _ORACLE_CACHE[method_name]
    fit = fitted()
    model = {
        Q_SHARED: fit.q_shared,
        STAGE_SPECIFIC: fit.stage_specific,
        DWOLS_SHARED: fit.dwols,
    }.get(method_name)
    value = rollout_value(model.greedy_policy(), n=ROLLOUT_SAMPLES) if model else None
    _ORACLE_CACHE[method_name] = value
    return value


def _fit_all() -> FittedEstimators:
    cohort = generate_ra_cohort(COHORT_SIZE, COHORT_SEED)
    train, holdout = train_test_split(cohort, HOLDOUT_FRACTION)

    q_shared = QLearningModel(train, share_blip=True)
    stage_specific = QLearningModel(train, share_blip=False)
    dwols = DWOLSModel(train)

    # The oracle rollout is a simulation-only diagnostic: it needs the generating
    # process, so it can never exist outside this prototype, and no recommendation
    # depends on it. Computing it here would put ~0.5s of Monte Carlo on the cold
    # start of every process that serves a single patient. `scorecard()` fills it
    # in on demand instead.
    scores = {
        Q_SHARED: evaluate_policy(
            Q_SHARED, q_shared.greedy_policy(), q_shared.predict_outcome, holdout
        ),
        STAGE_SPECIFIC: evaluate_policy(
            STAGE_SPECIFIC, stage_specific.greedy_policy(), stage_specific.predict_outcome, holdout
        ),
        DWOLS_SHARED: evaluate_policy(
            DWOLS_SHARED, dwols.greedy_policy(), dwols.predict_outcome, holdout
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
