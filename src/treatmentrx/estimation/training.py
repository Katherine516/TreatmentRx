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

import math
import threading
from dataclasses import dataclass

from treatmentrx.estimation.dwols import DWOLS_METHOD, DWOLSModel
from treatmentrx.estimation.q_learning import (
    DEFAULT_POOLING_RIDGE,
    Q_POOLED_METHOD,
    Q_SHARED_METHOD,
    STAGE_SPECIFIC_METHOD,
    QLearningModel,
)
from treatmentrx.estimation.inference import joint_bootstrap
from treatmentrx.estimation.propensity import PropensityModel
from treatmentrx.estimation.specification import specification_report
from treatmentrx.feedback.offline_evaluation import (
    PolicyScore,
    estimand_values,
    evaluate_policy,
    sequential_dr_sensitivity,
)
from treatmentrx.domain import CalibrationReport
from treatmentrx.scientific import EvaluationPartitionContract
from treatmentrx.simulation.ra_cohort import (
    CohortTrajectory,
    behaviour_policy,
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
Q_POOLED = Q_POOLED_METHOD

# The estimators that serve a patient. Four are fitted and scored — `cli
# evaluate`, `stability`, `misspecification` and `coverage` all need the
# endpoints to compare against — but only these two are averaged into a
# recommendation or an interval.
#
# **Why not the shared blip.** Its ψ is a single vector used at every stage, so
# with stage-varying delayed effects it converges to a stage-pooled compromise:
# measured against the true value-to-go contrast it runs -0.075 at stage 0,
# +0.009 at stage 1 and +0.067 at the terminal stage, where 91% of the patients
# the pipeline is asked about sit — a property of `simulated_bundles`, which
# exports a whole trajectory so the patient always presents at their last visit,
# rather than of a clinic; see `coverage.SERVED_STAGE_INDICES`. It is the right
# weighting for this comparison and the wrong one to generalise from
# and where the truth ranges 0.008 to 0.088. The bias is comparable to the
# effect. No weighting repairs that — model averaging is only coherent when the
# members estimate the same quantity — and dropping it took contrast coverage
# from 74% to 97% pooled and 45% to 95% at the worst patient, at unchanged
# interval width and no cost to the decision (rollout 2.1467 -> 2.1473,
# oracle-arm agreement 0.888 -> 0.900).
#
# **Why the pooled fit rather than the stage-specific one.** Dropping the shared
# blip threw away the stable end of a bias-variance axis. `Q-Pooled` is that axis
# made continuous — a shared level plus penalized per-stage deviations — and it
# beats both endpoints: terminal parameter error 0.037 against 0.047
# stage-specific and 0.098 shared, and lower total blip error than either at
# every curvature. Ensemble coverage 97% -> 98%, worst patient 95% -> 97% — both
# measured before `ArmFit.cross_covariance` was kept, so they price the swap
# against the interval of the day. The ensemble's absolute coverage now reads
# 95.0% at SE/spread 1.04, at nominal rather than above it; the swap itself has
# not been re-priced and the comparison above is the one that justified it.
SERVING_ENSEMBLE = (Q_POOLED, DWOLS_SHARED)


@dataclass(frozen=True)
class FittedEstimators:
    cohort: list[CohortTrajectory]
    train: list[CohortTrajectory]
    holdout: list[CohortTrajectory]
    q_shared: QLearningModel
    stage_specific: QLearningModel
    pooled: QLearningModel
    dwols: DWOLSModel
    # Fit on `train` and applied to `holdout`: the held-out estimate must not
    # need a probability only the simulator knows.
    propensity: PropensityModel
    # Whether any named candidate covariate looks like an effect modifier the
    # blip basis is missing. Computed here rather than on request because a
    # flagged basis invalidates every contrast the ensemble reports, and a
    # caveat nobody runs is a caveat nobody sees.
    specification: dict[str, object]
    scores: dict[str, PolicyScore]

    @property
    def calibration(self) -> CalibrationReport:
        """Calibration of the selected estimator on held-out data.

        Selection goes through `best_score()` rather than an argmax, so a
        re-seed that reorders three indistinguishable policy values does not
        silently change which model's calibration the validation ladder reads.
        """
        return best_score().calibration


_FITTED: FittedEstimators | None = None
_FIT_LOCK = threading.Lock()
_ORACLE_CACHE: dict[str, float | None] = {}
_ESTIMAND_CACHE: dict[str, tuple[float, float]] | None = None
_JOINT_BOOTSTRAP = None


def training_cohort() -> list[CohortTrajectory]:
    """The training split — the only data any estimator is allowed to be fit on."""
    train, _ = train_test_split(generate_ra_cohort(COHORT_SIZE, COHORT_SEED), HOLDOUT_FRACTION)
    return train


def fitted() -> FittedEstimators:
    """The one fitted ensemble for this process. Fit lazily, exactly once.

    The lock is what makes "exactly once" true under `service.py`, which runs a
    thread per request: measured, four concurrent cold callers each ran the whole
    fit, so the process briefly held four ensembles and three of them were
    discarded. Nothing was *wrong* afterwards — the fit is deterministic, so they
    agreed — which is precisely why it would never have been noticed. A plain
    `Lock` rather than an `RLock` on purpose: `_fit_all` does not re-enter here
    today, and if that ever changes it should deadlock loudly rather than quietly
    go back to fitting twice.
    """
    global _FITTED
    if _FITTED is None:
        with _FIT_LOCK:
            if _FITTED is None:
                _FITTED = _fit_all()
    return _FITTED


def serving_models() -> dict[str, object]:
    """The *deployed* models `SERVING_ENSEMBLE` names, from the cached fit.

    Invariant 19 in one place, for the fit that actually serves patients.
    `coverage._serving_models` is its sibling and builds fresh models from a
    supplied cohort — that is what a coverage replication needs and is not what
    a study of the deployed agent needs. Both raise on a member they cannot
    build, because the failure to design against is a silent collapse to a
    subset: the numbers stay plausible and the ensemble's name stays on them.
    """
    fit = fitted()
    available = {
        Q_POOLED: fit.pooled,
        Q_SHARED: fit.q_shared,
        STAGE_SPECIFIC: fit.stage_specific,
        DWOLS_SHARED: fit.dwols,
    }
    missing = [name for name in SERVING_ENSEMBLE if name not in available]
    if missing:
        raise ValueError(f"no deployed model for serving estimator(s): {missing}")
    return {name: available[name] for name in SERVING_ENSEMBLE}


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
    their covariance. Without this it falls back on the perfect-correlation upper
    bound, which is wider than it needs to be. This measures the covariance
    instead, and on a single-patient study it looked like a free improvement:
    intervals 1.4x narrower, recovering 9 of 150 patients from an unnecessary
    equipoise.

    **Off by default, on measurement rather than caution.** Swept across the
    patient grid at n=280 (`coverage.joint_replicate_sweep`):

        interval                        pooled   worst patient  SE/spread  width
        joint bootstrap,  25 replicates   87%         75%          0.96    0.0607
        joint bootstrap,  50 replicates   91%         80%          0.96    0.0638
        joint bootstrap, 100 replicates   92%         80%          0.95    0.0653
        joint bootstrap, 200 replicates   93%         85%          0.94    0.0662
        correlation bound (pre-covariance) 98%        96%          1.27    0.0759
        correlation bound (deployed)      95%         93%          1.04    0.0663

    Coverage climbs with the draw count *and the interval widens as it does*,
    which is a percentile estimate stabilising rather than the method changing —
    SE/spread is ~0.95 throughout, so the standard error was always honest and
    only the quantiles were crude. That resolves an earlier 92% that could have
    been either.

    At 200 replicates it is within Monte Carlo error of nominal on the pooled
    figure but reaches only 85% at the worst patient. An earlier reading that the
    bootstrap was 1.43x narrower came from the collapsed-ensemble defect
    described in `feedback/coverage.py`: with the ensemble silently reduced to
    dWOLS alone the parameter space is smaller and the interval correspondingly
    narrower. Corrected, it was 1.15x.

    **Keeping the dWOLS cross-arm covariance closed the question outright.** The
    bound's width fell from 0.0759 to 0.0663 against the bootstrap's 0.0662, so
    the narrowness advantage is now nil — the two intervals are the same width to
    three decimals. The bound covers 95% pooled and 93% at its worst patient; the
    bootstrap covers 93% and 85%. There is no longer a trade to weigh: the bound
    is better on coverage and costs nothing on width. Enable the joint fit for
    analysis; do not serve on it.
    """
    global _JOINT_BOOTSTRAP
    fit = fitted()

    def refit_all(sample):
        return {
            Q_SHARED: fit.q_shared.refit(sample),
            STAGE_SPECIFIC: fit.stage_specific.refit(sample),
            Q_POOLED: fit.pooled.refit(sample),
            DWOLS_SHARED: fit.dwols.refit(sample),
        }

    _JOINT_BOOTSTRAP = joint_bootstrap(
        refit_all,
        fit.train,
        {
            Q_SHARED: fit.q_shared.flat_parameters(),
            STAGE_SPECIFIC: fit.stage_specific.flat_parameters(),
            Q_POOLED: fit.pooled.flat_parameters(),
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


# When two estimators' held-out values are not separated, the ordering carries no
# information and something has to break the tie deterministically. Preferring the
# simpler model is a stated policy, not a measurement — which is the point: it is
# visible in the audit as a tie-break rather than disguised as a ranking. Lower is
# preferred.
INTERPRETABILITY_ORDER = {
    DWOLS_SHARED: 0,
    Q_POOLED: 1,
    STAGE_SPECIFIC: 2,
    Q_SHARED: 3,
}


def policy_value_for(method_name: str) -> float:
    """Held-out IPW policy value for an estimator, used as its `policy_value`."""
    score = fitted().scores.get(method_name)
    return score.ipw_policy_value if score else 0.0


def best_score() -> PolicyScore:
    """The estimator to use wherever a single one is needed, and why.

    Four call sites used to take `max(..., key=ipw_policy_value)` over three
    numbers that `cli stability` reports as indistinguishable — the lead is 0.006
    against a 0.024 combined standard error. An argmax over noise is not a
    selection; it just makes an arbitrary choice look measured, and it can flip
    on a re-seed while every downstream number moves with it.

    So: the top scorer wins only if its held-out interval clears the runner-up's.
    Otherwise the contenders are tied and `INTERPRETABILITY_ORDER` breaks it,
    which is deterministic and stated. `ranking_is_resolved()` reports which of
    the two happened.

    Only `SERVING_ENSEMBLE` members are eligible. Whatever this returns defines
    the estimands and supplies the calibration the validation ladder gates on, so
    it has to be a model that actually serves patients.
    """
    ranked = _serving_scores()
    if len(ranked) < 2 or ranked[0].beats(ranked[1]):
        return ranked[0]
    return sorted(ranked, key=lambda s: INTERPRETABILITY_ORDER.get(s.estimator, 99))[0]


def ranking_is_resolved() -> bool:
    """True when the leading estimator's held-out interval clears the runner-up's."""
    ranked = _serving_scores()
    return len(ranked) > 1 and ranked[0].beats(ranked[1])


def _serving_scores() -> list[PolicyScore]:
    scores = fitted().scores
    return sorted(
        (scores[name] for name in SERVING_ENSEMBLE if name in scores),
        key=lambda score: score.ipw_policy_value,
        reverse=True,
    )


def score_for(method_name: str) -> PolicyScore | None:
    return fitted().scores.get(method_name)


def holdout_calibration() -> CalibrationReport:
    return fitted().calibration


def holdout_outcome_sd() -> float:
    """Spread of the outcome on held-out rows — the scale a contrast is read against.

    A contrast of 0.04 means nothing until you know what the outcome's own
    spread is, and standardising by it is what turns an effect into a quantity
    an E-value can be computed from. Measured on the evaluation split rather
    than the patient in front of you: this is a **population** property, which
    is the side of invariant 14 it belongs on. `AssumptionSensitivity` divides
    one patient's contrast by it, which is Cohen's d and is the intended mixing,
    not the forbidden one.
    """
    key = "holdout:outcome_sd"
    if key not in _ORACLE_CACHE:
        outcomes = [stage.outcome for t in fitted().holdout for stage in t.stages]
        mean = sum(outcomes) / len(outcomes)
        variance = sum((y - mean) ** 2 for y in outcomes) / max(len(outcomes) - 1, 1)
        _ORACLE_CACHE[key] = math.sqrt(variance)
    return _ORACLE_CACHE[key]


def evaluation_partition() -> EvaluationPartitionContract:
    """The partition this build actually has, stated rather than implied.

    `EvaluationPartitionContract` was exported and unit-tested and never once
    constructed, so the package advertised a locked final test that did not
    exist. This builds the real thing: a training split and an evaluation split,
    no tuning partition, **no final test**. `has_final_test` is False and is
    reported on the model card.

    Why that is worth saying out loud rather than leaving to a reader who counts
    two lists: the evaluation split does four jobs at once. It sets the
    model-averaging weights, it selects the serving estimator through
    `best_score()`, it supplies the calibration the validation ladder gates on,
    and it is the held-out policy value on the card. Each of those is defensible
    on its own; together they mean the headline numbers come from a split that
    has been looked at many times, with nothing untouched left to check them
    against. The Monte Carlo studies in `feedback/` do draw fresh seeds — that
    part is fine and is the reason the tuned constants are not overfit to this
    split — but no amount of fresh seeds gives back a held-back test set.

    The ids are the cohort's `patient_index`, which is unique within a cohort and
    is what the split partitions.
    """
    fit = fitted()
    return EvaluationPartitionContract(
        training=frozenset(str(t.patient_index) for t in fit.train),
        # The held-out split is used to evaluate, to weight and to select, so it
        # is named `calibration` rather than `final_test`: a partition that has
        # informed model development is not a test set, whatever it is called.
        tuning=frozenset(),
        calibration=frozenset(str(t.patient_index) for t in fit.holdout),
        final_test=frozenset(),
    )


def propensity_comparison() -> dict[str, object]:
    """What using an estimated propensity instead of the generator's costs.

    Simulation-only, and the point of it: `CohortStage.propensity` is oracle
    knowledge, so every held-out number computed from it would be unavailable in
    a real deployment. Scoring both ways says whether the evaluation *depends* on
    that knowledge or merely used it because it was there.

    Measured on the deployed split, the fitted propensity moves each estimator's
    held-out value by +0.006 to +0.011 — under a sixth of the interval width, and
    in the same direction for every estimator, so the ordering does not change.
    That is the claim this supports: the evaluation is reproducible without the
    generator. It is *not* a claim about unmeasured confounding, which is the
    part a simulation cannot test at all.
    """
    fit = fitted()
    models = {
        Q_SHARED: fit.q_shared,
        STAGE_SPECIFIC: fit.stage_specific,
        Q_POOLED: fit.pooled,
        DWOLS_SHARED: fit.dwols,
    }
    rows = {}
    for name, model in models.items():
        oracle = evaluate_policy(
            name, model.greedy_policy(), model.predict_outcome, fit.holdout,
            with_intervals=False,
        )
        rows[name] = {
            "oracle_propensity": oracle.ipw_policy_value,
            "fitted_propensity": fit.scores[name].ipw_policy_value,
            "shift": round(
                fit.scores[name].ipw_policy_value - oracle.ipw_policy_value, 4
            ),
            "oracle_effective_sample_size": oracle.effective_sample_size,
            "fitted_effective_sample_size": fit.scores[name].effective_sample_size,
        }
    return {
        "model": {
            "converged": fit.propensity.fit.converged,
            "iterations": fit.propensity.fit.iterations,
            "rows": fit.propensity.fit.n_rows,
        },
        "calibration_vs_generator": {
            "train": fit.propensity.calibration(fit.train),
            "holdout": fit.propensity.calibration(fit.holdout),
        },
        "per_estimator": rows,
        "note": (
            "The deployed scores use the fitted propensity. The oracle column "
            "exists only in simulation and is here to size the difference, not "
            "because anything reads it."
        ),
    }


def basis_specification() -> dict[str, object]:
    """Does any named candidate look like a missing effect modifier?

    Model-level and computed once per fit. Read `flagged`: a non-empty list means
    the reported contrasts are covariate-averaged rather than this patient's, and
    — the part that makes it urgent — the interval will not show it. An omitted
    modifier is a bias in the estimand, and a standard error computed under the
    wrong basis cannot see one.
    """
    return fitted().specification


def basis_is_flagged() -> bool:
    return bool(fitted().specification.get("flagged"))


# Whether a flagged basis should stop the agent recommending at all.
#
# False, deliberately, and it is a policy rather than an oversight. The test is a
# *falsification* test with an actionable fix — "add this covariate and refit" —
# not a permanent property of the data, and turning a diagnostic into a silent
# behaviour change is the pattern this repo keeps removing. What a flag does
# instead is block advancement on the validation ladder, raise the epistemic
# uncertainty, and appear on every clinician card. Flip this to True for a
# deployment that would rather withhold than caveat.
BLOCK_ON_FLAGGED_BASIS = False


def deployment_readiness() -> dict[str, object]:
    """Model-level facts the validation ladder gates on.

    Every value here describes the *policy*, measured on held-out patients. The
    ladder previously read a patient's own switching-aware OPE, whose effective
    sample size is at least 1 for anybody with a visit — so the gate could not
    close. Whether a model may leave silent mode is not a fact about one patient.

    `live_data` is False and stays False in this build. A held-out split of the
    simulated cohort the model was fit on is not live data, and the SILENT gate
    reads "OPE + calibration stable **on live data**". Reporting it as satisfied
    because the held-out numbers look good is precisely the rung-skip the ladder
    exists to prevent.

    **Two effective sample sizes, and the gate needs both.** This used to report
    only `ope_effective_sample_size`, the *per-decision* one — 75.5 on the
    deployed holdout, comfortably over `MIN_OPE_EFFECTIVE_SAMPLE`. But what
    advances a rung is a three-stage regime, and the regime's own value is
    estimated on 44 rows at an effective sample of 14.6. A gate that reads the
    per-decision number to decide whether a sequential regime may be deployed is
    answering an easier question than the one it is named for, which is the
    defect invariant 25 is about.

    **The doubly-robust estimate does not retire that gate, and saying why is
    the point.** `sequential_regime_value_doubly_robust` augments with the
    fitted Q-function, uses all 120 holdout trajectories rather than five, and
    lands within half a standard error of the known truth. So the regime's value
    *is* identified — under the outcome model. The assumption-light IPW estimate
    is what could falsify that model, and at an effective sample of 14.6 it
    cannot: a doubly-robust estimate whose IPW check has no power is a
    model-based estimate wearing a robustness label. Both are reported, the
    gate reads the IPW one, and the blocker says which of the two facts it is
    stating.
    """
    best = best_score()
    calibration = holdout_calibration()
    sequential = best.sequential
    doubly_robust = best.sequential_dr
    return {
        "estimator": best.estimator,
        "ope_effective_sample_size": best.effective_sample_size,
        # The regime's own effective sample, which is what a *sequential* policy
        # is deployed on. `None` means nobody computed it, and the ladder words
        # that differently from "computed and failing".
        "sequential_ope_effective_sample_size": (
            sequential.effective_sample_size if sequential else None
        ),
        "sequential_value_interval": (
            list(sequential.interval) if sequential and sequential.interval else None
        ),
        # The doubly-robust counterpart, reported so the blocker can say what is
        # actually known about the regime's value rather than only what is not.
        "sequential_dr_value": doubly_robust.value if doubly_robust else None,
        "sequential_dr_interval": (
            list(doubly_robust.interval)
            if doubly_robust and doubly_robust.interval
            else None
        ),
        "sequential_dr_trajectories": (
            doubly_robust.n_trajectories if doubly_robust else None
        ),
        "regime_consistent_trajectories": (
            list(sequential.consistent_trajectories) if sequential else None
        ),
        "ope_improvement": best.improvement,
        "ope_improvement_lower": (
            best.improvement_interval[0] if best.improvement_interval else None
        ),
        "calibration_passed": calibration.passed,
        "expected_calibration_error": calibration.expected_calibration_error,
        "blip_basis_unflagged": not basis_is_flagged(),
        "flagged_modifiers": list(fitted().specification.get("flagged", [])),
        "live_data": False,
    }


def holdout_estimands() -> dict[str, tuple[float, float]]:
    """ITT / per-protocol / as-treated for the best-scoring estimator's policy.

    Model-level and measured on held-out patients, cached because the estimands
    describe the policy rather than the patient in front of you.
    """
    global _ESTIMAND_CACHE
    if _ESTIMAND_CACHE is None:
        fit = fitted()
        model = {
            Q_SHARED: fit.q_shared,
            STAGE_SPECIFIC: fit.stage_specific,
            Q_POOLED: fit.pooled,
            DWOLS_SHARED: fit.dwols,
        }[best_score().estimator]
        _ESTIMAND_CACHE = estimand_values(
            model.greedy_policy(), fit.holdout, propensity=fit.propensity.propensity
        )
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
    return [_with_oracle(score) for score in scores]


def _with_oracle(score: PolicyScore) -> dict[str, object]:
    """Attach both oracle rollouts, and close the DR estimate's comparison.

    Filled in here rather than at fit time: the rollouts need the generating
    process and ~0.5s of Monte Carlo each, which has no business on the cold
    start of a process that serves one patient. The consequence is that
    `SequentialDRValue.covers_oracle` is `None` until someone asks for the
    oracle, which is the honest state — absent, not false.
    """
    payload = score.as_dict()
    uncensored = oracle_uncensored_value(score.estimator)
    payload["oracle_rollout_value"] = oracle_rollout_value(score.estimator)
    payload["oracle_uncensored_value"] = uncensored
    doubly_robust = payload.get("sequential_regime_value_doubly_robust")
    if doubly_robust is not None and uncensored is not None:
        interval = doubly_robust.get("interval")
        doubly_robust["oracle_uncensored_value"] = uncensored
        doubly_robust["covers_oracle"] = bool(
            interval and interval[0] <= uncensored <= interval[1]
        )
    return payload


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
        Q_POOLED: fit.pooled,
        DWOLS_SHARED: fit.dwols,
    }.get(method_name)
    value = rollout_value(model.greedy_policy(), n=ROLLOUT_SAMPLES) if model else None
    _ORACLE_CACHE[method_name] = value
    return value


def behaviour_uncensored_value() -> float:
    """What the behaviour policy is worth on the DR estimate's scale.

    The doubly-robust sequential value is a value-to-go under full follow-up, so
    the thing it claims to beat has to be measured the same way.
    `PolicyScore.behaviour_value` is the per-decision observational mean and is
    not that quantity — comparing the two would be invariant 14 again.

    Simulation-only, like every rollout here, and cached for the same reason.
    """
    key = "behaviour:uncensored"
    if key not in _ORACLE_CACHE:
        _ORACLE_CACHE[key] = rollout_value(
            behaviour_policy, n=ROLLOUT_SAMPLES, dropout=False
        )
    return _ORACLE_CACHE[key]


def dr_sensitivity(method_name: str | None = None) -> dict[str, object]:
    """How wrong the outcome model would have to be to overturn the DR claim.

    Not computed at fit time: it re-runs the DR recursion at a dozen
    perturbations plus a bisection, and nothing on the serving path reads it.
    `cli evaluate` asks for it.
    """
    fit = fitted()
    name = method_name or best_score().estimator
    model = {
        Q_SHARED: fit.q_shared,
        STAGE_SPECIFIC: fit.stage_specific,
        Q_POOLED: fit.pooled,
        DWOLS_SHARED: fit.dwols,
    }[name]
    # dWOLS has no value-to-go Q-function to perturb, so it borrows the pooled
    # one exactly as its DR estimate does. `augmentation_model` on that estimate
    # is where the borrowing is already recorded.
    augmentation = model if isinstance(model, QLearningModel) else fit.pooled
    return sequential_dr_sensitivity(
        model.greedy_policy(),
        fit.holdout,
        augmentation,
        behaviour_uncensored_value(),
        propensity=fit.propensity.propensity,
    ).as_dict()


def oracle_uncensored_value(method_name: str) -> float | None:
    """The same rollout with dropout switched off — a *different* quantity.

    `oracle_rollout_value` is what a patient actually accrues: dropout is
    simulated, so a policy that drives toxicity is penalised twice, once through
    the delayed cost and again through the visits it loses. That is the right
    yardstick for the per-decision IPW value.

    It is the wrong one for `sequential_regime_value_doubly_robust`. That
    estimator augments with a Q-model fit under IPCW, which targets the
    value-to-go *had the patient stayed in care*. On the deployed split the two
    truths are 2.1469 and 2.2938; the 0.147 between them is retention, and
    scoring the DR estimate against the smaller one would charge it for a gap it
    is not estimating. Invariant 14, in a place where both numbers are model-level
    and only the follow-up assumption differs.
    """
    key = f"uncensored:{method_name}"
    if key in _ORACLE_CACHE:
        return _ORACLE_CACHE[key]
    fit = fitted()
    model = {
        Q_SHARED: fit.q_shared,
        STAGE_SPECIFIC: fit.stage_specific,
        Q_POOLED: fit.pooled,
        DWOLS_SHARED: fit.dwols,
    }.get(method_name)
    value = (
        rollout_value(model.greedy_policy(), n=ROLLOUT_SAMPLES, dropout=False)
        if model
        else None
    )
    _ORACLE_CACHE[key] = value
    return value


def _fit_all() -> FittedEstimators:
    cohort = generate_ra_cohort(COHORT_SIZE, COHORT_SEED)
    train, holdout = train_test_split(cohort, HOLDOUT_FRACTION)

    q_shared = QLearningModel(train, share_blip=True)
    stage_specific = QLearningModel(train, share_blip=False)
    # The serving Q-learning model. The two endpoints above stay fitted as the
    # comparators every study in `feedback/` measures against.
    pooled = QLearningModel(train, share_blip=True, pooling_ridge=DEFAULT_POOLING_RIDGE)
    dwols = DWOLSModel(train)
    # The behaviour policy's assignment probabilities, estimated rather than
    # read off the generator. `CohortStage.propensity` is oracle knowledge; the
    # held-out score is what the validation ladder gates on, so it has to be
    # computable from the data an analyst would actually hold.
    propensity = PropensityModel(train)
    # 0.18s on the deployed split. Cheap enough to be unconditional, and the
    # alternative — a CLI nobody remembers to run — is how a model ships with a
    # basis that `cli misspecification --omitted-modifier` says costs 65 points
    # of interval coverage.
    specification = specification_report(train)

    # The oracle rollout is a simulation-only diagnostic: it needs the generating
    # process, so it can never exist outside this prototype, and no recommendation
    # depends on it. Computing it here would put ~0.5s of Monte Carlo on the cold
    # start of every process that serves a single patient. `scorecard()` fills it
    # in on demand instead.
    #
    # The doubly-robust sequential value needs a *value-to-go* Q-function to
    # augment with. Each Q-learning model supplies its own, for which the
    # backward-induction `max` and the evaluated regime coincide. dWOLS's
    # `raw_q` is a single-visit outcome, so it borrows `pooled` — legitimate
    # (double robustness does not require the augmenting model to be Q^d) and
    # named in `augmentation_model`, because it costs efficiency and shifts the
    # estimator's weight onto the propensity model.
    scores = {
        name: evaluate_policy(
            name,
            model.greedy_policy(),
            model.predict_outcome,
            holdout,
            propensity=propensity.propensity,
            q_model=model if isinstance(model, QLearningModel) else pooled,
            augmentation_model=name if isinstance(model, QLearningModel) else Q_POOLED,
        )
        for name, model in (
            (Q_SHARED, q_shared),
            (STAGE_SPECIFIC, stage_specific),
            (Q_POOLED, pooled),
            (DWOLS_SHARED, dwols),
        )
    }

    return FittedEstimators(
        cohort=cohort,
        train=train,
        holdout=holdout,
        q_shared=q_shared,
        stage_specific=stage_specific,
        pooled=pooled,
        dwols=dwols,
        propensity=propensity,
        specification=specification,
        scores=scores,
    )
