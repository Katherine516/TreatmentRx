"""Held-out offline evaluation of a fitted treatment policy.

`switching_aware_ope.py` answers "what is this *patient's* trajectory worth?".
This module answers the model-level question the validation ladder actually
gates on: **is the learned policy better than the one that generated the data,
on patients the estimator never saw?**

Three quantities, deliberately kept separate:

* **IPW policy value** — self-normalised (Hajek) inverse-probability estimate on
  held-out observational rows. This is the estimate available in the real world.
* **Oracle rollout value** — expected reward under the generating process
  itself. Available only in simulation; it exists so the IPW estimate can be
  checked against a known answer rather than trusted.
* **Calibration** — predicted outcome for the arm the patient actually received
  versus what was observed, on held-out rows. Calibration is a property of the
  model measured on data it was not fit to, never a patient's own history
  compared against itself.

**Two values, and they are not interchangeable.** `ipw_policy_value` weights
each decision point by its own propensity, which is the right estimand for "how
good is the arm choice at this visit". `sequential_policy_value` requires
agreement at every prior decision and weights by the cumulative propensity
product, which is the value of the *regime* — what a three-stage agent actually
claims. Both are reported with their denominators, their weight diagnostics and
their intervals, because the gap between them (0.086) is larger than the whole
claimed gain over the behaviour policy (0.066), and because the variance
blow-up the per-decision version avoids is real: 88 rows at effective sample
75.5 against 44 rows at 14.6. That blow-up is the finding, not a reason to
report only the comfortable number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from treatmentrx.decision.calibration import CalibrationEvaluator
from treatmentrx.domain import CalibrationReport
from treatmentrx.estimation.inference import normal_critical_value
from treatmentrx.simulation.ra_cohort import CohortTrajectory, optimal_arm

# Propensity floor for the weights; mirrors the clipping used when fitting.
_PROPENSITY_FLOOR = 0.02

# Below this the sequential value is a ratio of a handful of heavily weighted
# rows and should be read as "not identified at this sample size" rather than as
# an estimate. Matches `validation_ladder.MIN_OPE_EFFECTIVE_SAMPLE`, because a
# deployment question and an evaluation question should not disagree about what
# counts as enough effective sample.
MIN_SEQUENTIAL_EFFECTIVE_SAMPLE = 30.0


@dataclass(frozen=True)
class PolicyScore:
    """Held-out scorecard for one estimator's greedy policy."""

    estimator: str
    ipw_policy_value: float
    behaviour_value: float
    effective_sample_size: float
    agreement_rate: float
    optimal_arm_rate: float
    calibration: CalibrationReport
    n_holdout_stages: int
    oracle_rollout_value: float | None = None
    # The regime's own value, beside the per-decision one. None until measured;
    # an absent sequential estimate and an unidentified one are different facts.
    sequential: SequentialValue | None = None
    # The same estimand, estimated doubly robustly. On a different scale from
    # `ipw_policy_value` and `sequential` — value-to-go rather than per-visit —
    # which `SequentialDRValue.scale` states rather than leaving to be inferred.
    sequential_dr: SequentialDRValue | None = None
    # The weights behind `ipw_policy_value`. An IPW estimate is a weighted mean;
    # its weights are part of the result, not an implementation detail.
    weights: WeightDiagnostics | None = None
    notes: list[str] = field(default_factory=list)
    # Percentile interval for `ipw_policy_value` and for `improvement`, from
    # resampling held-out trajectories. Both are `None` until measured, because
    # an absent interval and a wide one are different facts and a consumer that
    # takes an argmax needs to be able to tell them apart.
    value_interval: tuple[float, float] | None = None
    improvement_interval: tuple[float, float] | None = None

    @property
    def improvement(self) -> float:
        """Estimated per-stage gain over the clinician policy in the data."""
        return round(self.ipw_policy_value - self.behaviour_value, 4)

    @property
    def value_standard_error(self) -> float:
        """Half-width of the value interval over the 95% normal quantile."""
        if self.value_interval is None:
            return 0.0
        return (self.value_interval[1] - self.value_interval[0]) / (2.0 * 1.96)

    def beats(self, other: PolicyScore) -> bool:
        """Is this estimator's held-out value separated from `other`'s?

        Two estimators whose intervals overlap are not ranked; picking between
        them is a coin flip dressed as a measurement. `cli stability` reaches the
        same verdict by refitting across seeds — this is the cheap version that
        can run on the inference path.
        """
        if self.value_interval is None or other.value_interval is None:
            return False
        return self.value_interval[0] > other.value_interval[1]

    def as_dict(self) -> dict[str, object]:
        return {
            "estimator": self.estimator,
            "ipw_policy_value": self.ipw_policy_value,
            "ipw_policy_value_interval": (
                [round(v, 4) for v in self.value_interval] if self.value_interval else None
            ),
            "behaviour_value": self.behaviour_value,
            "improvement": self.improvement,
            "improvement_interval": (
                [round(v, 4) for v in self.improvement_interval]
                if self.improvement_interval
                else None
            ),
            "effective_sample_size": self.effective_sample_size,
            "sequential_regime_value": (
                self.sequential.as_dict() if self.sequential else None
            ),
            "sequential_regime_value_doubly_robust": (
                self.sequential_dr.as_dict() if self.sequential_dr else None
            ),
            "weight_diagnostics": self.weights.as_dict() if self.weights else None,
            "agreement_rate": self.agreement_rate,
            "myopic_oracle_agreement_rate": self.optimal_arm_rate,
            "expected_calibration_error": self.calibration.expected_calibration_error,
            "calibration_passed": self.calibration.passed,
            "n_holdout_stages": self.n_holdout_stages,
            "oracle_rollout_value": self.oracle_rollout_value,
            # `notes` was populated and never emitted, so the one line saying
            # the regime's own value is not identified — the whole point of
            # computing it — reached nobody. `cli evaluate` and the model card
            # both render this dict.
            "notes": list(self.notes),
        }


# Replicates for the held-out evaluation bootstrap. No refit is involved — the
# policy is already fitted — so this is a few hundred passes over ~120
# trajectories and is cheap enough to cache on the inference path.
EVALUATION_REPLICATES = 400
EVALUATION_SEED = 31


def _holdout_rows(cohort: list[CohortTrajectory]):
    for trajectory in cohort:
        for position, stage in enumerate(trajectory.stages):
            yield position, stage


def _hajek(weights: list[float], outcomes: list[float]) -> tuple[float, float]:
    """Self-normalised IPW value and its effective sample size."""
    if not weights:
        return 0.0, 0.0
    total = sum(weights)
    value = sum(w * y for w, y in zip(weights, outcomes)) / total
    ess = (total ** 2) / sum(w * w for w in weights)
    return value, ess


def estimand_values(
    policy, holdout: list[CohortTrajectory], propensity=None
) -> dict[str, tuple[float, float]]:
    """ITT, per-protocol and as-treated, each computed on held-out data.

    These are *model-level* quantities and have to be measured on a population
    where the deviations are actually observed. Scaling a policy value by one
    patient's adherence fraction produces a number that is neither the effect of
    the regime nor the effect on that patient.

    * **ITT** — value of the learned policy as assigned, over every held-out
      decision point, deviations included.
    * **Per-protocol** — the same, restricted to decision points where the
      clinician did not switch away from the previous arm. Informative only if
      switching is not itself outcome-driven, which here it is: the number is
      reported so that assumption is visible, not because it is safe.
    * **As-treated** — the value actually realised under whatever was given.
    """
    itt_weights: list[float] = []
    itt_outcomes: list[float] = []
    protocol_weights: list[float] = []
    protocol_outcomes: list[float] = []
    realised: list[float] = []

    for trajectory in holdout:
        for position, stage in enumerate(trajectory.stages):
            realised.append(stage.outcome)
            if policy(stage.features, position) != stage.arm:
                continue
            weight = 1.0 / _propensity_of(stage, propensity)
            itt_weights.append(weight)
            itt_outcomes.append(stage.outcome)
            switched = position > 0 and trajectory.stages[position - 1].arm != stage.arm
            if not switched:
                protocol_weights.append(weight)
                protocol_outcomes.append(stage.outcome)

    itt_value, itt_ess = _hajek(itt_weights, itt_outcomes)
    protocol_value, protocol_ess = _hajek(protocol_weights, protocol_outcomes)
    as_treated = sum(realised) / len(realised) if realised else 0.0
    return {
        "ITT": (round(itt_value, 4), round(itt_ess, 2)),
        "per_protocol": (round(protocol_value, 4), round(protocol_ess, 2)),
        "as_treated": (round(as_treated, 4), float(len(realised))),
    }


def _propensity_of(stage, propensity) -> float:
    """The probability this stage's arm was given, true or estimated.

    `propensity=None` reads `stage.propensity`, which the simulator supplies and
    no real deployment observes. Passing a fitted model is what makes the
    held-out estimate reproducible outside a simulation.
    """
    if propensity is None:
        return max(stage.propensity, _PROPENSITY_FLOOR)
    return max(propensity(stage.features, stage.arm), _PROPENSITY_FLOOR)


@dataclass(frozen=True)
class WeightDiagnostics:
    """The shape of the inverse-probability weights behind an IPW number.

    An IPW estimate is a weighted mean, and a weighted mean is only as good as
    its weights. Nothing here reported them: a value of 0.74 with an effective
    sample of 75 looks the same whether the weights are flat or whether three
    rows carry a third of the mass. `max_weight` and `share_at_floor` are the
    two facts that distinguish those, and `positivity_ok` is the standing
    question an IPW design has to answer rather than assume.

    The floor is `_PROPENSITY_FLOOR`; a row sitting on it is one where the model
    thinks this arm was nearly impossible for this patient, so its outcome is
    being asked to stand in for a counterfactual the data barely observed.
    """

    rows: int
    effective_sample_size: float
    max_weight: float
    mean_weight: float
    share_at_floor: float
    top_share: float

    @property
    def positivity_ok(self) -> bool:
        """No single row dominating, and the floor rarely binding."""
        return self.top_share <= 0.10 and self.share_at_floor <= 0.05

    def as_dict(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "effective_sample_size": round(self.effective_sample_size, 2),
            "efficiency": round(self.effective_sample_size / self.rows, 3) if self.rows else 0.0,
            "max_weight": round(self.max_weight, 2),
            "mean_weight": round(self.mean_weight, 2),
            "share_of_mass_in_heaviest_row": round(self.top_share, 4),
            "share_of_rows_at_the_propensity_floor": round(self.share_at_floor, 4),
            "positivity_ok": self.positivity_ok,
        }


def weight_diagnostics(weights: list[float]) -> WeightDiagnostics:
    """Summarise an IPW weight vector, including how often the floor binds."""
    if not weights:
        return WeightDiagnostics(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    total = sum(weights)
    _, ess = _hajek(weights, [0.0] * len(weights))
    ceiling = 1.0 / _PROPENSITY_FLOOR
    at_floor = sum(1 for w in weights if w >= ceiling - 1e-9)
    return WeightDiagnostics(
        rows=len(weights),
        effective_sample_size=ess,
        max_weight=max(weights),
        mean_weight=total / len(weights),
        share_at_floor=at_floor / len(weights),
        top_share=max(weights) / total,
    )


@dataclass(frozen=True)
class SequentialValue:
    """The value of the *regime*, estimated the way a DTR requires.

    `evaluate_policy` scores every stage-row independently: it keeps rows where
    the policy agreed with the arm actually given and weights by that one
    stage's propensity. That is a **per-decision** quantity, and for a sequential
    regime it is not the regime's value. A patient who deviated at stage 1 still
    contributes their stage-2 row — but under the regime being evaluated, that
    patient would never have reached that state.

    The standard estimator requires agreement at *every* decision up to t and
    weights by the cumulative product of propensities. It is reported beside the
    per-decision number rather than instead of it, because the two answer
    different questions and the honest thing is to show both with their
    denominators:

        per-decision   value 0.7377   92 rows   ESS 78.6
        sequential     value 0.7935   47 rows   ESS 15.9

    The difference (0.056) is comparable to the whole claimed gain over the
    behaviour policy (0.066), and the sequential effective sample falls below
    `MIN_OPE_EFFECTIVE_SAMPLE`. That is not a defect in this estimator — it is
    what a three-stage regime costs in an observational cohort this size, and it
    was previously invisible because nothing computed it.
    """

    value: float
    effective_sample_size: float
    matched_rows: int
    consistent_trajectories: tuple[int, ...]
    n_trajectories: int
    max_weight: float
    weights: WeightDiagnostics | None = None
    # Trajectory-clustered percentile interval. `None` only when the resample
    # could not produce two usable draws — which is itself a finding at this
    # effective sample, so it is not silently rendered as a point estimate.
    interval: tuple[float, float] | None = None
    # Share of bootstrap resamples that contained no regime-consistent row at
    # all. At 3 surviving trajectories out of 120 this is not a rounding detail.
    degenerate_resample_share: float = 0.0

    @property
    def identified(self) -> bool:
        """Is there enough effective sample to read this as an estimate?"""
        return self.effective_sample_size >= MIN_SEQUENTIAL_EFFECTIVE_SAMPLE

    @property
    def interval_width(self) -> float | None:
        """How wide the regime's value actually is, rather than how small its ESS is.

        `identified` compares an effective sample against a threshold, which
        tells a reader the estimate is untrustworthy without telling them by how
        much. On a 0-1 outcome scale an interval spanning most of the range is
        the same statement, said in the units the value is reported in.
        """
        if self.interval is None:
            return None
        return self.interval[1] - self.interval[0]

    def as_dict(self) -> dict[str, object]:
        return {
            "value": round(self.value, 4),
            "effective_sample_size": round(self.effective_sample_size, 2),
            "matched_rows": self.matched_rows,
            "consistent_trajectories": list(self.consistent_trajectories),
            "n_trajectories": self.n_trajectories,
            "max_weight": round(self.max_weight, 2),
            "weight_diagnostics": self.weights.as_dict() if self.weights else None,
            "interval": list(self.interval) if self.interval else None,
            "interval_width": (
                round(self.interval_width, 4) if self.interval_width is not None else None
            ),
            "degenerate_resample_share": round(self.degenerate_resample_share, 4),
            "identified": self.identified,
            "note": (
                "Agreement required at every decision up to each stage, weighted "
                "by the cumulative propensity product. This is the value of the "
                "regime; `ipw_policy_value` is the per-decision quantity and the "
                "two are not interchangeable. The interval is a percentile "
                "bootstrap over held-out trajectories with the policy held "
                "fixed, so it is the precision of this evaluation and not the "
                "variability of the fit — and at this effective sample a "
                "percentile interval is itself not validated, which is why "
                "`identified` is reported beside it rather than replaced by it."
            ),
        }


def _sequential_aggregates(policy, holdout: list[CohortTrajectory], propensity=None):
    """Per-trajectory partial sums over the regime-consistent prefix.

    Split out of `sequential_policy_value` for the same reason
    `_trajectory_aggregates` is: the trajectory is the independent unit, so a
    bootstrap has to resample whole trajectories, and every quantity in the
    estimator is a sum over that trajectory's stages. Computing the prefix once
    and re-weighting it is exact, not an approximation.

    A trajectory that deviates at its first decision contributes `(0, 0, [])`
    and still occupies a slot in the resample — it is part of the holdout, and
    dropping it would condition on the very agreement being measured.
    """
    aggregates = []
    surviving = [0] * max((len(t.stages) for t in holdout), default=0)

    for trajectory in holdout:
        cumulative = 1.0
        sum_w = sum_wy = 0.0
        weights: list[float] = []
        for position, stage in enumerate(trajectory.stages):
            if policy(stage.features, position) != stage.arm:
                break
            cumulative *= _propensity_of(stage, propensity)
            surviving[position] += 1
            weight = 1.0 / cumulative
            sum_w += weight
            sum_wy += weight * stage.outcome
            weights.append(weight)
        aggregates.append((sum_w, sum_wy, weights))

    return aggregates, tuple(surviving)


def sequential_intervals(
    aggregates,
    replicates: int = EVALUATION_REPLICATES,
    seed: int = EVALUATION_SEED,
    alpha: float = 0.05,
) -> tuple[tuple[float, float] | None, float]:
    """Percentile interval for the regime's value, clustered by trajectory.

    **Why this is not optional.** `identified` compares the effective sample
    against a threshold and returns a boolean. A reader told "0.8263, not
    identified" knows the number is untrustworthy but not by how much, and the
    natural thing to do with a point estimate that carries no interval is to
    quote it. The interval says the same thing in the units the value is
    reported in.

    **Why it is the same machinery as `evaluation_intervals` and not the same
    call.** That one resamples every trajectory's *matched rows* under
    per-decision weights; this one resamples the regime-consistent *prefix*
    under cumulative weights, and the two differ in which rows exist at all —
    88 against 44 on the deployed holdout. Sharing the resample loop and not
    the aggregates is the only way both stay honest.

    The share of resamples with no consistent row at all is returned beside the
    interval rather than quietly skipped. With three trajectories surviving to
    the terminal decision, a resample drawing none of them is a routine event
    and a reader should be told how routine.
    """
    import random

    n = len(aggregates)
    if n < 2:
        return None, 0.0
    rng = random.Random(seed)
    values: list[float] = []
    degenerate = 0
    for _ in range(replicates):
        total_w = total_wy = 0.0
        for _ in range(n):
            sum_w, sum_wy, _ = aggregates[rng.randrange(n)]
            total_w += sum_w
            total_wy += sum_wy
        if total_w <= 0.0:
            degenerate += 1
            continue
        values.append(total_wy / total_w)
    share = degenerate / replicates if replicates else 0.0
    if len(values) < 2:
        return None, share
    return _percentile_interval(values, alpha), share


def sequential_policy_value(
    policy,
    holdout: list[CohortTrajectory],
    propensity=None,
    with_interval: bool = True,
) -> SequentialValue:
    """Self-normalised IPW for a dynamic regime, with cumulative weights.

    Stops contributing a trajectory the moment it deviates: after a disagreement
    the remaining stages are no longer on the regime's path, so including them
    would evaluate a history the regime never produces. `consistent_trajectories`
    reports how many survive each stage, which is the diagnostic that makes the
    effective sample legible — at this cohort size it falls off a cliff.

    The interval is a trajectory-clustered percentile bootstrap. Like
    `evaluation_intervals` it holds the policy fixed, so it measures the
    precision of this evaluation rather than the variability of the fit.
    """
    aggregates, surviving = _sequential_aggregates(policy, holdout, propensity)

    # The point estimate pools rows, the bootstrap resamples trajectories, and
    # both read the same prefixes — so the interval cannot end up describing a
    # different set of rows than the value it brackets.
    flat_weights: list[float] = []
    flat_outcomes: list[float] = []
    for (_, _, prefix), trajectory in zip(aggregates, holdout):
        for offset, weight in enumerate(prefix):
            flat_weights.append(weight)
            flat_outcomes.append(trajectory.stages[offset].outcome)

    value, ess = _hajek(flat_weights, flat_outcomes)
    interval, degenerate = (
        sequential_intervals(aggregates) if with_interval else (None, 0.0)
    )
    return SequentialValue(
        value=value,
        effective_sample_size=ess,
        matched_rows=len(flat_weights),
        consistent_trajectories=surviving,
        n_trajectories=len(holdout),
        max_weight=max(flat_weights) if flat_weights else 0.0,
        weights=weight_diagnostics(flat_weights),
        interval=interval,
        degenerate_resample_share=degenerate,
    )


@dataclass(frozen=True)
class SequentialDRValue:
    """The regime's value estimated doubly robustly, on the value-to-go scale.

    **Why the IPW version needed replacing rather than explaining.**
    `sequential_policy_value` throws a trajectory away at its first deviation, so
    on the deployed holdout five of 120 trajectories carry the whole estimate and
    the heaviest of them carries 35% of the weight. That is not a defect in the
    estimator — it is what pure inverse weighting costs over three decisions with
    six arms — but it is fixable, because the fitted Q-functions the agent
    already serves from can supply the value where a trajectory leaves the
    regime's path.

    The estimator is the standard doubly-robust backward recursion
    (Murphy 2001; Bang & Robins 2005; Jiang & Li 2016), run per trajectory:

        V_{T+1} = 0
        V_t     = Q(x_t, d(x_t))
                  + 1{a_t = d(x_t)} / pi_t(a_t | x_t) * (y_t + V_{t+1} - Q(x_t, a_t))

    Where the observed arm matches the regime the residual is weighted in; where
    it does not, the indicator is zero and the trajectory contributes
    `Q(x_t, d(x_t))` — the model's own value for the arm the regime would have
    given. So **every trajectory contributes**, and the cumulative propensity
    product never appears as an explicit weight. Measured on the deployed
    holdout:

        estimator                value      SE     95% interval    traj   ESS
        sequential DR (AIPW)    2.3278   0.0673   (2.196, 2.460)    120   120.0
        sequential IPW (Hajek)  2.2864   0.2345   (1.713, 2.542)      5     3.9

    against a known truth of **2.2938**. Both cover it; the DR interval is
    **3.1x narrower**, its effective sample is the whole holdout, and its
    heaviest trajectory carries **1.5-1.7%** of the estimate against the IPW
    estimate's **16.7%** — the same quantity, measured the same way. Both
    serving estimators land within **0.07 and 0.52 standard errors** of their
    own oracle.

    **Which truth, and this is the part to read twice.** The augmenting Q-model
    is fit with IPCW, so it estimates the value-to-go *had the patient stayed in
    care* — `rollout_value(policy, dropout=False)` = 2.2938. The oracle figure
    reported beside every policy value, `oracle_rollout_value`, is the other
    quantity: what a patient actually accrues once dropout is simulated, 2.1469.
    The 0.147 between them is retention, not error, and comparing this estimate
    against `oracle_rollout_value` would charge the estimator for it. That is
    invariant 14 in a new place, and it is why `oracle_uncensored_value` is
    carried here rather than left to a reader to find.

    Double robustness means consistent if *either* the Q-model or the propensity
    model is right. One honesty note on the Q-model: it is fit by backward
    induction under its own `max`, so it is Q^d only for the policy that is that
    model's own greedy rule. Used to evaluate a different policy it is an
    approximation, which costs efficiency and not consistency — the estimator
    still leans on the propensity model in that case.
    """

    value: float
    standard_error: float
    interval: tuple[float, float] | None
    n_trajectories: int
    matched_decisions: int
    total_decisions: int
    #: Share of the estimate carried by its single heaviest trajectory — the
    #: same quantity `WeightDiagnostics.top_share` reports, so the IPW and DR
    #: concentrations can be compared directly.
    max_influence_share: float
    augmentation_model: str
    oracle_uncensored_value: float | None = None

    @property
    def identified(self) -> bool:
        """Every trajectory contributes, so the question is precision, not support."""
        return self.n_trajectories >= MIN_SEQUENTIAL_EFFECTIVE_SAMPLE

    @property
    def covers_oracle(self) -> bool | None:
        if self.interval is None or self.oracle_uncensored_value is None:
            return None
        return self.interval[0] <= self.oracle_uncensored_value <= self.interval[1]

    def as_dict(self) -> dict[str, object]:
        return {
            "value": round(self.value, 4),
            "standard_error": round(self.standard_error, 4),
            "interval": [round(v, 4) for v in self.interval] if self.interval else None,
            "n_trajectories": self.n_trajectories,
            "matched_decisions": self.matched_decisions,
            "total_decisions": self.total_decisions,
            "max_influence_share": round(self.max_influence_share, 4),
            "augmentation_model": self.augmentation_model,
            "oracle_uncensored_value": self.oracle_uncensored_value,
            "covers_oracle": self.covers_oracle,
            "identified": self.identified,
            "scale": "expected total response over the horizon (value-to-go)",
            "note": (
                "Doubly-robust backward recursion: every trajectory contributes, "
                "with the fitted Q-function supplying the value wherever the "
                "observed arm left the regime's path. The estimand is the "
                "value-to-go under full follow-up, because the augmenting "
                "Q-model is IPCW-weighted — compare it against "
                "`oracle_uncensored_value`, never against `oracle_rollout_value`, "
                "which is what a patient accrues once dropout is simulated and is "
                "smaller by the retention gap."
            ),
        }


def sequential_dr_value(
    policy,
    holdout: list[CohortTrajectory],
    q_model,
    propensity=None,
    alpha: float = 0.05,
    augmentation_model: str = "",
    oracle_uncensored_value: float | None = None,
) -> SequentialDRValue:
    """Doubly-robust value of a dynamic regime. See `SequentialDRValue`.

    `q_model.raw_q(features, arm, stage_index)` must be a *value-to-go* on the
    sum scale — this stage's outcome plus the remaining horizon — which is what
    `QLearningModel.raw_q` is and what `DWOLSModel.raw_q` is not. Passing a
    single-visit outcome model here would silently estimate a different quantity
    at every stage but the last.

    Each trajectory yields one influence value, so they are i.i.d. across
    patients and the standard error is the plain one. No cluster correction is
    needed and none is applied: the trajectory *is* the cluster.
    """
    contributions: list[float] = []
    matched = 0
    total = 0

    for trajectory in holdout:
        value = 0.0
        # Backward, because V_t reads V_{t+1}. The agreement tally rides along
        # rather than taking a second pass: `policy` scores every arm on the
        # menu, so evaluating it twice per stage would put a few tenths of a
        # second of avoidable work on every cold start.
        for position in reversed(range(len(trajectory.stages))):
            stage = trajectory.stages[position]
            chosen = policy(stage.features, position)
            regime_q = q_model.raw_q(stage.features, chosen, position)
            if stage.arm == chosen:
                matched += 1
                residual = stage.outcome + value - q_model.raw_q(
                    stage.features, stage.arm, position
                )
                value = regime_q + residual / _propensity_of(stage, propensity)
            else:
                # The trajectory left the regime's path here. The model supplies
                # the value of the arm the regime would have given, which is the
                # whole reason this estimator keeps the patient at all.
                value = regime_q
        contributions.append(value)
        total += len(trajectory.stages)

    n = len(contributions)
    if n == 0:
        return SequentialDRValue(0.0, 0.0, None, 0, 0, 0, 0.0, augmentation_model)

    mean = sum(contributions) / n
    if n < 2:
        return SequentialDRValue(
            mean, 0.0, None, n, matched, total, 1.0, augmentation_model,
            oracle_uncensored_value,
        )
    variance = sum((value - mean) ** 2 for value in contributions) / (n - 1)
    standard_error = math.sqrt(variance / n)
    margin = normal_critical_value(alpha) * standard_error
    # The share of the estimate one trajectory carries — deliberately the same
    # quantity `WeightDiagnostics.top_share` reports for the IPW estimators, so
    # the two are comparable rather than merely both small-looking. Absolute
    # values because a heavily-weighted residual can push one contribution
    # negative (two do here), and a signed share would then flatter the
    # concentration. Uniform would be 1/120 = 0.83%; this runs 1.5-1.7% across
    # the serving ensemble against the IPW estimate's 16.7%.
    magnitudes = [abs(value) for value in contributions]
    spread = sum(magnitudes)
    return SequentialDRValue(
        value=mean,
        standard_error=standard_error,
        interval=(mean - margin, mean + margin),
        n_trajectories=n,
        matched_decisions=matched,
        total_decisions=total,
        max_influence_share=(max(magnitudes) / spread) if spread else 0.0,
        augmentation_model=augmentation_model,
        oracle_uncensored_value=oracle_uncensored_value,
    )


class _ScaledBlipQ:
    """A Q-function whose every blip is scaled by `1 + gamma`.

    The treatment-free surface is left alone deliberately: it cancels out of a
    contrast and is not what the claim under test rests on. Scaling the blip is
    the same axis `generate_ra_cohort(..., blip_modifier=)` and the transfer
    study's estimand shift bend, so a gamma here is comparable to a
    misspecification this repo has already priced.
    """

    def __init__(self, model, gamma: float) -> None:
        self.model = model
        self.gamma = gamma

    def raw_q(self, features: dict[str, float], arm: str, stage_index: int) -> float:
        return self.model.treatment_free(features, stage_index) + (
            1.0 + self.gamma
        ) * self.model.blip(arm, features, stage_index)


# How far the outcome model is perturbed. The range brackets the 1.5x estimand
# shift `cli transfer` applies, which is the largest misspecification this repo
# has a measured monitor for.
DR_SENSITIVITY_GAMMAS = (-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.4, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class SequentialDRSensitivity:
    """How wrong the Q-model has to be before the regime's advantage disappears.

    **Why this is the analysis the DR estimate needs.** Double robustness is
    consistent if *either* the outcome model or the propensity model is right,
    and on this holdout only one of those is checkable: the inverse-weighted
    estimate has an effective sample of 14.6 and cannot falsify anything. So the
    useful question is not whether the Q-model is right — nothing here can answer
    that — but how wrong it would have to be to matter.

    Misspecification is parameterised as a proportional error in every blip:
    `Q_gamma = treatment_free + (1 + gamma) * blip`. The regime under evaluation
    is held **fixed**, so this measures the error in the value estimate rather
    than quietly evaluating a different policy.

    Measured on the deployed holdout, against a behaviour policy worth 1.9032 on
    the same value-to-go scale:

        gamma    DR value    gain over behaviour      z
        +0.00     2.3278          +0.4246           6.31
        +0.25     2.2243          +0.3211           3.33
        +0.50     2.1208          +0.2176           1.42   <- gone
        +1.00     1.9139          +0.0107           0.04

    **The tipping point is near +0.4**, and where it lands is the point. A
    proportional blip error of 50% is exactly the shift `cli transfer` applies in
    its estimand-shifted row — the row where calibration rises to 0.051 against
    0.002-0.006 everywhere else while policy value gets *better*. So the
    misspecification that would overturn this claim is one the deployment monitor
    already detects, and it detects it through calibration rather than value.
    That is not a coincidence worth glossing: it is the same finding from the
    other end.

    One caveat on the comparison. The transfer row shifts the *truth* upward
    while the model stays put; gamma shifts the *model* while the truth stays
    put. Both are a 1.5x mismatch between the two, and calibration reads the gap
    between predicted and observed either way, but the direction is not identical
    and the magnitudes need not match exactly.

    A second reading, almost free: the standard error is minimised **near** gamma
    = 0 (0.067, against 0.15 at either +/-0.5). A badly wrong Q-model degrades
    the augmentation's precision as well as its centre, so the estimator's own
    interval carries a weak signal about the model it leans on.

    *Near*, not at. The minimum sits exactly at 0 only when the augmenting model
    is the evaluated policy's own; for the deployed pairing — dWOLS's regime
    scored with `Q-Pooled`'s value function — it lands at +0.1, because the best
    control variate for someone else's policy is not their unmodified one. That
    is the efficiency cost the borrowing already carries, showing up in a second
    place.
    """

    gammas: tuple[float, ...]
    values: tuple[float, ...]
    standard_errors: tuple[float, ...]
    benchmark: float
    tipping_point: float | None
    alpha: float = 0.05

    @property
    def rows(self) -> tuple[dict[str, object], ...]:
        critical = normal_critical_value(self.alpha)
        out = []
        for gamma, value, se in zip(self.gammas, self.values, self.standard_errors):
            gain = value - self.benchmark
            out.append(
                {
                    "gamma": round(gamma, 3),
                    "value": round(value, 4),
                    "standard_error": round(se, 4),
                    "gain_over_benchmark": round(gain, 4),
                    "gain_interval": [
                        round(gain - critical * se, 4),
                        round(gain + critical * se, 4),
                    ],
                    "separated_from_zero": bool(gain - critical * se > 0.0),
                }
            )
        return tuple(out)

    def as_dict(self) -> dict[str, object]:
        return {
            "benchmark": round(self.benchmark, 4),
            "parameterisation": (
                "every estimated blip scaled by (1 + gamma); the treatment-free "
                "surface and the evaluated regime are both held fixed"
            ),
            "tipping_point": (
                round(self.tipping_point, 3) if self.tipping_point is not None else None
            ),
            "rows": list(self.rows),
            "note": (
                "How wrong the outcome model would have to be before the regime's "
                "advantage over the behaviour policy stops being separated from "
                "zero. This matters because the doubly-robust estimate leans on "
                "that model and the inverse-weighted estimate that could falsify "
                "it has an effective sample of 14.6. A tipping point near +0.4 "
                "sits at the same magnitude as `cli transfer`'s estimand-shifted "
                "row, where calibration rises an order of magnitude while policy "
                "value improves — so the misspecification that would overturn "
                "this claim is one calibration already catches and value does not."
            ),
        }


def sequential_dr_sensitivity(
    policy,
    holdout: list[CohortTrajectory],
    q_model,
    benchmark: float,
    propensity=None,
    gammas: tuple[float, ...] = DR_SENSITIVITY_GAMMAS,
    alpha: float = 0.05,
) -> SequentialDRSensitivity:
    """Sweep proportional outcome-model error. See `SequentialDRSensitivity`.

    `benchmark` is the value the regime is being claimed to beat, on the same
    value-to-go scale — `training.behaviour_uncensored_value()` supplies it, and
    like every rollout here it is simulation-only.
    """
    values: list[float] = []
    errors: list[float] = []
    for gamma in gammas:
        estimate = sequential_dr_value(
            policy,
            holdout,
            _ScaledBlipQ(q_model, gamma) if gamma else q_model,
            propensity,
            alpha=alpha,
        )
        values.append(estimate.value)
        errors.append(estimate.standard_error)

    return SequentialDRSensitivity(
        gammas=tuple(gammas),
        values=tuple(values),
        standard_errors=tuple(errors),
        benchmark=benchmark,
        tipping_point=_tipping_point(
            policy, holdout, q_model, benchmark, propensity, alpha
        ),
        alpha=alpha,
    )


def _tipping_point(
    policy, holdout, q_model, benchmark, propensity, alpha, upper: float = 3.0
) -> float | None:
    """Smallest positive gamma at which the gain stops excluding zero.

    Bisected rather than read off the sweep, because the sweep's grid is chosen
    for legibility and the tipping point should not move when someone adds a row
    to it. `None` means the claim survives every perturbation up to `upper`,
    which is a tripling of every effect and well past useful.
    """
    critical = normal_critical_value(alpha)

    def separated(gamma: float) -> bool:
        estimate = sequential_dr_value(
            policy,
            holdout,
            _ScaledBlipQ(q_model, gamma) if gamma else q_model,
            propensity,
            alpha=alpha,
        )
        return estimate.value - benchmark - critical * estimate.standard_error > 0.0

    if not separated(0.0):
        return 0.0
    if separated(upper):
        return None
    low, high = 0.0, upper
    for _ in range(14):
        middle = (low + high) / 2.0
        if separated(middle):
            low = middle
        else:
            high = middle
    return high


def _trajectory_aggregates(policy, holdout: list[CohortTrajectory], propensity=None):
    """Per-trajectory partial sums for the Hajek value and the behaviour mean.

    The policy is already fitted and the stages do not change, so which arm it
    picks at each visit is the same in every resample. Evaluating it once per
    stage here — rather than once per stage *per replicate* — is what makes the
    bootstrap below cost milliseconds instead of seconds, and it is exact: the
    resample only ever re-weights whole trajectories, and every quantity in the
    estimator is a sum over stages.
    """
    aggregates = []
    for trajectory in holdout:
        sum_w = sum_wy = sum_y = 0.0
        n_stages = 0
        for position, stage in enumerate(trajectory.stages):
            sum_y += stage.outcome
            n_stages += 1
            if policy(stage.features, position) == stage.arm:
                weight = 1.0 / _propensity_of(stage, propensity)
                sum_w += weight
                sum_wy += weight * stage.outcome
        aggregates.append((sum_w, sum_wy, sum_y, n_stages))
    return aggregates


def evaluation_intervals(
    policy,
    holdout: list[CohortTrajectory],
    replicates: int = EVALUATION_REPLICATES,
    seed: int = EVALUATION_SEED,
    alpha: float = 0.05,
    propensity=None,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Percentile intervals for the held-out policy value and its improvement.

    Trajectories are resampled with replacement — clustering by patient, for the
    same reason the sandwich does — and the whole Hajek calculation is redone on
    each resample. The policy itself is *not* refitted, so this measures the
    precision of the held-out **evaluation**, not the sampling variability of the
    fit. That is the right quantity for the question it answers: given this
    holdout, is the measured gain over the behaviour policy distinguishable from
    zero, and are two estimators' values distinguishable from each other?

    The fit's own variability is a larger and separate question, and the honest
    answer to it is `cli stability`, which refits across seeds. This is the part
    that is cheap enough to compute on every process start.
    """
    import random

    n = len(holdout)
    if n < 2:
        return None
    aggregates = _trajectory_aggregates(policy, holdout, propensity)
    rng = random.Random(seed)
    values: list[float] = []
    improvements: list[float] = []
    for _ in range(replicates):
        total_w = total_wy = total_y = 0.0
        total_stages = 0
        for _ in range(n):
            sum_w, sum_wy, sum_y, n_stages = aggregates[rng.randrange(n)]
            total_w += sum_w
            total_wy += sum_wy
            total_y += sum_y
            total_stages += n_stages
        if total_w <= 0.0 or total_stages == 0:
            continue
        value = total_wy / total_w
        values.append(value)
        improvements.append(value - total_y / total_stages)
    if len(values) < 2:
        return None
    return _percentile_interval(values, alpha), _percentile_interval(improvements, alpha)


def _percentile_interval(draws: list[float], alpha: float) -> tuple[float, float]:
    ordered = sorted(draws)
    return (_quantile(ordered, alpha / 2.0), _quantile(ordered, 1.0 - alpha / 2.0))


def _quantile(ordered: list[float], q: float) -> float:
    import math

    position = q * (len(ordered) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def evaluate_policy(
    estimator: str,
    policy,
    predict_outcome,
    holdout: list[CohortTrajectory],
    calibration_bins: int = 10,
    oracle_rollout_value: float | None = None,
    with_intervals: bool = True,
    propensity=None,
    q_model=None,
    augmentation_model: str = "",
    oracle_uncensored_value: float | None = None,
) -> PolicyScore:
    """Score `policy` on held-out trajectories.

    `policy(features, stage_index) -> arm` is the decision rule under test.
    `predict_outcome(features, arm, stage_index) -> float` is the model's
    predicted outcome, used only for calibration.
    """
    matched_weights: list[float] = []
    matched_outcomes: list[float] = []
    observed_outcomes: list[float] = []
    predictions: list[float] = []
    agreements = 0
    optimal_hits = 0
    total = 0

    for position, stage in _holdout_rows(holdout):
        total += 1
        chosen = policy(stage.features, position)
        observed_outcomes.append(stage.outcome)
        predictions.append(predict_outcome(stage.features, stage.arm, position))
        if chosen == optimal_arm(stage.features):
            optimal_hits += 1
        if chosen == stage.arm:
            agreements += 1
            matched_weights.append(1.0 / _propensity_of(stage, propensity))
            matched_outcomes.append(stage.outcome)

    if matched_weights:
        total_weight = sum(matched_weights)
        ipw_value = sum(w * y for w, y in zip(matched_weights, matched_outcomes)) / total_weight
        ess = (total_weight ** 2) / sum(w * w for w in matched_weights)
    else:
        ipw_value = 0.0
        ess = 0.0

    behaviour_value = sum(observed_outcomes) / total if total else 0.0
    calibration = CalibrationEvaluator().evaluate(
        predicted_probabilities=predictions,
        observed_outcomes=observed_outcomes,
        bins=calibration_bins,
    )

    sequential = sequential_policy_value(policy, holdout, propensity)
    # The doubly-robust counterpart. Optional because it needs a value-to-go
    # Q-function to augment with, and only the Q-learning models have one —
    # dWOLS's `raw_q` is a single-visit outcome.
    sequential_dr = (
        sequential_dr_value(
            policy,
            holdout,
            q_model,
            propensity,
            augmentation_model=augmentation_model or type(q_model).__name__,
            oracle_uncensored_value=oracle_uncensored_value,
        )
        if q_model is not None
        else None
    )

    notes: list[str] = []
    if ess < 30:
        notes.append("effective sample size is small; the IPW value is high-variance")
    if not sequential.identified:
        span = (
            f" (95% interval {sequential.interval[0]:.4f} to "
            f"{sequential.interval[1]:.4f}, width {sequential.interval_width:.3f})"
            if sequential.interval
            else " (no interval: the resample could not produce two usable draws)"
        )
        notes.append(
            f"ipw_policy_value is per-decision; the regime's own value is "
            f"{sequential.value:.4f}{span} at an effective sample of "
            f"{sequential.effective_sample_size:.1f}, which is not enough to read "
            f"as an estimate — only {sequential.consistent_trajectories[-1]} of "
            f"{sequential.n_trajectories} trajectories follow the regime to the end"
        )
    if agreements == 0:
        notes.append("policy never agreed with the observed arm; IPW value is not identified")

    intervals = (
        evaluation_intervals(policy, holdout, propensity=propensity)
        if with_intervals
        else None
    )
    value_interval, improvement_interval = intervals if intervals else (None, None)
    if improvement_interval is not None and improvement_interval[0] <= 0.0:
        notes.append(
            "the gain over the behaviour policy is not separated from zero on this holdout"
        )

    return PolicyScore(
        estimator=estimator,
        sequential=sequential,
        sequential_dr=sequential_dr,
        weights=weight_diagnostics(matched_weights),
        ipw_policy_value=round(ipw_value, 4),
        behaviour_value=round(behaviour_value, 4),
        effective_sample_size=round(ess, 2),
        agreement_rate=round(agreements / total, 4) if total else 0.0,
        optimal_arm_rate=round(optimal_hits / total, 4) if total else 0.0,
        calibration=calibration,
        n_holdout_stages=total,
        oracle_rollout_value=oracle_rollout_value,
        notes=notes,
        value_interval=value_interval,
        improvement_interval=improvement_interval,
    )
