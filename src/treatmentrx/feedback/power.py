"""How much data does it take before the agent will actually recommend?

The most visible property of this system is how often it declines: at the
deployed training size it abstains on roughly four fifths of the patients it sees. Read
without context that looks like a defect, and the tempting fix — lowering the
action bar or narrowing the interval — would be the wrong one, because the
abstention is *earned*: the patients it declines to separate really do have
close arms (`feedback/audit.audit_decision`).

The right question is not whether to abstain less but what it would cost to be
able to. That is a sample-size question, and this module answers it by refitting
the whole ensemble at a range of cohort sizes and measuring the decision the
system actually makes.

**The relationship was invisible until recently.** dWOLS kept its own module-level
cache, so `training.reset()` refit the Q-learning half of the serving ensemble and
left the dWOLS half at whatever cohort it first saw. The averaged standard error
then appeared to shrink at n^-0.15 — near enough to flat to suggest a floor in the
method. With one owner for the fit it shrinks at n^-0.45, close to the sqrt(n) a
correctly specified estimator is entitled to, and abstention falls steeply with n.

Everything here refits from scratch, so it is a CLI command, never something on
the inference path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from treatmentrx.data import DataContractError, DataLayer
from treatmentrx.domain import RecommendationStatus
from treatmentrx.simulation.fhir_export import simulated_bundles

# Cohort sizes to sweep, before the holdout split. The deployed default sits in
# the middle so the curve brackets it on both sides.
DEFAULT_SIZES = (140, 280, 400, 800, 1600)

# Sixty scored patients was too few for a stable headline abstention estimate.
# The 240-patient reference evaluation keeps Monte Carlo uncertainty smaller;
# its result must be regenerated whenever the decision threshold or its
# multiplicity correction changes.
DEFAULT_PATIENTS = 240
DEFAULT_SEED = 909

# What fraction of patients a deployment might want a recommendation for. Not a
# clinical standard — a reference point, so "how much data" has an answer.
TARGET_ABSTENTION = 0.30


@dataclass(frozen=True)
class PowerPoint:
    """One training size, and what the decision layer does at it."""

    cohort_size: int
    train_size: int
    patients: int
    equipoise_rate: float
    mean_standard_error: float
    mean_abs_difference: float

    @property
    def mean_z(self) -> float:
        if self.mean_standard_error <= 0.0:
            return 0.0
        return self.mean_abs_difference / self.mean_standard_error

    def as_dict(self) -> dict[str, object]:
        return {
            "cohort_size": self.cohort_size,
            "train_size": self.train_size,
            "patients_scored": self.patients,
            "equipoise_rate": round(self.equipoise_rate, 4),
            "recommend_rate": round(1.0 - self.equipoise_rate, 4),
            "mean_contrast_standard_error": round(self.mean_standard_error, 5),
            "mean_abs_contrast": round(self.mean_abs_difference, 5),
            "mean_z": round(self.mean_z, 3),
        }


def _measure(states, decision_layer, estimation_layer) -> tuple[float, float, float]:
    equipoise = 0
    errors: list[float] = []
    differences: list[float] = []
    for state in states:
        decision = decision_layer.decide(state, estimation_layer.estimate(state))
        if decision.status is RecommendationStatus.EQUIPOISE:
            equipoise += 1
        if decision.contrast is not None:
            errors.append(decision.contrast.standard_error)
            differences.append(abs(decision.contrast.difference))
    if not errors:
        return equipoise / max(len(states), 1), 0.0, 0.0
    return (
        equipoise / len(states),
        sum(errors) / len(errors),
        sum(differences) / len(differences),
    )


def power_curve(
    sizes: tuple[int, ...] = DEFAULT_SIZES,
    n_patients: int = DEFAULT_PATIENTS,
    seed: int = DEFAULT_SEED,
) -> list[PowerPoint]:
    """Refit the ensemble at each cohort size and score the same patients.

    The patient set is held fixed across sizes — the same people, a differently
    sized training cohort — so the only thing moving is how well the parameters
    are determined. Building the states once also keeps Layer 1 out of the
    comparison entirely.
    """
    from treatmentrx.decision import DecisionLayer
    from treatmentrx.estimation import EstimationLayer, training

    data_layer = DataLayer()
    states = []
    for bundle in simulated_bundles(n_patients, seed=seed):
        try:
            states.append(data_layer.build_patient_state(bundle))
        except DataContractError:
            continue
    if not states:
        raise ValueError("no patients survived the data contract")

    original = training.COHORT_SIZE
    points: list[PowerPoint] = []
    try:
        for size in sizes:
            training.COHORT_SIZE = size
            training.reset()
            equipoise, error, difference = _measure(
                states, DecisionLayer(), EstimationLayer()
            )
            points.append(
                PowerPoint(
                    cohort_size=size,
                    train_size=len(training.fitted().train),
                    patients=len(states),
                    equipoise_rate=equipoise,
                    mean_standard_error=error,
                    mean_abs_difference=difference,
                )
            )
    finally:
        # The sweep mutates a module global; a caller that runs this in the same
        # process as anything else must get the default fit back.
        training.COHORT_SIZE = original
        training.reset()
    return points


def shrinkage_exponent(points: list[PowerPoint]) -> float | None:
    """Fitted `p` in `SE ~ n^-p`, by least squares on the log-log points.

    A correctly specified estimator earns p = 0.5. Materially below that means
    something in the interval is not responding to sample size — which is what a
    stale half of the ensemble looked like, at p = 0.15.
    """
    usable = [p for p in points if p.mean_standard_error > 0.0 and p.train_size > 0]
    if len(usable) < 2:
        return None
    xs = [math.log(p.train_size) for p in usable]
    ys = [math.log(p.mean_standard_error) for p in usable]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator <= 0.0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
    return -slope


def required_train_size(points: list[PowerPoint], target: float = TARGET_ABSTENTION) -> int | None:
    """Training size at which the abstention rate would reach `target`.

    Interpolated on log(n) between the two measured points that bracket the
    target, and extrapolated from the last pair only when the curve has not
    reached it. Extrapolation is the weaker half of this and is labelled as such
    in the report — the honest use is "roughly this order", not a study size.
    """
    usable = sorted(points, key=lambda p: p.train_size)
    if len(usable) < 2:
        return None
    for earlier, later in zip(usable, usable[1:]):
        if earlier.equipoise_rate >= target >= later.equipoise_rate:
            span = earlier.equipoise_rate - later.equipoise_rate
            if span <= 0:
                return later.train_size
            fraction = (earlier.equipoise_rate - target) / span
            log_n = math.log(earlier.train_size) + fraction * (
                math.log(later.train_size) - math.log(earlier.train_size)
            )
            return int(round(math.exp(log_n)))
    if usable[-1].equipoise_rate > target:
        earlier, later = usable[-2], usable[-1]
        span = earlier.equipoise_rate - later.equipoise_rate
        if span <= 0:
            return None
        steps = (later.equipoise_rate - target) / span
        ratio = math.log(later.train_size) - math.log(earlier.train_size)
        return int(round(math.exp(math.log(later.train_size) + steps * ratio)))
    return usable[0].train_size


def final_test_feasibility(
    held_back: tuple[float, ...] = (0.25, 0.4, 0.5),
) -> dict[str, object]:
    """Could this cohort afford a locked final test? Measured, not asserted.

    `training.evaluation_partition()` reports the partition this build has — 280
    training, 120 evaluation, **no final test** — and says the evaluation split
    does four jobs at once. The obvious remedy is a three-way split. This prices
    it before anyone does it.

    The two halves pull against each other. Carve off enough to confirm something
    and the evaluation split, which sets the model-averaging weights and selects
    the serving estimator, gets noisier. Carve off little enough to leave that
    intact and the final test cannot confirm anything. Measured on the deployed
    holdout:

        split                       n    per-decision ESS   sequential ESS
        today, all 120 evaluate   120          73.0              14.0
        hold back 25%: evaluate    90          52.7              10.5
        hold back 25%: FINAL       30          20.4               5.4
        hold back 40%: FINAL       48          30.3               6.2
        hold back 50%: FINAL       60          39.4               7.5

    **The answer depends on which quantity you want confirmed, and the two
    disagree.** A final test large enough to identify the *per-decision* value
    does exist: hold back 40% and it reaches 30.3 against
    `MIN_OPE_EFFECTIVE_SAMPLE = 30`. But that is bought by taking the evaluation
    split from 73 to 42.9, and the per-decision value is not what the agent
    deploys.

    For the **regime's own** value — the sequential quantity invariant 39 is
    about — no split works at all: 5.4, 6.2, 7.5, against the same threshold of
    30. The confirmation would itself be unidentified, and `identified` is the
    field this repo already uses to refuse exactly that reading.

    So: affordable for the easier quantity at a real cost, not affordable for the
    one that matters. The useful output is the size at which that changes.
    Effective sample runs roughly linearly in the split size here, so the
    requirement is arithmetic once the two constraints are written down: keep
    today's evaluation precision, and give the final test enough sample to be
    identified on the quantity being confirmed.

    This is `cli power` asking its own question about a different resource. That
    study says ~1,490 trajectories buy 30% abstention; this one says a cohort
    several times the current size is what buys a test set worth locking. Both
    are statements about data rather than method.
    """
    from treatmentrx.estimation import training
    from treatmentrx.feedback.offline_evaluation import (
        MIN_SEQUENTIAL_EFFECTIVE_SAMPLE,
        evaluate_policy,
        sequential_policy_value,
    )

    fit = training.fitted()
    model = fit.pooled
    policy = model.greedy_policy()
    propensity = fit.propensity.propensity
    holdout = fit.holdout
    threshold = MIN_SEQUENTIAL_EFFECTIVE_SAMPLE

    def measure(subset):
        if len(subset) < 5:
            return None
        score = evaluate_policy(
            "Q-Pooled",
            policy,
            model.predict_outcome,
            subset,
            propensity=propensity,
            with_intervals=False,
            q_model=model,
        )
        sequential = sequential_policy_value(
            policy, subset, propensity, with_interval=False
        )
        return {
            "n": len(subset),
            "per_decision_ess": round(score.effective_sample_size, 1),
            "sequential_ess": round(sequential.effective_sample_size, 1),
            "per_decision_identified": score.effective_sample_size >= threshold,
            "sequential_identified": sequential.identified,
        }

    baseline = measure(holdout)
    splits = []
    for fraction in held_back:
        cut = int(len(holdout) * (1.0 - fraction))
        evaluation, final = measure(holdout[:cut]), measure(holdout[cut:])
        splits.append(
            {
                "held_back_fraction": fraction,
                "evaluation": evaluation,
                "final_test": final,
                # The point of the whole study: a final test that cannot identify
                # the quantity it is meant to confirm is not a confirmation.
                "final_test_identifies_per_decision_value": bool(
                    final and final["per_decision_identified"]
                ),
                # The quantity that actually matters: the agent deploys a
                # regime, not a per-decision rule.
                "final_test_identifies_the_regime": bool(
                    final and final["sequential_identified"]
                ),
            }
        )

    # Effective sample runs close to linear in the split size on this holdout, so
    # the two constraints reduce to arithmetic. Stated as a rate rather than a
    # fitted curve because three points do not earn a curve.
    per_decision_rate = baseline["per_decision_ess"] / len(holdout)
    sequential_rate = baseline["sequential_ess"] / len(holdout)
    fraction = training.HOLDOUT_FRACTION
    needed_per_decision = math.ceil(
        (len(holdout) + threshold / per_decision_rate) / fraction
    )
    needed_sequential = math.ceil(
        (len(holdout) + threshold / sequential_rate) / fraction
    )

    return {
        "cohort_size": training.COHORT_SIZE,
        "holdout_trajectories": len(holdout),
        "identification_threshold": threshold,
        "today": baseline,
        "splits": splits,
        "affordable_for_the_per_decision_value": any(
            s["final_test_identifies_per_decision_value"] for s in splits
        ),
        "affordable_for_the_regime": any(
            s["final_test_identifies_the_regime"] for s in splits
        ),
        "cohort_for_identified_final_test": {
            "per_decision_value": needed_per_decision,
            "sequential_value": needed_sequential,
            "assumption": (
                "effective sample scales linearly in the split size, and the "
                "evaluation split keeps the "
                f"{len(holdout)} trajectories it has today"
            ),
        },
        "verdict": (
            f"Not for the quantity that matters. Holding back 40-50% of the "
            f"{len(holdout)}-trajectory holdout does give a final test that "
            f"identifies the *per-decision* value, but it costs the evaluation "
            f"split its precision (effective sample 73 down to 43) and the "
            f"per-decision value is not what the agent deploys. For the regime's "
            f"own value no split works: every candidate final test lands at 5-8 "
            f"against a threshold of {threshold:.0f}, so the confirmation would "
            f"itself be unidentified. Keeping today's evaluation precision and "
            f"adding an identified final test needs roughly "
            f"{needed_per_decision:,} trajectories for the per-decision value and "
            f"{needed_sequential:,} for the regime's own — against "
            f"{training.COHORT_SIZE} today. Reported rather than acted on: "
            f"`COHORT_SIZE` is a stated choice near the low end, and raising it "
            f"changes the headline abstention rate, which is a separate decision."
        ),
    }


def power_report(
    sizes: tuple[int, ...] = DEFAULT_SIZES,
    n_patients: int = DEFAULT_PATIENTS,
    seed: int = DEFAULT_SEED,
    target: float = TARGET_ABSTENTION,
) -> dict[str, object]:
    from treatmentrx.estimation import training

    points = power_curve(sizes, n_patients, seed)
    exponent = shrinkage_exponent(points)
    needed = required_train_size(points, target)
    deployed = next(
        (p for p in points if p.cohort_size == training.COHORT_SIZE), None
    )
    extrapolated = needed is not None and needed > max(p.train_size for p in points)
    return {
        "patients_scored": points[0].patients if points else 0,
        "deployed_train_size": len(training.fitted().train),
        "curve": [point.as_dict() for point in points],
        "shrinkage_exponent": round(exponent, 3) if exponent is not None else None,
        "target_abstention": target,
        "train_size_for_target": needed,
        "target_is_extrapolated": extrapolated,
        # The same question about a different resource: this study prices sample
        # size against abstention, that one prices it against having a test set
        # worth locking. `evaluation_partition()` reports the absence; this says
        # what closing it would take.
        "final_test_feasibility": final_test_feasibility(),
        "verdict": _verdict(points, exponent, needed, deployed, target, extrapolated),
    }


def _verdict(points, exponent, needed, deployed, target, extrapolated) -> str:
    if not points:
        return "no measurable points"
    lines = []
    if deployed is not None:
        lines.append(
            f"At the deployed training size ({deployed.train_size}) the agent "
            f"recommends for {1 - deployed.equipoise_rate:.0%} of patients and "
            f"abstains for {deployed.equipoise_rate:.0%}."
        )
    first, last = points[0], points[-1]
    lines.append(
        f"Abstention falls from {first.equipoise_rate:.0%} at n={first.train_size} "
        f"to {last.equipoise_rate:.0%} at n={last.train_size}, while the mean "
        f"contrast stays flat ({first.mean_abs_difference:.3f} to "
        f"{last.mean_abs_difference:.3f}) — the effect is not changing, the "
        f"precision is."
    )
    if exponent is not None:
        if exponent < 0.35:
            lines.append(
                f"The standard error shrinks at only n^-{exponent:.2f} against the "
                f"n^-0.50 a correctly specified estimator earns. Something in the "
                f"interval is not responding to sample size; check that every "
                f"serving estimator is actually being refit."
            )
        else:
            lines.append(
                f"The standard error shrinks at n^-{exponent:.2f}, close to the "
                f"n^-0.50 a correctly specified estimator earns, so abstention is a "
                f"sample-size choice rather than a property of the method."
            )
    if needed is not None:
        qualifier = " (extrapolated beyond the measured range)" if extrapolated else ""
        lines.append(
            f"Reaching {target:.0%} abstention needs roughly {needed} training "
            f"trajectories{qualifier}."
        )
    return " ".join(lines)


__all__ = [
    "DEFAULT_PATIENTS",
    "DEFAULT_SEED",
    "DEFAULT_SIZES",
    "TARGET_ABSTENTION",
    "PowerPoint",
    "power_curve",
    "power_report",
    "required_train_size",
    "shrinkage_exponent",
]
