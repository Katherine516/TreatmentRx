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
