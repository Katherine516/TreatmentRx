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

Per-stage rather than per-trajectory weighting is a documented simplification:
each decision point is weighted by its own propensity, which is the right
estimand for "how good is the arm choice at this visit" and avoids the variance
blow-up of multiplying propensities down a trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from treatmentrx.decision.calibration import CalibrationEvaluator
from treatmentrx.domain import CalibrationReport
from treatmentrx.simulation.ra_cohort import CohortTrajectory, optimal_arm

# Propensity floor for the weights; mirrors the clipping used when fitting.
_PROPENSITY_FLOOR = 0.02


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
    notes: list[str] = field(default_factory=list)

    @property
    def improvement(self) -> float:
        """Estimated per-stage gain over the clinician policy in the data."""
        return round(self.ipw_policy_value - self.behaviour_value, 4)

    def as_dict(self) -> dict[str, object]:
        return {
            "estimator": self.estimator,
            "ipw_policy_value": self.ipw_policy_value,
            "behaviour_value": self.behaviour_value,
            "improvement": self.improvement,
            "effective_sample_size": self.effective_sample_size,
            "agreement_rate": self.agreement_rate,
            "optimal_arm_rate": self.optimal_arm_rate,
            "expected_calibration_error": self.calibration.expected_calibration_error,
            "calibration_passed": self.calibration.passed,
            "n_holdout_stages": self.n_holdout_stages,
            "oracle_rollout_value": self.oracle_rollout_value,
        }


def _holdout_rows(cohort: list[CohortTrajectory]):
    for trajectory in cohort:
        for position, stage in enumerate(trajectory.stages):
            yield position, stage


def evaluate_policy(
    estimator: str,
    policy,
    predict_outcome,
    holdout: list[CohortTrajectory],
    calibration_bins: int = 10,
    oracle_rollout_value: float | None = None,
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
            matched_weights.append(1.0 / max(stage.propensity, _PROPENSITY_FLOOR))
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

    notes: list[str] = []
    if ess < 30:
        notes.append("effective sample size is small; the IPW value is high-variance")
    if agreements == 0:
        notes.append("policy never agreed with the observed arm; IPW value is not identified")

    return PolicyScore(
        estimator=estimator,
        ipw_policy_value=round(ipw_value, 4),
        behaviour_value=round(behaviour_value, 4),
        effective_sample_size=round(ess, 2),
        agreement_rate=round(agreements / total, 4) if total else 0.0,
        optimal_arm_rate=round(optimal_hits / total, 4) if total else 0.0,
        calibration=calibration,
        n_holdout_stages=total,
        oracle_rollout_value=oracle_rollout_value,
        notes=notes,
    )
