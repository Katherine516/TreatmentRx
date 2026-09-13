"""Randomized-trial analyses, separated from individual DTR recommendations."""

from treatmentrx.trials.precision import (
    ContinuousTrialRecord,
    LinearTrialEstimate,
    PrecisionComparison,
    PrognosticAdjustmentContract,
    RandomizedTrialPrecisionAnalyzer,
    TrialPrecisionError,
    continuous_trial_estimand,
    simulate_precision_power,
)

__all__ = [
    "ContinuousTrialRecord",
    "LinearTrialEstimate",
    "PrecisionComparison",
    "PrognosticAdjustmentContract",
    "RandomizedTrialPrecisionAnalyzer",
    "TrialPrecisionError",
    "continuous_trial_estimand",
    "simulate_precision_power",
]
