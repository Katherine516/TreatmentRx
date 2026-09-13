"""Randomized precision analysis is useful but cannot recommend treatment."""

import unittest

from treatmentrx.scientific import ScientificMode, ra_dtr_estimand
from treatmentrx.trials import (
    ContinuousTrialRecord,
    PrognosticAdjustmentContract,
    RandomizedTrialPrecisionAnalyzer,
    TrialPrecisionError,
    continuous_trial_estimand,
    simulate_precision_power,
)


def records(n=80):
    result = []
    for index in range(n):
        treatment = index % 2
        score = (index - n / 2) / 12.0
        noise = ((index * 17) % 11 - 5) / 10.0
        outcome = 0.35 * treatment + 0.8 * score + noise
        result.append(ContinuousTrialRecord(outcome, treatment, score))
    return result


def contract(endpoint="365-day response", horizon_days=365):
    return PrognosticAdjustmentContract(
        artifact_id="external-prognostic-score",
        artifact_version="1.0",
        endpoint=endpoint,
        horizon_days=horizon_days,
        reference_treatment="control",
        independence_strategy="external_frozen",
        externally_validated=True,
    )


class TrialPrecisionTests(unittest.TestCase):
    def test_prognostic_adjustment_reduces_standard_error(self):
        result = RandomizedTrialPrecisionAnalyzer().analyze(
            records(),
            continuous_trial_estimand("365-day response", 365),
            contract(),
            planning_effect=0.35,
        )
        self.assertLess(result.adjusted.standard_error, result.unadjusted.standard_error)
        self.assertLess(result.required_sample_size_adjusted, result.required_sample_size_unadjusted)
        self.assertFalse(result.as_dict()["can_recommend_individual_treatment"])

    def test_dtr_estimand_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "randomized_trial"):
            RandomizedTrialPrecisionAnalyzer().analyze(
                records(),
                ra_dtr_estimand(("continue-current", "other")),
                contract(),
            )

    def test_each_arm_needs_data(self):
        invalid = [ContinuousTrialRecord(float(i), 0, float(i)) for i in range(8)]
        with self.assertRaises(TrialPrecisionError):
            RandomizedTrialPrecisionAnalyzer().analyze(
                invalid,
                continuous_trial_estimand("outcome", 30),
                contract("outcome", 30),
            )

    def test_endpoint_mismatch_fails_closed(self):
        with self.assertRaisesRegex(TrialPrecisionError, "endpoint"):
            RandomizedTrialPrecisionAnalyzer().analyze(
                records(),
                continuous_trial_estimand("365-day response", 365),
                contract(endpoint="different endpoint"),
            )

    def test_null_simulation_is_deterministic_and_reports_monte_carlo_error(self):
        first = simulate_precision_power(60, 0.0, 1.0, 1.0, replicates=80, seed=7)
        second = simulate_precision_power(60, 0.0, 1.0, 1.0, replicates=80, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["operating_mode"], ScientificMode.RANDOMIZED_TRIAL.value)
        self.assertIn("prognostic_adjusted", first["monte_carlo_standard_error"])
        self.assertFalse(first["can_recommend_individual_treatment"])


if __name__ == "__main__":
    unittest.main()
