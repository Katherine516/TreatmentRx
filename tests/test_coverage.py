"""Empirical coverage of the intervals, and the audit harness itself.

Coverage is deliberately measured at small replication counts here — the full
study is `treatmentrx.cli coverage`. The assertions are the ones that hold
robustly at this precision, not the point estimates.
"""

import unittest

from treatmentrx.estimation.q_learning import DEFAULT_BLIP_RIDGE, QLearningModel
from treatmentrx.feedback import coverage
from treatmentrx.feedback.audit import (
    audit_decision,
    audit_explanation,
    audit_governance,
    audit_ingestion,
    audit_safety,
)
from treatmentrx.simulation.ra_cohort import generate_ra_cohort


class CoverageStudyTests(unittest.TestCase):
    """One study shared across the assertions — each replication is a full refit.

    Deliberately small: the full study is `treatmentrx.cli coverage`. What is
    asserted here is the direction and mechanism, which hold at this size; the
    point estimates need the CLI's replication count to be worth quoting.
    """

    @classmethod
    def setUpClass(cls):
        cls.result = coverage.sandwich_coverage(replications=25, n=180, share_blip=False)

    def test_the_reference_estimand_is_a_real_effect(self):
        """A coverage study against a zero effect would prove nothing."""
        self.assertGreater(coverage.reference_truth(), 0.05)

    def test_the_sandwich_standard_error_is_too_small(self):
        """The finding this study exists to record.

        The sandwich treats the pseudo-outcomes as fixed, so it reports about
        80% of the estimator's actual spread. Asserted on the SE/SD ratio rather
        than the coverage tally: coverage at thirty replications is a proportion
        of thirty Bernoulli draws and swings several points on noise, while the
        ratio is a quotient of two means and barely moves.
        """
        self.assertLess(self.result.se_to_sd_ratio, 0.95)
        self.assertGreater(self.result.se_to_sd_ratio, 0.5)
        self.assertLess(self.result.coverage, coverage.NOMINAL)
        self.assertIn("too narrow", coverage.verdict([self.result]))

    def test_the_default_penalty_keeps_bias_small(self):
        """Which is why the default is 0.25 rather than 1.0."""
        self.assertLess(abs(self.result.bias), 0.02)

    def test_coverage_result_reports_its_own_monte_carlo_error(self):
        self.assertGreater(self.result.monte_carlo_error, 0.0)
        self.assertLess(self.result.monte_carlo_error, 0.2)


class RidgeDefaultTests(unittest.TestCase):
    def test_the_default_penalty_beats_the_previous_one_at_small_n(self):
        """78 blip parameters on 88 trajectories is what the bootstrap resamples to."""
        from treatmentrx.simulation.ra_cohort import REFERENCE_ARM, TREATMENT_ARMS, TRUE_BLIPS

        def worst_error(ridge):
            total = 0.0
            for seed in range(6):
                model = QLearningModel(
                    generate_ra_cohort(88, seed=500 + seed),
                    share_blip=False,
                    blip_ridge=ridge,
                    compute_covariance=False,
                )
                terminal = model.n_stages - 1
                total += max(
                    abs(model.blip_parameters(arm, terminal)[name] - truth)
                    for arm in TREATMENT_ARMS
                    if arm != REFERENCE_ARM
                    for name, truth in zip(model.blip_parameters(arm, terminal), TRUE_BLIPS[arm])
                )
            return total / 6

        self.assertLess(worst_error(DEFAULT_BLIP_RIDGE), worst_error(1.0))
        self.assertLess(worst_error(DEFAULT_BLIP_RIDGE), worst_error(0.0))


class AuditHarnessTests(unittest.TestCase):
    def test_ingestion_recovers_what_was_generated(self):
        metrics = audit_ingestion(n=15).metrics
        self.assertEqual(metrics["stage_count_exact"], 1.0)
        self.assertEqual(metrics["visit_interval_exact"], 1.0)
        self.assertEqual(
            metrics["switch_detection_recall"],
            1.0,
            "a change of arm is a switch; detection must not depend on free text",
        )

    def test_abstention_is_earned(self):
        """When the agent declines to separate arms, they should be close."""
        metrics = audit_decision(n=40).metrics
        self.assertTrue(metrics["abstention_is_earned"])
        self.assertGreater(metrics["optimal_arm_rate"], 0.7)
        self.assertLess(metrics["mean_regret_vs_oracle"], 0.02)

    def test_safety_catches_contraindications_without_over_removing(self):
        metrics = audit_safety().metrics
        self.assertEqual(metrics["contraindication_recall"], 1.0)
        self.assertEqual(metrics["spurious_removals"], 0)
        self.assertEqual(metrics["healthy_patient_removals"], 0)

    def test_explanations_decompose_the_model_exactly(self):
        metrics = audit_explanation(n=10).metrics
        self.assertEqual(metrics["attribution_sums_to_advantage"], 1.0)
        self.assertEqual(metrics["phi_leaks_into_narrative"], 0)
        self.assertFalse(metrics["memory_changed_q_values"])
        self.assertTrue(metrics["memory_changed_narrative"])

    def test_governance_keeps_the_gates_closed(self):
        metrics = audit_governance().metrics
        self.assertTrue(metrics["estimands_are_model_level"])
        self.assertTrue(metrics["estimands_are_distinct"])
        self.assertFalse(metrics["retraining_allowed"])


if __name__ == "__main__":
    unittest.main()
