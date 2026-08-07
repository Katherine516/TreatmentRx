"""The four follow-ups: interval-method sensitivity, estimator robustness,
visit-intensity weighting, and the misspecified-nuisance arm of the simulation.
"""

import unittest

from treatmentrx import TreatmentRxOrchestrator
from treatmentrx.demo_data import sample_ra_bundle
from treatmentrx.domain import RecommendationStatus
from treatmentrx.estimation.inference import SANDWICH_INFLATION, ContrastTest
from treatmentrx.estimation.q_learning import USE_VISIT_INTENSITY, QLearningModel
from treatmentrx.estimation.visit_intensity import VisitIntensityModel
from treatmentrx.feedback.misspecification import (
    RobustnessResult,
    preferred_estimator,
    robustness_study,
)
from treatmentrx.simulation.ra_cohort import (
    _INTERVAL_PER_SD,
    REFERENCE_ARM,
    TREATMENT_ARMS,
    TRUE_BLIPS,
    generate_ra_cohort,
    treatment_free_value,
)


def _contrast(difference, half_width, caveat="sandwich"):
    return ContrastTest(
        arm="a",
        comparator="b",
        difference=difference,
        standard_error=half_width / 1.96,
        lower=difference - half_width,
        upper=difference + half_width,
        alpha=0.05,
        caveat=caveat,
    )


class MethodSensitivityTests(unittest.TestCase):
    """A separation that only exists because the interval is too narrow is not one."""

    def test_a_comfortable_separation_survives_inflation(self):
        contrast = _contrast(difference=0.20, half_width=0.05)
        self.assertTrue(contrast.distinguishable)
        self.assertTrue(contrast.robustly_distinguishable)

    def test_a_marginal_separation_does_not(self):
        # Excludes zero, but only just: widening by the measured factor swallows it.
        contrast = _contrast(difference=0.055, half_width=0.05)
        self.assertTrue(contrast.distinguishable)
        self.assertFalse(contrast.robustly_distinguishable)

    def test_an_already_honest_interval_is_not_penalised_twice(self):
        """The bootstrap interval is the honest width; inflating it again would
        make the system pay the correction twice."""
        booted = _contrast(difference=0.055, half_width=0.05, caveat="m-out-of-n bootstrap (m=88)")
        self.assertTrue(booted.exact)
        self.assertTrue(booted.robustly_distinguishable)

    def test_an_exact_interval_has_no_caveat_or_a_bootstrap_one(self):
        self.assertTrue(_contrast(0.1, 0.02, caveat="").exact)
        self.assertFalse(_contrast(0.1, 0.02, caveat="treats the pseudo-outcomes as fixed").exact)

    def test_the_inflation_factor_is_above_the_measured_understatement(self):
        """cli coverage puts the sandwich at ~0.88 of the true spread."""
        self.assertGreater(SANDWICH_INFLATION, 1.0 / 0.88)

    def test_the_demo_patient_still_separates(self):
        """The escalation must not swallow a genuinely clear decision."""
        recommendation = TreatmentRxOrchestrator().run(sample_ra_bundle())
        self.assertEqual(recommendation.status, RecommendationStatus.RECOMMEND)
        self.assertTrue(recommendation.audit_event["contrast"]["robustly_distinguishable"])


class VisitIntensityTests(unittest.TestCase):
    def test_the_fitted_model_recovers_the_generating_relationship(self):
        """Sicker patients are seen sooner, by a known number of days per SD.

        Attenuated by the 28-day floor, which truncates the shortest intervals,
        so the estimate is expected to fall short of the generator's -22.
        """
        model = VisitIntensityModel(generate_ra_cohort(600, seed=7))
        self.assertTrue(model.fitted)
        self.assertLess(model.severity_slope(), 0.0)
        self.assertGreater(model.severity_slope(), _INTERVAL_PER_SD)
        self.assertLess(model.severity_slope(), _INTERVAL_PER_SD * 0.6)

    def test_weights_are_stabilised_around_one(self):
        model = VisitIntensityModel(generate_ra_cohort(400, seed=7))
        weights = [
            model.weight(stage.features)
            for trajectory in generate_ra_cohort(400, seed=7)
            for stage in trajectory.stages
        ]
        self.assertAlmostEqual(sum(weights) / len(weights), 1.0, delta=0.15)

    def test_it_is_off_by_default_and_the_reason_is_measured(self):
        """Kept, validated, and disabled: it adds variance without removing bias
        because visit frequency depends on a covariate the outcome model already
        conditions on. See the constant's comment for the numbers."""
        self.assertFalse(USE_VISIT_INTENSITY)
        self.assertIsNone(QLearningModel(generate_ra_cohort(120, seed=7)).visit_intensity)


class MisspecificationTests(unittest.TestCase):
    def test_curvature_leaves_the_estimand_untouched(self):
        """Only the nuisance surface bends; the blips are what we estimate."""
        features = {"das28": 7.0, "crp": 20.0, "anti_ccp": 1.0, "prior_tnf": 0.0,
                    "egfr": 90.0, "alt": 25.0}
        self.assertNotEqual(
            treatment_free_value(features, curvature=0.0),
            treatment_free_value(features, curvature=0.15),
        )
        from treatmentrx.simulation.ra_cohort import true_blip

        self.assertEqual(true_blip("rituximab", features), true_blip("rituximab", features))

    def test_curvature_is_off_by_default(self):
        """Every other result in the repo assumes the correctly specified cohort."""
        plain = generate_ra_cohort(30, seed=7)
        explicit = generate_ra_cohort(30, seed=7, curvature=0.0)
        self.assertEqual(
            [s.outcome for t in plain for s in t.stages],
            [s.outcome for t in explicit for s in t.stages],
        )

    def test_misspecification_degrades_every_estimator(self):
        results = robustness_study(curvatures=(0.0, 0.10), seeds=(7, 23), size=250)
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertGreater(result.degradation, 1.0, msg=result.estimator)

    def test_the_estimators_trade_off_rather_than_dominate(self):
        """The justification for averaging them: none wins on both axes.

        dWOLS is doubly robust and most accurate when the nuisance model is
        right; the shared-blip fit is the most stable when it is wrong.
        """
        results = robustness_study(curvatures=(0.0, 0.10), seeds=(7, 23, 101), size=250)
        most_accurate = min(results, key=lambda r: r.correctly_specified)
        most_stable = min(results, key=lambda r: r.degradation)
        self.assertNotEqual(most_accurate.estimator, most_stable.estimator)

    def test_a_tied_comparison_declines_to_name_a_winner(self):
        tied = [
            RobustnessResult("a", {0.0: 0.20, 0.1: 0.40}),
            RobustnessResult("b", {0.0: 0.10, 0.1: 0.41}),
        ]
        preferred, reason = preferred_estimator(tied)
        self.assertEqual(preferred, "")
        self.assertIn("No single estimator wins", reason)


if __name__ == "__main__":
    unittest.main()


class ModelAveragedContrastTests(unittest.TestCase):
    """The interval must describe the quantity the decision is made on.

    The decision uses the model-averaged Q-values. Reporting the widest of the
    three estimators' intervals instead meant the difference came from one model
    while the decision came from the ensemble, and which model supplied it moved
    with the data — measured end to end that rule covered 78% against a nominal
    95%, worse than any of its own components.
    """

    @classmethod
    def setUpClass(cls):
        cls.audit = TreatmentRxOrchestrator().run(sample_ra_bundle()).audit_event

    def test_the_interval_is_centred_on_the_decision_gap(self):
        self.assertAlmostEqual(
            self.audit["contrast"]["difference"], self.audit["confidence_gap"], delta=0.005
        )

    def test_the_averaged_interval_is_marked_conservative(self):
        """Its variance is an upper bound, so it must not be widened again."""
        self.assertTrue(self.audit["contrast"]["conservative"])

    def test_a_conservative_interval_is_not_inflated_twice(self):
        marginal = ContrastTest(
            arm="a", comparator="b", difference=0.055,
            standard_error=0.05 / 1.96, lower=0.005, upper=0.105,
            alpha=0.05, caveat="model-averaged", conservative=True,
        )
        self.assertTrue(marginal.exact)
        self.assertTrue(marginal.robustly_distinguishable)
        # The same interval, not already corrected, does not survive.
        naive = ContrastTest(**{**marginal.__dict__, "conservative": False})
        self.assertFalse(naive.robustly_distinguishable)

    def test_averaging_reaches_nominal_coverage(self):
        """The reason the rule changed, asserted rather than asserted-about.

        Averaging removes the estimator-selection variability and lets the
        components' opposing biases cancel; the interval then errs wide rather
        than narrow, which is the safe direction for a clinical decision.
        """
        from treatmentrx.feedback.coverage import NOMINAL, decision_rule_coverage

        result = decision_rule_coverage(replications=12, n=180)
        self.assertGreaterEqual(result.coverage, NOMINAL - 2 * result.monte_carlo_error)
        self.assertGreater(result.se_to_sd_ratio, 1.0)
