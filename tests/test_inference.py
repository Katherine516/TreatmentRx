"""Standard errors, contrast tests, and resampling stability.

The claim under test is that the intervals are *standard errors of something*
rather than a heuristic that merely narrows with more data.
"""

import unittest

from treatmentrx.data import DataLayer
from treatmentrx.decision import DecisionLayer
from treatmentrx.demo_data import sample_ra_bundle
from treatmentrx.domain import RecommendationStatus
from treatmentrx.estimation import EstimationLayer, training
from treatmentrx.estimation.features import model_features, stage_index
from treatmentrx.estimation.inference import ContrastTest, contrast_test
from treatmentrx.estimation.q_learning import QLearningModel
from treatmentrx.feedback.stability import (
    is_ranking_resolved,
    kfold_scores,
    seed_sweep,
    summarise,
)
from treatmentrx.simulation.ra_cohort import generate_ra_cohort


def _demo_state():
    return DataLayer().build_patient_state(sample_ra_bundle())


class StandardErrorTests(unittest.TestCase):
    def test_standard_errors_shrink_with_the_square_root_of_n(self):
        """The defining property of a standard error, and what the old
        support-size heuristic only imitated."""
        small = QLearningModel(generate_ra_cohort(150, seed=5))
        large = QLearningModel(generate_ra_cohort(600, seed=5))
        features = model_features(_demo_state().stages)

        small_se = small.blip_standard_error("rituximab", features, 0)
        large_se = large.blip_standard_error("rituximab", features, 0)
        self.assertGreater(small_se, large_se)
        # Quadrupling n should roughly halve the standard error.
        self.assertLess(large_se, small_se * 0.75)

    def test_every_arm_reports_a_positive_standard_error(self):
        model = training.fitted().q_shared
        features = model_features(_demo_state().stages)
        for arm in model.blip_arms:
            self.assertGreater(model.blip_standard_error(arm, features, 0), 0.0, msg=arm)

    def test_reference_arm_has_no_blip_uncertainty(self):
        """Its blip is zero by construction, not estimated."""
        model = training.fitted().q_shared
        features = model_features(_demo_state().stages)
        self.assertEqual(model.blip_standard_error("continue-current", features, 0), 0.0)

    def test_confidence_band_is_derived_from_the_standard_error(self):
        state = _demo_state()
        estimates = EstimationLayer().estimate(state)
        model = training.fitted().q_shared
        features = model_features(state.stages)
        for estimate in estimates:
            low, high = estimate.confidence_band
            self.assertLess(low, high, msg=estimate.estimator)
            self.assertLess(high - low, 0.5, msg=f"{estimate.estimator}: implausibly wide band")
        # The band must use the patient's own stage index — the demo patient is
        # at the terminal stage, where the value-to-go rescaling is a no-op.
        index = stage_index(state.stages, model.n_stages)
        expected = 2 * 1.96 * model.blip_standard_error(estimates[0].recommended_arm, features, index)
        low, high = estimates[0].confidence_band
        self.assertAlmostEqual(high - low, expected, places=2)


class ContrastTests(unittest.TestCase):
    def test_demo_patient_separates_rituximab_from_the_runner_up(self):
        model = training.fitted().q_shared
        features = model_features(_demo_state().stages)
        contrast = model.contrast("rituximab", "IL-6 inhibitor", features, 0)
        self.assertIsInstance(contrast, ContrastTest)
        self.assertGreater(contrast.difference, 0)
        self.assertTrue(contrast.distinguishable)
        self.assertLess(contrast.lower, contrast.difference)
        self.assertGreater(contrast.upper, contrast.difference)

    def test_an_arm_against_itself_is_never_distinguishable(self):
        model = training.fitted().q_shared
        features = model_features(_demo_state().stages)
        contrast = model.contrast("rituximab", "rituximab", features, 0)
        self.assertAlmostEqual(contrast.difference, 0.0, places=9)
        self.assertFalse(contrast.distinguishable)

    def test_only_a_regular_stage_reports_an_uncaveated_interval(self):
        """The sandwich is exact at a stage-specific terminal block and nowhere else.

        With a *shared* blip there is no fully regular stage: one parameter
        vector is fit jointly from every stage's rows, and the earlier rows carry
        pseudo-outcomes. Its terminal interval inherits that, so it is caveated
        too — which the ordinary reading of "terminal stages are fine" misses.
        """
        features = model_features(_demo_state().stages)
        shared = training.fitted().q_shared
        stage_specific = training.fitted().stage_specific
        terminal = stage_specific.n_stages - 1

        self.assertIn("pseudo-outcomes", shared.contrast("rituximab", "IL-6 inhibitor", features, 0).caveat)
        self.assertIn(
            "pseudo-outcomes",
            shared.contrast("rituximab", "IL-6 inhibitor", features, terminal).caveat,
            msg="a shared blip has no regular stage",
        )
        self.assertIn(
            "pseudo-outcomes",
            stage_specific.contrast("rituximab", "IL-6 inhibitor", features, 0).caveat,
        )
        self.assertEqual(
            stage_specific.contrast("rituximab", "IL-6 inhibitor", features, terminal).caveat,
            "",
        )

    def test_an_indistinguishable_contrast_forces_equipoise(self):
        """A gap that clears the clinical bar but sits inside its own interval
        must not be reported as a recommendation."""
        state = _demo_state()
        estimates = EstimationLayer().estimate(state)
        layer = DecisionLayer()
        decision = layer.decide(state, estimates)
        self.assertEqual(decision.status, RecommendationStatus.RECOMMEND)

        flat = contrast_test(
            [[1.0]], [1.0], difference=0.001, arm="a", comparator="b"
        )
        self.assertFalse(flat.distinguishable)
        status, rationale = layer._status(state, decision.uncertainty, decision.goal_decision, flat)
        self.assertEqual(status, RecommendationStatus.EQUIPOISE)
        self.assertIn("includes zero", rationale)

    def test_decision_layer_takes_the_widest_interval(self):
        """If any estimator cannot separate the arms, the system does not claim
        separation."""
        state = _demo_state()
        estimates = EstimationLayer().estimate(state)
        layer = DecisionLayer()
        chosen = layer._contrast(state, layer.bma.aggregate(estimates))
        ordered = sorted(chosen.arm for chosen in [chosen])
        every = [
            estimator.contrast(state.stages, chosen.arm, chosen.comparator)
            for estimator in layer.estimators
        ]
        self.assertEqual(chosen.standard_error, max(t.standard_error for t in every))
        self.assertTrue(ordered)


class StabilityTests(unittest.TestCase):
    """Small configurations — the full sweep is the `stability` CLI command."""

    def test_cross_validation_scores_every_fold(self):
        runs = kfold_scores(generate_ra_cohort(120, seed=5), folds=3)
        self.assertEqual(len(runs), 3)
        results = summarise(runs)
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertEqual(result.policy_value.n, 3)
            self.assertGreater(result.policy_value.mean, 0.0)

    def test_seed_sweep_varies_the_cohort(self):
        runs = seed_sweep(seeds=(3, 9), size=120)
        self.assertEqual(len(runs), 2)
        self.assertNotEqual(runs[0], runs[1], "different seeds produced identical scores")

    def test_first_place_rates_sum_to_one(self):
        results = summarise(kfold_scores(generate_ra_cohort(120, seed=5), folds=3))
        self.assertAlmostEqual(sum(r.first_place_rate for r in results), 1.0, places=6)

    def test_verdict_refuses_to_rank_indistinguishable_estimators(self):
        results = summarise(kfold_scores(generate_ra_cohort(150, seed=5), folds=3))
        resolved, verdict = is_ranking_resolved(results)
        self.assertIsInstance(resolved, bool)
        if not resolved:
            self.assertIn("not distinguishable", verdict)
            self.assertIn("keep averaging", verdict)


if __name__ == "__main__":
    unittest.main()
