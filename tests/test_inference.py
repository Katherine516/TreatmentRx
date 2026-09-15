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
        features = model_features(state.stages)
        for estimate in estimates:
            low, high = estimate.confidence_band
            self.assertLess(low, high, msg=estimate.estimator)
            self.assertLess(high - low, 0.5, msg=f"{estimate.estimator}: implausibly wide band")

        # Checked against whichever Q-learning model is serving, rather than a
        # named one: the band must be that estimator's own standard error, and
        # the pairing is what would break if the two drifted apart.
        fit = training.fitted()
        by_name = {
            training.STAGE_SPECIFIC: fit.stage_specific,
            training.Q_SHARED: fit.q_shared,
            training.Q_POOLED: fit.pooled,
        }
        checked = 0
        for estimate in estimates:
            model = by_name.get(estimate.estimator)
            if model is None:
                continue
            checked += 1
            # The band uses the patient's own stage index — the demo patient is
            # at the terminal stage, where the value-to-go rescaling is a no-op.
            index = stage_index(state.stages, model.n_stages)
            expected = 2 * 1.96 * model.blip_standard_error(
                estimate.recommended_arm, features, index
            )
            low, high = estimate.confidence_band
            self.assertAlmostEqual(high - low, expected, places=2, msg=estimate.estimator)
        self.assertGreater(checked, 0)


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

    def test_decision_layer_averages_rather_than_picking_one_estimator(self):
        """The interval describes the averaged contrast the decision is made on.

        Picking the single widest interval instead let the reported difference
        come from one model while the decision came from the ensemble, and made
        which model supplied it depend on the data.
        """
        state = _demo_state()
        estimates = EstimationLayer().estimate(state)
        layer = DecisionLayer()
        selected = layer.bma.aggregate(estimates)
        weights = {
            name.split(":", 1)[1]: value
            for name, value in selected.coefficients.items()
            if name.startswith("bma_weight:")
        }
        chosen = layer._contrast(state, selected, weights)
        components = [
            estimator.contrast(state.stages, chosen.arm, chosen.comparator)
            for estimator in layer.estimators
        ]

        # An average, so it sits inside the range of its components rather than
        # equalling any one of them.
        self.assertLessEqual(chosen.standard_error, max(c.standard_error for c in components))
        self.assertGreaterEqual(chosen.standard_error, min(c.standard_error for c in components))
        self.assertTrue(chosen.conservative)


class StabilityTests(unittest.TestCase):
    """Small configurations — the full sweep is the `stability` CLI command."""

    def test_cross_validation_scores_every_fold(self):
        runs = kfold_scores(generate_ra_cohort(120, seed=5), folds=3)
        self.assertEqual(len(runs), 3)
        results = summarise(runs)
        # One row per estimator the sweep fits, whatever that set currently is —
        # hard-coding the count is how the serving model got left out of studies
        # elsewhere in this repo.
        self.assertEqual({r.estimator for r in results}, set(runs[0]))
        for result in results:
            self.assertEqual(result.policy_value.n, 3)
            self.assertGreater(result.policy_value.mean, 0.0)

    def test_the_sweep_covers_every_serving_estimator(self):
        from treatmentrx.estimation import training

        runs = kfold_scores(generate_ra_cohort(120, seed=5), folds=2)
        self.assertTrue(
            set(training.SERVING_ENSEMBLE) <= set(runs[0]),
            f"stability does not score {set(training.SERVING_ENSEMBLE) - set(runs[0])}",
        )

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



class StudentTCriticalValueTests(unittest.TestCase):
    """An exact t quantile, because a normal one is not honest at small n.

    The randomized precision module rejected at 0.101 unadjusted and 0.138
    adjusted at n=8 — its own accepted minimum — against a nominal 0.05, and the
    *adjusted* fit was worse because it spends a further degree of freedom the
    normal quantile cannot see. The degrees of freedom were already computed and
    reported there and then not used.
    """

    #: Published two-sided 0.05 critical values.
    KNOWN = {1: 12.706, 2: 4.303, 5: 2.571, 10: 2.228, 30: 2.042, 100: 1.984}

    def test_it_matches_published_tables(self):
        from treatmentrx.estimation.inference import student_t_critical_value

        for degrees, published in self.KNOWN.items():
            with self.subTest(df=degrees):
                self.assertAlmostEqual(
                    student_t_critical_value(0.05, degrees), published, places=3
                )

    def test_other_alphas_match_too(self):
        from treatmentrx.estimation.inference import student_t_critical_value

        self.assertAlmostEqual(student_t_critical_value(0.01, 5), 4.032, places=3)
        self.assertAlmostEqual(student_t_critical_value(0.10, 20), 1.725, places=3)

    def test_it_converges_to_the_normal_quantile(self):
        from treatmentrx.estimation.inference import (
            normal_critical_value,
            student_t_critical_value,
        )

        self.assertAlmostEqual(
            student_t_critical_value(0.05, 100_000),
            normal_critical_value(0.05),
            places=4,
        )

    def test_it_is_always_wider_than_the_normal_one(self):
        """The whole point: the standard error is itself estimated."""
        from treatmentrx.estimation.inference import (
            normal_critical_value,
            student_t_critical_value,
        )

        normal = normal_critical_value(0.05)
        for degrees in (1, 3, 8, 25, 200):
            with self.subTest(df=degrees):
                self.assertGreater(student_t_critical_value(0.05, degrees), normal)

    def test_the_incomplete_beta_is_exact_at_known_points(self):
        """`I_x(a,b)` underpins the quantile; pinned against closed forms."""
        from treatmentrx.estimation.inference import regularized_incomplete_beta

        # I_x(1,1) = x, and I_x(a,b) = 1 - I_{1-x}(b,a).
        for x in (0.1, 0.35, 0.5, 0.9):
            with self.subTest(x=x):
                self.assertAlmostEqual(regularized_incomplete_beta(1.0, 1.0, x), x, places=12)
                self.assertAlmostEqual(
                    regularized_incomplete_beta(2.5, 3.5, x),
                    1.0 - regularized_incomplete_beta(3.5, 2.5, 1.0 - x),
                    places=12,
                )
        self.assertEqual(regularized_incomplete_beta(2.0, 3.0, 0.0), 0.0)
        self.assertEqual(regularized_incomplete_beta(2.0, 3.0, 1.0), 1.0)

    def test_the_trial_module_uses_it(self):
        """A 95% interval that behaves like an 86% one is the defect this closes."""
        from treatmentrx.trials.precision import simulate_precision_power

        result = simulate_precision_power(8, 0.0, 1.0, 1.0, replicates=1500, seed=7)
        for arm, rate in result["rejection_rate"].items():
            with self.subTest(arm=arm):
                self.assertLess(
                    rate, 0.085, f"{arm} type I error {rate:.3f} at the accepted minimum n"
                )

if __name__ == "__main__":
    unittest.main()
