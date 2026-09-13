"""The regime's own value, and the weights behind every IPW number.

`evaluate_policy` scores stage-rows independently and weights by one stage's
propensity. For a sequential regime that is a per-decision quantity, not the
regime's value: a patient who deviated at stage 1 still contributed their
stage-2 row, from a history the regime would never have produced.
"""

import unittest

from treatmentrx.estimation import training
from treatmentrx.estimation.q_learning import QLearningModel
from treatmentrx.feedback.offline_evaluation import (
    MIN_SEQUENTIAL_EFFECTIVE_SAMPLE,
    sequential_policy_value,
    weight_diagnostics,
)


def _serving_policy():
    models = training.serving_models()
    arms = training.fitted().pooled.arms

    def policy(features, stage_index):
        totals = {arm: 0.0 for arm in arms}
        for model in models.values():
            values = (
                model.q_values(features, stage_index)
                if isinstance(model, QLearningModel)
                else model.q_values(features)
            )
            for arm in arms:
                totals[arm] += values.get(arm, 0.0)
        return max(totals, key=totals.get)

    return policy


class SequentialValueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fit = training.fitted()
        cls.fit = fit
        cls.sequential = sequential_policy_value(
            _serving_policy(), fit.holdout, fit.propensity.propensity
        )

    def test_consistency_is_required_at_every_prior_decision(self):
        """A trajectory stops contributing the moment it leaves the regime's
        path, so survivors can only fall as the horizon grows."""
        surviving = self.sequential.consistent_trajectories
        self.assertTrue(surviving)
        for earlier, later in zip(surviving, surviving[1:]):
            self.assertLessEqual(later, earlier)
        self.assertLessEqual(surviving[0], self.sequential.n_trajectories)

    def test_it_uses_a_smaller_effective_sample_than_the_per_decision_value(self):
        """The point of the whole estimator. Requiring agreement at every prior
        decision and multiplying propensities can only shrink the effective
        sample — and it shrinks it a long way here."""
        per_decision = self.fit.scores[training.DWOLS_SHARED].effective_sample_size
        self.assertLess(self.sequential.effective_sample_size, per_decision)

    def test_it_reports_whether_it_is_identified_at_all(self):
        """At this cohort size it is not, and that has to be a field rather than
        something a reader infers from a small number."""
        self.assertEqual(
            self.sequential.identified,
            self.sequential.effective_sample_size >= MIN_SEQUENTIAL_EFFECTIVE_SAMPLE,
        )
        self.assertFalse(
            self.sequential.identified,
            "if this starts passing, the note on the scorecard needs rewriting too",
        )

    def test_the_scorecard_says_which_quantity_it_is_reporting(self):
        for name in training.SERVING_ENSEMBLE:
            with self.subTest(estimator=name):
                score = self.fit.scores[name]
                self.assertIsNotNone(score.sequential)
                self.assertTrue(
                    any("per-decision" in note for note in score.notes),
                    "an unidentified regime value has to be said out loud",
                )


class SequentialIntervalTests(unittest.TestCase):
    """The regime's value has to carry an interval, not just a verdict.

    `identified` compares an effective sample against a threshold and returns a
    boolean. A reader handed "0.8263, not identified" knows the number is
    untrustworthy but not by how much, and the natural thing to do with a bare
    point estimate is to quote it.
    """

    @classmethod
    def setUpClass(cls):
        fit = training.fitted()
        cls.fit = fit
        cls.sequential = sequential_policy_value(
            _serving_policy(), fit.holdout, fit.propensity.propensity
        )

    def test_the_value_comes_with_an_interval(self):
        self.assertIsNotNone(self.sequential.interval)
        low, high = self.sequential.interval
        self.assertLess(low, high)
        self.assertAlmostEqual(
            self.sequential.interval_width, high - low, places=9
        )

    def test_the_interval_brackets_the_point_estimate(self):
        low, high = self.sequential.interval
        self.assertLessEqual(low, self.sequential.value)
        self.assertLessEqual(self.sequential.value, high)

    def test_the_interval_and_the_value_read_the_same_rows(self):
        """Both are built from one pass of `_sequential_aggregates`.

        A bootstrap that recomputed the consistent prefix under its own rules
        could bracket a different set of rows than the value it is printed
        beside, and nothing would look wrong.
        """
        from treatmentrx.feedback.offline_evaluation import _sequential_aggregates

        aggregates, surviving = _sequential_aggregates(
            _serving_policy(), self.fit.holdout, self.fit.propensity.propensity
        )
        self.assertEqual(surviving, self.sequential.consistent_trajectories)
        self.assertEqual(
            sum(len(prefix) for _, _, prefix in aggregates),
            self.sequential.matched_rows,
        )

    def test_the_bootstrap_resamples_trajectories_not_rows(self):
        """The trajectory is the independent unit, so it is the resampling unit.

        Resampling rows would treat three stages of one patient as three
        independent observations and report an interval too narrow by roughly
        the within-patient correlation — the same error the cluster-robust
        sandwich exists to avoid.
        """
        from treatmentrx.feedback.offline_evaluation import (
            _sequential_aggregates,
            sequential_intervals,
        )

        aggregates, _ = _sequential_aggregates(
            _serving_policy(), self.fit.holdout, self.fit.propensity.propensity
        )
        self.assertEqual(len(aggregates), len(self.fit.holdout))
        interval, _ = sequential_intervals(aggregates, replicates=200, seed=5)
        self.assertIsNotNone(interval)

        # A trajectory that never matched still occupies a slot: dropping it
        # would condition on the agreement being measured.
        empty = sum(1 for _, _, prefix in aggregates if not prefix)
        self.assertGreater(empty, 0, "every holdout trajectory matched; check the fixture")

    def test_a_holdout_with_no_agreement_yields_no_interval(self):
        """No consistent row anywhere is 'not estimable', not 'zero'."""
        from treatmentrx.feedback.offline_evaluation import sequential_intervals

        interval, share = sequential_intervals(
            [(0.0, 0.0, []) for _ in range(20)], replicates=50, seed=1
        )
        self.assertIsNone(interval)
        self.assertEqual(share, 1.0)

    def test_the_degenerate_share_is_reported_rather_than_skipped(self):
        payload = self.sequential.as_dict()
        self.assertIn("degenerate_resample_share", payload)
        self.assertGreaterEqual(payload["degenerate_resample_share"], 0.0)
        self.assertLessEqual(payload["degenerate_resample_share"], 1.0)

    def test_the_unidentified_note_reaches_the_scorecard_dict(self):
        """`notes` was populated and never emitted by `as_dict`.

        The line saying the regime's value is not identified is the whole point
        of computing it, and `cli evaluate` and the model card both render this
        dict.
        """
        payload = training.best_score().as_dict()
        self.assertIn("notes", payload)
        self.assertTrue(
            any("regime's own value" in note for note in payload["notes"]),
            payload["notes"],
        )


class SequentialDRTests(unittest.TestCase):
    """The doubly-robust value, and the reason it does not retire the IPW one.

    Pure inverse weighting discards a trajectory at its first deviation: five of
    120 survive on the deployed holdout and the heaviest carries 35% of the
    weight. The DR recursion lets the fitted Q-function supply the value where a
    trajectory leaves the regime's path, so every trajectory contributes.
    """

    @classmethod
    def setUpClass(cls):
        from treatmentrx.feedback.offline_evaluation import sequential_dr_value
        from treatmentrx.simulation.ra_cohort import rollout_value

        fit = training.fitted()
        cls.fit = fit
        cls.policy = fit.pooled.greedy_policy()
        cls.truth = rollout_value(cls.policy, n=4000, seed=101, dropout=False)
        cls.observed_truth = rollout_value(cls.policy, n=4000, seed=101, dropout=True)
        cls.dr = sequential_dr_value(
            cls.policy,
            fit.holdout,
            fit.pooled,
            fit.propensity.propensity,
            augmentation_model="Q-Pooled",
            oracle_uncensored_value=cls.truth,
        )
        cls.ipw = sequential_policy_value(
            cls.policy, fit.holdout, fit.propensity.propensity
        )

    def test_every_trajectory_contributes(self):
        """The whole point. IPW keeps a handful; this keeps all of them."""
        self.assertEqual(self.dr.n_trajectories, len(self.fit.holdout))
        self.assertLess(
            self.ipw.consistent_trajectories[-1], self.dr.n_trajectories / 10
        )

    def test_it_recovers_the_known_value(self):
        """Checked against the generating process, not asserted."""
        self.assertIsNotNone(self.dr.interval)
        self.assertTrue(
            self.dr.covers_oracle,
            f"{self.dr.interval} does not contain {self.truth}",
        )
        self.assertLess(abs(self.dr.value - self.truth), 2.0 * self.dr.standard_error)

    def test_it_is_scored_against_the_uncensored_truth(self):
        """The two oracles are different quantities and mixing them is a bug.

        The augmenting Q-model is IPCW-weighted, so it targets the value-to-go
        had the patient stayed in care. `oracle_rollout_value` is what a patient
        accrues once dropout is simulated — smaller by the retention gap, which
        on this cohort is about 0.15 on a value near 2.3. Scoring the DR estimate
        against it would charge the estimator for retention.
        """
        self.assertGreater(self.truth, self.observed_truth)
        self.assertGreater(self.truth - self.observed_truth, 0.05)
        self.assertEqual(self.dr.oracle_uncensored_value, self.truth)

    def test_no_single_patient_dominates_it(self):
        """Compared like for like: the share one unit carries, both ways.

        `WeightDiagnostics.top_share` and `SequentialDRValue.max_influence_share`
        are deliberately the same quantity, so this is a comparison and not two
        numbers that both happen to look small. Uniform over 120 trajectories
        would be 0.83%; the DR estimate runs about 1.5% and the IPW estimate's
        heaviest row carries a sixth of its mass.
        """
        uniform = 1.0 / self.dr.n_trajectories
        self.assertLess(self.dr.max_influence_share, 5.0 * uniform)
        self.assertLess(
            self.dr.max_influence_share,
            self.ipw.weights.top_share / 5.0,
            "the augmentation did not reduce the concentration",
        )

    def test_it_is_far_more_precise_than_the_ipw_estimate(self):
        """Both are legitimate; only one of them can be read at this sample size."""
        self.assertTrue(self.dr.identified)
        self.assertFalse(self.ipw.identified)

    def test_a_deviating_trajectory_contributes_the_models_own_value(self):
        """Where the observed arm is never the regime's, the estimate is the plug-in.

        This is the branch that keeps the trajectory, and it has to be exactly
        `Q(x, d(x))` — any weighting there would be inverse-weighting a residual
        that was never observed.
        """
        from dataclasses import replace

        from treatmentrx.feedback.offline_evaluation import sequential_dr_value

        trajectory = self.fit.holdout[0]
        first = trajectory.stages[0]
        never = lambda features, stage_index: "__not-an-arm__"
        single = replace(trajectory, stages=trajectory.stages[:1])
        result = sequential_dr_value(
            never, [single], self.fit.pooled, self.fit.propensity.propensity
        )
        self.assertAlmostEqual(
            result.value,
            self.fit.pooled.raw_q(first.features, "__not-an-arm__", 0),
            places=9,
        )

    def test_the_scorecard_carries_both_and_says_which_scale(self):
        payload = training.best_score().as_dict()
        doubly_robust = payload["sequential_regime_value_doubly_robust"]
        self.assertIsNotNone(doubly_robust)
        self.assertIn("value-to-go", doubly_robust["scale"])
        # The per-visit Hajek value and the sum-scale value-to-go are different
        # quantities; if they ever come out equal, one of them has changed scale.
        self.assertNotAlmostEqual(
            doubly_robust["value"], payload["sequential_regime_value"]["value"], places=2
        )

    def test_the_ladder_blocker_names_what_is_known_as_well_as_what_is_not(self):
        """A blocker that only says "not identified" hides a measured estimate."""
        from treatmentrx.feedback.validation_ladder import ValidationLadder, ValidationRung

        readiness = training.deployment_readiness()
        status = ValidationLadder().assess(ValidationRung.SILENT, readiness)
        sequential = [b for b in status.blockers if "outcome model" in b]
        self.assertTrue(sequential, status.blockers)
        self.assertIn("doubly-robust estimate", sequential[0])


class WeightDiagnosticTests(unittest.TestCase):
    def test_flat_weights_are_fully_efficient(self):
        diagnostics = weight_diagnostics([1.0] * 40)
        self.assertAlmostEqual(diagnostics.effective_sample_size, 40.0, places=6)
        self.assertAlmostEqual(diagnostics.top_share, 1 / 40, places=6)
        self.assertTrue(diagnostics.positivity_ok)

    def test_one_dominating_row_is_caught(self):
        """ESS alone can look acceptable while a single row carries the mean."""
        diagnostics = weight_diagnostics([1.0] * 30 + [60.0])
        self.assertGreater(diagnostics.top_share, 0.10)
        self.assertFalse(diagnostics.positivity_ok)

    def test_rows_pinned_to_the_propensity_floor_are_counted(self):
        from treatmentrx.feedback.offline_evaluation import _PROPENSITY_FLOOR

        ceiling = 1.0 / _PROPENSITY_FLOOR
        diagnostics = weight_diagnostics([1.0] * 9 + [ceiling])
        self.assertAlmostEqual(diagnostics.share_at_floor, 0.1, places=6)

    def test_both_estimators_report_their_weights(self):
        score = training.fitted().scores[training.DWOLS_SHARED]
        self.assertIsNotNone(score.weights)
        self.assertIsNotNone(score.sequential.weights)
        # The score rounds for display; the diagnostic keeps full precision.
        self.assertAlmostEqual(
            round(score.weights.effective_sample_size, 2),
            score.effective_sample_size,
            places=6,
        )

    def test_the_sequential_weights_are_the_worse_of_the_two(self):
        """Cumulative products concentrate mass. This is the diagnostic that
        explains *why* the sequential effective sample collapses, rather than
        leaving a reader to guess."""
        score = training.fitted().scores[training.DWOLS_SHARED]
        self.assertGreater(score.sequential.weights.max_weight, score.weights.max_weight)
        self.assertGreater(score.sequential.weights.top_share, score.weights.top_share)


if __name__ == "__main__":
    unittest.main()
