"""The m-out-of-n bootstrap for the non-regular, non-terminal stages.

The ordinary bootstrap is *inconsistent* where the optimal arm is ambiguous,
because the `max` in the pseudo-outcome is not smooth there. Taking a resample
smaller than the sample is the standard repair, and the resample size has to
adapt to how non-regular the problem actually is.
"""

import unittest

from treatmentrx.estimation import training
from treatmentrx.estimation.inference import (
    BootstrapDistribution,
    adaptive_resample_size,
    m_out_of_n_bootstrap,
)
from treatmentrx.estimation.q_learning import QLearningModel
from treatmentrx.simulation.ra_cohort import generate_ra_cohort

_REPLICATES = 30  # the full run is `treatmentrx inference`

# One cohort and one fit for the whole module. Each replicate is a full refit,
# so re-fitting per test is the difference between a fast suite and a slow one.
_COHORT = generate_ra_cohort(150, seed=5)
_MODEL = QLearningModel(_COHORT)
_DISTRIBUTION = _MODEL.fit_bootstrap(_COHORT, replicates=80)
_FEATURES = _COHORT[0].stages[-1].features


class ResampleSizeTests(unittest.TestCase):
    def test_a_regular_problem_uses_the_whole_sample(self):
        """With no patient near a tie the estimator is regular and m = n."""
        self.assertEqual(adaptive_resample_size(400, non_regularity=0.0), 400)

    def test_resample_shrinks_as_non_regularity_rises(self):
        sizes = [adaptive_resample_size(400, p) for p in (0.0, 0.25, 0.5, 0.75, 1.0)]
        self.assertEqual(sizes, sorted(sizes, reverse=True))
        self.assertLess(sizes[-1], sizes[0])

    def test_resample_never_exceeds_n_or_collapses(self):
        for p in (0.0, 0.5, 1.0):
            m = adaptive_resample_size(60, p)
            self.assertLessEqual(m, 60)
            self.assertGreaterEqual(m, 25)


class NonRegularityTests(unittest.TestCase):
    def test_measured_non_regularity_is_a_proportion(self):
        self.assertTrue(0.0 <= _MODEL.non_regularity(_COHORT) <= 1.0)

    def test_stage_specific_fit_is_more_non_regular_than_the_shared_one(self):
        """More blip parameters on the same rows means noisier, closer contrasts."""
        stage_specific = QLearningModel(_COHORT, share_blip=False)
        self.assertGreater(
            stage_specific.non_regularity(_COHORT), _MODEL.non_regularity(_COHORT)
        )


class BootstrapDistributionTests(unittest.TestCase):
    def test_bootstrap_reruns_the_whole_procedure(self):
        self.assertIsInstance(_DISTRIBUTION, BootstrapDistribution)
        self.assertGreater(_DISTRIBUTION.replicates, 40)
        self.assertEqual(len(_DISTRIBUTION.draws[0]), _MODEL.n_features)
        self.assertLessEqual(_DISTRIBUTION.m, _DISTRIBUTION.n)

    def test_draws_differ_from_each_other(self):
        """A bootstrap whose replicates all agree is not resampling anything."""
        first = [draw[0] for draw in _DISTRIBUTION.draws]
        self.assertGreater(max(first) - min(first), 1e-6)

    def test_resampling_is_by_trajectory_not_by_visit(self):
        """Clusters must stay intact; a resample is patients, not rows."""
        seen = []

        def spy(sample):
            seen.append(sample)
            return _MODEL.refit(sample)

        m_out_of_n_bootstrap(spy, _COHORT, _MODEL._beta, 0.3, replicates=3, seed=1)
        for sample in seen:
            self.assertTrue(all(hasattr(item, "stages") for item in sample))

    def test_bootstrap_is_deterministic_for_a_seed(self):
        """Also guards a subtle feedback loop.

        The resample size depends on the measured non-regularity, which is a
        plug-in from the fitted intervals. If that measurement used the attached
        bootstrap rather than the sandwich, re-running it would pick a different
        m and produce different draws.
        """
        first = _MODEL.fit_bootstrap(_COHORT, replicates=8, seed=99)
        second = _MODEL.fit_bootstrap(_COHORT, replicates=8, seed=99)
        self.assertEqual(first.m, second.m)
        self.assertEqual(first.draws, second.draws)


class BootstrapIntervalTests(unittest.TestCase):
    """One fit and one bootstrap shared across the suite; the refits are the cost.

    Enough replicates that the resampled standard error is not itself noise —
    at a couple of dozen draws the comparison against the sandwich is a coin
    flip and would make a flaky test out of a real effect.
    """

    @classmethod
    def setUpClass(cls):
        cls.model, cls.cohort, cls.features = _MODEL, _COHORT, _FEATURES
        cls.model.attach_bootstrap(None)
        cls.before = cls.model.contrast("rituximab", "IL-6 inhibitor", cls.features, 0)
        cls.sandwich = cls.model.sandwich_contrast("rituximab", "IL-6 inhibitor", cls.features, 0)
        cls.model.attach_bootstrap(_DISTRIBUTION)
        cls.booted = cls.model.bootstrap_contrast("rituximab", "IL-6 inhibitor", cls.features, 0)

    def test_bootstrap_interval_is_wider_than_the_sandwich(self):
        """The point of the exercise: the sandwich understates the interval.

        It treats the pseudo-outcomes as fixed data when they are themselves
        estimated, so its interval is optimistic wherever backward induction is
        involved.
        """
        self.assertGreater(self.booted.standard_error, self.sandwich.standard_error)
        self.assertEqual(round(self.booted.difference, 9), round(self.sandwich.difference, 9))

    def test_contrast_prefers_the_bootstrap_once_attached(self):
        after = self.model.contrast("rituximab", "IL-6 inhibitor", self.features, 0)
        self.assertNotIn("m-out-of-n", self.before.caveat)
        self.assertIn("m-out-of-n", after.caveat)
        self.assertEqual(after.standard_error, self.booted.standard_error)

    def test_interval_brackets_the_point_estimate(self):
        self.assertLess(self.booted.lower, self.booted.difference)
        self.assertGreater(self.booted.upper, self.booted.difference)

    def test_reported_on_the_per_remaining_visit_scale(self):
        """Same rescaling as `q_values`, or the interval would not match the point."""
        terminal = self.model.bootstrap_contrast(
            "rituximab", "IL-6 inhibitor", self.features, self.model.n_stages - 1
        )
        self.assertAlmostEqual(
            self.booted.difference * self.model.n_stages, terminal.difference, places=6
        )


class TrainingIntegrationTests(unittest.TestCase):
    def test_enable_bootstrap_inference_reports_its_setup(self):
        report = training.enable_bootstrap_inference(replicates=6)
        for name, entry in report.items():
            self.assertLessEqual(entry["m"], entry["n"], msg=name)
            self.assertTrue(0.0 <= entry["non_regularity"] <= 1.0, msg=name)
            self.assertGreater(entry["replicates"], 0, msg=name)
        # Leave the shared cache as the rest of the suite expects it — detaching
        # the draws is enough, and keeps the three fitted models.
        training.disable_bootstrap_inference()


if __name__ == "__main__":
    unittest.main()
