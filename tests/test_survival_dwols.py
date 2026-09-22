"""The survival blip estimator, scored against the truth it was built for.

`test_survival_cohort.py` checks the generator produces what it declares. This
file asks the next question: can the estimator recover it? The parameters are
known, so every assertion here is a quantity — a recovered `psi`, a recovered
shape, a bias that a weighting removes — rather than a check that the code ran.

Two properties are deliberately **not** pinned here and are recorded in
`survival_dwols`'s module docstring instead, because a test that cannot fail
reliably is worse than a measurement that is written down. The
double-robustness rescue is a 0.037 effect on a 0.675 base, which needs more
seeds than this suite can afford to separate from noise. And the
covariate-dependent censoring bias is a property of the generator's
administrative horizon, not something this estimator can be asked to fix.
"""

from __future__ import annotations

import math
import statistics
import unittest
from dataclasses import replace

from treatmentrx.estimation import survival_dwols as sd
from treatmentrx.simulation import survival_cohort as sc

_N = 2000
_SEEDS = (71, 72, 73, 74, 75)
# The weighting studies need more: `_pooled_bias` takes the absolute value of a
# mean, which its own sampling noise inflates upward, so at five seeds the
# comparison is not merely noisy but biased in both arms by different amounts.
# Measured, five seeds reported IPCW making the bias *worse* (0.439 -> 0.510)
# while three independent eight-seed blocks all reported it better
# (0.489 -> 0.326, 0.653 -> 0.466, 0.640 -> 0.474).
_WEIGHTING_SEEDS = tuple(range(71, 79))
_PARAMETERS = [
    (arm, term)
    for arm in sc.SURVIVAL_ARMS
    if arm != sc.SURVIVAL_REFERENCE_ARM
    for term in sc.HAZARD_BASIS
]


def _true(arm: str, term: str) -> float:
    return dict(zip(sc.HAZARD_BASIS, sc.TRUE_LOG_HAZARD_RATIOS[arm]))[term]


def _pooled_bias(models) -> dict[tuple, float]:
    """Mean *signed* error per parameter, which is what a weighting can remove.

    Mean absolute error sums a bias and a sampling spread; at this cohort size
    the spread dominates, so scoring a weighting by it measures mostly noise.
    That is invariant 40's distinction on this estimator.
    """
    per_seed = [
        {key: model.blip_parameters(key[0])[key[1]] - _true(*key) for key in _PARAMETERS}
        for model in models
    ]
    return {
        key: statistics.mean(errors[key] for errors in per_seed) for key in _PARAMETERS
    }


class RecoveryTests(unittest.TestCase):
    """Does it get the parameters back when nothing is misspecified?"""

    @classmethod
    def setUpClass(cls):
        cls.cohorts = [sc.generate_survival_cohort(_N, seed=s) for s in _SEEDS]
        cls.models = [sd.SurvivalBlipModel(c) for c in cls.cohorts]

    def test_the_weibull_shape_is_recovered_from_the_residuals(self):
        """Nothing supplies the shape; it comes from the Gumbel spread.

        `sd = pi / (k sqrt 6)` is a property of the Weibull family, so getting
        it back is what says the log-time model is the right one — and the shape
        multiplies every `psi`, so an error here rescales the whole estimand.
        """
        shapes = [model.shape for model in self.models]
        self.assertAlmostEqual(
            statistics.mean(shapes),
            sc.WEIBULL_SHAPE,
            delta=0.05,
            msg=f"shapes {[round(s, 4) for s in shapes]} against {sc.WEIBULL_SHAPE}",
        )

    def test_every_blip_parameter_is_recovered(self):
        """Pooled over seeds, because one cohort's draw is not a bias."""
        bias = _pooled_bias(self.models)
        worst = max(bias.items(), key=lambda kv: abs(kv[1]))
        self.assertLess(
            abs(worst[1]),
            0.12,
            f"worst parameter bias {worst[1]:+.4f} on {worst[0]}",
        )

    def test_the_reference_arm_is_the_identifying_zero(self):
        """`tau_ref = 0` by construction is what every other arm is measured
        against; a fit for it would mean the contrast had lost its anchor."""
        for model in self.models:
            self.assertNotIn(sc.SURVIVAL_REFERENCE_ARM, model.fits)
            self.assertEqual(
                model.log_hazard_ratio(sc.SURVIVAL_REFERENCE_ARM, {"biomarker_std": 1.0}),
                0.0,
            )

    def test_the_standard_error_is_positive_and_shrinks_with_the_cohort(self):
        """A standard error that does not respond to sample size is not one."""
        features = {"biomarker_std": 0.4, "marker_positive": 1.0, "prior_line": 1.0}
        small = sd.SurvivalBlipModel(sc.generate_survival_cohort(500, seed=91))
        large = sd.SurvivalBlipModel(sc.generate_survival_cohort(4000, seed=91))
        for arm in ("arm-a", "arm-b", "arm-c"):
            with self.subTest(arm=arm):
                self.assertGreater(small.standard_error(arm, features), 0.0)
                self.assertLess(
                    large.standard_error(arm, features),
                    small.standard_error(arm, features),
                )


class DirectionTests(unittest.TestCase):
    """The single easiest thing to get wrong when porting onto a hazard."""

    @classmethod
    def setUpClass(cls):
        cls.model = sd.SurvivalBlipModel(sc.generate_survival_cohort(_N, seed=71))

    def test_it_recommends_the_lowest_hazard_not_the_highest(self):
        """`q_values` run one way and a hazard runs the other. An argmax here
        would recommend the worst arm while looking entirely reasonable, so it
        is asserted against the model's own numbers rather than trusted."""
        for seed_features in (
            {"biomarker_std": -1.2, "marker_positive": 1.0, "prior_line": 0.0},
            {"biomarker_std": 0.8, "marker_positive": 0.0, "prior_line": 2.0},
            {"biomarker_std": 0.0, "marker_positive": 1.0, "prior_line": 1.0},
        ):
            chosen = self.model.recommend(seed_features)
            ratios = {
                arm: self.model.log_hazard_ratio(arm, seed_features)
                for arm in self.model.arms
            }
            with self.subTest(features=seed_features):
                self.assertEqual(ratios[chosen], min(ratios.values()))

    def test_it_usually_picks_the_arm_the_generator_says_is_best(self):
        """Recovery of parameters is not the same as recovery of the decision,
        so the decision is scored too."""
        agreed = total = 0
        for index in range(200):
            features = {
                "biomarker_std": -2.0 + 0.02 * index,
                "marker_positive": float(index % 2),
                "prior_line": float(index % 3),
            }
            truth = min(
                sc.SURVIVAL_ARMS,
                key=lambda a: (sc.true_log_hazard_ratio(a, features), a),
            )
            agreed += self.model.recommend(features) == truth
            total += 1
        self.assertGreater(agreed / total, 0.8, f"agreed on {agreed}/{total}")


class WeightingTests(unittest.TestCase):
    """The two weights, each scored on the thing it is there to do."""

    def test_the_dwols_weight_balances_the_covariates(self):
        """The mechanism, asserted directly rather than through its effect.

        `|A - pi|` satisfies `pi w(1,X) = (1-pi) w(0,X)` pointwise, so the
        weighted covariate means in the two arms must agree. That balance is the
        whole reason a wrong treatment-free surface can be survived, and it is
        cheap and stable to check, where the survival benefit itself is a 0.037
        effect needing more seeds than this suite can spend.
        """
        trajectories = sc.generate_survival_cohort(4000, seed=71)
        model = sd.SurvivalBlipModel(trajectories)
        rows = model._rows_for(trajectories, "arm-b")
        weights = [
            abs(row.assignment - p) * row.censor_weight
            for row, p in zip(rows, model._propensities(rows))
        ]

        def imbalance(row_weights):
            worst = 0.0
            for name in ("biomarker_std", "marker_positive", "prior_line",
                         "performance_status"):
                totals = {0.0: [0.0, 0.0], 1.0: [0.0, 0.0]}
                for row, weight in zip(rows, row_weights):
                    totals[row.assignment][0] += weight * row.features.get(name, 0.0)
                    totals[row.assignment][1] += weight
                means = [
                    totals[side][0] / totals[side][1] if totals[side][1] else 0.0
                    for side in (0.0, 1.0)
                ]
                worst = max(worst, abs(means[0] - means[1]))
            return worst

        unweighted = imbalance([1.0] * len(rows))
        weighted = imbalance(weights)
        self.assertLess(
            weighted,
            unweighted / 2.0,
            f"weighted imbalance {weighted:.4f} against unweighted {unweighted:.4f}",
        )

    def test_ipcw_removes_bias_on_the_complete_case_comparator(self):
        """IPCW is the *comparator* path's correction, not the deployed one.

        Buckley-James keeps the censored rows and handles them by imputation, so
        there is no censoring weight left to remove — monkeypatching it away
        changes nothing, which is how this test caught its own premise going
        stale when the default changed. It now builds the comparator explicitly.
        """
        cohorts = [sc.generate_survival_cohort(_N, seed=s) for s in _WEIGHTING_SEEDS]
        with_ipcw = sum(abs(v) for v in _pooled_bias(
            [sd.SurvivalBlipModel(c, use_buckley_james=False) for c in cohorts]).values())

        original = sd.SurvivalBlipModel._rows_for

        def without(self, trajectories, arm):
            return [
                sd._Row(**dict(row.__dict__, censor_weight=1.0))
                for row in original(self, trajectories, arm)
            ]

        try:
            sd.SurvivalBlipModel._rows_for = without
            naive = sum(abs(v) for v in _pooled_bias(
                [sd.SurvivalBlipModel(c, use_buckley_james=False) for c in cohorts]).values())
        finally:
            sd.SurvivalBlipModel._rows_for = original

        self.assertGreater(
            naive,
            with_ipcw * 1.2,
            f"IPCW bias {with_ipcw:.4f} against complete-case {naive:.4f}",
        )


class BuckleyJamesTests(unittest.TestCase):
    """Keeping the censored rows rather than weighting the survivors.

    Invariant 77. The censored rows are not missing at random — they are
    right-censored, and what is known is that the event came later than the time
    recorded. Discarding them and reweighting the rest cannot recover that when
    the truncation is administrative, which invariant 76 measured.
    """

    SEEDS = (71, 72, 73, 74, 75, 76)
    SIZE = 1500

    @classmethod
    def setUpClass(cls):
        cls.cohorts = [sc.generate_survival_cohort(cls.SIZE, seed=s) for s in cls.SEEDS]
        cls.imputed = [sd.SurvivalBlipModel(c) for c in cls.cohorts]
        cls.complete = [
            sd.SurvivalBlipModel(c, use_buckley_james=False) for c in cls.cohorts
        ]

    def test_it_is_the_default(self):
        """The comparator has to be asked for by name, not stumbled into."""
        self.assertTrue(self.imputed[0].use_buckley_james)
        self.assertTrue(all(fit.imputed for fit in self.imputed[0].fits.values()))
        self.assertFalse(any(fit.imputed for fit in self.complete[0].fits.values()))

    def test_the_censored_rows_are_kept(self):
        """The whole mechanism in one assertion: the comparator throws them
        away and this does not."""
        kept = self.imputed[0]._rows_for(self.cohorts[0], "arm-b")
        dropped = self.complete[0]._rows_for(self.cohorts[0], "arm-b")
        self.assertGreater(len(kept), len(dropped))
        self.assertTrue(any(not row.event_observed for row in kept))
        self.assertTrue(all(row.event_observed for row in dropped))
        self.assertTrue(all(row.censor_weight == 1.0 for row in kept),
                        "imputation and a censoring weight would count it twice")

    def test_it_recovers_the_parameters_better_than_the_complete_case(self):
        """The reason it is the default, scored as a bias pooled over seeds."""
        imputed = sum(abs(v) for v in _pooled_bias(self.imputed).values())
        complete = sum(abs(v) for v in _pooled_bias(self.complete).values())
        self.assertLess(
            imputed,
            complete,
            f"Buckley-James bias {imputed:.4f} against complete case {complete:.4f}",
        )

    def test_the_shape_comes_from_the_recorded_times(self):
        """The bug this nearly shipped with.

        The shape describes the error distribution, and a censoring indicator
        only means something against the time actually observed. Scored against
        the *imputed* residuals instead — which is what the first integration
        did — it reads about 1.343 against a true 1.4. From the recorded ones it
        reads about 1.403. That is a 4% scale error on every published `psi`,
        because `psi = -shape * coefficient`.
        """
        shapes = [model.shape for model in self.imputed]
        self.assertAlmostEqual(
            statistics.mean(shapes),
            sc.WEIBULL_SHAPE,
            delta=0.04,
            msg=f"shapes {[round(s, 4) for s in shapes]}",
        )

    def test_the_inflation_is_applied_to_buckley_james_and_only_to_it(self):
        """The wiring, which is what a test can settle here.

        Whether `_BJ_SE_INFLATION` is *calibrated* is a 20-refit measurement and
        is recorded in the constant's comment, not asserted here: at the six
        seeds this suite can afford, `pstdev` underestimates the sampling spread
        badly enough that the comparison passes whatever the constant says. The
        first version of this test did exactly that — it compared a reported SE
        against a spread of 0.039 where 20 refits give 0.102, so it could not
        have failed for the right reason. Invariant 25's defect, found by
        writing a second test that contradicted the first.
        """
        probe = {"biomarker_std": 0.3, "marker_positive": 1.0, "prior_line": 1.0,
                 "performance_status": 0.35}
        self.assertGreater(sd._BJ_SE_INFLATION, 1.0)
        for arm in ("arm-a", "arm-b", "arm-c"):
            imputed_fit = self.imputed[0].fits[arm]
            complete_fit = self.complete[0].fits[arm]
            with self.subTest(arm=arm):
                self.assertTrue(imputed_fit.imputed)
                self.assertFalse(complete_fit.imputed)
                # Same fit, inflation off: the published number must be exactly
                # the constant times the uninflated one, so a later change
                # cannot quietly drop it.
                bare = replace(imputed_fit, imputed=False)
                self.assertAlmostEqual(
                    imputed_fit.standard_error(probe),
                    sd._BJ_SE_INFLATION * bare.standard_error(probe),
                    places=12,
                )
                self.assertGreater(bare.standard_error(probe), 0.0)


class CensoringCurveTests(unittest.TestCase):
    """The roles are swapped from the usual Kaplan-Meier, which is easy to get
    backwards and produces a plausible-looking curve either way."""

    @classmethod
    def setUpClass(cls):
        cls.trajectories = sc.generate_survival_cohort(_N, seed=71)
        cls.curve = sd.kaplan_meier_censoring(cls.trajectories)

    def test_it_is_a_survival_curve(self):
        values = [survival for _, survival in self.curve]
        self.assertTrue(all(0.0 <= v <= 1.0 for v in values))
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertLess(values[-1], 1.0, "nothing was ever censored")

    def test_it_estimates_censoring_and_not_progression(self):
        """Which endings drive the curve, counted rather than eyeballed.

        The roles are swapped from the usual estimate, and a curve fit the wrong
        way round looks entirely plausible — it is monotone, starts at 1 and
        ends near 0 either way. What distinguishes them is *how many times it
        steps down*: once per censoring event here, and once per progression if
        the roles were reversed. Progression is the majority of endings, so the
        two counts are far apart.
        """
        stages = [s for t in self.trajectories for s in t.stages]
        censored = sum(s.cause != "progression" for s in stages)
        progressed = len(stages) - censored
        self.assertGreater(progressed, censored, "the fixture has changed shape")

        drops = sum(
            1
            for (_, before), (_, after) in zip(self.curve, self.curve[1:])
            if after < before
        )
        # The first step can itself be a drop, which the pairwise walk above
        # cannot see.
        drops += self.curve[0][1] < 1.0
        self.assertLessEqual(drops, censored)
        self.assertGreater(drops, censored * 0.5, f"{drops} drops, {censored} censored")

    def test_the_curve_reaching_zero_drops_no_rows(self):
        """`_rows_for` skips a row whose `S_C(t)` is zero, and the final step of
        any Kaplan-Meier estimate is zero when the last observation is an event
        for that curve. Worth pinning that this is the tail and not a silent
        loss of data — and that the weights it produces stay bounded."""
        self.assertEqual(self.curve[-1][1], 0.0)
        model = sd.SurvivalBlipModel(self.trajectories)
        events = [
            stage
            for t in self.trajectories
            for stage in t.stages
            if stage.arm in ("arm-b", model.reference)
            and stage.cause == "progression"
            and stage.months > 0.0
        ]
        dropped = sum(
            1
            for stage in events
            if sd._censoring_survival(self.curve, stage.months) <= 0.0
        )
        self.assertEqual(dropped, 0, f"{dropped} of {len(events)} rows dropped")
        weights = [row.censor_weight for row in model._rows_for(self.trajectories, "arm-b")]
        self.assertLess(max(weights) / sum(weights), 0.01, "one row carries the fit")

    def test_the_weight_is_finite_and_at_least_one(self):
        """`1 / S_C(t)` upweights a survivor to stand in for those lost; a
        weight below 1 would mean it was standing in for less than itself."""
        for _, survival in self.curve:
            if survival > 0.0:
                self.assertGreaterEqual(1.0 / survival, 1.0)


class ScaleTests(unittest.TestCase):
    """`psi = -k * coefficient`, and both halves of that have to be right."""

    def test_a_longer_time_is_a_lower_hazard(self):
        """The sign convention, asserted on the arithmetic rather than assumed.

        A positive log-time coefficient means the arm buys time, which is a
        *negative* log-hazard ratio. Getting this backwards flips every
        recommendation in the package.
        """
        model = sd.SurvivalBlipModel(sc.generate_survival_cohort(_N, seed=71))
        n_free = len(sd.SURVIVAL_TREATMENT_FREE_BASIS)
        for arm, fit in model.fits.items():
            with self.subTest(arm=arm):
                for offset, psi in enumerate(fit.psi):
                    coefficient = fit.aft_coefficients[n_free + offset]
                    self.assertAlmostEqual(psi, -fit.shape * coefficient, places=12)

    def test_the_declared_hazard_ratio_is_on_the_scale_the_generator_uses(self):
        """End to end: the estimate and `true_log_hazard_ratio` must be
        comparable numbers, not merely correlated ones."""
        model = sd.SurvivalBlipModel(sc.generate_survival_cohort(4000, seed=71))
        features = {"biomarker_std": 0.3, "marker_positive": 1.0, "prior_line": 1.0}
        for arm in ("arm-a", "arm-b", "arm-c"):
            with self.subTest(arm=arm):
                self.assertAlmostEqual(
                    model.log_hazard_ratio(arm, features),
                    sc.true_log_hazard_ratio(arm, features),
                    delta=0.25,
                )


if __name__ == "__main__":
    unittest.main()
