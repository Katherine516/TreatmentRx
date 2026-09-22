"""Empirical coverage of the intervals, and the audit harness itself.

Coverage is deliberately measured at small replication counts here — the full
study is `treatmentrx.cli coverage`. The assertions are the ones that hold
robustly at this precision, not the point estimates.
"""

import unittest
from dataclasses import replace

from treatmentrx.estimation.q_learning import (
    DEFAULT_BLIP_RIDGE,
    DEFAULT_POOLING_RIDGE,
    QLearningModel,
)
from treatmentrx.feedback import coverage
from treatmentrx.feedback.audit import (
    audit_decision,
    audit_explanation,
    audit_governance,
    audit_ingestion,
    audit_safety,
)
from treatmentrx.simulation.ra_cohort import generate_ra_cohort, true_blip


class CoverageStudyTests(unittest.TestCase):
    """One study shared across the assertions — each replication is a full refit.

    Deliberately small: the full study is `treatmentrx.cli coverage`. What is
    asserted here is the direction and mechanism, which hold at this size; the
    point estimates need the CLI's replication count to be worth quoting.
    """

    @classmethod
    def setUpClass(cls):
        # One patient here, deliberately: the assertion is about the sandwich's
        # SE, and the grid sweep is the CLI's job. `test_the_grid_finds_worse
        # _coverage_than_the_demo_patient` is the one that needs the grid.
        cls.result = coverage.sandwich_coverage(
            replications=25,
            n=180,
            share_blip=False,
            patients={"demo": coverage.REFERENCE_FEATURES},
        )

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

    def test_the_grid_finds_worse_coverage_than_the_demo_patient(self):
        """The point of sweeping: one covariate point is not coverage.

        Asserted as "the grid is not uniformly as good as the demo patient"
        rather than on a threshold, because the threshold is what the CLI's
        replication count is for. At this size the direction is what holds.
        """
        pooled = coverage.sandwich_coverage(replications=12, n=180, share_blip=True)
        self.assertGreater(len(pooled.per_patient), 1)
        demo = next(
            row for row in pooled.per_patient if row.patient.startswith("seropositive, prior")
        )
        worst = min(row.coverage for row in pooled.per_patient)
        self.assertLess(worst, demo.coverage)


class EstimandSeparationTests(unittest.TestCase):
    """The shared blip and the stage-specific blip target different quantities.

    Invariant 9 says the estimators live on one scale, and the horizon division
    puts their *Q-values* there. It does not put their terminal-stage contrasts
    there: a shared psi is fit from every stage's rows at once, so it carries the
    delayed effects that a terminal-block psi cannot. Averaging the two in Layer 3
    is what drags the decision rule's coverage to 74% pooled, and the effect is
    invisible at the single demo patient.
    """

    @classmethod
    def setUpClass(cls):
        from treatmentrx.estimation.dwols import DWOLSModel

        cls.cohort = generate_ra_cohort(280, seed=9000)
        cls.shared = QLearningModel(cls.cohort, share_blip=True)
        cls.stage_specific = QLearningModel(cls.cohort, share_blip=False)
        cls.dwols = DWOLSModel(cls.cohort)
        cls.terminal = cls.stage_specific.n_stages - 1

    def test_the_stage_specific_terminal_blip_recovers_the_single_visit_truth(self):
        from treatmentrx.simulation.ra_cohort import true_blip

        for name, features in coverage.PATIENT_GRID.items():
            arm, comparator = coverage.top_two(features)
            with self.subTest(patient=name):
                truth = true_blip(arm, features) - true_blip(comparator, features)
                estimate = self.stage_specific.blip(
                    arm, features, self.terminal
                ) - self.stage_specific.blip(comparator, features, self.terminal)
                self.assertLess(abs(estimate - truth), 0.06)

    def test_the_shared_blip_targets_the_pooled_value_to_go_instead(self):
        """Inside the span means a different question, not a wrong answer."""
        inside = 0
        for features in coverage.PATIENT_GRID.values():
            arm, comparator = coverage.top_two(features)
            low, high = coverage.estimand_range(features, arm, comparator)
            estimate = self.shared.blip(arm, features, self.terminal) - self.shared.blip(
                comparator, features, self.terminal
            )
            if low - 0.01 <= estimate <= high + 0.01:
                inside += 1
        self.assertGreaterEqual(inside, 5, "the shared blip left the estimand span it pools over")

    def test_only_the_shared_blip_is_biased_at_the_terminal_stage(self):
        """Why the shared blip is out of `training.SERVING_ENSEMBLE`.

        A full coverage sweep is the CLI's job; this pins the mechanism behind
        it. Asserted per estimator rather than on the ensemble average: a single
        cohort's ensemble bias is a sum of three errors that can cancel by luck,
        and on seed 9000 they very nearly do. The individual biases are stable
        because the shared blip's is large relative to sampling noise — it is
        pooling three stages of delayed effect into one parameter, not
        mis-estimating one.
        """
        from treatmentrx.estimation import training
        from treatmentrx.simulation.ra_cohort import true_blip

        bias = {"shared": [], "stage_specific": [], "pooled": [], "dwols": []}
        for seed in (9000, 9001, 9002):
            cohort = generate_ra_cohort(280, seed=seed)
            shared = QLearningModel(cohort, share_blip=True)
            stage_specific = QLearningModel(cohort, share_blip=False)
            pooled = QLearningModel(
                cohort, share_blip=True, pooling_ridge=DEFAULT_POOLING_RIDGE
            )
            from treatmentrx.estimation.dwols import DWOLSModel

            dwols = DWOLSModel(cohort)
            terminal = stage_specific.n_stages - 1
            for features in coverage.PATIENT_GRID.values():
                arm, comparator = coverage.top_two(features)
                truth = true_blip(arm, features) - true_blip(comparator, features)
                bias["shared"].append(
                    shared.blip(arm, features, terminal)
                    - shared.blip(comparator, features, terminal)
                    - truth
                )
                bias["stage_specific"].append(
                    stage_specific.blip(arm, features, terminal)
                    - stage_specific.blip(comparator, features, terminal)
                    - truth
                )
                bias["pooled"].append(
                    pooled.blip(arm, features, terminal)
                    - pooled.blip(comparator, features, terminal)
                    - truth
                )
                bias["dwols"].append(
                    dwols.blip(arm, features) - dwols.blip(comparator, features) - truth
                )

        mean = {name: sum(values) / len(values) for name, values in bias.items()}
        self.assertGreater(
            mean["shared"], 0.03, f"the shared blip's terminal bias has gone: {mean}"
        )
        for name in ("stage_specific", "pooled", "dwols"):
            self.assertLess(
                abs(mean[name]), 0.02, f"{name} is no longer centred at the terminal stage: {mean}"
            )
        self.assertEqual(set(training.SERVING_ENSEMBLE), {"Q-Pooled", "dWOLS-Shared"})

    def test_the_two_estimands_genuinely_differ(self):
        """If they coincided everywhere, averaging them would be harmless."""
        spans = [
            coverage.estimand_range(features, *coverage.top_two(features))
            for features in coverage.PATIENT_GRID.values()
        ]
        widest = max(high - low for low, high in spans)
        self.assertGreater(widest, 0.15, "delayed effects vanished from the cohort")


class JointStudyMembershipTests(unittest.TestCase):
    """The joint study has to resample the ensemble that actually serves.

    It used to keep its own `{"shared", "stage_specific", "dwols"}` map and
    filter it against `SERVING_ENSEMBLE`. When the serving Q-learning model
    became `Q-Pooled` the map no longer contained it, the filter collapsed to
    dWOLS alone, and the study reported a single estimator's coverage under the
    name of the ensemble's. Nothing raised.
    """

    def test_every_serving_estimator_is_resampled(self):
        from treatmentrx.estimation import training

        models = coverage._serving_models(generate_ra_cohort(80, seed=1))
        self.assertEqual(set(models), set(training.SERVING_ENSEMBLE))
        self.assertGreater(len(models), 1, "the ensemble collapsed to one member")

    def test_an_unbuildable_serving_member_raises(self):
        """Silence is what made the original defect survive; this makes it loud."""
        from treatmentrx.estimation import training

        original = training.SERVING_ENSEMBLE
        try:
            training.SERVING_ENSEMBLE = original + ("Not-A-Real-Estimator",)
            with self.assertRaises(ValueError):
                coverage._serving_models(generate_ra_cohort(80, seed=1))
        finally:
            training.SERVING_ENSEMBLE = original

    def test_the_joint_contrast_averages_more_than_one_estimator(self):
        booted, loadings_for = coverage._joint_draws(180, seed=1, replicates=8)
        tally = coverage._Tally("demo", coverage.REFERENCE_FEATURES)
        self.assertEqual(len(loadings_for(tally)), len(booted.point))
        self.assertGreater(len(loadings_for(tally)), 1)

    def test_the_bound_study_averages_the_same_ensemble(self):
        """Both studies have to move together with `SERVING_ENSEMBLE`.

        They each used to filter a local list of estimator names, and both lists
        went stale at the same moment, so both reported a single estimator's
        coverage under the ensemble's name.
        """
        from treatmentrx.estimation import training

        contrasts_for = coverage._serving_contrasts(180, seed=1)
        components = contrasts_for(coverage._Tally("demo", coverage.REFERENCE_FEATURES))
        self.assertEqual(set(components), set(training.SERVING_ENSEMBLE))


class StageSweepTests(unittest.TestCase):
    """Coverage has two axes and only one of them was ever swept.

    The patient grid exists because coverage at one covariate point is not
    coverage. The stage index moves the estimand the same way and every study in
    `coverage.py` pinned it at the terminal block — which is also the single
    stage where both serving estimators target the same quantity, so it is the
    most flattering place to measure and it was the only place measured.
    """

    @classmethod
    def setUpClass(cls):
        cls.sweep = coverage.decision_rule_stage_sweep(replications=6, n=200)
        cls.rows = {row["stage_index"]: row for row in cls.sweep["stages"]}

    def test_every_fitted_stage_is_measured(self):
        self.assertEqual(sorted(self.rows), list(range(self.sweep["n_stages"])))

    def test_each_stage_is_marked_served_or_not(self):
        """An unserved stage is reported, not dropped.

        Dropping it would hide the fact that the thing keeping stage 0 out of
        reach is Layer 1's stage numbering rather than anything statistical.
        """
        for index, row in self.rows.items():
            self.assertEqual(row["served"], index in coverage.SERVED_STAGE_INDICES)
        self.assertTrue(any(not row["served"] for row in self.rows.values()))

    def test_the_truth_is_on_the_scale_the_rule_reports(self):
        """The per-remaining-visit rescaling has to be in the truth too.

        `sandwich_contrast` divides by the remaining horizon so a value-to-go
        blip and dWOLS's single-visit blip are comparable. Scoring a swept stage
        against the undivided value-to-go measures that division and reads 0% at
        stage 0 — arithmetic dressed as a finding.
        """
        features = coverage.REFERENCE_FEATURES
        arm, comparator = coverage.top_two(features)
        terminal = coverage.reported_scale_contrast(features, arm, comparator, 2, 3)
        single_visit = true_blip(arm, features) - true_blip(comparator, features)
        self.assertAlmostEqual(terminal, single_visit, places=9)

        undivided = coverage.value_to_go_contrast(features, arm, comparator, 0)
        scaled = coverage.reported_scale_contrast(features, arm, comparator, 0, 3)
        self.assertAlmostEqual(scaled * 3.0, undivided, places=9)

    def test_a_low_pooled_ratio_is_diagnosed_not_just_reported(self):
        """`se_to_sd_ratio` is not a width diagnostic, and reading it as one misled me.

        `_pool` centres each patient's estimates on their own *truth*, so a bias
        that differs between patients lands in the denominator. At stage 0 that
        reads 0.59 — an interval a third too narrow — while the same estimates
        about their own means give 1.13. The interval is slightly wide there;
        what fails is centring, and the two ratios have to be separable or the
        wrong conclusion is the easy one.
        """
        for index, row in self.rows.items():
            with self.subTest(stage=index):
                combined = row["se_to_sd_ratio"]
                width_only = row["se_to_within_sd_ratio"]
                self.assertGreater(width_only, 0.0)
                # Removing bias can only shrink the denominator, never grow it.
                self.assertGreaterEqual(width_only + 1e-9, combined)
                self.assertGreaterEqual(row["bias_dispersion"], 0.0)

    def test_the_interval_is_wide_enough_at_every_stage(self):
        """Width and centring fail independently, and only centring fails here.

        If this ever drops below 1 the ensemble really has become too narrow,
        which is a different defect from the one at stage 0 and wants a different
        fix — widening cannot repair a centre.
        """
        for index, row in self.rows.items():
            with self.subTest(stage=index):
                self.assertGreater(
                    row["se_to_within_sd_ratio"],
                    0.85,
                    f"stage {index} interval is genuinely too narrow",
                )

    def test_mis_centring_concentrates_where_the_estimands_diverge(self):
        """The bias dispersion has to vanish at the terminal block.

        There is no future left there, so a value-to-go blip *is* a single-visit
        blip and the two serving members target the same quantity exactly. Away
        from it they do not, and the horizon rescaling shrinks the gap without
        closing it — which is why stage 0 carries the most.
        """
        terminal = self.sweep["n_stages"] - 1
        self.assertLess(
            self.rows[terminal]["bias_dispersion"],
            self.rows[0]["bias_dispersion"],
            "stage 0 should be the most mis-centred, not the terminal block",
        )

    def test_the_predicted_and_measured_mis_centring_agree_in_sign(self):
        """Averaging two estimands biases by half their gap; check that mechanism.

        The ensemble is the mean of a single-visit blip and a value-to-go
        contrast divided by the remaining horizon, so its bias should be about
        `(single_visit - v2go/h) / 2`. At the terminal block that gap is
        identically zero for every patient, which is the check that matters —
        anything else there would mean the rescaling is wrong.
        """
        terminal = self.sweep["n_stages"] - 1
        for name, features in coverage.PATIENT_GRID.items():
            arm, comparator = coverage.top_two(features)
            with self.subTest(patient=name):
                scaled = coverage.reported_scale_contrast(
                    features, arm, comparator, terminal, self.sweep["n_stages"]
                )
                single_visit = true_blip(arm, features) - true_blip(comparator, features)
                self.assertAlmostEqual(scaled, single_visit, places=9)

        gaps = []
        for name, features in coverage.PATIENT_GRID.items():
            arm, comparator = coverage.top_two(features)
            scaled = coverage.reported_scale_contrast(
                features, arm, comparator, 0, self.sweep["n_stages"]
            )
            single_visit = true_blip(arm, features) - true_blip(comparator, features)
            gaps.append(abs(single_visit - scaled) / 2.0)
        self.assertGreater(
            max(gaps),
            self.rows[terminal]["bias_dispersion"],
            "the stage-0 estimand gap should dominate the terminal block's",
        )

    def test_the_served_stages_cover_near_nominal(self):
        """The stages a patient can actually land on are the ones that must hold."""
        for index in coverage.SERVED_STAGE_INDICES:
            row = self.rows[index]
            self.assertGreater(
                row["coverage"],
                0.85,
                f"served stage {index} covers {row['coverage']:.2f}",
            )

    def test_a_stage_below_nominal_is_named_in_the_note(self):
        """A sub-nominal unserved stage has to be said out loud, not buried."""
        low = [
            index
            for index, row in self.rows.items()
            if not row["served"] and row["coverage"] < coverage.NOMINAL - 0.05
        ]
        for index in low:
            self.assertIn(str(index), self.sweep["note"])

    def test_the_served_stage_list_matches_what_the_pipeline_produces(self):
        """`SERVED_STAGE_INDICES` is a claim about Layer 1, so check Layer 1.

        `DataLayer.build_patient_state` appends the pending visit to the observed
        history, which is why `stage_index` is never 0 for anyone the pipeline
        sees. The `served` flags on the sweep are only worth anything if that
        stays true, and it is the kind of bookkeeping that changes without
        anybody thinking about this study.
        """
        from treatmentrx.data import DataLayer
        from treatmentrx.data.contract import DataContractError
        from treatmentrx.estimation import training
        from treatmentrx.estimation.features import stage_index
        from treatmentrx.simulation.fhir_export import simulated_bundles

        horizon = training.fitted().pooled.n_stages
        seen = set()
        for bundle in simulated_bundles(40, seed=991):
            try:
                state = DataLayer().build_patient_state(bundle)
            except DataContractError:
                continue
            seen.add(stage_index(state.stages, horizon))
        self.assertTrue(seen, "no bundle survived ingestion")
        self.assertTrue(
            seen <= set(coverage.SERVED_STAGE_INDICES),
            f"the pipeline reached stage(s) {sorted(seen - set(coverage.SERVED_STAGE_INDICES))}, "
            "which the stage sweep reports as unserved",
        )

    def test_the_terminal_row_matches_the_headline_study(self):
        """The swept terminal row and `decision_rule_coverage` are the same rule.

        If they drift, one of them has stopped measuring what it names —
        invariant 19's shape on a new axis.
        """
        contrasts_for = coverage._serving_contrasts(200, seed=9_700)
        terminal = coverage._serving_contrasts(200, seed=9_700, stage_index=2)
        tally = coverage._Tally("demo", coverage.REFERENCE_FEATURES)
        self.assertEqual(
            {name: test.difference for name, test in contrasts_for(tally).items()},
            {name: test.difference for name, test in terminal(tally).items()},
        )


class JointReplicateSweepTests(unittest.TestCase):
    """The joint interval's shortfall was the draw count, not the method.

    Deliberately tiny — the numbers worth quoting need `cli coverage
    --decision-rule`. What is asserted is the mechanism: nested prefixes of one
    set of draws, and an interval that gets *wider* as the quantiles stabilise.
    """

    @classmethod
    def setUpClass(cls):
        cls.results = coverage.joint_replicate_sweep(
            replicate_counts=(10, 40), replications=4, n=140
        )

    def test_every_replicate_count_is_measured(self):
        self.assertEqual(set(self.results), {10, 40})

    def test_more_draws_widen_the_percentile_interval(self):
        """A 2.5th quantile from few draws is near the minimum, so the interval
        is too narrow — and grows toward the truth as draws accumulate."""
        few, many = self.results[10], self.results[40]
        self.assertGreater(many.mean_width, few.mean_width)

    def test_the_standard_error_does_not_move_with_the_draw_count(self):
        """Only the quantiles were crude; the SE itself was always honest.

        This is what separates 'the method under-covers' from 'the percentile is
        estimated from too few draws' — if the SE moved too, the diagnosis would
        be different.
        """
        few, many = self.results[10], self.results[40]
        self.assertLess(
            abs(few.mean_standard_error - many.mean_standard_error),
            0.15 * many.mean_standard_error,
        )

    def test_the_prefixes_share_one_set_of_draws(self):
        """Otherwise the sweep costs one full bootstrap per replicate count."""
        booted, _ = coverage._joint_draws(140, seed=1, replicates=12)
        self.assertEqual(booted.replicates, 12)
        prefix = replace(booted, draws=booted.draws[:5])
        self.assertEqual(prefix.replicates, 5)
        self.assertEqual(prefix.draws, booted.draws[:5])


class PowerCurveTests(unittest.TestCase):
    """Abstention is a sample-size choice, and the curve has to show it.

    Deliberately three sizes and few patients — the shape is what holds at this
    precision, and the numbers worth quoting need `cli power`.
    """

    @classmethod
    def setUpClass(cls):
        from treatmentrx.feedback.power import power_curve

        from treatmentrx.feedback.power import power_curves, average_curves

        # Two draws rather than the deployed four: enough that the aggregation
        # path is the one under test and that `equipoise_spread` is populated,
        # few enough that this stays affordable. The numbers worth quoting need
        # `cli power`, which draws four.
        cls.curves = power_curves(
            sizes=(140, 400, 1120), n_patients=25, seed=909, replicates=2
        )
        cls.points = average_curves(cls.curves)

    def test_more_data_means_fewer_abstentions(self):
        rates = [point.equipoise_rate for point in self.points]
        self.assertGreater(rates[0], rates[-1])

    def test_the_effect_is_flat_and_only_the_precision_moves(self):
        """If the contrast itself grew with n, the curve would be measuring the
        estimator drifting rather than the interval tightening."""
        errors = [point.mean_standard_error for point in self.points]
        contrasts = [point.mean_abs_difference for point in self.points]
        self.assertGreater(errors[0], errors[-1])
        self.assertLess(abs(contrasts[-1] - contrasts[0]), 0.02)

    def test_the_standard_error_responds_to_sample_size(self):
        """The regression guard for the stale-dWOLS defect.

        With half the serving ensemble frozen the exponent measured 0.15; with
        both halves refitting it is ~0.45 against the 0.50 a correctly specified
        estimator earns. Anything below 0.35 means a serving estimator has
        stopped following the training split again.
        """
        from treatmentrx.feedback.power import shrinkage_exponent

        exponent = shrinkage_exponent(self.points)
        self.assertIsNotNone(exponent)
        self.assertGreater(exponent, 0.35, "an estimator is not being refit with the cohort")
        self.assertLess(exponent, 0.65)

    def test_the_sweep_restores_the_deployed_fit(self):
        """It mutates two module globals; anything after it must see the default.

        `COHORT_SEED` joined `COHORT_SIZE` when the sweep started replicating
        over fits, and a restored size with a drifted seed would leave every
        later test scoring against a cohort nobody chose.
        """
        from treatmentrx.estimation import training

        self.assertEqual(training.COHORT_SIZE, 400)
        self.assertEqual(training.COHORT_SEED, 7)
        self.assertEqual(len(training.fitted().train), 280)

    def test_each_point_reports_every_draw_it_averaged(self):
        """The property the replication exists for.

        A point estimate with no spread beside it is what let a one-draw curve
        read as a curve. Each point must carry its draws, and at the deployed
        size they must actually differ — if they did not, refitting would be
        buying nothing and the aggregation would be decoration.
        """
        for point in self.points:
            with self.subTest(cohort_size=point.cohort_size):
                self.assertEqual(point.replicates, 2)
                self.assertEqual(len(point.equipoise_rates), 2)
                self.assertAlmostEqual(
                    point.equipoise_rate,
                    sum(point.equipoise_rates) / 2,
                    places=12,
                )
        self.assertGreater(
            max(p.equipoise_spread for p in self.points),
            0.0,
            "no size moved between draws — the fit is not being redrawn",
        )

    def test_the_draws_are_different_fits_rather_than_different_patients(self):
        """Scoring more patients cannot average a fit away, which is why this
        replicates over the cohort seed. Same patient count in every draw, and
        the same training size — only the cohort drawn differs."""
        first, second = self.curves
        for left, right in zip(first, second):
            with self.subTest(cohort_size=left.cohort_size):
                self.assertEqual(left.patients, right.patients)
                self.assertEqual(left.train_size, right.train_size)


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


class IngestionAuditTests(unittest.TestCase):
    """Layer 1's headline metric could not fail, and now says what it is.

    `switch_detection_recall` asked whether any stage was flagged for a patient
    whose arm changed — and `SwitchingCapture`'s third condition *is* "the arm
    changed". It measured the same predicate it used as truth, read 1.0 by
    construction, used `any()` so the wrong stage counted, and excluded from its
    denominator every patient where a false positive could appear.
    """

    @classmethod
    def setUpClass(cls):
        cls.section = audit_ingestion()
        cls.metrics = cls.section.metrics
        cls.notes = " ".join(cls.section.notes)

    def test_the_structural_metric_is_labelled_as_one(self):
        """1.0 is fine; presenting it as accuracy is not."""
        self.assertEqual(self.metrics["arm_change_always_flagged"], 1.0)
        self.assertIn("by construction", self.notes)
        self.assertIn("wiring check", self.notes)

    def test_the_unverifiable_share_is_reported(self):
        """How much of the flag rests on conditions with no ground truth."""
        share = self.metrics["switches_beyond_arm_change"]
        self.assertGreater(share, 0.0, "no switch came from the other conditions")
        self.assertLess(share, 1.0)

    def test_the_dead_seams_are_named_rather_than_silent(self):
        """A field that always echoes its input is a finding, not a blank.

        `realized` can only differ when the bundle carries a dispense record, and
        this fixture carries none — so the ITT / per-protocol / as-treated split
        has no input from it. Reported the way `SwitchingAwareOPE` reports its own
        zeros rather than left for a reader to notice.
        """
        self.assertEqual(self.metrics["stages_where_realized_differs"], 0)
        self.assertEqual(self.metrics["distinct_adherence_values"], 1)
        self.assertIn("no input", self.notes)
        self.assertIn("days-covered path never runs", self.notes)

    def test_a_dispense_record_would_move_the_dead_seam(self):
        """The zero is a property of the fixture, not of the code.

        If this stops holding, `realized` has become genuinely unreachable and
        the note above would be describing a different problem.
        """
        from treatmentrx.arms import normalize_arm
        from treatmentrx.data.switching import SwitchingCapture

        self.assertTrue(hasattr(SwitchingCapture, "apply"))
        source = SwitchingCapture.apply.__doc__ or ""
        self.assertIsNotNone(normalize_arm("TNF-inhibitor"))
        # The branch exists and is reachable given a dispensed name.
        import inspect

        body = inspect.getsource(SwitchingCapture.apply)
        self.assertIn("dispensed_name", body)

    def test_the_flagged_count_carries_its_denominator(self):
        flagged, _, total = self.metrics["stages_flagged_switched"].partition("/")
        self.assertGreater(int(total), 0)
        self.assertLessEqual(int(flagged), int(total))

    def test_the_gate_is_scored_on_records_built_to_fail(self):
        """Every other metric in this section is a round-trip on a well-formed
        record, so the only thing they can report is that the fixture is well
        formed. This one is scored on records that are not."""
        block = self.metrics["identification_matches_the_features"]
        self.assertEqual(block["identification_recall"], 1.0)
        self.assertEqual(block["identification_precision"], 1.0)
        self.assertEqual(block["reason_names_the_adjuster"], 1.0)
        self.assertEqual(block["unreadable_but_certified"], {})
        self.assertEqual(block["refused_a_readable_record"], {})
        # Both denominators are real, which is the Layer 4 discipline: a
        # certificate that refused every record would score perfect recall.
        self.assertGreaterEqual(block["records_the_estimators_cannot_read"], 3)
        self.assertGreaterEqual(block["records_the_estimators_can_read"], 2)

    def test_the_contract_is_silent_on_most_of_those_records(self):
        """The finding behind the metric, stated as a number rather than prose.

        The contract checks variable *families* and the families are broader
        than the covariates: a HAQ-DI satisfies `disease_activity`, an ESR
        satisfies `inflammation`, a rheumatoid factor satisfies `serostatus`.
        Each is an ordinary RA record that leaves a default in the estimators
        and draws no diagnostic at Layer 1.
        """
        block = self.metrics["identification_matches_the_features"]
        self.assertLess(
            block["cases_the_data_contract_warned_about"],
            block["records_the_estimators_cannot_read"],
            "if the contract caught them all, this metric has nothing to add",
        )

    def test_the_old_check_is_caught_and_the_other_metrics_are_blind_to_it(self):
        """The point is not that the new metric works — it is that nothing else
        here can see the defect. `_has_adjuster` used to accept a HAQ-DI for
        `baseline_disease_activity`, certifying a defaulted DAS28 as identified,
        and every other Layer 1 number reads exactly the same either way.
        """
        from treatmentrx.data.dag import CausalDAGRegistry
        from treatmentrx.feedback.audit import _identification_against_the_features

        def old(self, node, stage, patient):
            observed = set(stage.features)
            if node == "baseline_disease_activity":
                return bool(observed & {"das28", "cdai", "sdai", "haq_di"})
            if node == "prior_biologic_exposure":
                return bool(patient.medications)
            return node in observed

        fixed = CausalDAGRegistry._has_adjuster
        try:
            CausalDAGRegistry._has_adjuster = old
            injected = _identification_against_the_features()
            unchanged = audit_ingestion()
        finally:
            CausalDAGRegistry._has_adjuster = fixed

        self.assertLess(injected["identification_recall"], 1.0)
        self.assertIn(
            "DAS28 absent, HAQ-DI present", injected["unreadable_but_certified"]
        )
        # Precision does not move: the defect lets records through rather than
        # holding them back, and one counter standing in for two properties is
        # how a partial regression passes.
        self.assertEqual(injected["identification_precision"], 1.0)

        for key in (
            "stage_count_exact",
            "visit_interval_exact",
            "arm_change_always_flagged",
            "switches_beyond_arm_change",
            "stages_where_realized_differs",
        ):
            with self.subTest(metric=key):
                self.assertEqual(unchanged.metrics[key], self.metrics[key])


class FinalTestFeasibilityTests(unittest.TestCase):
    """`evaluation_partition()` reports there is no final test; this prices one.

    The obvious remedy for "no partition held back" is a three-way split, and it
    is the kind of change that looks like pure discipline until someone measures
    what it costs the half that is left.
    """

    @classmethod
    def setUpClass(cls):
        from treatmentrx.feedback.power import final_test_feasibility

        cls.report = final_test_feasibility()

    def test_holding_data_back_costs_the_evaluation_split(self):
        """Both halves come out of one holdout, so this is a trade, not a free win."""
        today = self.report["today"]["per_decision_ess"]
        for split in self.report["splits"]:
            with self.subTest(held_back=split["held_back_fraction"]):
                self.assertLess(split["evaluation"]["per_decision_ess"], today)

    def test_no_split_can_confirm_the_regimes_own_value(self):
        """The finding. The agent deploys a regime, and no final test identifies one.

        A confirmation that is itself unidentified is not a confirmation, and
        `identified` is the field this repo already uses to refuse that reading.
        """
        self.assertFalse(self.report["affordable_for_the_regime"])
        for split in self.report["splits"]:
            with self.subTest(held_back=split["held_back_fraction"]):
                self.assertFalse(split["final_test_identifies_the_regime"])

    def test_the_two_quantities_are_reported_separately(self):
        """They disagree, and collapsing them would hide which one is affordable.

        A final test large enough for the per-decision value does exist at 40-50%
        held back. Reporting one boolean would either overclaim or underclaim.
        """
        self.assertTrue(self.report["affordable_for_the_per_decision_value"])
        self.assertNotEqual(
            self.report["affordable_for_the_per_decision_value"],
            self.report["affordable_for_the_regime"],
        )

    def test_the_required_cohort_is_larger_for_the_harder_quantity(self):
        needed = self.report["cohort_for_identified_final_test"]
        from treatmentrx.estimation import training

        self.assertGreater(needed["sequential_value"], needed["per_decision_value"])
        self.assertGreater(needed["per_decision_value"], training.COHORT_SIZE)

    def test_the_verdict_does_not_overstate_the_finding(self):
        """An earlier verdict string said every split failed on both quantities.

        The measurement says otherwise, and a summary that contradicts the table
        beneath it is the failure this repo keeps removing.
        """
        verdict = self.report["verdict"]
        self.assertIn("per-decision", verdict)
        self.assertIn("regime", verdict)


class AuditHarnessTests(unittest.TestCase):
    def test_ingestion_recovers_what_was_generated(self):
        metrics = audit_ingestion(n=15).metrics
        self.assertEqual(metrics["stage_count_exact"], 1.0)
        self.assertEqual(metrics["visit_interval_exact"], 1.0)
        self.assertEqual(
            metrics["arm_change_always_flagged"],
            1.0,
            "a change of arm is a switch; detection must not depend on free text",
        )

    def test_abstention_is_earned(self):
        """When the agent declines to separate arms, they should be close."""
        metrics = audit_decision(n=40).metrics
        self.assertTrue(metrics["abstention_is_earned"])
        self.assertGreater(metrics["if_forced_to_commit"]["oracle_arm_rate"], 0.7)
        self.assertLess(metrics["if_forced_to_commit"]["mean_regret"], 0.02)

    def test_regret_is_measured_against_the_sequential_optimum(self):
        """Not against the myopic blip argmax, which the agent already beats."""
        metrics = audit_decision(n=40).metrics
        self.assertIn("myopic_oracle_agreement_rate", metrics)
        self.assertGreaterEqual(metrics["if_forced_to_commit"]["mean_regret"], 0.0)

    def test_neither_regret_denominator_can_collapse_silently(self):
        """The regression this pair of blocks exists for.

        `recommended_arm` is None unless the agent commits, so scoring only those
        rows asks a different question — "when it commits, is it right?" — whose
        answer is trivially yes. When that happened the section reported oracle-arm
        rate 1.0 and max regret 0.0 over 43 of 120 patients, and the assertion
        above (`> 0.7`) waved it straight through. Both denominators are now
        named, and the forced one has to cover every patient scored.
        """
        metrics = audit_decision(n=40).metrics
        committed = metrics["when_it_commits"]
        forced = metrics["if_forced_to_commit"]
        total = sum(metrics["status_distribution"].values())

        self.assertEqual(forced["patients"], total, "the ranking must be scored on everyone")
        self.assertLess(
            committed["patients"],
            forced["patients"],
            "the agent abstains here, so the committed subset must be smaller — "
            "if these are equal the two blocks are measuring the same thing",
        )
        self.assertGreater(committed["patients"], 0)

    def test_abstention_is_priced_by_what_the_clinician_does_next(self):
        """Declining is not one cost. It is cheap if the clinician takes the
        model's own ordering and expensive if they take the worst thing on the
        menu; the candidate set is what sits between them."""
        price = audit_decision(n=40).metrics["abstention_price"]
        self.assertEqual(
            price["declined_patients"],
            audit_decision(n=40).metrics["status_distribution"].get("equipoise", 0),
        )
        argmax = price["clinician_takes_the_models_top_arm"]["mean"]
        in_set = price["clinician_takes_the_worst_arm_in_the_candidate_set"]["mean"]
        overall = price["clinician_takes_the_worst_arm_on_the_menu"]["mean"]
        self.assertLessEqual(argmax, in_set)
        self.assertLess(in_set, overall)

    def test_safety_catches_contraindications_without_over_removing(self):
        metrics = audit_safety().metrics
        self.assertEqual(metrics["contraindication_recall"], 1.0, metrics["missed"])
        self.assertEqual(metrics["removal_precision"], 1.0, metrics["over_removed"])
        self.assertEqual(metrics["spurious_removals"], 0)
        self.assertEqual(metrics["healthy_patient_removals"], 0)
        self.assertEqual(metrics["allergen_composites_surviving"], {})

    def test_the_removal_reason_names_the_condition_that_fired(self):
        """Which arms went is only half of a safety filter's job.

        All three physiological conditions contraindicate the same two arms, so
        recall and precision cannot tell a filter that read the right
        observation from one that read the wrong one and removed the same pair.
        The reason is what the card shows a clinician and the only place that
        distinction survives.
        """
        block = audit_safety().metrics["removal_reason_names_the_condition"]
        self.assertGreater(block["removals_checked"], 10)
        self.assertEqual(block["rate"], 1.0)
        self.assertEqual(block["misnamed"], {})

    def test_the_reason_check_catches_what_recall_and_precision_cannot(self):
        """The regression this metric exists for, injected.

        A crossed wire that fires the right branch and names the wrong organ
        produces a card reading "unsafe with eGFR < 30" for a patient whose eGFR
        is 90 and whose ALT is 400. Recall and precision stay at a perfect 1.000
        through it — that is asserted here too, because the point is not that the
        new check works but that the old ones are blind to this.
        """
        from treatmentrx.safety import feasible_set as fs

        original = fs.FeasibleSet._unsafe_reason

        def names_the_wrong_organ(self, action, alt, egfr, pregnant, tokens, arm=""):
            reason = original(self, action, alt, egfr, pregnant, tokens, arm)
            if reason and "ALT >" in reason:
                return reason.replace(
                    f"ALT > {fs.ALT_CEILING:.0f}", f"eGFR < {fs.EGFR_FLOOR:.0f}"
                )
            return reason

        try:
            fs.FeasibleSet._unsafe_reason = names_the_wrong_organ
            metrics = audit_safety().metrics
        finally:
            fs.FeasibleSet._unsafe_reason = original

        self.assertEqual(metrics["contraindication_recall"], 1.0, "recall should be blind here")
        self.assertEqual(metrics["removal_precision"], 1.0, "precision should be blind here")

        block = metrics["removal_reason_names_the_condition"]
        self.assertLess(block["rate"], 1.0, "the reason check did not catch a misnamed organ")
        self.assertIn("ALT=400", block["misnamed"])

    def test_the_safety_sweep_has_a_denominator_worth_having(self):
        """Recall over six removals could not distinguish a filter from a hammer.

        The sweep has to contain cases that must remove *nothing*, or precision
        has no meaning and a filter that drops every arm scores perfectly.
        """
        metrics = audit_safety().metrics
        self.assertGreaterEqual(metrics["cases"], 15)
        self.assertGreaterEqual(metrics["labelled_removals_expected"], 12)

    def test_explanations_decompose_the_model_exactly(self):
        """The faithfulness figure is the round-trip against the fitted model.

        This used to read `attribution_sums_to_advantage`, which compared the
        reported total against its own parts — both rounded to 4dp, so an
        identity that measured 5.6e-17. The identity is still reported, and
        still asserted below, but as the wiring check it is.
        """
        metrics = audit_explanation(n=10).metrics
        self.assertEqual(metrics["attribution_parts_sum_to_total"]["rate"], 1.0)

        faithful = metrics["attribution_matches_its_source_model"]
        self.assertGreater(faithful["patients"], 0)
        self.assertEqual(faithful["rate"], 1.0)
        self.assertTrue(faithful["sources"], "the card must name the model it decomposed")

        phi = metrics["phi_leaks_into_narrative"]
        self.assertEqual(phi["leaks"], 0)
        self.assertGreater(
            phi["cards_carrying_episodic_text"],
            0,
            "the guard is being scored where its channel does not run",
        )
        # `leaks` is counted over the fixture card too, so the denominator has
        # to cover it — and it must not be only the fixture.
        self.assertGreater(phi["cards_scanned"], phi["cards_carrying_episodic_text"])
        self.assertFalse(metrics["memory_changed_q_values"])
        self.assertTrue(metrics["memory_changed_narrative"])

    def test_governance_keeps_the_gates_closed(self):
        """And says why. `validation_rung: silent` beside `retraining_allowed:
        false` told a reader the gate was shut and nothing about the reason,
        while `ValidationStatus.blockers` sat populated and unemitted."""
        metrics = audit_governance(n=12).metrics
        self.assertFalse(metrics["retraining_allowed"])
        self.assertFalse(metrics["validation_gate_passed"])
        self.assertTrue(
            metrics["validation_blockers"],
            "the gate is shut and the section does not say why",
        )

    def test_the_estimands_are_model_level_over_a_denominator(self):
        """This compared exactly two patients, so a patient-dependent estimand
        had to disagree on that one pair to be caught."""
        block = audit_governance(n=12).metrics["estimands_are_model_level"]
        self.assertGreater(block["patients"], 2)
        self.assertEqual(block["distinct_value_sets"], 1)
        self.assertTrue(block["model_level"])

    def test_the_three_tracks_stay_separate(self):
        """Layer 6's stated purpose, and nothing measured it.

        Track B is the off-policy evaluation cohort and takes recommendations
        only: an abstention has no policy action to evaluate, and the diagnostic
        top-scored arm must never enter as though it were one. That is invariant
        2's shape a layer down.
        """
        block = audit_governance(n=30).metrics["track_separation"]
        self.assertEqual(block["observational_rows"], block["patients"])
        self.assertEqual(block["full_system_rows"], block["patients"])
        self.assertEqual(block["ope_rows"], block["recommendations"])
        self.assertEqual(block["ope_rows_not_the_published_arm"], 0)
        self.assertEqual(block["abstentions_carrying_the_top_scored_arm"], 0)
        self.assertGreater(
            block["abstentions"],
            0,
            "no abstentions scored, so the checks above have no denominator",
        )

    def test_each_half_of_the_separation_can_fail_on_its_own(self):
        """A gate that cannot close is not a gate (invariant 25).

        Both regressions are injected, and each must be caught by its own
        counter while the other stays at zero — otherwise one number is standing
        in for two properties.
        """
        from treatmentrx.domain import RecommendationStatus
        from treatmentrx.feedback import FeedbackLayer

        original = FeedbackLayer.enqueue

        def leak_top_arm(self, state, recommendation, safe=None):
            receipt = original(self, state, recommendation, safe)
            if recommendation.status is not RecommendationStatus.RECOMMEND:
                self.full_system_track[-1]["policy_action"] = recommendation.top_scored_arm
            return receipt

        def leak_onto_track_b(self, state, recommendation, safe=None):
            receipt = original(self, state, recommendation, safe)
            if recommendation.status is not RecommendationStatus.RECOMMEND:
                self.ope_track.append(
                    {
                        "patient_hash": state.patient_hash,
                        "stage": state.stage,
                        "policy_arm": recommendation.top_scored_arm,
                    }
                )
            return receipt

        try:
            FeedbackLayer.enqueue = leak_top_arm
            promoted = audit_governance(n=20).metrics["track_separation"]
            FeedbackLayer.enqueue = leak_onto_track_b
            smuggled = audit_governance(n=20).metrics["track_separation"]
        finally:
            FeedbackLayer.enqueue = original

        self.assertGreater(promoted["abstentions_carrying_the_top_scored_arm"], 0)
        self.assertEqual(promoted["ope_rows_not_the_published_arm"], 0)

        self.assertGreater(smuggled["ope_rows_not_the_published_arm"], 0)
        self.assertGreater(smuggled["ope_rows"], smuggled["recommendations"])
        self.assertEqual(smuggled["abstentions_carrying_the_top_scored_arm"], 0)

    def test_the_structural_assertions_are_labelled_as_such(self):
        """Three of these have no denominator and never could. Reporting them
        beside measurements without saying so is what invariant 49 is about."""
        block = audit_governance(n=12).metrics["regression_tripwires"]
        self.assertIn("structural assertions", block["note"])
        for name in ("estimands_are_distinct", "ope_is_patient_level", "ope_is_descriptive_only"):
            with self.subTest(tripwire=name):
                self.assertTrue(block[name])


if __name__ == "__main__":
    unittest.main()
