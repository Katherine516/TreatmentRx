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


class OmittedEffectModifierTests(unittest.TestCase):
    """The other kind of wrong, and the one nothing used to test.

    `curvature` bends the nuisance surface, which double robustness survives.
    An omitted *effect modifier* bends the estimand, which nothing survives —
    and it is what a real cohort will have, because the true modifiers will not
    be exactly `anti_ccp` and `prior_tnf`.
    """

    def test_the_modifier_is_inert_at_zero(self):
        """Every other result in the repo assumes the default cohort is unchanged."""
        from treatmentrx.simulation.ra_cohort import generate_ra_cohort, true_blip

        plain = generate_ra_cohort(40, seed=3)
        explicit = generate_ra_cohort(40, seed=3, blip_modifier=0.0)
        self.assertEqual(
            [s.outcome for t in plain for s in t.stages],
            [s.outcome for t in explicit for s in t.stages],
        )
        features = {"das28": 5.2, "crp": 90.0, "anti_ccp": 1.0, "prior_tnf": 0.0,
                    "egfr": 82.0, "alt": 25.0}
        self.assertEqual(true_blip("IL-6 inhibitor", features),
                         true_blip("IL-6 inhibitor", features, 0.0))

    def test_it_bends_the_estimand_not_the_nuisance_surface(self):
        """A modifier that only moved the outcome level would be absorbed by the
        treatment-free model and prove nothing."""
        from treatmentrx.simulation.ra_cohort import true_blip

        low = {"das28": 5.2, "crp": 8.0, "anti_ccp": 0.0, "prior_tnf": 0.0,
               "egfr": 82.0, "alt": 25.0}
        high = dict(low, crp=100.0)
        # Same arm, same blip basis values, different true effect.
        self.assertEqual(true_blip("IL-6 inhibitor", low), true_blip("IL-6 inhibitor", high))
        self.assertNotEqual(
            true_blip("IL-6 inhibitor", low, 0.1), true_blip("IL-6 inhibitor", high, 0.1)
        )

    def test_coverage_collapses_and_the_interval_does_not_notice(self):
        """The finding: an omitted modifier is bias, and the SE cannot see it.

        Small configuration — the quotable numbers are `cli misspecification
        --omitted-modifier`. What holds at this size is the direction and the
        mechanism: coverage falls *and* SE/spread falls with it, which is what
        distinguishes a bias from an interval that is merely too narrow.
        """
        from treatmentrx.feedback.misspecification import omitted_modifier_study

        results = omitted_modifier_study(modifiers=(0.0, 0.2), replications=5, n=180)
        clean, bent = results[0.0], results[0.2]
        self.assertGreater(clean["coverage"], 0.8)
        self.assertLess(bent["coverage"], 0.6)
        self.assertGreater(abs(bent["bias"]), 4 * abs(clean["bias"]))
        self.assertLess(bent["se_to_sd_ratio"], clean["se_to_sd_ratio"])

    def test_the_study_grid_varies_the_omitted_covariate(self):
        """`coverage.PATIENT_GRID` holds CRP fixed, so it is structurally blind
        to CRP as an effect modifier. This study needs its own grid."""
        from treatmentrx.feedback import coverage
        from treatmentrx.feedback.misspecification import _modifier_grid

        self.assertEqual(len({p["crp"] for p in coverage.PATIENT_GRID.values()}), 1)
        self.assertGreater(len({p["crp"] for p in _modifier_grid().values()}), 1)


class SpecificationTestCalibrationTests(unittest.TestCase):
    """A specification test is only useful if its error rates are known.

    The first working version rejected on 33% of null cohorts against a nominal
    5% — it corrected multiplicity over arms while running covariates times arms
    tests, and took the sandwich at face value when this repo measures it at
    0.86-0.90 of the actual spread. A test that cries wolf a third of the time
    sends an analyst chasing a modifier that is not there.

    Deliberately few seeds here; the quotable rates are in the module docstring.
    What is asserted is that the test is silent on a null cohort and fires on a
    real modifier, which is the property that makes it worth running at all.
    """

    def test_the_threshold_corrects_for_every_test_performed(self):
        from treatmentrx.estimation.inference import SANDWICH_INFLATION
        from treatmentrx.estimation.specification import rejection_threshold

        over_arms_only = rejection_threshold(0.05, 1, 5)
        over_everything = rejection_threshold(0.05, 3, 5)
        self.assertGreater(over_everything, over_arms_only)
        # And the sandwich correction is in there, not just multiplicity.
        self.assertGreater(rejection_threshold(0.05, 1, 1), 1.96 * SANDWICH_INFLATION * 0.99)

    def test_it_is_silent_when_the_basis_is_correct(self):
        from treatmentrx.estimation.specification import specification_report

        flagged = 0
        for seed in (7000, 7001, 7002, 7003, 7004, 7005):
            report = specification_report(generate_ra_cohort(280, seed=seed))
            if report["flagged"]:
                flagged += 1
        self.assertLessEqual(flagged, 1, "the null false-positive rate has drifted up")

    def test_it_finds_a_modifier_that_is_really_there(self):
        """Silence is only worth something if the test can speak."""
        from treatmentrx.estimation.specification import specification_report

        for seed in (7100, 7101, 7102):
            report = specification_report(
                generate_ra_cohort(280, seed=seed, blip_modifier=0.10)
            )
            with self.subTest(seed=seed):
                self.assertIn("crp_std", report["flagged"])

    def test_the_statistic_grows_with_the_modifier(self):
        from treatmentrx.estimation.specification import specification_report

        z = [
            specification_report(
                generate_ra_cohort(280, seed=9000, blip_modifier=modifier)
            )["candidates"]["crp_std"]["max_abs_z"]
            for modifier in (0.0, 0.05, 0.20)
        ]
        self.assertEqual(z, sorted(z), f"the statistic is not monotone in the modifier: {z}")

    def test_a_mediator_is_not_offered_as_a_candidate(self):
        """ALT is caused by the previous arm, so interacting it with the current
        one measures mediation. It rejected at z=4.35 on a cohort with no ALT
        effect modification at all, which is how it was found."""
        from treatmentrx.estimation.specification import (
            CANDIDATE_MODIFIERS,
            EXCLUDED_CANDIDATES,
        )

        self.assertIn("alt_excess", EXCLUDED_CANDIDATES)
        self.assertNotIn("alt_excess", CANDIDATE_MODIFIERS)
        self.assertFalse(set(CANDIDATE_MODIFIERS) & set(EXCLUDED_CANDIDATES))

    def test_alt_really_is_downstream_of_treatment_in_this_cohort(self):
        """The justification for excluding it, as a measurement rather than a claim."""
        from treatmentrx.simulation.ra_cohort import HEPATOTOXIC_ARMS

        cohort = generate_ra_cohort(300, seed=9000)
        after_hepatotoxic, after_other = [], []
        for trajectory in cohort:
            for previous, current in zip(trajectory.stages, trajectory.stages[1:]):
                target = (
                    after_hepatotoxic
                    if previous.arm in HEPATOTOXIC_ARMS
                    else after_other
                )
                target.append(current.features["alt"])
        mean_hepatotoxic = sum(after_hepatotoxic) / len(after_hepatotoxic)
        mean_other = sum(after_other) / len(after_other)
        self.assertGreater(mean_hepatotoxic, mean_other + 20.0)


class CausalCertificateTests(unittest.TestCase):
    """The certificate has to describe the model, not the record.

    `_has_adjuster` counted a `birthDate` as adjusting for age and any
    medication as adjusting for steroid use, so `identified` was true for
    essentially every patient while no basis carried either variable.
    """

    def test_the_adjustment_set_is_derived_from_the_graph(self):
        """Not hand-listed. A declared set drifts from the edges it claims to
        summarise; a derived one cannot."""
        from treatmentrx.data.dag import CausalDAGRegistry, minimal_backdoor_set

        dag = CausalDAGRegistry()._ra_v1()
        derived = minimal_backdoor_set(dag)
        self.assertEqual(
            derived, set(dag.adjustment_set) | set(dag.unmodelled_confounders)
        )

    def test_the_backdoor_criterion_finds_the_confounders(self):
        from treatmentrx.data.dag import CausalDAGRegistry, backdoor_paths

        dag = CausalDAGRegistry()._ra_v1()
        paths = backdoor_paths(dag.edges, "treatment", "ra_response")
        self.assertTrue(paths)
        # Every backdoor path here is treatment <- confounder -> outcome.
        for path in paths:
            self.assertEqual(len(path), 3, path)
            self.assertEqual(path[0], "treatment")
            self.assertEqual(path[-1], "ra_response")

    def test_colliders_are_not_adjusted_for(self):
        """Conditioning on one opens a path rather than closing it."""
        from treatmentrx.data.dag import CausalDAGRegistry

        dag = CausalDAGRegistry()._ra_v1()
        for collider in dag.colliders:
            self.assertNotIn(collider, dag.adjustment_set)
            self.assertNotIn(collider, dag.unmodelled_confounders)

    def test_the_model_set_does_not_satisfy_the_backdoor_criterion(self):
        """The honest verdict, and it should stay False until the basis grows.

        Four confounders the graph requires are in no estimator basis, so the
        adjustment the model performs is incomplete. Reporting otherwise is what
        the old presence-check did.
        """
        from treatmentrx.data.dag import CausalDAGRegistry, satisfies_backdoor

        dag = CausalDAGRegistry()._ra_v1()
        identified, open_paths = satisfies_backdoor(dag, set(dag.adjustment_set))
        self.assertFalse(identified)
        self.assertEqual(
            {path[1] for path in open_paths}, set(dag.unmodelled_confounders)
        )
        # ...and the full derived set does satisfy it, which is what makes the
        # failure a statement about the model rather than about the graph.
        full = set(dag.adjustment_set) | set(dag.unmodelled_confounders)
        self.assertTrue(satisfies_backdoor(dag, full)[0])

    def test_every_claimed_adjuster_is_in_a_model_basis(self):
        from treatmentrx.data.dag import MODELLED_BY, CausalDAGRegistry
        from treatmentrx.estimation.basis import TREATMENT_FREE_BASIS

        dag = CausalDAGRegistry()._ra_v1()
        for node in dag.adjustment_set:
            with self.subTest(node=node):
                self.assertIn(node, MODELLED_BY, f"{node} maps to no basis term")
                self.assertIn(MODELLED_BY[node], TREATMENT_FREE_BASIS)

    def test_the_unadjusted_confounders_are_named(self):
        """Silence is what made the old verdict readable as 'adjusted for'."""
        from treatmentrx.data.dag import CausalDAGRegistry

        dag = CausalDAGRegistry()._ra_v1()
        self.assertEqual(
            set(dag.unmodelled_confounders), {"age", "gender", "steroid_use", "comorbidity_burden"}
        )
        self.assertFalse(set(dag.adjustment_set) & set(dag.unmodelled_confounders))

    def test_every_patient_is_told_what_was_not_adjusted_for(self):
        from treatmentrx.data import DataLayer
        from treatmentrx.demo_data import sample_ra_bundle

        state = DataLayer().build_patient_state(sample_ra_bundle())
        warning = next(
            d for d in state.diagnostics if d.name == "unmodelled_confounders"
        )
        self.assertEqual(warning.severity, "warning")
        for node in ("age", "gender", "steroid_use", "comorbidity_burden"):
            self.assertIn(node, warning.message)

    def test_identifiability_can_actually_fail(self):
        """A verdict that cannot fail is not a verdict.

        A record with no disease-activity measurement runs the estimators on a
        default; the effect is not identified for that patient, and the old
        check would have passed them on a `birthDate`.

        **Dropping the DAS28 alone is enough, and it did not used to be.** This
        test withheld the HAQ-DI as well, because `_has_adjuster` accepted it for
        `baseline_disease_activity` — a code that produces no DAS28 at all, so
        the estimators ran on `FEATURE_DEFAULTS["das28"]` under a certificate
        saying the effect was identified. The record kept here is the ordinary
        one: a HAQ-DI recorded and no DAS28.
        """
        import copy

        from treatmentrx.data import DataLayer
        from treatmentrx.data.dag import CausalDAGRegistry
        from treatmentrx.demo_data import sample_ra_bundle
        from treatmentrx.estimation.features import FEATURE_DEFAULTS, model_features

        bundle = copy.deepcopy(sample_ra_bundle())
        bundle["entry"] = [
            entry
            for entry in bundle["entry"]
            if not (
                entry["resource"].get("resourceType") == "Observation"
                and entry["resource"]["code"]["text"].upper() == "DAS28"
            )
        ]
        haq = [
            entry for entry in bundle["entry"]
            if entry["resource"].get("resourceType") == "Observation"
            and entry["resource"]["code"]["text"].upper() == "HAQ-DI"
        ]
        self.assertTrue(haq, "the point of this record is that a family member survives")

        patient = DataLayer().fhir.parse_bundle(bundle)
        stages = DataLayer().stage_builder.build(patient)
        result = CausalDAGRegistry().validate(patient, stages)
        self.assertFalse(result.identified)
        self.assertIn("baseline_disease_activity", result.blocked_reason)
        # And the reason it must not be identified, measured rather than assumed:
        # the covariate the estimators read is the default.
        self.assertEqual(
            model_features(stages)["das28"], FEATURE_DEFAULTS["das28"]
        )


class JointBootstrapTests(unittest.TestCase):
    """Measuring the estimators' covariance instead of bounding it.

    The averaged interval has to account for how the three estimators co-vary.
    Fitting them separately leaves that unmeasured, and the only defensible
    fallback is the perfect-correlation upper bound — which runs about 1.29x the
    averaged estimator's actual spread, so every interval is a quarter wider than
    it needs to be.
    """

    @classmethod
    def setUpClass(cls):
        from treatmentrx.estimation import training

        cls.training = training
        cls.cohort = generate_ra_cohort(150, seed=5)

    def tearDown(self):
        # Turning the mode off is enough; discarding the fits would refit three
        # models between every test in this class for no reason.
        self.training.disable_joint_inference()

    def test_loadings_reproduce_each_estimator_contrast(self):
        """The flat-vector view has to be the same quantity, or the resampled
        contrast is of something else entirely."""
        from treatmentrx.estimation import linalg
        from treatmentrx.estimation.dwols import DWOLSModel

        features = self.cohort[0].stages[-1].features
        dwols = DWOLSModel(self.cohort)
        direct = dwols.blip("rituximab", features) - dwols.blip("IL-6 inhibitor", features)
        loaded = linalg.dot(
            dwols.contrast_loading("rituximab", "IL-6 inhibitor", features),
            dwols.flat_parameters(),
        )
        self.assertAlmostEqual(direct, loaded, places=12)

        model = QLearningModel(self.cohort, share_blip=False)
        terminal = model.n_stages - 1
        direct_q = model.blip("rituximab", features, terminal) - model.blip(
            "IL-6 inhibitor", features, terminal
        )
        loaded_q = linalg.dot(
            model.contrast_loading("rituximab", "IL-6 inhibitor", features, terminal),
            model.flat_parameters(),
        )
        self.assertAlmostEqual(direct_q, loaded_q, places=12)

    def test_every_estimator_is_refit_on_the_same_resample(self):
        """Independent resamples would destroy the correlation being measured."""
        bootstrap = self.training.enable_joint_inference(replicates=6)
        self.assertGreater(bootstrap.replicates, 0)
        for draw in bootstrap.draws:
            self.assertEqual(
                sorted(draw),
                sorted([
                    self.training.Q_SHARED,
                    self.training.STAGE_SPECIFIC,
                    self.training.Q_POOLED,
                    self.training.DWOLS_SHARED,
                ]),
            )

    def test_the_measured_interval_is_narrower_than_the_bound(self):
        """The whole point: the bound assumes perfect correlation and the truth
        is high but not perfect, so measuring it recovers real width."""
        orchestrator = TreatmentRxOrchestrator()
        bound = orchestrator.run(sample_ra_bundle()).audit_event["contrast"]
        self.training.enable_joint_inference(replicates=40)
        measured = orchestrator.run(sample_ra_bundle()).audit_event["contrast"]

        self.assertLess(measured["standard_error"], bound["standard_error"])
        self.assertIn("joint m-out-of-n", measured["caveat"])
        self.assertTrue(measured["conservative"])

    def test_it_is_opt_in_and_the_fallback_is_the_bound(self):
        """An interval that silently degrades is worse than one visibly wide."""
        self.assertIsNone(self.training.joint_inference())
        contrast = TreatmentRxOrchestrator().run(sample_ra_bundle()).audit_event["contrast"]
        self.assertIn("weighted sum", contrast["caveat"])

    def test_the_mode_can_be_turned_off_without_discarding_the_fits(self):
        self.training.enable_joint_inference(replicates=4)
        fitted_before = self.training.fitted()
        self.assertIsNotNone(self.training.joint_inference())

        self.training.disable_joint_inference()
        self.assertIsNone(self.training.joint_inference())
        self.assertIs(self.training.fitted(), fitted_before, "the fits were thrown away")

    def test_a_full_reset_clears_the_joint_draws_too(self):
        self.training.enable_joint_inference(replicates=4)
        self.training.reset()
        self.assertIsNone(self.training.joint_inference())

    def test_reset_refits_every_serving_estimator(self):
        """`reset()` has to reach the models that actually serve patients.

        dWOLS used to keep its own module-level cache, so `reset()` refit the
        Q-learning half and left the dWOLS half at whatever cohort it first saw.
        Nothing failed — the two agree on the default cohort — but the serving
        ensemble was half stale, and a sample-size sweep silently measured a
        standard error that could not shrink.
        """
        from treatmentrx.estimation import dwols

        self.assertIs(self.training.fitted().dwols, dwols.fitted_model())
        original = self.training.COHORT_SIZE
        try:
            self.training.COHORT_SIZE = original + 60
            self.training.reset()
            self.assertIs(self.training.fitted().dwols, dwols.fitted_model())
            self.assertEqual(
                dwols.fitted_model().n_train,
                len(self.training.fitted().train),
                "the serving dWOLS fit did not follow the training split",
            )
        finally:
            self.training.COHORT_SIZE = original
            self.training.reset()


class AutomaticSpecificationCheckTests(unittest.TestCase):
    """The basis check runs on every fit, not only when someone remembers to look.

    `cli misspecification --omitted-modifier` measures an omitted effect modifier
    costing up to 65 points of interval coverage, with the interval *narrowing*
    as it happens. A caveat that only appears in a command nobody runs is not a
    caveat.
    """

    @classmethod
    def setUpClass(cls):
        from treatmentrx.estimation import training

        cls.training = training

    def tearDown(self):
        import treatmentrx.estimation.specification as specification

        specification.rejection_threshold = self._original_threshold
        self.training.reset()

    def setUp(self):
        import treatmentrx.estimation.specification as specification

        self._original_threshold = specification.rejection_threshold

    def _force_flag(self):
        """Lower the bar so the wiring can be exercised on the default cohort.

        The deployed basis is correct, so nothing flags — which is the right
        answer and a useless test. Dropping the threshold makes the *propagation*
        observable without pretending the cohort is misspecified.
        """
        import treatmentrx.estimation.specification as specification

        specification.rejection_threshold = lambda *args, **kwargs: 2.0
        self.training.reset()

    def test_it_is_computed_at_fit_time(self):
        report = self.training.basis_specification()
        self.assertIn("candidates", report)
        self.assertIs(report, self.training.fitted().specification)

    def test_the_deployed_basis_is_not_flagged(self):
        self.assertFalse(self.training.basis_is_flagged())
        self.assertEqual(self.training.basis_specification()["flagged"], [])

    def test_a_flag_blocks_the_validation_ladder(self):
        from treatmentrx.domain import ValidationRung
        from treatmentrx.feedback.validation_ladder import ValidationLadder

        self._force_flag()
        readiness = self.training.deployment_readiness()
        self.assertFalse(readiness["blip_basis_unflagged"])
        status = ValidationLadder().assess(ValidationRung.SILENT, readiness)
        self.assertFalse(status.gate_passed)
        self.assertTrue(any("blip basis" in blocker for blocker in status.blockers))

    def test_a_flag_reaches_every_patient_as_an_uncertainty_flag(self):
        from treatmentrx.demo_data import sample_ra_bundle
        from treatmentrx.orchestrator import TreatmentRxOrchestrator

        self._force_flag()
        recommendation = TreatmentRxOrchestrator().run(sample_ra_bundle())
        flags = recommendation.audit_event["uncertainty"]["flags"]
        self.assertTrue(any(flag.startswith("blip_basis_may_omit:") for flag in flags))

    def test_a_flag_reaches_the_clinician_card(self):
        """The reader who acts on the interval is the one who needs to know."""
        from treatmentrx.demo_data import sample_ra_bundle
        from treatmentrx.orchestrator import TreatmentRxOrchestrator

        self._force_flag()
        card = TreatmentRxOrchestrator().run(sample_ra_bundle()).clinician_card
        self.assertIn("CAVEAT", card)
        self.assertIn("missing an effect modifier", card)

    def test_a_clean_basis_adds_no_caveat(self):
        from treatmentrx.demo_data import sample_ra_bundle
        from treatmentrx.orchestrator import TreatmentRxOrchestrator

        card = TreatmentRxOrchestrator().run(sample_ra_bundle()).clinician_card
        self.assertNotIn("CAVEAT", card)

    def test_a_flag_does_not_silently_change_the_decision(self):
        """Stated policy, not an oversight — see `BLOCK_ON_FLAGGED_BASIS`.

        The test is a falsification test with an actionable fix. Turning a
        diagnostic into a silent behaviour change is the pattern this repo keeps
        removing; it makes the flag loud instead.
        """
        from treatmentrx.demo_data import sample_ra_bundle
        from treatmentrx.orchestrator import TreatmentRxOrchestrator

        self.assertFalse(self.training.BLOCK_ON_FLAGGED_BASIS)
        clean = TreatmentRxOrchestrator().run(sample_ra_bundle())
        self._force_flag()
        flagged = TreatmentRxOrchestrator().run(sample_ra_bundle())
        self.assertEqual(clean.recommended_arm, flagged.recommended_arm)
        self.assertEqual(clean.status, flagged.status)
