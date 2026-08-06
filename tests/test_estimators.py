"""Statistical validation of the Layer 4 estimators.

Every test here asserts something about *correctness of the numbers*, not about
plumbing: parameter recovery under confounding, that adjustment beats the naive
comparison, that backward induction beats a myopic fit on a delayed-toxicity
generating process, and that the learned policies beat the clinician policy that
produced the data.
"""

import unittest

from treatmentrx.demo_data import sample_ra_bundle
from treatmentrx.data.stages import StageHistoryBuilder
from treatmentrx.data.fhir import FHIRAdapter
from treatmentrx.estimation import training
from treatmentrx.estimation.dwols import DWOLSModel, DWOLSSharedEstimator
from treatmentrx.estimation.estimators import QSharedEstimator, StageSpecificQEstimator
from treatmentrx.estimation.features import model_features, prior_tnf_exposure
from treatmentrx.estimation.q_learning import QLearningModel, myopic_model
from treatmentrx.feedback.offline_evaluation import evaluate_policy
from treatmentrx.domain import RegimeAssignment, RegimeType
from treatmentrx.simulation.ra_cohort import (
    HEPATOTOXIC_ARMS,
    HIGH_BURDEN_ARMS,
    REFERENCE_ARM,
    TREATMENT_ARMS,
    TRUE_BLIPS,
    assignment_probabilities,
    behaviour_policy,
    generate_ra_cohort,
    myopic_optimal_policy,
    rollout_value,
    true_blip,
)

# Every arm now carries some delayed component — hepatotoxic arms cost ALT at
# the next visit, high-burden arms are abandoned unless they work — so only the
# *terminal*-stage blip is the single-visit quantity `TRUE_BLIPS` describes.
_ACTIVE_ARMS = tuple(arm for arm in TREATMENT_ARMS if arm != REFERENCE_ARM)
# Arms with neither a toxicity cost nor a retention effect.
_CLEAN_ARMS = tuple(
    arm
    for arm in _ACTIVE_ARMS
    if arm not in HEPATOTOXIC_ARMS and arm not in HIGH_BURDEN_ARMS
)

_ASSIGNMENT = RegimeAssignment(
    regime_type=RegimeType.SPTR, reason="test", shared_bic=1.0, stage_specific_bic=1.0
)


def _demo_stages():
    patient = FHIRAdapter().parse_bundle(sample_ra_bundle())
    return patient, StageHistoryBuilder().build(patient)


class CohortTests(unittest.TestCase):
    def test_positivity_holds_for_every_arm(self):
        cohort = generate_ra_cohort(200, seed=3)
        worst = min(
            min(assignment_probabilities(stage.features).values())
            for trajectory in cohort
            for stage in trajectory.stages
        )
        # Without positivity no inverse-probability estimator is identified.
        self.assertGreater(worst, 0.01)

    def test_assignment_is_confounded(self):
        """Sicker patients really are steered toward the escalation arms."""
        cohort = generate_ra_cohort(300, seed=3)
        stages = [stage for trajectory in cohort for stage in trajectory.stages]
        il6 = [s.features["das28"] for s in stages if s.arm == "IL-6 inhibitor"]
        continued = [s.features["das28"] for s in stages if s.arm == REFERENCE_ARM]
        self.assertGreater(sum(il6) / len(il6), sum(continued) / len(continued) + 0.5)

    def test_generation_is_deterministic(self):
        first = generate_ra_cohort(50, seed=11)
        second = generate_ra_cohort(50, seed=11)
        self.assertEqual(
            [stage.outcome for t in first for stage in t.stages],
            [stage.outcome for t in second for stage in t.stages],
        )


class BlipRecoveryTests(unittest.TestCase):
    def test_dwols_recovers_every_arm_blip_under_confounding(self):
        model = DWOLSModel(generate_ra_cohort(400, seed=7))
        for arm in TREATMENT_ARMS:
            if arm == REFERENCE_ARM:
                continue
            estimated = model.blip_parameters(arm)
            for name, truth in zip(estimated, TRUE_BLIPS[arm]):
                self.assertAlmostEqual(
                    estimated[name], truth, delta=0.03, msg=f"{arm}:{name}"
                )

    def test_terminal_stage_blips_are_recovered_for_every_arm(self):
        """The terminal stage is where the blip is a single-visit effect.

        Earlier stages legitimately differ — they carry the delayed consequences
        of the choice — so this is the stage `TRUE_BLIPS` can be checked against.
        """
        model = QLearningModel(generate_ra_cohort(400, seed=7), share_blip=False)
        terminal = model.n_stages - 1
        for arm in _ACTIVE_ARMS:
            estimated = model.blip_parameters(arm, terminal)
            for name, truth in zip(estimated, TRUE_BLIPS[arm]):
                self.assertAlmostEqual(
                    estimated[name], truth, delta=0.04, msg=f"{arm}:{name}"
                )

    def test_shared_blip_recovers_arms_without_delayed_effects(self):
        model = QLearningModel(generate_ra_cohort(400, seed=7), share_blip=True)
        for arm in _CLEAN_ARMS:
            estimated = model.blip_parameters(arm)
            for name, truth in zip(estimated, TRUE_BLIPS[arm]):
                self.assertAlmostEqual(
                    estimated[name], truth, delta=0.04, msg=f"{arm}:{name}"
                )

    def test_adjustment_beats_the_naive_comparison(self):
        """The whole point of the causal machinery, stated as a test."""
        cohort = generate_ra_cohort(400, seed=7)
        rows = [(s.features, s.arm, s.outcome) for t in cohort for s in t.stages]
        reference = [y for _, arm, y in rows if arm == REFERENCE_ARM]
        reference_mean = sum(reference) / len(reference)
        model = DWOLSModel(cohort)

        for arm in _CLEAN_ARMS:
            observed = [y for _, a, y in rows if a == arm]
            naive = sum(observed) / len(observed) - reference_mean
            truth = sum(true_blip(arm, f) for f, _, _ in rows) / len(rows)
            fitted = sum(model.blip(arm, f) for f, _, _ in rows) / len(rows)
            self.assertLess(
                abs(fitted - truth),
                abs(naive - truth),
                msg=f"{arm}: adjusted estimate should be closer to truth than the naive contrast",
            )

    def test_stage_specific_blip_absorbs_the_delayed_toxicity_cost(self):
        """A hepatotoxic arm is worth less at stage 1 than at the last stage.

        The ALT it raises is only paid for at the next visit, so a sequential
        estimator must price that in and a single-visit one cannot.
        """
        model = QLearningModel(generate_ra_cohort(400, seed=7), share_blip=False)
        for arm in sorted(HEPATOTOXIC_ARMS):
            first = model.blip_parameters(arm, 0)["intercept"]
            terminal = model.blip_parameters(arm, model.n_stages - 1)["intercept"]
            self.assertLess(first, terminal - 0.01, msg=arm)

    def test_retention_benefit_raises_the_value_of_earlier_decisions(self):
        """A burdensome arm that works keeps the patient in care.

        Those retained visits are extra accrued benefit that only a
        multi-stage estimator can see, so the earlier-stage blip for a
        high-burden arm sits *above* its single-visit effect — the mirror image
        of the delayed toxicity cost.
        """
        model = QLearningModel(generate_ra_cohort(400, seed=7), share_blip=False)
        terminal_index = model.n_stages - 1
        for arm in sorted(HIGH_BURDEN_ARMS):
            first = model.blip_parameters(arm, 0)["intercept"]
            terminal = model.blip_parameters(arm, terminal_index)["intercept"]
            self.assertGreater(first, terminal + 0.005, msg=arm)


class PolicyValueTests(unittest.TestCase):
    def test_every_estimator_beats_the_clinician_policy(self):
        behaviour = rollout_value(behaviour_policy, n=3000, seed=202)
        for name, score in training.fitted().scores.items():
            self.assertGreater(score.ipw_policy_value, score.behaviour_value, msg=name)
            self.assertGreater(score.oracle_rollout_value, behaviour, msg=name)

    def test_backward_induction_beats_a_myopic_fit(self):
        train = training.fitted().train
        backward = QLearningModel(train, share_blip=True)
        myopic = myopic_model(train, share_blip=True)
        self.assertGreater(
            rollout_value(backward.greedy_policy(), n=3000, seed=303),
            rollout_value(myopic.greedy_policy(), n=3000, seed=303),
        )

    def test_backward_induction_avoids_hepatotoxic_arms_at_the_first_stage(self):
        train = training.fitted().train
        backward = QLearningModel(train, share_blip=True)
        myopic = myopic_model(train, share_blip=True)
        patients = [stage.features for trajectory in train for stage in trajectory.stages[:1]]
        backward_hits = sum(1 for f in patients if backward.recommend(f, 0) in HEPATOTOXIC_ARMS)
        myopic_hits = sum(1 for f in patients if myopic.recommend(f, 0) in HEPATOTOXIC_ARMS)
        self.assertLess(backward_hits, myopic_hits)

    def test_ipw_and_oracle_agree_on_the_ranking_against_behaviour(self):
        """The observational estimate must not disagree with the known answer."""
        for name, score in training.fitted().scores.items():
            self.assertGreater(score.improvement, 0.0, msg=name)
            self.assertGreater(score.effective_sample_size, 10.0, msg=name)

    def test_policy_score_reports_no_agreement_honestly(self):
        holdout = training.fitted().holdout
        never_taken = evaluate_policy(
            "never-agrees",
            lambda features, stage_index: "not-a-real-arm",
            lambda features, arm, stage_index: 0.5,
            holdout,
        )
        self.assertEqual(never_taken.ipw_policy_value, 0.0)
        self.assertIn(
            "policy never agreed with the observed arm; IPW value is not identified",
            never_taken.notes,
        )


class CalibrationTests(unittest.TestCase):
    def test_holdout_calibration_is_measured_not_assumed(self):
        report = training.holdout_calibration()
        # A non-degenerate reliability curve: predictions must actually land in
        # more than one bin, otherwise ECE is meaningless.
        populated = [b for b in report.reliability_bins if b["count"] > 0]
        self.assertGreater(len(populated), 1)
        self.assertGreater(report.expected_calibration_error, 0.0)
        self.assertTrue(report.passed)

    def test_miscalibrated_predictions_are_caught(self):
        holdout = training.fitted().holdout
        biased = evaluate_policy(
            "always-optimistic",
            training.fitted().q_shared.greedy_policy(),
            lambda features, arm, stage_index: 0.95,
            holdout,
        )
        self.assertFalse(biased.calibration.passed)


class EstimatorFacadeTests(unittest.TestCase):
    def test_estimators_share_one_scale_and_one_menu(self):
        _, stages = _demo_stages()
        results = [
            QSharedEstimator().fit_predict(stages, _ASSIGNMENT, ["das28"]),
            StageSpecificQEstimator().fit_predict(stages, _ASSIGNMENT, ["das28"]),
            DWOLSSharedEstimator().fit_predict(stages, _ASSIGNMENT, ["das28"]),
        ]
        menus = {tuple(sorted(result.q_values)) for result in results}
        self.assertEqual(len(menus), 1, "estimators must score the same arm menu")
        for result in results:
            for arm, value in result.q_values.items():
                self.assertTrue(0.0 <= value <= 1.0, msg=f"{result.estimator}:{arm}")

    def test_policy_value_is_the_held_out_score_not_a_self_report(self):
        _, stages = _demo_stages()
        result = QSharedEstimator().fit_predict(stages, _ASSIGNMENT, ["das28"])
        self.assertEqual(result.policy_value, training.policy_value_for(training.Q_SHARED))

    def test_blip_parameters_are_exposed_for_audit(self):
        _, stages = _demo_stages()
        result = DWOLSSharedEstimator().fit_predict(stages, _ASSIGNMENT, ["das28"])
        recommended = result.recommended_arm
        self.assertEqual(result.estimator, training.DWOLS_SHARED)
        self.assertTrue(
            any(key.startswith(f"psi:{recommended}:") for key in result.coefficients),
            msg=f"no blip parameters reported for {recommended}: {sorted(result.coefficients)}",
        )

    def test_seropositive_prior_tnf_failure_is_steered_to_rituximab(self):
        """The demo patient's clinical signature has a known right answer.

        Anti-CCP positive with a TNF loss of response: the generating process
        makes rituximab optimal, and the guideline knowledge base says the same.
        This is the end-to-end check that the covariate mapping, the fit, and the
        arm menu all line up.
        """
        _, stages = _demo_stages()
        features = model_features(stages)
        self.assertEqual(features["anti_ccp"], 1.0)
        self.assertEqual(features["prior_tnf"], 1.0)
        self.assertEqual(myopic_optimal_policy(features, 0), "rituximab")
        self.assertEqual(
            QSharedEstimator().fit_predict(stages, _ASSIGNMENT, ["das28"]).recommended_arm,
            "rituximab",
        )

    def test_blip_attributions_sum_to_the_estimated_advantage(self):
        """Explanations must decompose the model, not paraphrase it."""
        from treatmentrx.estimation.explainability import ModelExplainer

        _, stages = _demo_stages()
        result = QSharedEstimator().fit_predict(stages, _ASSIGNMENT, ["das28"])
        attribution = ModelExplainer().explain(result, [result], stages).attributions[0]
        self.assertEqual(attribution.action, result.recommended_arm)
        self.assertAlmostEqual(
            sum(attribution.contributions.values()), attribution.total_advantage, places=3
        )
        model = training.fitted().q_shared
        self.assertAlmostEqual(
            attribution.total_advantage,
            model.blip(result.recommended_arm, model_features(stages), 0),
            places=3,
        )

    def test_prior_tnf_excludes_the_open_stage(self):
        """Exposure must be counted from completed stages, not the pending one."""
        _, stages = _demo_stages()
        self.assertEqual(prior_tnf_exposure(stages), 1.0)
        self.assertEqual(prior_tnf_exposure(stages[:1]), 0.0)


if __name__ == "__main__":
    unittest.main()
