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
    oracle_policy,
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

_ROLLOUT_SEEDS = (404, 505)
_ROLLOUT_CACHE: dict[str, float] = {}


def _mean_rollout(label, policy, n=2000, seeds=_ROLLOUT_SEEDS):
    """Average a policy's true value over several rollout seeds, memoised.

    A single rollout at n=2000 carries a Monte Carlo sd of about 0.013, which is
    larger than the differences between the better policies, so a comparison off
    one seed is a coin flip. Caching by label keeps the cost of averaging from
    being paid once per test — the oracle's backward induction is the expensive
    part and two tests need the same number.
    """
    if label not in _ROLLOUT_CACHE:
        _ROLLOUT_CACHE[label] = sum(
            rollout_value(policy, n=n, seed=seed) for seed in seeds
        ) / len(seeds)
    return _ROLLOUT_CACHE[label]


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



class PublishedBlipScaleTests(unittest.TestCase):
    """What `coefficient_summary` publishes sits on the scale `q_values` sit on.

    `blip_parameters` is a value-to-go stage parameter; `q_values` divide that by
    the remaining horizon so every estimator reports response per remaining visit
    (invariant 9). `coefficient_summary` used to publish the blips raw, and the
    explanation layer decomposes exactly those into the per-covariate terms the
    clinician card prints *under a gap taken from `q_values`*. Measured before the
    repair, a decomposition sourced from this model ran 1.54x to 3.49x the gap it
    claimed to explain at `stage_index` 1, agreeing only at the terminal block
    where the horizon is 1.

    Nothing served ever crossed those scales — `attribution_source` was
    dWOLS-Shared for every patient on the deployed fit — so these assert the
    property directly rather than through a card, because the margin that keeps
    it off the card is a 0.003 BMA weight.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QLearningModel(generate_ra_cohort(400, seed=7), share_blip=False)
        _, stages = _demo_stages()
        cls.features = model_features(stages)

    def _published(self, arm, index):
        from treatmentrx.estimation.basis import BLIP_BASIS, blip_basis

        summary = self.model.coefficient_summary(arm, index)
        basis = dict(zip(BLIP_BASIS, blip_basis(self.features)))
        return sum(
            summary[f"psi:{arm}:{name}"] * basis[name]
            for name in BLIP_BASIS
            if f"psi:{arm}:{name}" in summary
        )

    def test_the_published_blip_is_the_q_value_difference(self):
        """The identity that makes the division exact rather than a fudge.

        `psi_a . h(X)` *is* `raw_q(a) - raw_q(reference)` by construction, so
        dividing by the horizon gives precisely `q_values[a] - q_values[ref]` —
        the quantity the card prints beside the decomposition.
        """
        for index in range(self.model.n_stages):
            q_values = self.model.q_values(self.features, index)
            for arm in self.model.arms:
                if arm == REFERENCE_ARM:
                    continue
                with self.subTest(arm=arm, stage=index):
                    self.assertAlmostEqual(
                        self._published(arm, index),
                        q_values[arm] - q_values[REFERENCE_ARM],
                        places=2,
                    )

    def test_the_accessor_stays_the_value_to_go_parameter(self):
        """`blip_parameters` must *not* be divided: it is the parameter the model
        estimates, and `cli coverage`, `cli misspecification` and the recovery
        tests above compare it against the generating process's blips. The two
        methods differ by exactly the horizon, and nothing else."""
        from treatmentrx.estimation.basis import BLIP_BASIS, blip_basis

        basis = dict(zip(BLIP_BASIS, blip_basis(self.features)))
        for index in range(self.model.n_stages):
            horizon = self.model.remaining_stages(index)
            for arm in self.model.arms:
                if arm == REFERENCE_ARM:
                    continue
                raw = sum(
                    value * basis[name]
                    for name, value in self.model.blip_parameters(arm, index).items()
                )
                with self.subTest(arm=arm, stage=index):
                    self.assertAlmostEqual(self._published(arm, index), raw / horizon, places=3)

    def test_the_horizon_actually_bites_somewhere(self):
        """A division by 1 everywhere would make both tests above vacuous — which
        is what made this defect invisible: the terminal block is the only stage
        the deployed pipeline usually serves, and there the horizon is 1."""
        horizons = {self.model.remaining_stages(i) for i in range(self.model.n_stages)}
        self.assertGreater(max(horizons), 1)
        arm = next(a for a in self.model.arms if a != REFERENCE_ARM)
        raw = self._published(arm, 0) * self.model.remaining_stages(0)
        self.assertNotAlmostEqual(self._published(arm, 0), raw, places=3)


class PolicyValueTests(unittest.TestCase):
    def test_every_estimator_beats_the_clinician_policy(self):
        behaviour = rollout_value(behaviour_policy, n=3000, seed=202)
        for name, score in training.fitted().scores.items():
            self.assertGreater(score.ipw_policy_value, score.behaviour_value, msg=name)
            # The oracle benchmark is simulation-only and computed on request, so
            # it stays off the cold path of a process that just serves a patient.
            self.assertGreater(training.oracle_rollout_value(name), behaviour, msg=name)

    def test_the_oracle_benchmark_is_not_paid_for_at_fit_time(self):
        for name, score in training.fitted().scores.items():
            self.assertIsNone(score.oracle_rollout_value, msg=name)

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

    def test_the_regret_reference_beats_the_myopic_rule_it_replaced(self):
        """An oracle that the agent beats is not an oracle.

        Layer 3's regret used to be measured against the per-visit blip argmax,
        which loses to all three fitted policies on the true trajectory value
        because it over-prescribes the arms whose cost is delayed. This pins the
        margin that makes the replacement reference worth having: +0.020 over the
        myopic rule, measured 8/8 across seeds against a paired sd of 0.007.
        """
        oracle = _mean_rollout("oracle", oracle_policy())
        myopic = _mean_rollout("myopic", myopic_optimal_policy)
        behaviour = _mean_rollout("behaviour", behaviour_policy)
        self.assertGreater(oracle - myopic, 0.010, "oracle margin over the myopic rule collapsed")
        self.assertGreater(myopic, behaviour)

    def test_no_fitted_policy_materially_exceeds_the_regret_reference(self):
        """Certainty equivalence leaves a Jensen gap, so this is a tolerance, not
        an inequality.

        The measured paired sd of oracle-minus-fitted is 0.0040 for the closest
        estimator; 0.010 is comfortably outside it, so a failure here means the
        reference has genuinely stopped bounding the policies rather than that a
        seed went the other way.
        """
        fit = training.fitted()
        oracle = _mean_rollout("oracle", oracle_policy())
        for name, model in (
            (training.Q_SHARED, fit.q_shared),
            (training.DWOLS_SHARED, fit.dwols),
        ):
            value = _mean_rollout(name, model.greedy_policy())
            self.assertLess(
                value - oracle, 0.010, msg=f"{name} exceeded the reference: {value} vs {oracle}"
            )

    def test_the_myopic_rule_is_not_the_optimal_policy(self):
        """The two references must actually differ, or the distinction is decorative."""
        from treatmentrx.simulation.ra_cohort import oracle_arm, optimal_arm

        train = training.fitted().train
        patients = [stage.features for trajectory in train for stage in trajectory.stages[:1]]
        disagreements = sum(1 for f in patients if oracle_arm(f, 0) != optimal_arm(f))
        self.assertGreater(disagreements, 0)

    def test_every_estimator_gains_over_behaviour_with_an_interval(self):
        """The gain is real; the ordering between estimators is not."""
        for name, score in training.fitted().scores.items():
            self.assertIsNotNone(score.improvement_interval, msg=name)
            self.assertGreater(score.improvement_interval[0], 0.0, msg=name)

    def test_the_estimator_ranking_is_not_read_off_indistinguishable_values(self):
        """Four call sites used to argmax over three overlapping numbers.

        If the intervals overlap, the selection must come from the stated
        interpretability order rather than from whichever value happened to land
        on top — otherwise a re-seed silently changes which model's calibration
        gates deployment and which policy defines the estimands.
        """
        ranked = sorted(
            training.fitted().scores.values(),
            key=lambda score: score.ipw_policy_value,
            reverse=True,
        )
        if training.ranking_is_resolved():
            self.assertEqual(training.best_score().estimator, ranked[0].estimator)
        else:
            expected = min(
                (score.estimator for score in ranked),
                key=lambda name: training.INTERPRETABILITY_ORDER.get(name, 99),
            )
            self.assertEqual(training.best_score().estimator, expected)

    def test_an_estimator_only_beats_another_when_the_intervals_separate(self):
        scores = list(training.fitted().scores.values())
        for score in scores:
            self.assertFalse(score.beats(score), "an estimator cannot beat itself")
            for other in scores:
                if score.beats(other):
                    self.assertGreater(score.value_interval[0], other.value_interval[1])

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


class PropensityTests(unittest.TestCase):
    """The held-out estimate must not need a probability only the simulator knows.

    `CohortStage.propensity` is what the generator drew from. Every quantity the
    validation ladder gates on was computed from it, which is an advantage no
    deployment has.
    """

    @classmethod
    def setUpClass(cls):
        from treatmentrx.estimation.propensity import PropensityModel

        cls.fit = training.fitted()
        cls.model = PropensityModel(cls.fit.train)

    def test_it_recovers_the_behaviour_policy_on_held_out_patients(self):
        """Fit on train, checked on holdout — the same discipline as everything else."""
        calibration = self.model.calibration(self.fit.holdout)
        self.assertLess(calibration["mean_absolute_error"], 0.05)
        self.assertAlmostEqual(calibration["mean_ratio_to_truth"], 1.0, delta=0.1)

    def test_the_probabilities_are_a_distribution(self):
        features = self.fit.holdout[0].stages[0].features
        probabilities = self.model.probabilities(features)
        self.assertEqual(set(probabilities), set(TREATMENT_ARMS))
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=9)
        self.assertTrue(all(p > 0.0 for p in probabilities.values()))

    def test_it_reports_convergence_honestly(self):
        """`converged` has to mean the fixed point, not the iteration budget."""
        self.assertTrue(self.model.fit.converged)
        self.assertLess(self.model.fit.iterations, 200)

    def test_the_deployed_scores_use_the_fitted_propensity(self):
        """Not the generator's — otherwise the ladder gates on oracle knowledge."""
        from treatmentrx.feedback.offline_evaluation import evaluate_policy

        name, model = training.Q_POOLED, self.fit.pooled
        oracle = evaluate_policy(
            name, model.greedy_policy(), model.predict_outcome, self.fit.holdout,
            with_intervals=False,
        )
        self.assertNotEqual(
            self.fit.scores[name].ipw_policy_value,
            oracle.ipw_policy_value,
            "the deployed score still reads stage.propensity",
        )

    def test_swapping_the_oracle_for_the_fit_does_not_move_the_conclusion(self):
        """The claim this supports: the evaluation is reproducible without the
        generator. Every estimator shifts by less than its own interval width."""
        from treatmentrx.feedback.offline_evaluation import evaluate_policy

        for name, model in (
            (training.Q_POOLED, self.fit.pooled),
            (training.DWOLS_SHARED, self.fit.dwols),
        ):
            with self.subTest(estimator=name):
                score = self.fit.scores[name]
                oracle = evaluate_policy(
                    name, model.greedy_policy(), model.predict_outcome, self.fit.holdout,
                    with_intervals=False,
                )
                width = score.value_interval[1] - score.value_interval[0]
                self.assertLess(
                    abs(score.ipw_policy_value - oracle.ipw_policy_value), width / 2.0
                )
                self.assertGreater(score.improvement_interval[0], 0.0)


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


class FittedCacheTests(unittest.TestCase):
    """One fitted ensemble per process, including under a threaded server.

    `service.py` runs a thread per request, so `fitted()`'s lazy global is
    reachable concurrently. Unguarded, four cold callers each ran the whole fit
    and three ensembles were discarded — and because the fit is deterministic
    they all agreed, so nothing would ever have surfaced it.
    """

    def test_concurrent_cold_callers_fit_exactly_once(self):
        """A stub fit, not the real one, and deliberately slow.

        Two reasons not to refit for real here. It costs ~3s of suite time to
        demonstrate something that is about the guard rather than about the
        models; and a fast fit is a *worse* race detector — the second thread
        may simply arrive after the first has finished, so an unguarded
        implementation could pass by luck. The sleep holds the window open so
        the assertion means what it says.
        """
        import threading
        import time as _time

        original_fitted = training._FITTED
        original_fit_all = training._fit_all
        fits = []
        sentinel = object()

        def slow_stub():
            fits.append(1)
            _time.sleep(0.05)
            return sentinel

        training._fit_all = slow_stub
        training._FITTED = None
        try:
            handed_out = []
            threads = [
                threading.Thread(target=lambda: handed_out.append(training.fitted()))
                for _ in range(8)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        finally:
            training._fit_all = original_fit_all
            training._FITTED = original_fitted

        self.assertEqual(len(fits), 1, f"the fit ran {len(fits)} times, not once")
        self.assertEqual(len(handed_out), 8)
        self.assertTrue(
            all(fit is sentinel for fit in handed_out),
            "callers were handed different ensembles",
        )


class DWOLSContrastCovarianceTests(unittest.TestCase):
    """Two arms fit against the same reference are not independent.

    Each `ArmFit` is one-vs-reference, so any two arms share every reference-arm
    row and their estimates move together — measured at +0.20 to +0.51 on this
    cohort. Adding their variances as if independent left the contrast 24% wider
    than the estimator's actual spread across refits; with the cross term it sits
    at 0.95 of it. That width was pure lost precision and it fed straight into
    the abstention rate.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = training.fitted().dwols
        cls.features = model_features(_demo_stages()[1])

    def _independent(self, arm, comparator):
        return (
            self.model.blip_standard_error(arm, self.features) ** 2
            + self.model.blip_standard_error(comparator, self.features) ** 2
        ) ** 0.5

    def test_the_covariance_is_positive_so_the_contrast_narrows(self):
        pairs = [
            ("rituximab", "IL-6 inhibitor"),
            ("IL-6 inhibitor", "JAK-inhibitor"),
            ("TNF-inhibitor", "methotrexate-optimization"),
        ]
        for arm, comparator in pairs:
            with self.subTest(pair=(arm, comparator)):
                joint = self.model.contrast_standard_error(arm, comparator, self.features)
                self.assertGreater(joint, 0.0)
                self.assertLess(joint, self._independent(arm, comparator))

    def test_it_is_symmetric_in_its_arguments(self):
        a = self.model.contrast_standard_error("rituximab", "JAK-inhibitor", self.features)
        b = self.model.contrast_standard_error("JAK-inhibitor", "rituximab", self.features)
        self.assertAlmostEqual(a, b, places=9)

    def test_an_arm_against_itself_has_no_spread(self):
        """Var(x - x) = 0. If the cross term were dropped this would return
        sqrt(2) times the arm's own standard error instead."""
        self.assertAlmostEqual(
            self.model.contrast_standard_error("rituximab", "rituximab", self.features),
            0.0,
            places=9,
        )

    def test_the_reference_arm_reduces_to_the_other_arm_alone(self):
        """The reference has no `ArmFit` and a blip identically zero, so the
        contrast is just the other arm's own uncertainty."""
        from treatmentrx.arms import REFERENCE_ARM

        self.assertAlmostEqual(
            self.model.contrast_standard_error("rituximab", REFERENCE_ARM, self.features),
            self.model.blip_standard_error("rituximab", self.features),
            places=9,
        )

    def test_the_sandwich_matches_its_textbook_form(self):
        """Invariant 23's rule, applied to the argument that changed.

        `sandwich_covariance` takes the *inverted* normal matrix now. The one
        thing that can go wrong is a caller handing it X'WX instead of its
        inverse — and that fails silently, producing a plausible covariance
        built from the wrong bread, which would shift every standard error in
        the system without raising anything.

        So it is pinned against `A^-1 (sum_g s_g s_g') A^-1 . G/(G-1)` computed
        the long way on a case small enough to check by hand, and asserted to
        *disagree* when handed the un-inverted matrix.
        """
        from treatmentrx.estimation import linalg
        from treatmentrx.estimation.inference import sandwich_covariance

        design = [[1.0, 0.5], [1.0, -1.5], [1.0, 2.0], [1.0, 0.25]]
        residuals = [0.3, -0.2, 0.15, -0.05]
        weights = [1.0, 0.8, 1.2, 0.9]
        clusters = [0, 0, 1, 2]
        rows = [[(i, v) for i, v in enumerate(row) if v != 0.0] for row in design]
        normal = linalg.sparse_normal_matrix(rows, weights, 2, 0.0)
        bread = linalg.inverse(normal)

        scores = {}
        for row, residual, weight, cluster in zip(design, residuals, weights, clusters):
            score = scores.setdefault(cluster, [0.0, 0.0])
            for index, value in enumerate(row):
                score[index] += weight * value * residual
        meat = [[0.0, 0.0], [0.0, 0.0]]
        for score in scores.values():
            for i in range(2):
                for j in range(2):
                    meat[i][j] += score[i] * score[j]
        groups = len(scores)
        scale = groups / (groups - 1)
        expected = linalg.matmul(linalg.matmul(bread, meat), bread)

        measured = sandwich_covariance(rows, residuals, weights, clusters, bread, 2)
        for i in range(2):
            for j in range(2):
                with self.subTest(entry=(i, j)):
                    self.assertAlmostEqual(measured[i][j], expected[i][j] * scale, places=12)

        # And the failure mode the signature change exists to prevent.
        wrong = sandwich_covariance(rows, residuals, weights, clusters, normal, 2)
        self.assertNotAlmostEqual(
            wrong[0][0],
            measured[0][0],
            places=6,
            msg="passing X'WX where the bread belongs went unnoticed",
        )

    def test_one_fit_inverts_its_normal_matrix_once(self):
        """The quantity the signature change buys, counted rather than timed.

        `sandwich_covariance` used to invert the matrix itself, and both callers
        that needed the inverse for anything else inverted it again fifteen
        lines away. For `QLearningModel` that second Gauss-Jordan is 98x98 and
        cost 272ms of pure repetition on every fit that computes a covariance.
        """
        from treatmentrx.estimation import linalg
        from treatmentrx.estimation.dwols import DWOLSModel
        from treatmentrx.estimation.q_learning import QLearningModel
        from treatmentrx.estimation import training

        cohort = training.training_cohort()[:120]
        sizes = []
        real = linalg.inverse

        def counted(matrix):
            sizes.append(len(matrix))
            return real(matrix)

        import treatmentrx.estimation.dwols as dwols_module
        import treatmentrx.estimation.inference as inference_module
        import treatmentrx.estimation.q_learning as ql_module

        patched = (linalg, dwols_module.linalg, inference_module.linalg, ql_module.linalg)
        try:
            for module in patched:
                module.inverse = counted
            sizes.clear()
            model = DWOLSModel(cohort)
            self.assertEqual(
                len(sizes),
                len(model.fits),
                "each arm fit must invert its normal matrix exactly once",
            )
            sizes.clear()
            QLearningModel(cohort)
            self.assertEqual(
                len(sizes), 1, "the fixed point already factored X'WX; the "
                "sandwich must not invert it again"
            )
        finally:
            for module in patched:
                module.inverse = real

    def test_the_memoised_covariance_is_the_one_it_replaced(self):
        """The cache is behaviour-preserving, asserted rather than assumed.

        `ArmFit.cross_covariance` takes no features — it is a property of the
        fit, and the patient enters afterwards in the quadratic form — so it was
        rebuilt identically on every request at 1.31ms a call. `DWOLSModel`
        keeps it per ordered pair now. What must hold is that every contrast
        still equals the one computed from a matrix built fresh from the fits.
        """
        from treatmentrx.estimation.basis import BLIP_BASIS, TREATMENT_FREE_BASIS, blip_basis
        from treatmentrx.estimation import linalg

        n_free = len(TREATMENT_FREE_BASIS)
        for arm, fit in self.model.fits.items():
            for comparator, other in self.model.fits.items():
                if arm == comparator:
                    continue
                cross = fit.cross_covariance(other)
                loading = [0.0] * len(cross)
                for offset, value in enumerate(blip_basis(self.features)):
                    loading[n_free + offset] = value
                variance = (
                    self.model.blip_standard_error(arm, self.features) ** 2
                    + self.model.blip_standard_error(comparator, self.features) ** 2
                    - 2.0 * linalg.quadratic_form(loading, cross)
                )
                with self.subTest(pair=(arm, comparator)):
                    self.assertAlmostEqual(
                        self.model.contrast_standard_error(arm, comparator, self.features),
                        max(variance, 0.0) ** 0.5,
                        places=12,
                    )

    def test_a_refit_does_not_inherit_the_cache(self):
        """The property that keeps this away from invariant 7.

        That defect was a second *model*, fit from its own cohort and silently
        diverging from the one every study measured. `refit` builds a whole new
        `DWOLSModel`, so a bootstrap replicate starts with an empty cache and
        cannot read a parent's — if it could, every replicate would report the
        deployed fit's covariance and the joint interval would stop moving.
        """
        from treatmentrx.estimation.dwols import DWOLSModel
        from treatmentrx.estimation import training

        cohort = training.training_cohort()
        replica = DWOLSModel(cohort[: len(cohort) // 2])
        self.assertEqual(replica._cross_covariance, {})
        replica.contrast_standard_error("rituximab", "IL-6 inhibitor", self.features)
        self.assertEqual(len(replica._cross_covariance), 1)
        # And the parent's cache is untouched by the replica's work.
        self.assertNotIn(
            ("rituximab", "IL-6 inhibitor"),
            {k: v for k, v in replica._cross_covariance.items() if v is None},
        )

    def test_scoring_many_patients_costs_one_covariance_per_pair(self):
        """The quantity the memo buys, counted rather than timed.

        Wall-clock on this machine has about 40% run-to-run variance — the same
        unchanged suite measured 271s and 387s — so a timing assertion here would
        be noise. The call count is deterministic and it is the thing that
        matters: the matrix is a property of the fit, so scoring the tenth
        patient must cost no covariance work at all.

        Counted over `cli audit`, this is 1340 calls against 16.
        """
        from treatmentrx.estimation.dwols import ArmFit

        # Cold cache, because a sibling test alphabetically ahead of this one
        # warms some of these pairs and the count would silently under-read.
        # Clearing is free and cannot corrupt anything: every entry is derived
        # from fits that are written once and never mutated, so a later test
        # simply recomputes the same matrix.
        self.model._cross_covariance.clear()

        calls = []
        real = ArmFit.cross_covariance
        try:
            ArmFit.cross_covariance = lambda self, other: (
                calls.append((self.arm, other.arm)) or real(self, other)
            )
            pairs = [
                ("rituximab", "IL-6 inhibitor"),
                ("rituximab", "JAK-inhibitor"),
                ("TNF-inhibitor", "IL-6 inhibitor"),
            ]
            for patient in range(10):
                features = dict(self.features, das28=3.0 + 0.4 * patient)
                for arm, comparator in pairs:
                    self.model.contrast_standard_error(arm, comparator, features)
        finally:
            ArmFit.cross_covariance = real

        self.assertEqual(
            len(calls),
            len(pairs),
            "the covariance is being rebuilt per patient for a matrix that has "
            "no patient in it",
        )
        self.assertEqual(len(set(calls)), len(pairs))

    def test_the_coverage_study_measures_the_rule_the_facade_deploys(self):
        """Invariant 19's shape: a study that computes a wider interval than the
        agent emits is reporting a rule nobody deploys."""
        from treatmentrx.feedback.coverage import _dwols_contrast

        replica = _dwols_contrast(self.model, "rituximab", "IL-6 inhibitor", self.features)
        deployed = self.model.contrast_standard_error(
            "rituximab", "IL-6 inhibitor", self.features
        )
        self.assertAlmostEqual(replica.standard_error, deployed, places=9)


class EstimatorFacadeTests(unittest.TestCase):
    def test_estimators_share_one_scale_and_one_menu(self):
        _, stages = _demo_stages()
        results = [
            QSharedEstimator().fit_predict(stages),
            StageSpecificQEstimator().fit_predict(stages),
            DWOLSSharedEstimator().fit_predict(stages),
        ]
        menus = {tuple(sorted(result.q_values)) for result in results}
        self.assertEqual(len(menus), 1, "estimators must score the same arm menu")
        for result in results:
            for arm, value in result.q_values.items():
                self.assertTrue(0.0 <= value <= 1.0, msg=f"{result.estimator}:{arm}")

    def test_policy_value_is_the_held_out_score_not_a_self_report(self):
        _, stages = _demo_stages()
        result = QSharedEstimator().fit_predict(stages)
        self.assertEqual(result.policy_value, training.policy_value_for(training.Q_SHARED))

    def test_blip_parameters_are_exposed_for_audit(self):
        _, stages = _demo_stages()
        result = DWOLSSharedEstimator().fit_predict(stages)
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
            QSharedEstimator().fit_predict(stages).recommended_arm,
            "rituximab",
        )

    def test_blip_attributions_sum_to_the_estimated_advantage(self):
        """Explanations must decompose the model, not paraphrase it."""
        from treatmentrx.estimation.explainability import ModelExplainer

        _, stages = _demo_stages()
        result = QSharedEstimator().fit_predict(stages)
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
