"""End-to-end workflow behaviour, and the defects a full run surfaced.

Each test here corresponds to something that went wrong when the whole pipeline
was driven over realistic and malformed inputs rather than the one demo patient.
"""

import copy
import unittest

from treatmentrx import TreatmentRxOrchestrator
from treatmentrx.data import DataContractError, DataLayer
from treatmentrx.diseases import UnsupportedDiseaseError
from treatmentrx.demo_data import sample_ra_bundle
from treatmentrx.domain import RecommendationStatus, StageRecord
from treatmentrx.estimation.actions import ARM_CANDIDATES, CompositeActionSpace
from treatmentrx.safety.feasible_set import ALT_CEILING, FeasibleSet


def _bundle(*resources):
    bundle = copy.deepcopy(sample_ra_bundle())
    for resource in resources:
        bundle["entry"].append({"resource": resource})
    return bundle


def _without(resource_type):
    bundle = copy.deepcopy(sample_ra_bundle())
    bundle["entry"] = [
        entry for entry in bundle["entry"] if entry["resource"].get("resourceType") != resource_type
    ]
    return bundle


def _observation(code, value, day=365):
    key = "valueBoolean" if isinstance(value, bool) else "valueQuantity"
    payload = value if isinstance(value, bool) else {"value": value}
    return {"resourceType": "Observation", "code": {"text": code}, key: payload, "effectiveDay": day}


def _allergy(text):
    return {"resourceType": "AllergyIntolerance", "code": {"text": text}}


def _replace_observation(code, value, unit):
    """Rewrite an observation the demo bundle already carries, unit included."""
    bundle = copy.deepcopy(sample_ra_bundle())
    for entry in bundle["entry"]:
        resource = entry["resource"]
        if (
            resource.get("resourceType") == "Observation"
            and resource["code"]["text"].lower() == code.lower()
        ):
            resource["valueQuantity"] = {"value": value, "unit": unit}
    return bundle


class EndpointTests(unittest.TestCase):
    """The reward is a choice, and the choice is now visible.

    `_stage_outcome` used to match keywords in a free-text field. That mapping is
    the objective function; changing it moves every Q-value, and nothing in the
    record said which mapping produced the number.
    """

    def test_the_eular_table_is_the_published_one(self):
        from treatmentrx.data.endpoints import GOOD, MODERATE, NONE, eular_response

        # (baseline, attained) -> expected, straight from the 1996 criteria.
        cases = {
            (6.0, 2.5): GOOD,      # improvement 3.5, low activity attained
            (6.0, 4.0): MODERATE,  # improvement 2.0, moderate activity attained
            (6.0, 5.6): NONE,      # improvement 0.4, below the minimal bar
            (5.0, 4.2): MODERATE,  # improvement 0.8, moderate activity attained
            (5.0, 4.6): NONE,      # improvement 0.4, below the minimal bar
            (6.0, 6.0): NONE,      # no improvement at all
        }
        for (baseline, attained), expected in cases.items():
            with self.subTest(baseline=baseline, attained=attained):
                self.assertEqual(eular_response(baseline, attained), expected)

    def test_a_large_improvement_to_high_activity_is_only_moderate(self):
        """The attained level matters, not just the change — which is the whole
        reason EULAR is a table rather than a threshold on the difference."""
        from treatmentrx.data.endpoints import GOOD, MODERATE, eular_response

        self.assertEqual(eular_response(9.0, 6.0), MODERATE)
        self.assertEqual(eular_response(5.0, 3.0), GOOD)

    def test_an_open_stage_has_no_measured_outcome(self):
        """Scoring the stage the patient is standing at would read the future."""
        from treatmentrx.data.endpoints import EULARResponseEndpoint

        state = DataLayer(endpoint=EULARResponseEndpoint()).build_patient_state(
            sample_ra_bundle()
        )
        self.assertIsNone(state.stages[-1].end_day)

    def test_the_endpoint_actually_changes_the_outcome(self):
        """Otherwise the seam is decorative.

        Needs a record that carries DAS28 *inside* each stage — the demo bundle
        has one measurement at the decision day, so the measured endpoint
        correctly falls back for every stage there. Simulated bundles carry one
        per visit, which is what a real longitudinal extract looks like.
        """
        from treatmentrx.data.endpoints import EULARResponseEndpoint
        from treatmentrx.simulation.fhir_export import simulated_bundles

        bundle = simulated_bundles(1, seed=4242)[0]
        text = DataLayer().build_patient_state(bundle)
        eular = DataLayer(endpoint=EULARResponseEndpoint()).build_patient_state(bundle)
        self.assertNotEqual(
            [stage.outcome for stage in text.stages],
            [stage.outcome for stage in eular.stages],
        )

    def test_the_two_endpoints_disagree_on_this_cohort_by_construction(self):
        """The reason EULAR is available and not the default.

        `ra_cohort` draws a synthetic 0..1 response and moves DAS28 by
        `3 * (outcome - 0.5)`, so an outcome of 0.55 improves DAS28 by 0.15 —
        which EULAR correctly calls no response. The scales are different, and
        reconciling them would mean re-tuning the simulation to suit an endpoint.
        """
        from treatmentrx.data.endpoints import EULARResponseEndpoint
        from treatmentrx.simulation.fhir_export import simulated_bundles

        text_layer = DataLayer()
        eular_layer = DataLayer(endpoint=EULARResponseEndpoint())
        agreements = total = 0
        for bundle in simulated_bundles(12, seed=4242):
            a = text_layer.build_patient_state(bundle)
            b = eular_layer.build_patient_state(bundle)
            for left, right in zip(a.stages, b.stages):
                total += 1
                agreements += left.outcome == right.outcome
        self.assertGreater(total, 0)
        self.assertLess(agreements / total, 0.6, "the endpoints stopped disagreeing")

    def test_the_measured_endpoint_falls_back_rather_than_inventing(self):
        """A stage with no DAS28 inside it is not scoreable from measurements."""
        from treatmentrx.data.endpoints import EULARResponseEndpoint, ResponseTextEndpoint

        endpoint = EULARResponseEndpoint()
        self.assertEqual(
            endpoint.score("good response", [], start_day=0, end_day=90),
            ResponseTextEndpoint().score("good response", [], 0, 90),
        )

    def test_the_default_is_the_text_endpoint(self):
        """Documented, because it is the surprising choice.

        EULAR is the better definition and it is *not* the default: this repo's
        generator draws a synthetic 0..1 response and moves DAS28 by
        `3 * (outcome - 0.5)`, so the two scales disagree by construction. Making
        EULAR the default here would mean re-tuning the simulation to suit it.
        """
        from treatmentrx.data.endpoints import DEFAULT_ENDPOINT, ResponseTextEndpoint

        self.assertIsInstance(DEFAULT_ENDPOINT, ResponseTextEndpoint)
        self.assertFalse(DEFAULT_ENDPOINT.measured)


class UnitHandlingTests(unittest.TestCase):
    """A number is only as good as its label, and the label was never read.

    CRP in mg/dL is an ordinary reporting convention and ten times the same
    concentration in mg/L. Modelled as given it enters `crp_std = (crp - 30)/25`
    an order of magnitude too small.
    """

    def _crp(self, state):
        return state.stages[-1].features["crp"]

    def test_a_recognised_equivalent_is_converted(self):
        canonical = DataLayer().build_patient_state(_replace_observation("CRP", 28, "mg/L"))
        deciliters = DataLayer().build_patient_state(_replace_observation("CRP", 2.8, "mg/dL"))
        self.assertAlmostEqual(self._crp(canonical), self._crp(deciliters), places=6)

    def test_the_conversion_is_recorded_rather_than_silent(self):
        state = DataLayer().build_patient_state(_replace_observation("CRP", 2.8, "mg/dL"))
        notes = [
            issue.message
            for issue in state.data_contract.issues
            if issue.field == "observations.crp.unit"
        ]
        self.assertTrue(notes, "the conversion left no trace in the report")
        self.assertIn("mg/dL", notes[0])
        self.assertIn("converted", notes[0])

    def test_an_incompatible_unit_is_rejected(self):
        """An enzyme activity is not a concentration; there is no conversion."""
        with self.assertRaises(DataContractError):
            DataLayer().build_patient_state(_replace_observation("CRP", 28, "U/L"))

    def test_an_unknown_unit_is_a_warning_not_a_rejection(self):
        """Assuming canonical is the only option; saying so is the point."""
        state = DataLayer().build_patient_state(_replace_observation("CRP", 28, "furlongs"))
        self.assertEqual(self._crp(state), 28.0)
        severities = {
            issue.severity
            for issue in state.data_contract.issues
            if issue.field == "observations.crp.unit"
        }
        self.assertEqual(severities, {"warning"})

    def test_a_missing_unit_is_flagged(self):
        state = DataLayer().build_patient_state(_replace_observation("CRP", 28, None))
        messages = [
            issue.message
            for issue in state.data_contract.issues
            if issue.field == "observations.crp.unit"
        ]
        self.assertTrue(any("no unit" in message for message in messages))

    def test_a_score_needs_no_unit(self):
        """DAS28 is dimensionless — an absent unit there is not an assumption."""
        state = DataLayer().build_patient_state(_replace_observation("DAS28", 5.2, None))
        self.assertEqual(
            [i for i in state.data_contract.issues if i.field == "observations.das28.unit"], []
        )

    def test_a_conversion_can_move_a_value_out_of_its_plausible_range(self):
        """Which is the point of checking units before ranges.

        60 mg/dL is 600 mg/L, outside `PLAUSIBLE_RANGES`. Read as mg/L it would
        pass the gate and be modelled.
        """
        with self.assertRaises(DataContractError):
            DataLayer().build_patient_state(_replace_observation("CRP", 60, "mg/dL"))
        DataLayer().build_patient_state(_replace_observation("CRP", 60, "mg/L"))


class DataContractGateTests(unittest.TestCase):
    """A record the contract rejects must fail *at* the contract.

    These inputs previously reached the stage builder and the DAG registry and
    crashed there with an untyped ValueError, even though the contract had
    already identified exactly what was wrong.
    """

    def test_no_treatment_history_raises_a_typed_error(self):
        with self.assertRaises(DataContractError) as caught:
            TreatmentRxOrchestrator().run(_without("MedicationRequest"))
        self.assertIn("treatment-line event", str(caught.exception))

    def test_non_ra_record_raises_a_typed_error(self):
        with self.assertRaises(UnsupportedDiseaseError) as caught:
            TreatmentRxOrchestrator().run(_without("Condition"))
        self.assertIn("supported", str(caught.exception).lower())

    def test_the_error_carries_the_report(self):
        with self.assertRaises(DataContractError) as caught:
            DataLayer().build_patient_state(_without("MedicationRequest"))
        self.assertTrue(caught.exception.issues)
        self.assertTrue(all(issue.severity == "error" for issue in caught.exception.issues))


class PlausibilityTests(unittest.TestCase):
    """Impossible values are corrupt records, not unusual patients."""

    def test_a_negative_disease_activity_score_is_rejected(self):
        with self.assertRaises(DataContractError) as caught:
            TreatmentRxOrchestrator().run(_bundle(_observation("DAS28", -5)))
        self.assertIn("physiologically possible", str(caught.exception))

    def test_an_impossible_high_score_is_rejected(self):
        with self.assertRaises(DataContractError):
            TreatmentRxOrchestrator().run(_bundle(_observation("DAS28", 12)))

    def test_an_extreme_but_possible_value_is_still_served(self):
        """Range checks must not swallow genuinely sick patients."""
        recommendation = TreatmentRxOrchestrator().run(_bundle(_observation("CRP", 300)))
        self.assertIn(recommendation.status, set(RecommendationStatus))

    def test_a_non_numeric_value_is_flagged_rather_than_silently_defaulted(self):
        bundle = _bundle(
            {
                "resourceType": "Observation",
                "code": {"text": "DAS28"},
                "valueString": "high",
                "effectiveDay": 365,
            }
        )
        state = DataLayer().build_patient_state(bundle)
        messages = [issue.message for issue in state.data_contract.issues]
        self.assertTrue(
            any("non-numeric" in message for message in messages),
            msg=f"no data-quality issue raised: {messages}",
        )


class HepaticSafetyTests(unittest.TestCase):
    """Hepatotoxic agents were guarded on pregnancy and kidneys but not liver."""

    def test_a_failing_liver_removes_the_hepatotoxic_arms(self):
        recommendation = TreatmentRxOrchestrator().run(_bundle(_observation("ALT", 400)))
        removed = recommendation.audit_event["removed_arms"]
        self.assertIn("methotrexate-optimization", removed)
        self.assertIn("JAK-inhibitor", removed)

    def test_methotrexate_is_removed_as_a_background_combination_too(self):
        """MTX is unsafe whether it is the arm or what the arm is added to."""
        state = DataLayer().build_patient_state(_bundle(_observation("ALT", 400)))
        candidates = CompositeActionSpace().candidates(["TNF-inhibitor"], state.stages)
        result = FeasibleSet().filter(candidates, state.stages, [])
        surviving = {action.combination for action in result.feasible}
        self.assertNotIn("MTX", surviving)
        self.assertIn(None, surviving, "the monotherapy option should survive")

    def test_a_healthy_liver_keeps_them(self):
        recommendation = TreatmentRxOrchestrator().run(sample_ra_bundle())
        self.assertEqual(recommendation.audit_event["removed_arms"], {})
        self.assertLess(ALT_CEILING, 400)


class CandidateBreadthTests(unittest.TestCase):
    def test_every_advanced_arm_has_a_monotherapy_option(self):
        """Otherwise one MTX contraindication removes a whole arm.

        A curated set that lists biologics only in combination turns any
        methotrexate contraindication into a blocked recommendation, for a
        patient who had a viable option the model simply never listed.
        """
        for arm, options in ARM_CANDIDATES.items():
            if arm in {"continue-current", "methotrexate-optimization"}:
                continue
            self.assertTrue(
                any(option.combination is None for option in options),
                msg=f"{arm} exists only as a combination",
            )

    def test_pregnancy_no_longer_blocks_a_patient_with_options(self):
        recommendation = TreatmentRxOrchestrator().run(_bundle(_observation("pregnant", True)))
        self.assertNotEqual(recommendation.status, RecommendationStatus.BLOCKED)
        removed = recommendation.audit_event["removed_arms"]
        self.assertIn("methotrexate-optimization", removed)
        self.assertIn("JAK-inhibitor", removed)
        self.assertNotIn("TNF-inhibitor", removed)


class AllergyMatchingTests(unittest.TestCase):
    def test_a_class_level_allergy_removes_the_arm(self):
        """Composites are named by molecule; allergies are often named by class."""
        recommendation = TreatmentRxOrchestrator().run(_bundle(_allergy("TNF-inhibitor")))
        self.assertIn("TNF-inhibitor", recommendation.audit_event["removed_arms"])

    def test_a_drug_level_allergy_still_removes_the_arm(self):
        recommendation = TreatmentRxOrchestrator().run(_bundle(_allergy("tocilizumab")))
        self.assertIn("IL-6 inhibitor", recommendation.audit_event["removed_arms"])

    def test_an_allergy_to_every_arm_blocks(self):
        recommendation = TreatmentRxOrchestrator().run(
            _bundle(*[_allergy(arm) for arm in ARM_CANDIDATES])
        )
        self.assertEqual(recommendation.status, RecommendationStatus.BLOCKED)
        self.assertIsNone(recommendation.recommended_arm)

    def test_a_blank_allergy_record_does_not_block_anyone(self):
        """An empty string is a substring of every arm name.

        A blank AllergyIntolerance is a data-quality artefact, not a clinical
        finding, and it used to take the patient out of service entirely —
        silently, because blocking is the fail-safe direction.
        """
        for blank in ("", "   "):
            with self.subTest(allergy=repr(blank)):
                recommendation = TreatmentRxOrchestrator().run(_bundle(_allergy(blank)))
                self.assertNotEqual(recommendation.status, RecommendationStatus.BLOCKED)
                self.assertNotIn(
                    "allergy_contraindication",
                    {flag.code for flag in recommendation.safety_flags},
                )


class ModelAndPatientLevelSeparationTests(unittest.TestCase):
    """Invariant 14, checked where it was still being broken.

    The competing-risk step scaled the estimator's held-out policy value by the
    patient's own event incidence, producing a number that described neither.
    """

    def test_the_policy_value_survives_the_competing_risk_step_unchanged(self):
        from treatmentrx.decision.bma import BayesianModelAverager
        from treatmentrx.estimation import EstimationLayer
        from treatmentrx.estimation.competing_risk_outcomes import CompetingRiskEndpoint

        state = DataLayer().build_patient_state(sample_ra_bundle())
        averaged = BayesianModelAverager().aggregate(EstimationLayer().estimate(state))
        adjusted = CompetingRiskEndpoint().adjust(
            averaged, state.stages, state.competing_risk_incidence
        )
        self.assertEqual(adjusted.policy_value, averaged.policy_value)
        self.assertIn("competing_risk_penalty", adjusted.coefficients)

    def test_the_penalty_is_still_computed_and_reported(self):
        """Separating it must not mean losing it."""
        from treatmentrx.estimation.competing_risk_outcomes import CompetingRiskEndpoint

        state = DataLayer().build_patient_state(sample_ra_bundle())
        self.assertTrue(state.competing_risk_incidence)
        from treatmentrx.decision.bma import BayesianModelAverager
        from treatmentrx.estimation import EstimationLayer

        averaged = BayesianModelAverager().aggregate(EstimationLayer().estimate(state))
        adjusted = CompetingRiskEndpoint().adjust(
            averaged, state.stages, {"death": 0.5, "dropout": 0.5}
        )
        self.assertAlmostEqual(adjusted.coefficients["competing_risk_penalty"], 0.375)

    def test_the_reported_policy_value_matches_the_held_out_score(self):
        """End to end: what the audit calls a held-out score must be one."""
        from treatmentrx.estimation import training

        recommendation = TreatmentRxOrchestrator().run(sample_ra_bundle())
        reported = recommendation.audit_event["ope"]["model_policy_value"]
        weights = recommendation.audit_event["model_weights"]
        expected = sum(
            weight * training.policy_value_for(name) for name, weight in weights.items()
        )
        self.assertAlmostEqual(reported, expected, places=2)


class ConfidenceBandTests(unittest.TestCase):
    """The band reports the standard error, not a heuristic on top of it."""

    def test_the_band_is_the_measured_interval_not_a_widened_one(self):
        from treatmentrx.decision.bma import BayesianModelAverager
        from treatmentrx.estimation import EstimationLayer
        from treatmentrx.estimation.belief_aware import BeliefAwareAdjuster

        state = DataLayer().build_patient_state(sample_ra_bundle())
        averaged = BayesianModelAverager().aggregate(EstimationLayer().estimate(state))
        adjusted = BeliefAwareAdjuster().adjust(averaged, state.stages)
        self.assertEqual(adjusted.confidence_band, averaged.confidence_band)

    def test_the_band_does_not_run_into_the_ceiling(self):
        """It used to: 0.5 * belief.uncertainty pushed the top end to exactly 1.0,
        where the number carries no information at all."""
        recommendation = TreatmentRxOrchestrator().run(sample_ra_bundle())
        low, high = recommendation.audit_event["confidence_band"]
        self.assertLess(high, 1.0)
        self.assertGreater(low, 0.0)
        self.assertLess(high - low, 0.2)

    def test_belief_uncertainty_is_still_reported_somewhere(self):
        """Dropping it from the band must not drop it from the output."""
        recommendation = TreatmentRxOrchestrator().run(sample_ra_bundle())
        flags = recommendation.audit_event["uncertainty"]["flags"]
        self.assertIn("uncertain_disease_activity_belief", flags)


class OutOfSupportTests(unittest.TestCase):
    """A measured zero is a clinical fact, not a missing value."""

    def test_an_anuric_patient_is_flagged_as_out_of_support(self):
        recommendation = TreatmentRxOrchestrator().run(_bundle(_observation("eGFR", 0)))
        self.assertIn(
            "outside_training_support",
            {flag.code for flag in recommendation.safety_flags},
        )

    def test_a_severe_but_nonzero_egfr_is_flagged_too(self):
        recommendation = TreatmentRxOrchestrator().run(_bundle(_observation("eGFR", 12)))
        self.assertIn(
            "outside_training_support",
            {flag.code for flag in recommendation.safety_flags},
        )

    def test_a_healthy_patient_is_not_flagged(self):
        recommendation = TreatmentRxOrchestrator().run(sample_ra_bundle())
        self.assertNotIn(
            "outside_training_support",
            {flag.code for flag in recommendation.safety_flags},
        )


class EstimandDimensionTests(unittest.TestCase):
    """Estimands are model-level; the patient's deviations are context only."""

    def test_estimands_do_not_depend_on_the_patient(self):
        """They describe the policy, so two patients must get the same values.

        They were previously the model's held-out policy value multiplied by the
        current patient's adherence fraction — a population number scaled by an
        individual one, which is neither quantity.
        """
        first = TreatmentRxOrchestrator().run(sample_ra_bundle())
        second = TreatmentRxOrchestrator().run(_bundle(_observation("CRP", 55)))
        self.assertEqual(
            {e.estimand: e.policy_value for e in first.estimands},
            {e.estimand: e.policy_value for e in second.estimands},
        )

    def test_the_three_estimands_are_genuinely_different(self):
        values = {e.estimand: e.policy_value for e in TreatmentRxOrchestrator().run(sample_ra_bundle()).estimands}
        self.assertEqual(set(values), {"ITT", "per_protocol", "as_treated"})
        self.assertNotEqual(values["ITT"], values["as_treated"])
        self.assertGreater(values["ITT"], values["as_treated"])

    def test_each_estimand_reports_its_own_effective_sample(self):
        for result in TreatmentRxOrchestrator().run(sample_ra_bundle()).estimands:
            self.assertGreater(result.n_effective, 0.0, msg=result.estimand)

    def test_ope_reports_patient_level_values_without_blending(self):
        ope = TreatmentRxOrchestrator().run(sample_ra_bundle()).audit_event["ope"]
        self.assertNotEqual(ope["observed_mean_outcome"], ope["model_policy_value"])
        self.assertIn("must not be combined", ope["note"])

    def test_the_patient_level_block_carries_no_invented_weighting(self):
        """`iptw_policy_value` reweighted one patient's three or four outcomes by
        `adherence x 0.5^switched x 0.7^rescue`. Measured over 728 stage-rows,
        adherence was identically 1.0 and rescue never fired, so the whole weight
        was one hand-set constant — and it moved the reported value by up to
        0.112. There is no population estimand for a single trajectory, so
        fitting the constant would not have fixed it."""
        ope = TreatmentRxOrchestrator().run(sample_ra_bundle()).audit_event["ope"]
        self.assertNotIn("iptw_policy_value", ope)
        self.assertNotIn("naive_policy_value", ope)
        self.assertIn("observed_mean_outcome", ope)
        self.assertIn("n_stages", ope)
        self.assertIn("stages_switched", ope)


class UncertaintyTests(unittest.TestCase):
    def test_every_out_of_distribution_term_can_fire_on_its_own(self):
        """A gate whose criteria cannot cross their own threshold is not a gate.

        The score routes to manual review above 0.75, so each term is normalised
        to reach 1.0 alone — and each threshold has to sit inside what the data
        contract actually admits, or the criterion is decoration.
        """
        from treatmentrx.decision.uncertainty import UncertaintyDecomposer
        from treatmentrx.data import DataLayer

        stages = DataLayer().build_patient_state(sample_ra_bundle()).stages
        decomposer = UncertaintyDecomposer()
        baseline = decomposer._ood_score(stages)
        self.assertLess(baseline, 0.75, "the demo patient is not out of distribution")

        for key, extreme in (("das28", 10.0), ("crp", 400.0), ("egfr", 2.0)):
            with self.subTest(covariate=key):
                shifted = list(stages)
                features = dict(shifted[-1].features)
                features[key] = extreme
                shifted[-1] = StageRecord(**(shifted[-1].__dict__ | {"features": features}))
                self.assertGreater(
                    decomposer._ood_score(shifted),
                    0.75,
                    f"{key}={extreme} is inside the contract's range but does not trip review",
                )

    def test_the_encoder_no_longer_contributes_to_the_score(self):
        """It never could. The encoder is a deterministic summariser of nine
        features normalised into [0, 1]; the removed term needed their RMS to
        exceed 0.85, and measured over 121 patients it topped out at 0.664. The
        term contributed exactly zero to every score ever produced."""
        import inspect

        from treatmentrx.decision.uncertainty import UncertaintyDecomposer

        source = inspect.getsource(UncertaintyDecomposer._ood_score)
        body = source.split('"""')[-1]
        self.assertNotIn("encoded_state", body)
        self.assertNotIn("vector", body)

    def test_epistemic_uncertainty_is_the_decision_standard_error(self):
        """Not `1/sqrt(visits this patient has had)`, which measures something else."""
        recommendation = TreatmentRxOrchestrator().run(sample_ra_bundle())
        audit = recommendation.audit_event
        self.assertAlmostEqual(
            audit["uncertainty"]["epistemic"], audit["contrast"]["standard_error"], places=6
        )

    def test_a_short_history_is_reported_separately(self):
        recommendation = TreatmentRxOrchestrator().run(sample_ra_bundle())
        flags = recommendation.audit_event["uncertainty"]["flags"]
        self.assertNotIn("limited_history", flags, "the demo patient has three stages")


class EValueTests(unittest.TestCase):
    """The E-value has to be about confounding, not a re-encoding of the Q-gap.

    It used to be `rr = q_values[0] / q_values[1]` fed into the E-value formula:
    a ratio of two nearly-equal bounded means, never referencing confounding.
    Across 60 patients it returned 1.000 to 1.617, with 11 under 1.1 — and a
    published E-value of 1.0 asserts that *no* unmeasured confounding is needed.
    """

    @classmethod
    def setUpClass(cls):
        from treatmentrx.orchestrator import TreatmentRxOrchestrator
        from treatmentrx.demo_data import sample_ra_bundle

        cls.recommendation = TreatmentRxOrchestrator().run(sample_ra_bundle())
        cls.sensitivity = cls.recommendation.explanation.sensitivity

    def test_a_zero_effect_needs_no_confounding(self):
        """The anchor of the whole scale, and the old version got it backwards."""
        from treatmentrx.estimation.explainability import _e_value

        self.assertEqual(_e_value(0.0, 0.15), 1.0)
        self.assertGreater(_e_value(0.05, 0.15), 1.0)

    def test_it_grows_with_the_effect_and_shrinks_with_the_spread(self):
        """Both directions, because an E-value is a standardised quantity."""
        from treatmentrx.estimation.explainability import _e_value

        self.assertGreater(_e_value(0.08, 0.15), _e_value(0.04, 0.15))
        self.assertGreater(_e_value(0.04, 0.10), _e_value(0.04, 0.20))

    def test_it_is_computed_from_the_contrast_the_decision_reports(self):
        """Not rebuilt from `q_values`. Invariant 16's principle, one layer over."""
        contrast = self.recommendation.audit_event.get("contrast") or {}
        self.assertAlmostEqual(
            self.sensitivity.contrast, contrast["difference"], places=4
        )

    def test_the_interval_bound_is_one_exactly_when_zero_is_inside(self):
        """The number a reader should quote, and the case that makes 1.0 mean something.

        If the interval already contains zero then no confounding is required to
        reach "no difference" — so 1.0 is the correct answer rather than an
        artifact of two near-equal numbers, which is what it used to be.
        """
        from treatmentrx.data.contract import DataContractError
        from treatmentrx.orchestrator import TreatmentRxOrchestrator
        from treatmentrx.simulation.fhir_export import simulated_bundles

        orchestrator = TreatmentRxOrchestrator()
        checked = 0
        for bundle in simulated_bundles(25, seed=991):
            try:
                recommendation = orchestrator.run(bundle)
            except DataContractError:
                continue
            contrast = recommendation.audit_event.get("contrast") or {}
            if "interval" not in contrast:
                continue
            low, high = contrast["interval"]
            spans_zero = low <= 0.0 <= high
            bound = recommendation.explanation.sensitivity.e_value_for_interval
            with self.subTest(interval=(low, high)):
                if spans_zero:
                    self.assertEqual(bound, 1.0)
                else:
                    self.assertGreater(bound, 1.0)
            checked += 1
        self.assertGreater(checked, 5, "no patient carried a contrast interval")

    def test_the_point_bound_is_never_below_the_interval_bound(self):
        """The limit nearest the null is closer to zero than the estimate is."""
        self.assertGreaterEqual(
            self.sensitivity.e_value, self.sensitivity.e_value_for_interval
        )

    def test_the_note_says_what_was_assumed(self):
        """A bound whose approximation is unstated is a number without a method."""
        note = self.sensitivity.note
        self.assertIn("VanderWeele", note)
        self.assertIn("continuous", note)
        self.assertIn(f"{self.sensitivity.outcome_sd:.4f}", note)

    def test_the_spread_is_a_population_quantity(self):
        """Invariant 14: one patient's contrast over the cohort's spread is Cohen's d.

        Taking the spread from the patient would make the denominator a
        patient-level quantity and the ratio meaningless.
        """
        from treatmentrx.estimation import training

        self.assertAlmostEqual(
            self.sensitivity.outcome_sd, round(training.holdout_outcome_sd(), 4), places=4
        )


if __name__ == "__main__":
    unittest.main()
