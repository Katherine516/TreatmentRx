"""End-to-end workflow behaviour, and the defects a full run surfaced.

Each test here corresponds to something that went wrong when the whole pipeline
was driven over realistic and malformed inputs rather than the one demo patient.
"""

import copy
import unittest

from treatmentrx import TreatmentRxOrchestrator
from treatmentrx.data import DataContractError, DataLayer
from treatmentrx.demo_data import sample_ra_bundle
from treatmentrx.domain import RecommendationStatus
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


class DataContractGateTests(unittest.TestCase):
    """A record the contract rejects must fail *at* the contract.

    These inputs previously reached the stage builder and the DAG registry and
    crashed there with an untyped ValueError, even though the contract had
    already identified exactly what was wrong.
    """

    def test_no_treatment_history_raises_a_typed_error(self):
        with self.assertRaises(DataContractError) as caught:
            TreatmentRxOrchestrator().run(_without("MedicationRequest"))
        self.assertIn("treatment event", str(caught.exception))

    def test_non_ra_record_raises_a_typed_error(self):
        with self.assertRaises(DataContractError) as caught:
            TreatmentRxOrchestrator().run(_without("Condition"))
        self.assertIn("rheumatoid", str(caught.exception).lower())

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
        self.assertNotEqual(ope["naive_policy_value"], ope["model_policy_value"])
        self.assertIn("must not be averaged together", ope["note"])


class UncertaintyTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
