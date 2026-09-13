"""Regression tests for the cross-layer boundaries a plausible happy path misses."""

import copy
import unittest

from treatmentrx import TreatmentRxOrchestrator, UnsupportedDiseaseError
from treatmentrx.data import DataContractError, DataLayer
from treatmentrx.data.fhir import FHIRAdapter
from treatmentrx.demo_data import sample_ra_bundle
from treatmentrx.diseases import DiseaseRegistry
from treatmentrx.domain import RecommendationStatus
from treatmentrx.feedback.switching_aware_ope import SwitchingAwareOPE
from treatmentrx.simulation.fhir_export import simulated_bundles


def _allergy(bundle, text):
    bundle["entry"].append(
        {
            "resource": {
                "resourceType": "AllergyIntolerance",
                "code": {"text": text},
            }
        }
    )
    return bundle


class RecommendationPublicationTests(unittest.TestCase):
    def test_equipoise_publishes_no_recommended_arm(self):
        orchestrator = TreatmentRxOrchestrator()
        recommendation = next(
            result
            for result in (
                orchestrator.run(bundle)
                for bundle in simulated_bundles(20, seed=1234)
            )
            if result.status is RecommendationStatus.EQUIPOISE
        )
        self.assertIsNone(recommendation.recommended_arm)
        self.assertIn(recommendation.top_scored_arm, recommendation.q_values)
        self.assertNotIn(recommendation.top_scored_arm, recommendation.patient_summary)

    def test_out_of_distribution_review_publishes_no_recommended_arm(self):
        bundle = copy.deepcopy(sample_ra_bundle())
        bundle["entry"].append(
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"text": "DAS28"},
                    "valueQuantity": {"value": 9.5, "unit": "score"},
                    "effectiveDay": 365,
                }
            }
        )
        recommendation = TreatmentRxOrchestrator().run(bundle)
        self.assertEqual(recommendation.status, RecommendationStatus.REVIEW)
        self.assertIsNone(recommendation.recommended_arm)
        self.assertIn(recommendation.top_scored_arm, recommendation.q_values)
        self.assertNotIn(recommendation.top_scored_arm, recommendation.patient_summary)


class DecisionPointContractTests(unittest.TestCase):
    def test_missing_current_decision_point_is_rejected(self):
        bundle = copy.deepcopy(sample_ra_bundle())
        bundle["entry"] = [
            entry
            for entry in bundle["entry"]
            if not (
                entry["resource"].get("resourceType") == "MedicationRequest"
                and entry["resource"].get("authoredOnDay") == 365
            )
        ]
        with self.assertRaises(DataContractError) as caught:
            TreatmentRxOrchestrator().run(bundle)
        self.assertIn("current decision", str(caught.exception))

    def test_known_response_on_open_decision_is_rejected(self):
        bundle = copy.deepcopy(sample_ra_bundle())
        for entry in bundle["entry"]:
            resource = entry["resource"]
            if resource.get("medicationCodeableConcept", {}).get("text") == "current decision point":
                resource["response"] = "good response"
        with self.assertRaises(DataContractError) as caught:
            TreatmentRxOrchestrator().run(bundle)
        self.assertIn("has not occurred", str(caught.exception))


class PatientScopeAndPrivacyTests(unittest.TestCase):
    def test_multiple_patient_resources_are_rejected(self):
        bundle = copy.deepcopy(sample_ra_bundle())
        bundle["entry"].append(
            {"resource": {"resourceType": "Patient", "id": "another-patient"}}
        )
        with self.assertRaisesRegex(ValueError, "exactly one Patient"):
            FHIRAdapter().parse_bundle(bundle)

    def test_mismatched_subject_reference_is_rejected(self):
        bundle = copy.deepcopy(sample_ra_bundle())
        bundle["entry"].append(
            {
                "resource": {
                    "resourceType": "Observation",
                    "subject": {"reference": "Patient/someone-else"},
                    "code": {"text": "CRP"},
                    "valueQuantity": {"value": 10, "unit": "mg/L"},
                    "effectiveDay": 365,
                }
            }
        )
        with self.assertRaisesRegex(ValueError, "not the bundle patient"):
            FHIRAdapter().parse_bundle(bundle)

    def test_cross_layer_state_contains_only_the_patient_hash(self):
        state = DataLayer().build_patient_state(sample_ra_bundle())
        self.assertFalse(hasattr(state, "raw_patient"))
        self.assertEqual({stage.patient_id for stage in state.stages}, {state.patient_hash})
        self.assertNotIn("patient-demo-001", repr(state))


class AllergyTextTests(unittest.TestCase):
    def test_common_allergy_prose_blocks_the_top_arm(self):
        for text in ("rituximab allergy", "Allergy to rituximab"):
            with self.subTest(text=text):
                recommendation = TreatmentRxOrchestrator().run(
                    _allergy(copy.deepcopy(sample_ra_bundle()), text)
                )
                self.assertEqual(recommendation.status, RecommendationStatus.BLOCKED)
                self.assertIsNone(recommendation.recommended_arm)
                self.assertIn("rituximab", recommendation.audit_event["removed_arms"])


class FeedbackOutcomeTests(unittest.TestCase):
    def test_open_stage_placeholder_is_not_reported_as_observed(self):
        state = DataLayer().build_patient_state(sample_ra_bundle())
        selected = TreatmentRxOrchestrator().estimation.estimate(state)[0]
        result = SwitchingAwareOPE().evaluate(state.stages, selected)
        completed = [stage for stage in state.stages if stage.end_day is not None]
        self.assertEqual(result.n_stages, len(completed))
        self.assertAlmostEqual(
            result.observed_mean_outcome,
            round(sum(stage.outcome for stage in completed) / len(completed), 3),
        )


class DiseaseIsolationTests(unittest.TestCase):
    def test_registry_exposes_only_the_implemented_ra_workflow(self):
        registry = DiseaseRegistry()
        self.assertEqual(registry.supported_ids(), ["rheumatoid_arthritis"])
        capability = registry.capabilities()[0]
        self.assertEqual(capability["disease_id"], "rheumatoid_arthritis")
        self.assertIn("rituximab", capability["treatment_arms"])

    def test_unsupported_disease_never_falls_back_to_ra(self):
        bundle = copy.deepcopy(sample_ra_bundle())
        for entry in bundle["entry"]:
            resource = entry["resource"]
            if resource.get("resourceType") == "Condition":
                resource["code"] = {"text": "Asthma"}
        with self.assertRaises(UnsupportedDiseaseError):
            TreatmentRxOrchestrator().run(bundle)

    def test_supported_disease_need_not_be_the_first_condition(self):
        bundle = copy.deepcopy(sample_ra_bundle())
        condition_index = next(
            index
            for index, entry in enumerate(bundle["entry"])
            if entry["resource"].get("resourceType") == "Condition"
        )
        bundle["entry"].insert(
            condition_index,
            {
                "resource": {
                    "resourceType": "Condition",
                    "code": {"text": "Hypertension"},
                }
            },
        )
        recommendation = TreatmentRxOrchestrator().run(bundle)
        self.assertEqual(
            recommendation.audit_event["disease_definition"]["disease_id"],
            "rheumatoid_arthritis",
        )

    def test_audit_pins_the_disease_definition(self):
        recommendation = TreatmentRxOrchestrator().run(sample_ra_bundle())
        definition = recommendation.audit_event["disease_definition"]
        self.assertEqual(definition["disease_id"], "rheumatoid_arthritis")
        self.assertEqual(
            definition,
            recommendation.provenance["disease_definition"],
        )


if __name__ == "__main__":
    unittest.main()
