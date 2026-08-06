import unittest

from precisionrx_agent.demo_data import sample_ra_bundle
from precisionrx_agent.fhir import FHIRAdapter
from precisionrx_agent.layer1_ingestion.ra_data_contract import RADataContract
from precisionrx_agent.layer2_encoder.baseline import GRUBaselineEncoder, HandcraftedFeatureEncoder
from precisionrx_agent.layer3_causal_dag.dag import CausalDAGRegistry
from precisionrx_agent.layer6_uncertainty.uncertainty import UncertaintyDecomposer
from precisionrx_agent.layer7_calibration.calibration import CalibrationEvaluator
from precisionrx_agent.layer1_ingestion.data_engineering import IPCWHandler, StageHistoryBuilder, VisitAligner
from precisionrx_agent.layer4_estimation.estimation import QSharedEstimator
from precisionrx_agent.layer4_estimation.regime import AdaptiveRegimeSelector
from precisionrx_agent.models import RecommendationStatus
from precisionrx_agent.pipeline import PrecisionRxAgent


class PrecisionRxPipelineTests(unittest.TestCase):
    def test_fhir_adapter_parses_patient_bundle(self) -> None:
        patient = FHIRAdapter().parse_bundle(sample_ra_bundle())

        self.assertEqual(patient.patient_id, "patient-demo-001")
        self.assertEqual(patient.disease, "Rheumatoid Arthritis")
        self.assertEqual(len(patient.medications), 3)
        self.assertTrue(any(observation.code == "DAS28" for observation in patient.observations))

    def test_agent_returns_structured_recommendation(self) -> None:
        recommendation = PrecisionRxAgent().recommend_from_fhir(sample_ra_bundle())

        self.assertIn(recommendation.status, set(RecommendationStatus))
        self.assertEqual(recommendation.statistical_output["regime_type"], "SPTR")
        self.assertIn("recommended_action", recommendation.statistical_output)
        self.assertIn("uncertainty", recommendation.audit_event)
        self.assertTrue(recommendation.audit_event["data_contract_passed"])
        self.assertTrue(recommendation.audit_event["dag_identified"])
        self.assertEqual(recommendation.audit_event["encoder_dimension"], 256)
        self.assertTrue(recommendation.clinician_rationale)
        self.assertNotIn("patient-demo-001", str(recommendation.patient_context))

    def test_safety_gate_blocks_allergy_conflict(self) -> None:
        # Record an allergy to whatever the model actually recommends, so the
        # test tracks the safety gate rather than a hard-coded arm.
        recommended = PrecisionRxAgent().recommend_from_fhir(sample_ra_bundle())
        bundle = sample_ra_bundle()
        bundle["entry"].append(
            {
                "resource": {
                    "resourceType": "AllergyIntolerance",
                    "code": {"text": recommended.statistical_output["recommended_action"]},
                }
            }
        )

        recommendation = PrecisionRxAgent().recommend_from_fhir(bundle)

        self.assertEqual(recommendation.status, RecommendationStatus.BLOCKED)
        self.assertTrue(recommendation.safety.contraindication_flag)
        self.assertIn("blocked", recommendation.clinician_rationale.lower())

    def test_ra_data_contract_maps_required_treatment_arms(self) -> None:
        patient = FHIRAdapter().parse_bundle(sample_ra_bundle())
        contract = RADataContract()
        report = contract.validate(patient)

        self.assertTrue(report.passed)
        self.assertEqual(contract.normalize_treatment_arm("adalimumab TNF inhibitor"), "TNF-inhibitor")
        self.assertEqual(contract.normalize_treatment_arm("tocilizumab"), "IL-6 inhibitor")
        self.assertEqual(contract.normalize_treatment_arm("upadacitinib"), "JAK-inhibitor")

    def test_ra_dag_returns_adjustment_set_and_causal_path(self) -> None:
        patient = FHIRAdapter().parse_bundle(sample_ra_bundle())
        result = CausalDAGRegistry().validate(patient, treatment="IL-6 inhibitor")

        self.assertTrue(result.identified)
        self.assertIn("baseline_disease_activity", result.adjustment_set)
        self.assertIn("colliders", result.causal_path_text)

    def test_baseline_encoders_create_stable_state_vectors(self) -> None:
        patient = FHIRAdapter().parse_bundle(sample_ra_bundle())
        stages = StageHistoryBuilder().build(patient)
        stages = VisitAligner().apply(stages, patient.encounters)
        stages = IPCWHandler().apply(stages)

        handcrafted = HandcraftedFeatureEncoder().encode(stages)
        gru_state = GRUBaselineEncoder().encode(stages)

        self.assertEqual(len(handcrafted.vector), 32)
        self.assertEqual(len(gru_state.vector), 256)
        self.assertIn("das28", handcrafted.feature_map)

    def test_uncertainty_and_calibration_reports_are_structured(self) -> None:
        patient = FHIRAdapter().parse_bundle(sample_ra_bundle())
        stages = StageHistoryBuilder().build(patient)
        assignment = AdaptiveRegimeSelector().select(stages)
        method = QSharedEstimator().fit_predict(stages, assignment, ["das28", "crp"])

        uncertainty = UncertaintyDecomposer().decompose(stages, method, [method])
        calibration = CalibrationEvaluator().evaluate([0.2, 0.7], [0.0, 1.0], bins=2)

        self.assertGreaterEqual(uncertainty.aleatoric, 0)
        self.assertIn(calibration.passed, {True, False})
        self.assertEqual(len(calibration.reliability_bins), 2)


if __name__ == "__main__":
    unittest.main()
