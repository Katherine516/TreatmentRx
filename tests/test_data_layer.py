"""Layer 1 — ingestion, stage construction, and the guards that must raise."""

import unittest

from treatmentrx.contracts import PatientState
from treatmentrx.data import DataLayer
from treatmentrx.data.contract import RADataContract
from treatmentrx.data.dag import CausalDAGRegistry
from treatmentrx.data.encoders import GRUBaselineEncoder, HandcraftedFeatureEncoder
from treatmentrx.data.fhir import FHIRAdapter
from treatmentrx.data.leakage import LeakageError, LeakageTestSuite, TemporalFirewall
from treatmentrx.data.stages import IPCWHandler, StageHistoryBuilder, VisitAligner
from treatmentrx.demo_data import sample_ra_bundle
from treatmentrx.domain import CareGoal, Observation, PatientRecord, StageRecord


def _patient():
    return FHIRAdapter().parse_bundle(sample_ra_bundle())


def _stages():
    patient = _patient()
    return patient, StageHistoryBuilder().build(patient)


class IngestionTests(unittest.TestCase):
    def test_fhir_adapter_parses_patient_bundle(self):
        patient = _patient()
        self.assertEqual(patient.patient_id, "patient-demo-001")
        self.assertEqual(patient.disease, "Rheumatoid Arthritis")
        self.assertEqual(len(patient.medications), 3)
        self.assertTrue(any(observation.code == "DAS28" for observation in patient.observations))

    def test_patient_id_never_leaves_the_data_layer(self):
        state = DataLayer().build_patient_state(sample_ra_bundle())
        self.assertNotIn("patient-demo-001", state.patient_hash)
        self.assertEqual(len(state.patient_hash), 16)

    def test_data_contract_maps_medications_to_the_shared_vocabulary(self):
        contract = RADataContract()
        report = contract.validate(_patient())
        self.assertTrue(report.passed)
        self.assertEqual(contract.normalize_treatment_arm("adalimumab TNF inhibitor"), "TNF-inhibitor")
        self.assertEqual(contract.normalize_treatment_arm("tocilizumab"), "IL-6 inhibitor")
        self.assertEqual(contract.normalize_treatment_arm("upadacitinib"), "JAK-inhibitor")
        self.assertEqual(contract.normalize_treatment_arm("rituximab"), "rituximab")

    def test_dag_returns_adjustment_set_and_causal_path(self):
        result = CausalDAGRegistry().validate(_patient(), treatment="IL-6 inhibitor")
        self.assertTrue(result.identified)
        self.assertIn("baseline_disease_activity", result.adjustment_set)
        self.assertIn("colliders", result.causal_path_text)

    def test_encoders_produce_stable_state_vectors(self):
        patient, stages = _stages()
        stages = IPCWHandler().apply(VisitAligner().apply(stages, patient.encounters))
        self.assertEqual(len(HandcraftedFeatureEncoder().encode(stages).vector), 32)
        self.assertEqual(len(GRUBaselineEncoder().encode(stages).vector), 256)


class ClinicalRealismTests(unittest.TestCase):
    """The v5.1 annotations must survive all the way into the PatientState."""

    def test_every_annotation_reaches_the_contract(self):
        state = DataLayer().build_patient_state(sample_ra_bundle())
        self.assertIsInstance(state, PatientState)
        latest = state.latest
        self.assertIsNotNone(latest.timing, "timing annotation was dropped")
        self.assertIsNotNone(latest.belief, "belief annotation was dropped")
        self.assertIsNotNone(latest.event, "competing-risk annotation was dropped")
        self.assertIsNotNone(latest.switching, "switching annotation was dropped")

    def test_timing_captures_irregular_intervals(self):
        state = DataLayer().build_patient_state(sample_ra_bundle())
        self.assertIsNone(state.stages[0].timing.time_since_last_treatment)
        self.assertIsNotNone(state.stages[1].timing.time_since_last_treatment)
        self.assertGreater(state.latest.timing.inverse_intensity_weight, 0)

    def test_switching_flags_loss_of_response(self):
        state = DataLayer().build_patient_state(sample_ra_bundle())
        self.assertTrue(any(s.switching and s.switching.switched for s in state.stages))

    def test_belief_is_bounded(self):
        belief = DataLayer().build_patient_state(sample_ra_bundle()).latest.belief
        self.assertTrue(0.0 <= belief.activity <= 1.0)
        self.assertTrue(0.0 <= belief.uncertainty <= 0.5)

    def test_care_goal_is_inferred_from_the_trajectory(self):
        state = DataLayer().build_patient_state(sample_ra_bundle())
        self.assertEqual(state.care_goal, CareGoal.INDUCTION)
        self.assertTrue(all(stage.care_goal == state.care_goal for stage in state.stages))

    def test_toxicity_overrides_the_inferred_goal(self):
        """A failing liver is not an induction conversation."""
        patient, stages = _stages()
        toxic = dict(stages[-1].features)
        toxic["alt"] = 180.0
        stages[-1] = StageRecord(**(stages[-1].__dict__ | {"features": toxic}))
        self.assertEqual(DataLayer().infer_care_goal(stages), CareGoal.TOXICITY_CONTROL)


class LeakageTests(unittest.TestCase):
    def test_firewall_catches_a_future_feature(self):
        patient, stages = _stages()
        tampered_features = dict(stages[0].features)
        tampered_features["future_marker"] = 1.0
        tampered = StageRecord(**(stages[0].__dict__ | {"features": tampered_features}))
        leaky = PatientRecord(
            patient_id=patient.patient_id,
            disease=patient.disease,
            demographics=patient.demographics,
            conditions=patient.conditions,
            allergies=patient.allergies,
            medications=patient.medications,
            observations=patient.observations
            + [Observation(code="future marker", value=1.0, days_from_baseline=10_000)],
            encounters=patient.encounters,
            outcomes=patient.outcomes,
        )
        self.assertTrue(TemporalFirewall().check(leaky, [tampered]))
        with self.assertRaises(LeakageError):
            TemporalFirewall().assert_clean(leaky, [tampered])

    def test_suite_passes_on_the_clean_demo(self):
        patient, stages = _stages()
        self.assertTrue(LeakageTestSuite().run(patient, stages).passed)

    def test_a_leaking_record_cannot_produce_a_patient_state(self):
        """The guard raises out of the pipeline; it is never a soft diagnostic."""
        bundle = sample_ra_bundle()
        bundle["entry"].append(
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"text": "future marker"},
                    "valueQuantity": {"value": 1.0},
                    "effectiveDay": 10_000,
                }
            }
        )
        layer = DataLayer()
        patient = FHIRAdapter().parse_bundle(bundle)
        stages = layer.stage_builder.build(patient)
        leaked = dict(stages[0].features)
        leaked["future_marker"] = 1.0
        stages[0] = StageRecord(**(stages[0].__dict__ | {"features": leaked}))
        with self.assertRaises(LeakageError):
            TemporalFirewall().assert_clean(patient, stages)


if __name__ == "__main__":
    unittest.main()
