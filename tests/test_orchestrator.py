"""End to end — the six layers in the one order they are allowed to run in."""

import unittest

from treatmentrx import TreatmentRxOrchestrator
from treatmentrx.contracts import Recommendation
from treatmentrx.demo_data import sample_ra_bundle
from treatmentrx.domain import OverrideChannel, RecommendationStatus
from treatmentrx.orchestrator import LAYER_ORDER


class OrchestratorTests(unittest.TestCase):
    def test_runs_all_six_layers_in_order(self):
        recommendation = TreatmentRxOrchestrator().run(sample_ra_bundle())
        self.assertIsInstance(recommendation, Recommendation)
        self.assertEqual(recommendation.provenance["layer_order"], list(LAYER_ORDER))
        self.assertTrue(recommendation.provenance["schema_validated"])
        self.assertIn(recommendation.status, set(RecommendationStatus))

    def test_recommendation_carries_every_layer_output(self):
        recommendation = TreatmentRxOrchestrator().run(sample_ra_bundle())
        audit = recommendation.audit_event
        self.assertTrue(audit["data_contract_passed"])
        self.assertTrue(audit["dag_identified"])
        self.assertEqual(audit["encoder_dimension"], 256)
        self.assertEqual(audit["calibration"]["measured_on"], "held-out cohort")
        self.assertIn("estimator_scorecard", audit)
        self.assertIn("competing_risk_incidence", audit)
        self.assertIn("iptw_policy_value", audit["ope"])
        self.assertIsNotNone(recommendation.explanation)
        self.assertIsNotNone(recommendation.validation)
        self.assertEqual(
            {result.estimand for result in recommendation.estimands},
            {"ITT", "per_protocol", "as_treated"},
        )

    def test_no_raw_patient_identifier_reaches_the_output(self):
        recommendation = TreatmentRxOrchestrator().run(sample_ra_bundle())
        rendered = repr(recommendation)
        self.assertNotIn("patient-demo-001", rendered)

    def test_the_only_wall_clock_value_is_the_audit_timestamp(self):
        """Determinism: two runs of the same commit differ only in the timestamp."""
        first = TreatmentRxOrchestrator().run(sample_ra_bundle()).audit_event
        second = TreatmentRxOrchestrator().run(sample_ra_bundle()).audit_event
        self.assertNotEqual(first.pop("timestamp"), None)
        second.pop("timestamp")
        self.assertEqual(first, second)

    def test_memory_continuity_and_override_loop(self):
        orchestrator = TreatmentRxOrchestrator()
        first = orchestrator.run(sample_ra_bundle())
        self.assertNotIn("Continuity", first.clinician_card)

        routing = orchestrator.submit_override(
            first, clinician_action="JAK-inhibitor", reason_text="local protocol prefers JAK"
        )
        self.assertEqual(routing.channel, OverrideChannel.GUIDELINE_CONFLICT)
        self.assertFalse(routing.influences_model)

        second = orchestrator.run(sample_ra_bundle())
        self.assertIn("Continuity", second.clinician_card)
        # Continuity may reframe the narrative; it may not move the recommendation.
        self.assertEqual(second.recommended_arm, first.recommended_arm)
        self.assertEqual(second.q_values, first.q_values)

    def test_seropositive_prior_tnf_failure_gets_rituximab(self):
        """The demo patient's clinical signature has a known right answer."""
        recommendation = TreatmentRxOrchestrator().run(sample_ra_bundle())
        self.assertEqual(recommendation.recommended_arm, "rituximab")
        self.assertEqual(recommendation.status, RecommendationStatus.RECOMMEND)

    def test_scored_menu_matches_the_contract_vocabulary(self):
        """One arm vocabulary: a naming mismatch must never look like a safety call."""
        recommendation = TreatmentRxOrchestrator().run(sample_ra_bundle())
        self.assertEqual(recommendation.audit_event["removed_arms"], {})
        self.assertEqual(len(recommendation.q_values), 6)


if __name__ == "__main__":
    unittest.main()
