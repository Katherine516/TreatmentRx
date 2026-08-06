import unittest

from precisionrx_agent.demo_data import sample_ra_bundle
from precisionrx_agent.pipeline import PrecisionRxAgent
from treatmentrx import TreatmentRxOrchestrator
from treatmentrx.agent import AgentLayer
from treatmentrx.contracts import (
    Decision,
    PatientState,
    Recommendation,
    RecommendationStatus,
    RegimeEstimate,
    SafeDecision,
)
from treatmentrx.data import DataLayer
from treatmentrx.decision import DecisionLayer
from treatmentrx.estimation import EstimationLayer
from treatmentrx.safety import SafetyLayer


class V5ArchitectureTests(unittest.TestCase):
    def test_layer_contract_flow_is_explicit(self):
        state = DataLayer().build_patient_state(sample_ra_bundle())
        estimates = EstimationLayer().estimate(state)
        decision = DecisionLayer().decide(state, estimates)
        safe = SafetyLayer().apply(decision, state)
        context = AgentLayer().build_context(safe)
        recommendation = AgentLayer().run_agents(context, safe)

        self.assertIsInstance(state, PatientState)
        self.assertTrue(state.diagnostics_passed)
        self.assertTrue(estimates)
        self.assertTrue(all(isinstance(estimate, RegimeEstimate) for estimate in estimates))
        self.assertIsInstance(decision, Decision)
        self.assertIsInstance(safe, SafeDecision)
        self.assertIsInstance(recommendation, Recommendation)
        self.assertIn(recommendation.status, set(RecommendationStatus))

    def test_thin_orchestrator_runs_all_six_layers(self):
        recommendation = TreatmentRxOrchestrator().run(sample_ra_bundle())

        self.assertIsInstance(recommendation, Recommendation)
        self.assertEqual(
            recommendation.provenance["layer_order"],
            ["data", "estimation", "decision", "safety", "agent", "feedback"],
        )
        self.assertTrue(recommendation.provenance["feedback"]["observational_enqueued"])
        self.assertTrue(recommendation.provenance["schema_validated"])
        self.assertIn("model", recommendation.provenance["versions"])

    def test_both_pipelines_agree_on_the_recommended_arm(self):
        """The v5 facade and the v5.1 pipeline must not disagree clinically.

        They previously did, silently: the data contract and the estimators used
        different names for the same arms, so the v5 safety layer treated the
        top-scored arm as infeasible and substituted the runner-up. A naming
        mismatch must never look like a safety decision.
        """
        facade = TreatmentRxOrchestrator().run(sample_ra_bundle())
        pipeline = PrecisionRxAgent().recommend_from_fhir(sample_ra_bundle())

        self.assertEqual(facade.recommended_arm, pipeline.statistical_output["recommended_action"])
        self.assertEqual(facade.status, pipeline.status)

    def test_contract_arm_vocabulary_matches_the_scored_menu(self):
        state = DataLayer().build_patient_state(sample_ra_bundle())
        estimates = EstimationLayer().estimate(state)
        for estimate in estimates:
            self.assertEqual(
                sorted(estimate.q_values),
                sorted(state.feasible_arms),
                msg=f"{estimate.estimator} scored a menu the data contract does not recognise",
            )

    def test_safety_layer_removes_jak_for_pregnancy_before_agent(self):
        bundle = sample_ra_bundle()
        bundle["entry"].append(
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"text": "pregnant"},
                    "valueBoolean": True,
                    "effectiveDay": 365,
                }
            }
        )
        state = DataLayer().build_patient_state(bundle)
        estimates = EstimationLayer().estimate(state)
        decision = DecisionLayer().decide(state, estimates)
        safe = SafetyLayer().apply(decision, state)

        self.assertNotIn("JAK-inhibitor", safe.feasible_arms)
        self.assertIn("JAK-inhibitor", safe.removed_arms)
        self.assertTrue(any(flag.affected_arm == "JAK-inhibitor" for flag in safe.safety_flags))


if __name__ == "__main__":
    unittest.main()
