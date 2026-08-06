import unittest

from precisionrx_agent.demo_data import sample_ra_bundle
from precisionrx_agent.layer1_ingestion.belief import BeliefStateFilter
from precisionrx_agent.layer1_ingestion.competing_risks import CompetingRiskBuilder
from precisionrx_agent.layer1_ingestion.data_engineering import StageHistoryBuilder
from precisionrx_agent.layer1_ingestion.fhir import FHIRAdapter
from precisionrx_agent.layer1_ingestion.leakage import LeakageTestSuite, TemporalFirewall, LeakageError
from precisionrx_agent.layer1_ingestion.switching import SwitchingCapture
from precisionrx_agent.layer1_ingestion.timing import TimingModel
from precisionrx_agent.layer4_estimation.actions import CompositeActionSpace
from precisionrx_agent.layer4_estimation.goal_conditioned import GoalConditionedThresholds
from precisionrx_agent.layer8_safety.feasible_set import FeasibleSet
from precisionrx_agent.layer9_memory_rag.memory import (
    EpisodicItem,
    EpisodicMemory,
    MemoryInfluenceError,
    SemanticKnowledgeBase,
    apply_memory,
)
from precisionrx_agent.layer11_feedback.override_governance import OverrideRouter
from precisionrx_agent.layer11_feedback.validation_ladder import ValidationLadder
from precisionrx_agent.pipeline import PrecisionRxAgent
from precisionrx_agent.shared.models import (
    CareGoal,
    MethodResult,
    Observation,
    OverrideChannel,
    OverrideRecord,
    PatientRecord,
    RegimeType,
    StageRecord,
    ValidationRung,
)


def _stages():
    patient = FHIRAdapter().parse_bundle(sample_ra_bundle())
    stages = StageHistoryBuilder().build(patient)
    return patient, stages


def _method_result(q_values):
    return MethodResult(
        method_name="test",
        regime_type=RegimeType.SPTR,
        recommended_action=max(q_values, key=q_values.get),
        q_values=q_values,
        policy_value=0.8,
        confidence_band=(0.6, 0.9),
        coefficients={"das28": 0.5},
        top_tailoring_variables=["das28"],
    )


class Layer1Tests(unittest.TestCase):
    def test_timing_adds_intervals_and_intensity_weights(self):
        patient, stages = _stages()
        annotated = TimingModel().apply(stages, patient.encounters)
        self.assertTrue(all(s.timing is not None for s in annotated))
        self.assertIsNone(annotated[0].timing.time_since_last_treatment)
        self.assertIsNotNone(annotated[1].timing.time_since_last_treatment)
        self.assertGreater(annotated[-1].timing.inverse_intensity_weight, 0)

    def test_switching_capture_flags_loss_of_response(self):
        patient, stages = _stages()
        annotated = SwitchingCapture().apply(stages, patient)
        # The TNF stage in the demo has a 'loss of response' discontinuation reason.
        self.assertTrue(any(s.switching and s.switching.switched for s in annotated))

    def test_belief_filter_produces_bounded_belief(self):
        _, stages = _stages()
        annotated = BeliefStateFilter().apply(stages)
        belief = annotated[-1].belief
        self.assertIsNotNone(belief)
        self.assertTrue(0.0 <= belief.activity <= 1.0)
        self.assertTrue(0.0 <= belief.uncertainty <= 0.5)

    def test_competing_risk_classifies_events(self):
        patient, stages = _stages()
        stages = SwitchingCapture().apply(stages, patient)
        annotated = CompetingRiskBuilder().apply(stages)
        self.assertTrue(all(s.event is not None for s in annotated))
        incidence = CompetingRiskBuilder().cumulative_incidence(annotated)
        self.assertIsInstance(incidence, dict)

    def test_leakage_firewall_catches_future_feature(self):
        patient, stages = _stages()
        # Inject a feature whose only observation is in the future of the decision.
        tampered_features = dict(stages[0].features)
        tampered_features["future_marker"] = 1.0
        tampered = StageRecord(**(stages[0].__dict__ | {"features": tampered_features}))
        patient2 = PatientRecord(
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
        violations = TemporalFirewall().check(patient2, [tampered])
        self.assertTrue(violations)
        with self.assertRaises(LeakageError):
            TemporalFirewall().assert_clean(patient2, [tampered])

    def test_leakage_suite_passes_on_clean_demo(self):
        patient, stages = _stages()
        report = LeakageTestSuite().run(patient, stages)
        self.assertTrue(report.passed)


class DecisionTests(unittest.TestCase):
    def test_goal_conditioned_threshold_differs_by_goal(self):
        result = _method_result({"a": 0.80, "b": 0.77})  # gap 0.03
        induction = GoalConditionedThresholds().decide(result, CareGoal.INDUCTION)
        qol = GoalConditionedThresholds().decide(result, CareGoal.QUALITY_OF_LIFE)
        self.assertTrue(induction.act)       # 0.03 >= 0.02
        self.assertFalse(qol.act)            # 0.03 < 0.10

    def test_feasible_set_removes_jak_in_pregnancy(self):
        _, stages = _stages()
        pregnant_features = dict(stages[-1].features)
        pregnant_features["pregnant"] = True
        stages[-1] = StageRecord(**(stages[-1].__dict__ | {"features": pregnant_features}))
        candidates = CompositeActionSpace().candidates(["JAK-inhibitor", "IL-6 inhibitor"], stages)
        result = FeasibleSet().filter(candidates, stages, [])
        labels = [a.drug for a in result.feasible]
        self.assertNotIn("upadacitinib", labels)
        self.assertTrue(any("pregnancy" in reason for _, reason in result.removed))


class MemoryTests(unittest.TestCase):
    def test_apply_memory_does_not_touch_statistical_output(self):
        memory = EpisodicMemory()
        kb = SemanticKnowledgeBase()
        bundle = {
            "patient_context": {"patient_id": "h1", "history_summary": "TNF inadequate response"},
            "statistical_output": {
                "recommended_action": "IL-6 inhibitor",
                "q_values": {"IL-6 inhibitor": 0.8},
                "policy_value": 0.8,
                "confidence_band": [0.6, 0.9],
                "safety_status": "recommend",
            },
        }
        before = dict(bundle["statistical_output"])
        out = apply_memory(bundle, memory, kb)
        self.assertEqual(out["statistical_output"], before)
        self.assertIn("memory", out)
        self.assertEqual(out["memory"]["provenance"]["influence"], "narrative_and_retrieval_only")

    def test_apply_memory_raises_if_q_value_mutated(self):
        # Simulate a tamper by monkeypatching kb.retrieve to mutate the bundle.
        memory = EpisodicMemory()
        kb = SemanticKnowledgeBase()
        bundle = {
            "patient_context": {"patient_id": "h1", "history_summary": ""},
            "statistical_output": {
                "recommended_action": "IL-6 inhibitor",
                "q_values": {"IL-6 inhibitor": 0.8},
                "policy_value": 0.8,
                "confidence_band": [0.6, 0.9],
                "safety_status": "recommend",
            },
        }

        original_retrieve = kb.retrieve

        def tampering_retrieve(query, k=2):
            bundle["statistical_output"]["policy_value"] = 0.99
            return original_retrieve(query, k)

        kb.retrieve = tampering_retrieve
        with self.assertRaises(MemoryInfluenceError):
            apply_memory(bundle, memory, kb)

    def test_episodic_memory_is_deterministic_and_resettable(self):
        memory = EpisodicMemory()
        memory.record("h1", EpisodicItem(stage=1, recommended_action="MTX", clinician_action=None,
                                         override_reason=None, outcome_summary="partial", preference="prefers oral"))
        self.assertEqual(memory.preferences("h1"), ["prefers oral"])
        self.assertEqual(memory.recall("h1"), memory.recall("h1"))
        memory.reset("h1")
        self.assertEqual(memory.recall("h1"), [])


class FeedbackTests(unittest.TestCase):
    def test_override_router_channels(self):
        router = OverrideRouter()
        usability = router.route(OverrideRecord("h", "IL-6", "TNF", "card was confusing"))
        safety = router.route(OverrideRecord("h", "JAK", "IL-6", "saw an unsafe interaction"))
        self.assertEqual(usability.channel, OverrideChannel.USABILITY)
        self.assertFalse(usability.influences_model)
        self.assertEqual(safety.channel, OverrideChannel.SAFETY_REVIEW)

    def test_override_influences_model_only_when_outcome_validated(self):
        router = OverrideRouter()
        unvalidated = router.route(OverrideRecord("h", "IL-6", "JAK", "model looks wrong here"))
        validated = router.route(
            OverrideRecord("h", "IL-6", "JAK", "model looks wrong here", outcome_confirmed_clinician=True)
        )
        self.assertFalse(unvalidated.influences_model)
        self.assertTrue(validated.influences_model)

    def test_validation_ladder_gates(self):
        ladder = ValidationLadder()
        blocked = ladder.assess(ValidationRung.SILENT, {"ope_stable": False, "calibration_passed": True})
        passed = ladder.assess(ValidationRung.SILENT, {"ope_stable": True, "calibration_passed": True})
        self.assertFalse(blocked.gate_passed)
        self.assertTrue(passed.gate_passed)
        self.assertEqual(passed.next_rung, ValidationRung.SHADOW)


class IntegrationTests(unittest.TestCase):
    def test_recommendation_carries_v51_outputs(self):
        rec = PrecisionRxAgent().recommend_from_fhir(sample_ra_bundle())
        self.assertIsNotNone(rec.explanation)
        self.assertEqual({e.estimand for e in rec.estimands}, {"ITT", "per_protocol", "as_treated"})
        self.assertIsNotNone(rec.validation)
        self.assertIn("care_goal", rec.audit_event)
        self.assertTrue(rec.audit_event["leakage_passed"])
        self.assertIn("iptw_policy_value", rec.audit_event["ope"])

    def test_memory_continuity_and_override_loop(self):
        agent = PrecisionRxAgent()
        bundle = sample_ra_bundle()
        first = agent.recommend_from_fhir(bundle)
        # First call has no prior context.
        self.assertIn("No prior", first.clinician_rationale + agent.memory.recent_summary(first.patient_hash))

        # Clinician overrides; it routes to a channel and is logged to episodic memory.
        routing = agent.submit_override(first, clinician_action="JAK-inhibitor", reason_text="local protocol prefers JAK")
        self.assertEqual(routing.channel, OverrideChannel.GUIDELINE_CONFLICT)
        self.assertFalse(routing.influences_model)

        # A subsequent recommendation now carries prior continuity in the narrative.
        second = agent.recommend_from_fhir(bundle)
        self.assertNotIn("No prior recorded encounters", second.clinician_rationale)


if __name__ == "__main__":
    unittest.main()
