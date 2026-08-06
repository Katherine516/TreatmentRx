"""Layers 3–6 — decision, safety, agent, feedback.

These tests assert the layer *contracts*: what each layer is allowed to change,
and what it must never change.
"""

import unittest

from treatmentrx.agent import AgentLayer
from treatmentrx.agent.memory import (
    EpisodicItem,
    EpisodicMemory,
    MemoryInfluenceError,
    SemanticKnowledgeBase,
    apply_memory,
)
from treatmentrx.contracts import Decision, PatientState, RegimeEstimate, SafeDecision
from treatmentrx.data import DataLayer
from treatmentrx.decision import DecisionLayer
from treatmentrx.demo_data import sample_ra_bundle
from treatmentrx.domain import (
    CareGoal,
    OverrideChannel,
    OverrideRecord,
    RecommendationStatus,
    RegimeType,
    StageRecord,
    ValidationRung,
)
from treatmentrx.estimation import EstimationLayer
from treatmentrx.estimation.actions import CompositeActionSpace
from treatmentrx.estimation.goal_conditioned import GoalConditionedThresholds
from treatmentrx.feedback import FeedbackLayer
from treatmentrx.feedback.override_governance import OverrideRouter
from treatmentrx.feedback.validation_ladder import ValidationLadder
from treatmentrx.safety import SafetyLayer
from treatmentrx.safety.feasible_set import FeasibleSet


def _estimate(q_values, name="test"):
    return RegimeEstimate(
        estimator=name,
        regime_type=RegimeType.SPTR,
        recommended_arm=max(q_values, key=q_values.get),
        q_values=q_values,
        policy_value=0.73,
        confidence_band=(0.6, 0.9),
        coefficients={"psi:rituximab:das28_std": 0.5},
        top_tailoring_variables=["das28"],
    )


def _pipeline_upto_safety(bundle=None):
    state = DataLayer().build_patient_state(bundle or sample_ra_bundle())
    estimates = EstimationLayer().estimate(state)
    decision = DecisionLayer().decide(state, estimates)
    return state, decision, SafetyLayer().apply(decision, state)


class DecisionLayerTests(unittest.TestCase):
    def test_layer_flow_produces_the_declared_contracts(self):
        state, decision, safe = _pipeline_upto_safety()
        self.assertIsInstance(state, PatientState)
        self.assertIsInstance(decision, Decision)
        self.assertIsInstance(safe, SafeDecision)
        self.assertTrue(all(isinstance(e, RegimeEstimate) for e in decision.estimates))

    def test_model_averaging_weights_every_estimator(self):
        _, decision, _ = _pipeline_upto_safety()
        self.assertEqual(len(decision.model_weights), 3)
        self.assertAlmostEqual(sum(decision.model_weights.values()), 1.0, places=3)

    def test_goal_conditioned_threshold_changes_the_verdict(self):
        estimate = _estimate({"a": 0.80, "b": 0.77})  # gap 0.03
        thresholds = GoalConditionedThresholds()
        self.assertTrue(thresholds.decide(estimate, CareGoal.INDUCTION).act)
        self.assertFalse(thresholds.decide(estimate, CareGoal.QUALITY_OF_LIFE).act)

    def test_uncertainty_reports_calibration_from_held_out_data(self):
        _, decision, _ = _pipeline_upto_safety()
        self.assertTrue(decision.uncertainty.calibrated)
        self.assertNotIn("uncalibrated_model", decision.uncertainty.flags)

    def test_decision_layer_refuses_to_guess_without_estimates(self):
        state = DataLayer().build_patient_state(sample_ra_bundle())
        with self.assertRaises(ValueError):
            DecisionLayer().decide(state, [])


class SafetyLayerTests(unittest.TestCase):
    def test_pregnancy_removes_jak_before_any_narrative(self):
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
        _, _, safe = _pipeline_upto_safety(bundle)
        self.assertNotIn("JAK-inhibitor", safe.feasible_arms)
        self.assertIn("JAK-inhibitor", safe.removed_arms)
        self.assertTrue(any(flag.affected_arm == "JAK-inhibitor" for flag in safe.safety_flags))

    def test_feasible_set_filters_composite_actions_not_labels(self):
        state = DataLayer().build_patient_state(sample_ra_bundle())
        stages = list(state.stages)
        pregnant = dict(stages[-1].features)
        pregnant["pregnant"] = True
        stages[-1] = StageRecord(**(stages[-1].__dict__ | {"features": pregnant}))
        candidates = CompositeActionSpace().candidates(["JAK-inhibitor", "IL-6 inhibitor"], stages)
        result = FeasibleSet().filter(candidates, stages, [])
        self.assertNotIn("upadacitinib", [action.drug for action in result.feasible])
        self.assertTrue(any("pregnancy" in reason for _, reason in result.removed))

    def test_allergy_to_the_recommended_arm_blocks(self):
        recommended = _pipeline_upto_safety()[2].decision.recommended_arm
        bundle = sample_ra_bundle()
        bundle["entry"].append(
            {"resource": {"resourceType": "AllergyIntolerance", "code": {"text": recommended}}}
        )
        _, _, safe = _pipeline_upto_safety(bundle)
        self.assertEqual(safe.status, RecommendationStatus.BLOCKED)
        self.assertTrue(safe.contraindicated)

    def test_an_infeasible_top_arm_is_never_silently_replaced(self):
        """A removed arm routes to review; it does not become a quiet downgrade."""
        state, decision, _ = _pipeline_upto_safety()
        stages = list(state.stages)
        blocked = dict(stages[-1].features)
        blocked["pregnant"] = True
        stages[-1] = StageRecord(**(stages[-1].__dict__ | {"features": blocked}))
        forced = Decision(**(decision.__dict__ | {"recommended_arm": "JAK-inhibitor"}))
        safe = SafetyLayer().apply(forced, PatientState(**(state.__dict__ | {"stages": stages})))

        self.assertEqual(safe.status, RecommendationStatus.BLOCKED)
        self.assertEqual(safe.decision.recommended_arm, "JAK-inhibitor")
        self.assertTrue(
            any(flag.code == "recommended_arm_infeasible" for flag in safe.safety_flags)
        )


class AgentLayerTests(unittest.TestCase):
    def test_memory_never_moves_a_statistical_quantity(self):
        bundle = {
            "patient_context": {"patient_id": "h1", "history_summary": "TNF inadequate response"},
            "statistical_output": {
                "recommended_arm": "IL-6 inhibitor",
                "q_values": {"IL-6 inhibitor": 0.8},
                "policy_value": 0.8,
                "confidence_band": [0.6, 0.9],
                "safety_status": "recommend",
            },
        }
        before = dict(bundle["statistical_output"])
        out = apply_memory(bundle, EpisodicMemory(), SemanticKnowledgeBase())
        self.assertEqual(out["statistical_output"], before)
        self.assertEqual(out["memory"]["provenance"]["influence"], "narrative_and_retrieval_only")

    def test_tampering_with_a_q_value_raises(self):
        memory, kb = EpisodicMemory(), SemanticKnowledgeBase()
        bundle = {
            "patient_context": {"patient_id": "h1", "history_summary": ""},
            "statistical_output": {
                "recommended_arm": "IL-6 inhibitor",
                "q_values": {"IL-6 inhibitor": 0.8},
                "policy_value": 0.8,
                "confidence_band": [0.6, 0.9],
                "safety_status": "recommend",
            },
        }
        original = kb.retrieve

        def tampering(query, k=2):
            bundle["statistical_output"]["policy_value"] = 0.99
            return original(query, k)

        kb.retrieve = tampering
        with self.assertRaises(MemoryInfluenceError):
            apply_memory(bundle, memory, kb)

    def test_episodic_memory_is_deterministic_and_resettable(self):
        memory = EpisodicMemory()
        memory.record(
            "h1",
            EpisodicItem(
                stage=1,
                recommended_arm="MTX",
                clinician_action=None,
                override_reason=None,
                outcome_summary="partial",
                preference="prefers oral",
            ),
        )
        self.assertEqual(memory.preferences("h1"), ["prefers oral"])
        self.assertEqual(memory.recall("h1"), memory.recall("h1"))
        memory.reset("h1")
        self.assertEqual(memory.recall("h1"), [])

    def test_a_blocked_decision_gets_no_treatment_narrative(self):
        recommended = _pipeline_upto_safety()[2].decision.recommended_arm
        bundle = sample_ra_bundle()
        bundle["entry"].append(
            {"resource": {"resourceType": "AllergyIntolerance", "code": {"text": recommended}}}
        )
        _, _, safe = _pipeline_upto_safety(bundle)
        agent = AgentLayer()
        recommendation = agent.run_agents(agent.build_context(safe), safe)
        self.assertEqual(recommendation.status, RecommendationStatus.BLOCKED)
        self.assertIsNone(recommendation.recommended_arm)
        self.assertIn("blocked", recommendation.clinician_card.lower())
        self.assertNotIn(recommended, recommendation.patient_summary)

    def test_clinician_card_renders_the_model_not_prose(self):
        _, _, safe = _pipeline_upto_safety()
        agent = AgentLayer()
        card = agent.run_agents(agent.build_context(safe), safe).clinician_card
        self.assertIn(safe.decision.recommended_arm, card)
        self.assertIn("Why not the alternatives", card)
        self.assertIn("Estimated advantage", card)
        self.assertIn("Uncertainty:", card)


class FeedbackLayerTests(unittest.TestCase):
    def test_override_routing_channels(self):
        router = OverrideRouter()
        usability = router.route(OverrideRecord("h", "IL-6", "TNF", "card was confusing"))
        safety = router.route(OverrideRecord("h", "JAK", "IL-6", "saw an unsafe interaction"))
        self.assertEqual(usability.channel, OverrideChannel.USABILITY)
        self.assertFalse(usability.influences_model)
        self.assertEqual(safety.channel, OverrideChannel.SAFETY_REVIEW)

    def test_only_outcome_validated_overrides_may_inform_the_model(self):
        router = OverrideRouter()
        unvalidated = router.route(OverrideRecord("h", "IL-6", "JAK", "model looks wrong here"))
        validated = router.route(
            OverrideRecord("h", "IL-6", "JAK", "model looks wrong here", outcome_confirmed_clinician=True)
        )
        self.assertFalse(unvalidated.influences_model)
        self.assertTrue(validated.influences_model)

    def test_validation_ladder_gates_each_rung(self):
        ladder = ValidationLadder()
        blocked = ladder.assess(ValidationRung.SILENT, {"ope_stable": False, "calibration_passed": True})
        passed = ladder.assess(ValidationRung.SILENT, {"ope_stable": True, "calibration_passed": True})
        self.assertFalse(blocked.gate_passed)
        self.assertTrue(passed.gate_passed)
        self.assertEqual(passed.next_rung, ValidationRung.SHADOW)

    def test_feedback_reports_all_three_estimands(self):
        state, _, safe = _pipeline_upto_safety()
        agent = AgentLayer()
        recommendation = agent.run_agents(agent.build_context(safe), safe)
        receipt = FeedbackLayer().enqueue(state, recommendation, safe)
        self.assertEqual(
            {result.estimand for result in receipt.estimands},
            {"ITT", "per_protocol", "as_treated"},
        )
        self.assertIn("iptw_policy_value", receipt.ope)

    def test_retraining_is_never_automatically_enabled(self):
        state, _, safe = _pipeline_upto_safety()
        agent = AgentLayer()
        recommendation = agent.run_agents(agent.build_context(safe), safe)
        self.assertFalse(FeedbackLayer().enqueue(state, recommendation, safe).retraining_allowed)


if __name__ == "__main__":
    unittest.main()
