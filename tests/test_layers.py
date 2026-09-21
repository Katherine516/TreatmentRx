"""Layers 3–6 — decision, safety, agent, feedback.

These tests assert the layer *contracts*: what each layer is allowed to change,
and what it must never change.
"""

import unittest
from dataclasses import replace

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
        from treatmentrx.estimation import training

        _, decision, _ = _pipeline_upto_safety()
        self.assertEqual(
            set(decision.model_weights), set(training.SERVING_ENSEMBLE)
        )
        self.assertAlmostEqual(sum(decision.model_weights.values()), 1.0, places=3)

    def test_the_interval_covers_the_same_ensemble_the_decision_used(self):
        """Invariant 16, as a check rather than a convention.

        `DecisionLayer.estimators` and `EstimationLayer.estimators` are separate
        tuples, so they can drift — and if they do, the reported contrast stops
        describing the Q-values it sits next to.
        """
        from treatmentrx.decision import DecisionLayer
        from treatmentrx.estimation import EstimationLayer, training

        serving = set(training.SERVING_ENSEMBLE)
        self.assertEqual(
            {estimator.method_name for estimator in EstimationLayer().estimators}, serving
        )
        self.assertEqual(
            {estimator.method_name for estimator in DecisionLayer().estimators}, serving
        )

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

        # Mirrors `SemanticKnowledgeBase.retrieve`. A double that drifts from
        # the interface it stands in for stops testing the thing it names —
        # this one caught its own drift when `subject` was added.
        def tampering(query, k=2, subject=""):
            bundle["statistical_output"]["policy_value"] = 0.99
            return original(query, k, subject)

        kb.retrieve = tampering
        with self.assertRaises(MemoryInfluenceError):
            apply_memory(bundle, memory, kb)

    def test_mutating_a_q_value_in_place_raises(self):
        """The nested containers are the hole a shallow snapshot leaves open.

        Replacing `statistical_output["policy_value"]` is caught by any snapshot.
        Editing the `q_values` dict the snapshot also points at is not, unless
        the snapshot is deep — and that is the mutation a real memory component
        holding the bundle would actually perform.
        """
        memory, kb = EpisodicMemory(), SemanticKnowledgeBase()
        bundle = {
            "patient_context": {"patient_id": "h1", "history_summary": ""},
            "statistical_output": {
                "recommended_arm": "IL-6 inhibitor",
                "q_values": {"IL-6 inhibitor": 0.8, "TNF-inhibitor": 0.5},
                "policy_value": 0.8,
                "confidence_band": [0.6, 0.9],
                "safety_status": "recommend",
            },
        }

        # Mirrors `SemanticKnowledgeBase.retrieve`. A double that drifts from
        # the interface it stands in for stops testing the thing it names —
        # this one caught its own drift when `subject` was added.
        def tampering(query, k=2, subject=""):
            bundle["statistical_output"]["q_values"]["TNF-inhibitor"] = 0.99
            return []

        kb.retrieve = tampering
        with self.assertRaises(MemoryInfluenceError):
            apply_memory(bundle, memory, kb)

    def test_mutating_the_confidence_band_in_place_raises(self):
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

        # Mirrors `SemanticKnowledgeBase.retrieve`. A double that drifts from
        # the interface it stands in for stops testing the thing it names —
        # this one caught its own drift when `subject` was added.
        def tampering(query, k=2, subject=""):
            bundle["statistical_output"]["confidence_band"][1] = 1.0
            return []

        kb.retrieve = tampering
        with self.assertRaises(MemoryInfluenceError):
            apply_memory(bundle, memory, kb)

    def test_the_tailoring_drivers_are_the_model_s_effect_modifiers(self):
        """The card used to answer this question twice, differently.

        "Tailoring drivers" came from a Layer 1 heuristic that ranked raw
        features by magnitude and skipped booleans; the attribution three lines
        below came from the fitted blip. On the demo patient the first listed
        `egfr, crp, das28, haq_di` and the second credited `anti_ccp` — which the
        first could not have selected at all.
        """
        from treatmentrx.orchestrator import TreatmentRxOrchestrator
        from treatmentrx.estimation.basis import BLIP_BASIS

        recommendation = TreatmentRxOrchestrator().run(sample_ra_bundle())
        drivers = [
            line for line in recommendation.clinician_card.split("\n\n")
            if line.startswith("Tailoring drivers:")
        ][0]
        named = {
            entry.split("=")[0].strip()
            for entry in drivers.removeprefix("Tailoring drivers:").rstrip(".").split(",")
        }
        self.assertTrue(named)
        self.assertTrue(
            named <= set(BLIP_BASIS),
            f"{named - set(BLIP_BASIS)} are not covariates the treatment effect varies over",
        )
        self.assertNotIn("intercept", named, "the intercept is not a tailoring variable")

        attribution = recommendation.explanation.attributions[0]
        self.assertTrue(
            named <= set(attribution.contributions),
            "the card names drivers the attribution does not decompose",
        )

    def test_the_card_never_claims_separation_the_decision_rejected(self):
        """Layer 5 renders Layer 3's verdict; it does not get a second opinion.

        `distinguishable` and `robustly_distinguishable` disagree exactly at the
        boundary where the sandwich is too narrow to trust — which is where a
        clinician most needs the two lines of the card to agree.
        """
        from treatmentrx.orchestrator import TreatmentRxOrchestrator
        from treatmentrx.simulation.fhir_export import simulated_bundles

        orchestrator = TreatmentRxOrchestrator()
        checked = 0
        for bundle in simulated_bundles(60, seed=909):
            recommendation = orchestrator.run(bundle)
            contrast = recommendation.audit_event.get("contrast")
            if contrast is None:
                continue
            checked += 1
            claims_separable = "— separable at this sample size" in recommendation.clinician_card
            self.assertEqual(
                claims_separable,
                contrast["robustly_distinguishable"],
                msg=f"card and contrast verdict disagree: {contrast}",
            )
            if recommendation.status is RecommendationStatus.EQUIPOISE:
                self.assertIn("EQUIPOISE", recommendation.clinician_card)
                self.assertNotIn(
                    f"recommend {recommendation.recommended_arm}.",
                    recommendation.clinician_card,
                )
        self.assertGreater(checked, 0)

    def test_a_renderer_that_tries_to_move_a_number_is_caught(self):
        """The guardrails are tested against a component that wants to break them.

        Everything else in this file exercises a deterministic renderer with no
        incentive to violate the boundary — which tests that the happy path is
        clean, not that the guard works. These cases are what an LLM in this
        position could plausibly do, and each one has to fail loudly.
        """
        attacks = {
            "replace the recommended arm": lambda b: b["statistical_output"].__setitem__(
                "recommended_arm", "JAK-inhibitor"
            ),
            "nudge one q-value": lambda b: b["statistical_output"]["q_values"].__setitem__(
                "TNF-inhibitor", 0.99
            ),
            "widen the band": lambda b: b["statistical_output"]["confidence_band"].__setitem__(
                1, 1.0
            ),
            "inflate the policy value": lambda b: b["statistical_output"].__setitem__(
                "policy_value", 0.99
            ),
            "clear a safety block": lambda b: b["statistical_output"].__setitem__(
                "safety_status", "recommend"
            ),
            "delete the band entirely": lambda b: b["statistical_output"].pop(
                "confidence_band"
            ),
            "add an arm that was never scored": lambda b: b["statistical_output"][
                "q_values"
            ].__setitem__("experimental-agent", 0.99),
        }
        for name, attack in attacks.items():
            with self.subTest(attack=name):
                bundle = {
                    "patient_context": {"patient_id": "h1", "history_summary": ""},
                    "statistical_output": {
                        "recommended_arm": "IL-6 inhibitor",
                        "q_values": {"IL-6 inhibitor": 0.8, "TNF-inhibitor": 0.5},
                        "policy_value": 0.8,
                        "confidence_band": [0.6, 0.9],
                        "safety_status": "blocked",
                    },
                }
                kb = SemanticKnowledgeBase()
                kb.retrieve = (
                    lambda query, k=2, subject="", _a=attack, _b=bundle: (_a(_b) or [])
                )
                with self.assertRaises(MemoryInfluenceError, msg=name):
                    apply_memory(bundle, EpisodicMemory(), kb)

    def test_memory_cannot_reach_the_narrative_around_a_block(self):
        """A blocked case gets no treatment narrative, whatever memory says.

        The block short-circuits before any rationale is composed, so there is no
        text for a memory component to influence — checked rather than assumed,
        because "we return early" is exactly the kind of claim that stops being
        true after a refactor.
        """
        state, _, safe = _pipeline_upto_safety()
        blocked = SafeDecision(
            decision=safe.decision,
            feasible_arms=[],
            feasible_actions=[],
            removed_arms={arm: "unsafe" for arm in safe.decision.q_values},
            safety_flags=[
                __import__("treatmentrx.domain", fromlist=["SafetyFlag"]).SafetyFlag(
                    "empty_feasible_set", "block", "No safe arm remains."
                )
            ],
            status=RecommendationStatus.BLOCKED,
            provenance=safe.provenance,
        )
        agent = AgentLayer()
        agent.memory.record(
            blocked.provenance["patient_hash"],
            EpisodicItem(
                stage=1,
                recommended_arm="JAK-inhibitor",
                clinician_action=None,
                override_reason=None,
                outcome_summary=None,
                preference="strongly prefers JAK-inhibitor",
            ),
        )
        recommendation = agent.run_agents(agent.build_context(blocked), blocked)
        self.assertEqual(recommendation.status, RecommendationStatus.BLOCKED)
        self.assertIsNone(recommendation.recommended_arm)
        card = recommendation.clinician_card
        # The guard is that *memory* cannot reach this card, and it is asserted
        # against the memory sections themselves. It used to be asserted against
        # the arm name memory mentions, which was a proxy: the blocked card now
        # carries the safety layer's removals list, so an arm name in the card no
        # longer distinguishes "memory leaked in" from "safety said why it went".
        # The two memory sections and the preference wording are exact.
        self.assertNotIn("prefers", card)
        self.assertNotIn("Continuity:", card)
        self.assertNotIn("Recorded patient preferences:", card)
        self.assertNotIn("recommend JAK-inhibitor", card)

    def test_the_card_names_no_arm_outside_the_scored_menu(self):
        """The layer renders a menu; it does not get to extend one."""
        from treatmentrx.arms import TREATMENT_ARMS
        from treatmentrx.orchestrator import TreatmentRxOrchestrator
        from treatmentrx.simulation.fhir_export import simulated_bundles

        known = {arm.lower() for arm in TREATMENT_ARMS}
        orchestrator = TreatmentRxOrchestrator()
        for bundle in simulated_bundles(15, seed=909):
            recommendation = orchestrator.run(bundle)
            if recommendation.recommended_arm is None:
                continue
            self.assertIn(recommendation.recommended_arm.lower(), known)
            for entry in recommendation.explanation.why_not:
                self.assertIn(entry.action.lower(), known)

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

    def test_an_unclassifiable_override_goes_to_a_human_not_the_model(self):
        """The unknown case used to default into the one model-influencing channel.

        Confirming the outcome is the second key, but the default direction was
        still backwards: a reason the taxonomy cannot read is the case a safety
        officer should see, not the case that trains the policy.
        """
        router = OverrideRouter()
        for reason in ("patient preference", "insurance denied it", "", "pt declined infusion"):
            with self.subTest(reason=reason):
                routing = router.route(
                    OverrideRecord("h", "IL-6", "JAK", reason, outcome_confirmed_clinician=True)
                )
                self.assertEqual(routing.channel, OverrideChannel.SAFETY_REVIEW)
                self.assertFalse(routing.influences_model)

    def test_a_guideline_conflict_reaches_the_guideline_channel(self):
        """"ui" is a substring of "guideline", "requires", "build" and "quality"."""
        router = OverrideRouter()
        for reason in ("guideline says otherwise", "local protocol requires MTX"):
            with self.subTest(reason=reason):
                routing = router.route(OverrideRecord("h", "IL-6", "JAK", reason))
                self.assertEqual(routing.channel, OverrideChannel.GUIDELINE_CONFLICT)

    def test_validation_ladder_gates_each_rung(self):
        ladder = ValidationLadder()
        satisfied = {
            "ope_effective_sample_size": 80.0,
            "sequential_ope_effective_sample_size": 60.0,
            "ope_improvement_lower": 0.04,
            "calibration_passed": True,
            "live_data": True,
        }
        passed = ladder.assess(ValidationRung.SILENT, satisfied)
        self.assertTrue(passed.gate_passed)
        self.assertEqual(passed.next_rung, ValidationRung.SHADOW)

    def test_every_silent_criterion_can_close_the_gate_on_its_own(self):
        """A gate that only one condition can close is not really four conditions.

        The previous SILENT gate read `effective_sample_size > 0` off a patient's
        own OPE, which is at least 1 for anyone with a visit, so it passed
        universally. Each criterion is checked here in isolation against an
        otherwise-satisfied set.
        """
        ladder = ValidationLadder()
        satisfied = {
            "ope_effective_sample_size": 80.0,
            "sequential_ope_effective_sample_size": 60.0,
            "ope_improvement_lower": 0.04,
            "calibration_passed": True,
            "live_data": True,
        }
        breaks = {
            "ope_effective_sample_size": 5.0,
            # The regime's own effective sample. It has to be able to close the
            # gate by itself, because on the deployed holdout it is the only
            # statistical criterion that actually fails.
            "sequential_ope_effective_sample_size": 5.0,
            "ope_improvement_lower": -0.01,
            "calibration_passed": False,
            "live_data": False,
        }
        for key, bad in breaks.items():
            with self.subTest(criterion=key):
                status = ladder.assess(ValidationRung.SILENT, satisfied | {key: bad})
                self.assertFalse(status.gate_passed, f"{key} did not close the gate")
                self.assertEqual(len(status.blockers), 1)

    def test_a_missing_measurement_is_a_blocker_not_a_pass(self):
        status = ValidationLadder().assess(ValidationRung.SILENT, {})
        self.assertFalse(status.gate_passed)
        self.assertEqual(len(status.blockers), 5)

    def test_this_build_can_never_pass_the_silent_gate(self):
        """`live_data` is structural: a held-out split of the training cohort is not
        live data, however good the numbers on it look."""
        from treatmentrx.estimation import training

        readiness = training.deployment_readiness()
        self.assertFalse(readiness["live_data"])
        status = ValidationLadder().assess(ValidationRung.SILENT, readiness)
        self.assertFalse(status.gate_passed)
        self.assertTrue(any("live data" in blocker for blocker in status.blockers))

    def test_the_silent_gate_reads_the_regimes_own_effective_sample(self):
        """The per-decision sample size is the easier question wearing the hard one's name.

        `ope_effective_sample_size` keeps each stage-row where the policy agreed
        with the arm given and weights by that stage's own propensity — 88 rows,
        effective sample 75.5. What leaves silent mode is a three-stage *regime*,
        whose value needs agreement at every prior decision and the cumulative
        propensity product: 44 rows, effective sample 14.6, three trajectories
        surviving to the end. The gate used to pass on the first number.
        """
        from treatmentrx.estimation import training

        readiness = training.deployment_readiness()
        self.assertLess(
            readiness["sequential_ope_effective_sample_size"],
            readiness["ope_effective_sample_size"],
            "the two effective samples have collapsed into one quantity",
        )
        status = ValidationLadder().assess(ValidationRung.SILENT, readiness)
        self.assertTrue(
            any("not identified" in blocker for blocker in status.blockers),
            status.blockers,
        )

    def test_the_silent_gate_fails_on_statistics_and_not_only_on_live_data(self):
        """Strip the structural blocker and the gate must still close.

        If `live_data` were the only thing holding SILENT shut, flipping it —
        which is what a first real cohort does — would advance a rung on a regime
        whose own value is not identified.
        """
        from treatmentrx.estimation import training

        readiness = dict(training.deployment_readiness())
        readiness["live_data"] = True
        status = ValidationLadder().assess(ValidationRung.SILENT, readiness)
        self.assertFalse(status.gate_passed, status.blockers)
        self.assertFalse(
            any("live data" in blocker for blocker in status.blockers),
            "the structural blocker was supposed to be removed for this check",
        )

    def test_an_unmeasured_safety_criterion_does_not_read_as_satisfied(self):
        """`safety_events` defaulted to 0, which *passes*.

        Fail-open on the one rung whose purpose is catching harm before it
        reaches a patient: with nothing measured, the shadow gate reported no
        safety events rather than no measurement. Same class as the override
        router's unmatched-reason default.
        """
        status = ValidationLadder().assess(ValidationRung.SHADOW, {})
        self.assertFalse(status.gate_passed)
        self.assertTrue(
            any("safety events" in b and "not measured" in b for b in status.blockers),
            status.blockers,
        )

    def test_unmeasured_and_failing_are_worded_differently(self):
        """A reader deciding whether to advance needs to tell them apart.

        "fairness checks not clean" was printed on every assessment ever made,
        and nothing in this codebase computes fairness at all — `cli subgroups`
        stratifies clinical covariates and says in its own docstring that it is
        not a fairness audit.
        """
        ladder = ValidationLadder()
        unmeasured = ladder.assess(ValidationRung.ADVISORY, {})
        failing = ladder.assess(
            ValidationRung.ADVISORY,
            {"clinician_utility": -1.0, "fairness_clean": False, "irb_approved": False},
        )
        self.assertTrue(all("not measured" in b for b in unmeasured.blockers))
        self.assertTrue(all("measured and not met" in b for b in failing.blockers))
        self.assertNotEqual(set(unmeasured.blockers), set(failing.blockers))

    def test_every_upper_rung_criterion_can_pass_when_measured(self):
        """The mirror of the SILENT test: a gate nothing can satisfy is not a gate
        either, and defaulting three ADVISORY criteria to False hid whether they
        were reachable at all."""
        ladder = ValidationLadder()
        for rung, satisfied in (
            (ValidationRung.SHADOW, {"safety_events": 0, "concordance": 0.8}),
            (
                ValidationRung.ADVISORY,
                {"clinician_utility": 0.2, "fairness_clean": True, "irb_approved": True},
            ),
            (ValidationRung.PRAGMATIC_TRIAL, {"trial_endpoint_met": True}),
        ):
            with self.subTest(rung=rung):
                status = ladder.assess(rung, satisfied)
                self.assertTrue(status.gate_passed, status.blockers)

    def test_feedback_reports_all_three_estimands(self):
        state, _, safe = _pipeline_upto_safety()
        agent = AgentLayer()
        recommendation = agent.run_agents(agent.build_context(safe), safe)
        receipt = FeedbackLayer().enqueue(state, recommendation, safe)
        self.assertEqual(
            {result.estimand for result in receipt.estimands},
            {"ITT", "per_protocol", "as_treated"},
        )
        self.assertIn("observed_mean_outcome", receipt.ope)

    def test_full_system_track_represents_abstention_as_usual_care(self):
        state, _, safe = _pipeline_upto_safety()
        agent = AgentLayer()
        recommendation = agent.run_agents(agent.build_context(safe), safe)
        abstention = replace(
            recommendation,
            status=RecommendationStatus.EQUIPOISE,
            recommended_arm=None,
        )
        feedback = FeedbackLayer()
        receipt = feedback.enqueue(state, abstention, safe)
        self.assertFalse(receipt.ope_track_enqueued)
        self.assertTrue(receipt.full_system_track_enqueued)
        self.assertEqual(
            feedback.full_system_track[-1]["policy_action"],
            "clinician-usual-care",
        )
        self.assertNotEqual(
            feedback.full_system_track[-1]["policy_action"],
            abstention.top_scored_arm,
        )

    def test_retraining_is_never_automatically_enabled(self):
        state, _, safe = _pipeline_upto_safety()
        agent = AgentLayer()
        recommendation = agent.run_agents(agent.build_context(safe), safe)
        self.assertFalse(FeedbackLayer().enqueue(state, recommendation, safe).retraining_allowed)



class ClampedRankingTests(unittest.TestCase):
    """The leader must not be chosen by a display quantity.

    `q_values` is clamped to `[Q_FLOOR, Q_CEILING]` and rounded to 3dp before
    anything downstream sees it, and both steps are many-to-one. A patient whose
    predicted response saturates the ceiling had two arms collapse to 0.99, the
    argmax fell through to dictionary order, and the decision then reported a
    *negative* contrast for its own top pair.
    """

    @classmethod
    def setUpClass(cls):
        from treatmentrx.data.contract import DataContractError
        from treatmentrx.orchestrator import TreatmentRxOrchestrator
        from treatmentrx.simulation.fhir_export import simulated_bundles

        orchestrator = TreatmentRxOrchestrator()
        cls.rows = []
        for bundle in simulated_bundles(60, seed=991):
            try:
                recommendation = orchestrator.run(bundle)
            except DataContractError:
                continue
            contrast = recommendation.audit_event.get("contrast") or {}
            if "difference" in contrast:
                cls.rows.append((recommendation, contrast))

    def test_the_contrast_describes_the_arm_the_decision_named(self):
        """One leader, chosen once. It used to be chosen twice, two ways."""
        for recommendation, contrast in self.rows:
            top = recommendation.top_scored_arm
            if top is None:
                continue
            with self.subTest(patient=recommendation.audit_event.get("patient_hash")):
                self.assertEqual(contrast["arm"], top)

    def test_a_saturated_patient_does_not_get_a_backwards_contrast(self):
        """The defect's signature: the interval says the runner-up is better.

        One residual case is expected and is a different thing — a genuine tie
        at mid-range where the two members disagree and the weighted vote breaks
        it. That one is honest; a clamp artefact is not.
        """
        backwards = [c for _, c in self.rows if c["difference"] < 0]
        self.assertLessEqual(
            len(backwards),
            1,
            f"{len(backwards)} patients have a contrast pointing away from their leader",
        )

    def test_the_estimators_rank_on_unclamped_values(self):
        """The information was never lost, only discarded at the facade."""
        from treatmentrx.estimation import training
        from treatmentrx.estimation.q_learning import Q_CEILING

        fit = training.fitted()
        features = {
            "das28": 2.0, "crp": 4.0, "anti_ccp": 1.0,
            "prior_tnf": 0.0, "egfr": 110.0, "alt": 12.0,
        }
        terminal = fit.pooled.n_stages - 1
        raw = {
            arm: fit.pooled.raw_q(features, arm, terminal) for arm in fit.pooled.arms
        }
        clamped = fit.pooled.q_values(features, terminal)
        self.assertEqual(
            fit.pooled.recommend(features, terminal), max(raw, key=raw.get)
        )
        self.assertTrue(
            all(value <= Q_CEILING for value in clamped.values()),
            "q_values is the display quantity and must stay inside the band",
        )

    def test_a_tie_is_broken_by_the_members_not_by_dict_order(self):
        """Deterministic, and using information rather than insertion order."""
        from treatmentrx.decision.bma import BayesianModelAverager
        from treatmentrx.contracts import RegimeEstimate
        from treatmentrx.domain import RegimeType

        def estimate(name, recommended):
            return RegimeEstimate(
                estimator=name,
                regime_type=RegimeType.DTR,
                recommended_arm=recommended,
                q_values={"zeta-arm": 0.99, "alpha-arm": 0.99},
                policy_value=0.7,
                confidence_band=(0.6, 0.8),
                coefficients={},
                top_tailoring_variables=[],
                estimand_fingerprint="same",
            )

        averaged = BayesianModelAverager().aggregate(
            [estimate("Q-Pooled", "zeta-arm"), estimate("dWOLS-Shared", "zeta-arm")]
        )
        self.assertEqual(
            averaged.recommended_arm,
            "zeta-arm",
            "a unanimous member vote must beat alphabetical order",
        )


class KnowledgeRetrievalTests(unittest.TestCase):
    """The keyword index has to match keywords, and cite the arm it is about.

    Two defects. Whitespace tokenisation meant `TNF-inhibitor` never matched the
    key `tnf inadequate response`, so four of six arms retrieved nothing for
    their own name. And ties scored one point each and fell back to
    `_KNOWLEDGE_BASE` order, so even a matching arm lost to history tokens: the
    demo patient is recommended rituximab, the knowledge base has a rituximab
    passage, and the card cited TNF and methotrexate instead.
    """

    def test_every_arm_with_a_passage_retrieves_it(self):
        from treatmentrx.agent.memory import SemanticKnowledgeBase
        from treatmentrx.arms import TREATMENT_ARMS

        knowledge = SemanticKnowledgeBase()
        # `continue-current` has no passage; the rest do, and each must find it.
        covered = [arm for arm in TREATMENT_ARMS if arm != "continue-current"]
        for arm in covered:
            with self.subTest(arm=arm):
                self.assertTrue(
                    knowledge.retrieve(arm, subject=arm),
                    f"{arm} retrieves nothing from its own name",
                )

    def test_the_hyphen_is_what_used_to_break_it(self):
        """Pins the mechanism, so a future tokeniser change fails loudly."""
        from treatmentrx.agent.memory import SemanticKnowledgeBase, _words

        self.assertEqual(_words("TNF-inhibitor"), {"tnf", "inhibitor"})
        self.assertNotEqual(set("TNF-inhibitor".lower().split()), {"tnf", "inhibitor"})
        knowledge = SemanticKnowledgeBase()
        passages = knowledge.retrieve("TNF-inhibitor", subject="TNF-inhibitor")
        self.assertIn("TNF", passages[0])

    def test_the_subject_outranks_the_surrounding_history(self):
        """Evidence printed under a recommendation should be about that arm."""
        from treatmentrx.agent.memory import SemanticKnowledgeBase

        knowledge = SemanticKnowledgeBase()
        query = "rituximab prior methotrexate then tnf inadequate response"
        passages = knowledge.retrieve(query, subject="rituximab")
        self.assertIn("rituximab", passages[0].lower())

    def test_the_served_recommendation_cites_its_own_arm(self):
        from treatmentrx.demo_data import sample_ra_bundle
        from treatmentrx.orchestrator import TreatmentRxOrchestrator

        recommendation = TreatmentRxOrchestrator().run(sample_ra_bundle())
        arm = recommendation.top_scored_arm
        self.assertTrue(recommendation.evidence)
        self.assertIn(
            arm.split("-")[0].lower(),
            recommendation.evidence[0].text.lower(),
            f"first citation is not about {arm}",
        )

    def test_retrieval_stays_deterministic(self):
        """Ties break by knowledge-base order, which is arbitrary but auditable."""
        from treatmentrx.agent.memory import SemanticKnowledgeBase

        knowledge = SemanticKnowledgeBase()
        first = knowledge.retrieve("methotrexate tnf", subject="")
        for _ in range(5):
            self.assertEqual(knowledge.retrieve("methotrexate tnf", subject=""), first)


class CareGoalGapTests(unittest.TestCase):
    """The care-goal bar must judge the difference the interval brackets.

    `GoalThresholds.decide` derived its gap from `q_values`, which is clamped to
    `[Q_FLOOR, Q_CEILING]` and rounded to 3dp. Measured over 120 patients, every
    one of the nine whose response saturated the ceiling had a top-two gap of
    exactly 0.0000 — arms the model distinguishes, reported to the clinician as
    identical and failing the action bar for an arithmetic reason. That is the
    fourth place the leader was being re-derived from a display quantity.
    """

    def test_the_clamp_no_longer_collapses_the_gap(self):
        from treatmentrx.data import DataLayer
        from treatmentrx.data.contract import DataContractError
        from treatmentrx.orchestrator import TreatmentRxOrchestrator
        from treatmentrx.simulation.fhir_export import simulated_bundles

        orchestrator = TreatmentRxOrchestrator()
        zero_gaps = scored = 0
        for bundle in simulated_bundles(60, seed=991):
            try:
                recommendation = orchestrator.run(bundle)
            except DataContractError:
                continue
            gap = recommendation.audit_event.get("confidence_gap")
            if gap is None:
                continue
            scored += 1
            zero_gaps += gap == 0.0
        self.assertGreater(scored, 10)
        self.assertEqual(
            zero_gaps, 0, "a clamped q_value is still collapsing the care-goal gap"
        )

    def test_the_card_reports_the_same_gap(self):
        """`confidence_gap` is what the clinician card prints as the Q-gap."""
        from treatmentrx.data import DataLayer
        from treatmentrx.decision import DecisionLayer
        from treatmentrx.demo_data import sample_ra_bundle
        from treatmentrx.estimation import EstimationLayer

        state = DataLayer().build_patient_state(sample_ra_bundle())
        decision = DecisionLayer().decide(state, EstimationLayer().estimate(state))
        self.assertAlmostEqual(
            decision.confidence_gap, decision.goal_decision.observed_gap, places=6
        )

    def test_the_bar_and_the_interval_read_the_same_number(self):
        """Invariant 10's two conditions are only comparable if they agree."""
        from treatmentrx.data import DataLayer
        from treatmentrx.decision import DecisionLayer
        from treatmentrx.demo_data import sample_ra_bundle
        from treatmentrx.estimation import EstimationLayer

        state = DataLayer().build_patient_state(sample_ra_bundle())
        decision = DecisionLayer().decide(state, EstimationLayer().estimate(state))
        self.assertIsNotNone(decision.contrast)
        self.assertAlmostEqual(
            decision.goal_decision.observed_gap,
            round(decision.contrast.difference, 4),
            places=4,
        )

    def test_the_gap_is_signed_not_absolute(self):
        """A leader scoring below its comparator must fail the bar, not clear it.

        The contrast can be negative in the one honest case where the two serving
        members disagree and the weighted vote picks the leader. Taking the
        magnitude would let that clear an action bar and rely on the interval
        condition to catch it downstream.
        """
        from treatmentrx.domain import CareGoal
        from treatmentrx.estimation.goal_conditioned import GoalConditionedThresholds

        goal = GoalConditionedThresholds()
        estimate = _regime_estimate_with_q_values({"a": 0.8, "b": 0.5})
        negative = goal.decide(estimate, CareGoal.INDUCTION, observed_gap=-0.30)
        self.assertLess(negative.observed_gap, 0.0)
        self.assertFalse(negative.act)

    def test_a_caller_without_a_contrast_still_works(self):
        """`cli misspecification` and the subgroup sweep construct estimates directly."""
        from treatmentrx.domain import CareGoal
        from treatmentrx.estimation.goal_conditioned import GoalConditionedThresholds

        estimate = _regime_estimate_with_q_values({"a": 0.8, "b": 0.5})
        fallback = GoalConditionedThresholds().decide(estimate, CareGoal.INDUCTION)
        self.assertAlmostEqual(fallback.observed_gap, 0.3, places=4)


def _regime_estimate_with_q_values(q_values):
    from treatmentrx.contracts import RegimeEstimate
    from treatmentrx.domain import RegimeType

    return RegimeEstimate(
        estimator="test",
        regime_type=RegimeType.DTR,
        recommended_arm=max(q_values, key=q_values.get),
        q_values=q_values,
        policy_value=0.7,
        confidence_band=(0.6, 0.8),
        coefficients={},
        top_tailoring_variables=[],
    )

if __name__ == "__main__":
    unittest.main()


class WhyNotScaleTests(unittest.TestCase):
    """The why-not gap is the decision's contrast, not a difference of Q-values.

    `q_values` is clamped to `[Q_FLOOR, Q_CEILING]` and rounded to 3dp, and both
    steps are many-to-one. For a patient whose predicted response saturates the
    ceiling three arms collapse to 0.99, and the card printed "TNF-inhibitor
    (gap 0.000)" one line under "Separation: methotrexate-optimization over
    TNF-inhibitor is +0.051 — separable at this sample size". Measured over 120
    patients: 4 printed exactly 0.000 under a separable verdict, and 29
    disagreed with the separation line by any amount, by up to 0.065. That is
    invariant 46's defect in a fifth place, on a served field.
    """

    @classmethod
    def setUpClass(cls):
        from treatmentrx.data.contract import DataContractError
        from treatmentrx.estimation import EstimationLayer
        from treatmentrx.simulation.fhir_export import simulated_bundles

        data, estimation, decision = DataLayer(), EstimationLayer(), DecisionLayer()
        cls.decisions = []
        for bundle in simulated_bundles(60, seed=4242):
            try:
                state = data.build_patient_state(bundle)
            except DataContractError:
                continue
            cls.decisions.append(decision.decide(state, estimation.estimate(state)))

    def test_the_gap_is_the_interval_the_separation_line_reports(self):
        """The runner-up's two numbers now agree by construction, not by luck."""
        checked = 0
        for decision in self.decisions:
            if decision.contrast is None or decision.explanation is None:
                continue
            for entry in decision.explanation.why_not:
                if entry.action != decision.contrast.comparator:
                    continue
                checked += 1
                with self.subTest(arm=entry.action):
                    # `q_gap` carries three decimals; that is the only slack.
                    self.assertAlmostEqual(
                        entry.q_gap, decision.contrast.difference, places=2
                    )
        self.assertGreater(checked, 10)

    def test_a_saturated_patient_does_not_get_a_zero_gap(self):
        """The mechanism, pinned where it bit.

        A separable verdict beside a gap of 0.000 is the contradiction; assert
        that no arm the decision could rule out is reported as tied with the
        leader.
        """
        from treatmentrx.estimation.q_learning import Q_CEILING, Q_FLOOR

        saturated = collapsed = 0
        for decision in self.decisions:
            if decision.explanation is None:
                continue
            if not any(
                value >= Q_CEILING or value <= Q_FLOOR
                for value in decision.q_values.values()
            ):
                continue
            saturated += 1
            for entry in decision.explanation.why_not:
                if entry.action in decision.candidate_arms:
                    continue  # genuinely not separable; a small gap is honest
                collapsed += entry.q_gap == 0.0
        self.assertGreater(saturated, 0, "no patient in this cohort hit the clamp")
        self.assertEqual(collapsed, 0, "a clamped q_value is still collapsing a why-not gap")

    def test_the_entries_are_ordered_by_the_gap_they_print(self):
        """The renderer shows the first two, so the order has to be the printed
        number rather than the display value it replaced."""
        for decision in self.decisions:
            if decision.explanation is None:
                continue
            gaps = [entry.q_gap for entry in decision.explanation.why_not]
            with self.subTest(arms=decision.recommended_arm):
                self.assertEqual(gaps, sorted(gaps))


class ComparatorSelectionTests(unittest.TestCase):
    """The arm the recommendation is justified against is chosen by the data.

    It came from `_ranked()[1]` — the argmax over the *clamped, rounded*
    non-leader `q_values` — so for a patient whose response saturates the ceiling
    three arms sit at 0.99 and dictionary order picked it. Measured over 120
    patients, 5 had a tied comparator and 4 of those were recommended on a
    separation (+0.051) that did not hold for the genuinely closest arm (+0.018).
    That was the last place invariant 46's display quantity reached a clinical
    output.
    """

    @classmethod
    def setUpClass(cls):
        from treatmentrx.data.contract import DataContractError
        from treatmentrx.estimation import EstimationLayer
        from treatmentrx.simulation.fhir_export import simulated_bundles

        data, estimation, decision = DataLayer(), EstimationLayer(), DecisionLayer()
        cls.decisions = []
        for bundle in simulated_bundles(60, seed=4242):
            try:
                state = data.build_patient_state(bundle)
            except DataContractError:
                continue
            cls.decisions.append(decision.decide(state, estimation.estimate(state)))

    def test_the_comparator_is_the_closest_arm_the_model_estimated(self):
        for decision in self.decisions:
            if decision.contrast is None or not decision.candidate_contrasts:
                continue
            closest = min(
                decision.candidate_contrasts.values(),
                key=lambda test: (test.difference, test.comparator),
            )
            with self.subTest(patient=decision.recommended_arm):
                self.assertEqual(decision.contrast.comparator, closest.comparator)
                self.assertAlmostEqual(
                    decision.contrast.difference, closest.difference, places=9
                )

    def test_a_clamped_tie_no_longer_chooses_it(self):
        """The mechanism. Where arms are tied in the display dict the comparator
        must still be the one the unclamped estimate says is nearest."""
        from treatmentrx.estimation.q_learning import Q_CEILING, Q_FLOOR

        saturated = 0
        for decision in self.decisions:
            if decision.contrast is None:
                continue
            tied = [
                arm for arm, value in decision.q_values.items()
                if arm != decision.recommended_arm
                and (value >= Q_CEILING or value <= Q_FLOOR)
            ]
            if len(tied) < 2:
                continue
            saturated += 1
            others = [
                test.difference for arm, test in decision.candidate_contrasts.items()
                if arm != decision.contrast.comparator
            ]
            with self.subTest(arms=tuple(tied)):
                self.assertTrue(
                    all(decision.contrast.difference <= other for other in others),
                    "a clamped tie is still supplying the comparator",
                )
        self.assertGreater(saturated, 0, "no patient in this cohort hit the clamp")

    def test_the_contrast_is_the_candidate_sets_own_interval(self):
        """One interval per pair, built once. Computing the runner-up's twice was
        how the separation line and the set could report different verdicts about
        the same arms."""
        for decision in self.decisions:
            if decision.contrast is None:
                continue
            with self.subTest(arm=decision.recommended_arm):
                self.assertIn(decision.contrast.comparator, decision.candidate_contrasts)
                self.assertIs(
                    decision.contrast,
                    decision.candidate_contrasts[decision.contrast.comparator],
                )

    def test_the_nearest_arm_is_not_always_the_least_separable(self):
        """Why this does *not* make RECOMMEND mean "separated from every arm".

        The smallest difference is not the smallest z, so an arm further away but
        less precisely estimated can still be the one the data cannot exclude.
        That is what `_not_excluded` reports, and pinning it here stops a future
        reader from collapsing the two rules together.
        """
        divergent = 0
        for decision in self.decisions:
            if decision.contrast is None or len(decision.candidate_contrasts) < 2:
                continue
            least_separable = min(
                decision.candidate_contrasts.values(),
                key=lambda test: (
                    abs(test.difference) / test.standard_error
                    if test.standard_error
                    else float("inf")
                ),
            )
            divergent += least_separable.comparator != decision.contrast.comparator
        self.assertGreater(
            divergent,
            0,
            "if these never diverge the two rules are the same rule and one should go",
        )


class CardMatchesItsStatusTests(unittest.TestCase):
    """Invariant 35, checked on the two blocks that were still missing it."""

    def _flagged_basis(self):
        """A real pipeline decision with the blip-basis flag forced on.

        The specification test does not fire on this build —
        `deployment_readiness()["blip_basis_unflagged"]` is True and
        `das28_squared` reaches 3.367 against a 3.669 threshold — so the flag has
        to be injected to render the branch at all.
        """
        from treatmentrx.estimation import EstimationLayer
        from treatmentrx.safety import SafetyLayer

        state = DataLayer().build_patient_state(sample_ra_bundle())
        decision = DecisionLayer().decide(state, EstimationLayer().estimate(state))
        safe = SafetyLayer().apply(decision, state)
        uncertainty = replace(
            decision.uncertainty,
            flags=list(decision.uncertainty.flags) + ["blip_basis_may_omit:crp_std"],
        )
        decision = replace(decision, uncertainty=uncertainty)
        return replace(safe, decision=decision)

    def test_the_blocked_card_carries_the_caveat_with_the_interval(self):
        """A caveat travels with the line it undercuts.

        `blocked_card` reports the separation interval — invariant 35 put it
        there, because the one status that escalates to a person was handing that
        person the least — but not the warning that the interval may be centred
        on the wrong quantity. The reviewer got the number without its qualifier.
        """
        from treatmentrx.agent.rationale import RationaleGenerator

        safe = self._flagged_basis()
        rationale = RationaleGenerator()
        blocked = rationale.blocked_card(safe, "blocked for the purposes of this test")
        self.assertIn("Separation:", blocked)
        self.assertIn("CAVEAT", blocked)
        self.assertIn("crp_std", blocked)

    def test_both_cards_caveat_the_same_interval(self):
        """The clinician card already did; the two must not diverge."""
        from treatmentrx.agent.rationale import RationaleGenerator

        safe = self._flagged_basis()
        rationale = RationaleGenerator()
        context = AgentLayer().build_context(safe)
        clinician = rationale.clinician_card(context, safe, "safety text", "evidence")
        blocked = rationale.blocked_card(safe, "safety text")
        for card in (clinician, blocked):
            self.assertIn("CAVEAT", card)

    def test_a_recommendation_always_clears_its_own_action_bar(self):
        """Why `patient_summary`'s hedge was removed rather than rewired.

        It appended " ...because the options are close" when
        `goal_decision.act` was False, and `DecisionLayer._status` returns
        RECOMMEND only *after* `act` is True — so the branch could not fire, on
        any patient, ever. Measured 41/41 on the audit cohort.

        The patients it was aimed at are real and are elsewhere: those who clear
        the care-goal bar but fail the interval condition are sent to the
        EQUIPOISE summary, which says the options are close in its first
        sentence. A hedge that can only fire where it is wrong is invariant 25 in
        patient-facing prose.
        """
        from treatmentrx.data.contract import DataContractError
        from treatmentrx.estimation import EstimationLayer
        from treatmentrx.simulation.fhir_export import simulated_bundles

        data, estimation, decision = DataLayer(), EstimationLayer(), DecisionLayer()
        recommended = close_but_declined = 0
        for bundle in simulated_bundles(60, seed=4242):
            try:
                state = data.build_patient_state(bundle)
            except DataContractError:
                continue
            made = decision.decide(state, estimation.estimate(state))
            if made.status is RecommendationStatus.RECOMMEND:
                recommended += 1
                self.assertTrue(
                    made.goal_decision.act,
                    "a recommendation that failed its own action bar would make "
                    "the removed hedge reachable again",
                )
            elif made.goal_decision.act:
                close_but_declined += 1
        self.assertGreater(recommended, 10)
        self.assertGreater(
            close_but_declined,
            0,
            "the patients the hedge was written for should still exist, and be declined",
        )


class AttributionSourceTests(unittest.TestCase):
    """The blip decomposition is the ensemble's, on the same weights as q_values.

    It used to be one member's, chosen by a **0.003** weight margin (0.4985 /
    0.5015) while the two members' blips for the same patient differed by up to
    **0.041** — the size of the contrast the decision reports. psi could not be
    averaged then, because dWOLS's is a single-visit blip and `Q-Pooled` published
    an undivided value-to-go stage psi. `coefficient_summary` puts both on the
    per-remaining-visit scale now, so the average is well defined — and it is the
    right quantity rather than an available one, because `_pair_contrast` builds
    the gap the card prints as exactly this weighted mean of the members'
    contrasts.
    """

    @classmethod
    def setUpClass(cls):
        from treatmentrx.orchestrator import TreatmentRxOrchestrator
        from treatmentrx.simulation.fhir_export import simulated_bundles

        from treatmentrx.data import DataLayer
        from treatmentrx.decision import DecisionLayer
        from treatmentrx.estimation import EstimationLayer

        orchestrator = TreatmentRxOrchestrator()
        cls.bundles = simulated_bundles(12, seed=4242)
        cls.recommendations = [orchestrator.run(bundle) for bundle in cls.bundles]
        data, estimation, decision = DataLayer(), EstimationLayer(), DecisionLayer()
        cls.decisions = {}
        for bundle in cls.bundles:
            state = data.build_patient_state(bundle)
            cls.decisions[state.patient_hash] = decision.decide(
                state, estimation.estimate(state)
            )

    def test_the_source_is_recorded_and_is_the_ensemble(self):
        """Not a member chosen by a hair's-breadth weight margin."""
        from treatmentrx.decision.bma import BMA_ENSEMBLE
        from treatmentrx.estimation import training

        for recommendation in self.recommendations:
            source = recommendation.audit_event["attribution_source"]
            with self.subTest(patient=recommendation.patient_hash):
                self.assertEqual(source, BMA_ENSEMBLE)
                self.assertNotIn(source, training.SERVING_ENSEMBLE)

    def test_the_published_blip_is_the_weighted_mean_of_the_members(self):
        """The averaging itself, read off the coefficients both card blocks use.

        Checked per basis term rather than through a dot product, so a blend that
        happened to agree on one patient's features would still fail.
        """
        for recommendation in self.recommendations:
            event = recommendation.audit_event
            weights = event["model_weights"]
            selected = recommendation.explanation.attributions[0].action
            estimate = self.decisions[recommendation.patient_hash]
            prefix = f"psi:{selected}:"
            members = {
                e.estimator: {
                    k[len(prefix):]: v for k, v in e.coefficients.items() if k.startswith(prefix)
                }
                for e in estimate.estimates
            }
            averaged = {
                k[len(prefix):]: v
                for k, v in estimate.selected.coefficients.items()
                if k.startswith(prefix)
            }
            self.assertTrue(averaged)
            for term, value in averaged.items():
                present = {n: w for n, w in weights.items() if term in members.get(n, {})}
                total = sum(present.values())
                expected = sum(w / total * members[n][term] for n, w in present.items())
                with self.subTest(patient=recommendation.patient_hash, term=term):
                    self.assertAlmostEqual(value, expected, places=3)

    def test_the_card_says_the_decomposition_is_the_ensembles(self):
        """It used to read "(dWOLS-Shared's blip, not the ensemble average)".
        Now that the blip *is* the ensemble average, that sentence would be
        false, and dropping it silently would leave a reader guessing which of
        the two models named on the line above they are looking at."""
        shown = 0
        for recommendation in self.recommendations:
            card = recommendation.clinician_card
            if "Estimated advantage of" not in card:
                continue
            shown += 1
            line = card.split("Estimated advantage of", 1)[1].split("\n\n", 1)[0]
            with self.subTest(patient=recommendation.patient_hash):
                self.assertIn("weighted average of the estimators", line)
                self.assertNotIn("not the ensemble average", line)
        self.assertGreater(shown, 5)

    def test_the_decomposition_matches_the_model_it_names(self):
        """The faithfulness check the audit's arithmetic identity stood in for.

        Recomputed from the fitted model rather than read back out of the
        coefficients the estimate carries, and it discriminates: against the
        named member the residual is the 4dp rounding floor, against the other
        serving member it is 261x larger.
        """
        from treatmentrx.estimation import training
        from treatmentrx.feedback.audit import _attribution_against_its_model

        named = other = 0.0
        checked = 0
        for bundle, recommendation in zip(self.bundles, self.recommendations):
            source, residual = _attribution_against_its_model(bundle, recommendation)
            if source is None:
                continue
            checked += 1
            named = max(named, residual)
            swapped = dict(recommendation.audit_event)
            swapped["attribution_source"] = (
                training.Q_POOLED if source == training.DWOLS_SHARED else training.DWOLS_SHARED
            )
            _, wrong = _attribution_against_its_model(
                bundle, replace(recommendation, audit_event=swapped)
            )
            other = max(other, wrong)
        self.assertGreater(checked, 5)
        self.assertLess(named, 1e-3, "the attribution does not match the model it names")
        self.assertGreater(
            other,
            1e-2,
            "the check does not discriminate between the two members, so it cannot fail",
        )


class CounterfactualGapTests(unittest.TestCase):
    """The sixth place a display quantity reached a served field.

    `q_values` are clamped to `[Q_FLOOR, Q_CEILING]` and rounded to 3dp.
    Invariants 46, 53, 54 and 55 removed a ranking re-derived from them in five
    places; `_counterfactuals` was the sixth, and it sat in the same function
    signature as two that were fixed — `_sensitivity` takes `contrast`,
    `_why_not` takes `arm_contrasts`, and this one sorted the dict.

    It decides `recommendation_changes` on `gap < 0.05`, so the clamp reached a
    boolean rather than a rounding. Measured over 120 patients: 6 had a display
    gap of exactly 0.0000 while the model distinguished the arms, the two gaps
    differ by up to 0.0648 — more than the threshold — and 2 patients got a
    different verdict.
    """

    THRESHOLD = 0.05

    @classmethod
    def setUpClass(cls):
        from treatmentrx.orchestrator import TreatmentRxOrchestrator
        from treatmentrx.simulation.fhir_export import trajectory_to_bundle
        from treatmentrx.simulation.ra_cohort import generate_ra_cohort

        orchestrator = TreatmentRxOrchestrator()
        cls.scored = []
        for trajectory in generate_ra_cohort(60, seed=4242):
            recommendation = orchestrator.run(trajectory_to_bundle(trajectory))
            contrast = recommendation.audit_event.get("contrast")
            if contrast is None:
                continue
            probes = {
                probe.covariate: probe
                for probe in recommendation.explanation.counterfactuals
            }
            cls.scored.append((recommendation, contrast, probes))

    def test_the_sweep_reaches_the_probe(self):
        self.assertGreater(len(self.scored), 20)
        self.assertTrue(any("crp" in probes for _, _, probes in self.scored))

    def test_the_probe_reads_the_decisions_own_gap(self):
        """Asserted against the contrast, not against the sorted dict — the two
        disagree by up to 0.0648 and the threshold is 0.05."""
        for recommendation, contrast, probes in self.scored:
            probe = probes.get("crp")
            if probe is None:
                continue
            crp = recommendation.audit_event.get("features", {}).get("crp")
            if not isinstance(crp, (int, float)):
                continue
            expected = float(crp) > 20 and abs(contrast["difference"]) < self.THRESHOLD
            with self.subTest(patient=recommendation.patient_hash):
                self.assertEqual(probe.recommendation_changes, expected)

    def test_the_two_gaps_genuinely_disagree_on_this_fixture(self):
        """Otherwise the test above passes by there being nothing to get wrong.

        The clamp does not only zero a gap — that is the case invariant 53
        measured, where *both* top arms saturate. When only the leader does, the
        display gap is **compressed** rather than zeroed, and compression alone
        crosses a 0.05 threshold. Measured over 120 patients, the two disagreeing
        cases both have the leader at exactly 0.99, the ceiling, with the
        runner-up at 0.989 and 0.980 — display gaps of 0.0010 and 0.0100 against
        contrasts of 0.0658 and 0.0715, a factor of seven on the *same* pair of
        arms.
        """
        disagreements = []
        for recommendation, contrast, _ in self.scored:
            ranked = sorted(recommendation.q_values.values(), reverse=True)
            display_gap = ranked[0] - ranked[1]
            if (display_gap < self.THRESHOLD) != (
                abs(contrast["difference"]) < self.THRESHOLD
            ):
                disagreements.append((display_gap, abs(contrast["difference"])))
        self.assertTrue(
            disagreements,
            "no patient here distinguishes the display gap from the contrast, so "
            "the probe would read the same either way",
        )
        for display_gap, real in disagreements:
            with self.subTest(display_gap=round(display_gap, 4)):
                self.assertLess(display_gap, real, "the clamp compresses, never widens")


class ExplanationAuditTests(unittest.TestCase):
    """Layer 5's audit read 1.0 / 0.0 / 0 and two of those could not do otherwise."""

    @classmethod
    def setUpClass(cls):
        from treatmentrx.feedback.audit import audit_explanation

        cls.section = audit_explanation(n=12)

    def test_the_identity_check_is_labelled_as_one(self):
        """`sum(parts) == total` is arithmetic: both sides are rounded to 4dp and
        the measured residual is 5.6e-17. Keeping it is fine; presenting it as a
        faithfulness rate was not."""
        block = self.section.metrics["attribution_parts_sum_to_total"]
        self.assertEqual(block["rate"], 1.0)
        self.assertIn("wiring check", block["note"])

    def test_the_faithfulness_figure_names_its_denominator_and_source(self):
        block = self.section.metrics["attribution_matches_its_source_model"]
        self.assertGreater(block["patients"], 0)
        self.assertTrue(block["sources"])

    def test_the_note_describes_the_source_it_measured(self):
        """The note under this metric said psi could not be averaged across
        members for a while after invariant 60 averaged them — three lines below
        a `sources` field reading `BMA Ensemble`. It is written from the measured
        value now, and this is what keeps the two from parting again."""
        from treatmentrx.decision.bma import BMA_ENSEMBLE

        block = self.section.metrics["attribution_matches_its_source_model"]
        note = " ".join(self.section.notes)
        if block["sources"] == [BMA_ENSEMBLE]:
            self.assertIn("BMA-weighted blip", note)
            self.assertNotIn("one member's blip", note)
        else:
            self.assertIn("one member's blip", note)
            self.assertNotIn("BMA-weighted blip", note)

    def test_the_phi_guard_is_scored_where_the_channel_runs(self):
        """It scanned 30 cards for a patient hash while the two sections that
        carry patient-specific free text — both fed from episodic memory — were
        empty for every one of them. The denominator is now reported and the
        memory path is exercised."""
        block = self.section.metrics["phi_leaks_into_narrative"]
        self.assertGreater(block["cards_scanned"], 0)
        self.assertGreater(
            block["cards_carrying_episodic_text"],
            0,
            "the guard is still being scored where its channel does not run",
        )
        self.assertEqual(block["leaks"], 0)

    def test_the_leak_count_and_its_denominator_cover_the_same_cards(self):
        """Invariant 36, in the block that was written to satisfy it.

        `leaks` is summed over the cohort loop *and* the fixture card, so
        reporting the loop's count alone as `cards_scanned` would put a count and
        its denominator over different populations — which is the defect this
        metric was rewritten to remove, reproduced one level in.
        """
        block = self.section.metrics["phi_leaks_into_narrative"]
        self.assertGreater(
            block["cards_scanned"],
            block["cards_carrying_episodic_text"],
            "the scan is only the fixture card, so the cohort loop is unreported",
        )
        self.assertIn("cohort cards", block["note"])

    def test_both_free_text_routes_reach_the_card(self):
        """The guard's denominator rests on this inventory being complete.

        Two episodic fields render as free text — `preference` under `Recorded
        patient preferences:` and `outcome_summary` interpolated into
        `Continuity:` — and `override_reason` is stored and never rendered. A
        fixture that fills one of the two leaves the other route unscanned, so
        the route list is pinned here rather than in a comment.
        """
        from treatmentrx.agent.memory import EpisodicItem
        from treatmentrx.demo_data import sample_ra_bundle
        from treatmentrx.feedback.audit import _episodic_sections
        from treatmentrx.orchestrator import TreatmentRxOrchestrator

        orchestrator = TreatmentRxOrchestrator()
        bundle = sample_ra_bundle()
        clean = orchestrator.run(bundle)
        self.assertEqual(
            _episodic_sections(clean.clinician_card),
            [],
            "a patient with no recorded history should carry no episodic section",
        )

        orchestrator.agent.memory.record(
            clean.patient_hash,
            EpisodicItem(
                stage=1,
                recommended_arm="TNF-inhibitor",
                clinician_action="TNF-inhibitor",
                override_reason="override reason free text",
                outcome_summary="outcome summary free text",
                preference="preference free text",
            ),
        )
        card = orchestrator.run(bundle).clinician_card
        self.assertIn("preference free text", card)
        self.assertIn("outcome summary free text", card)
        self.assertNotIn(
            "override reason free text",
            card,
            "override_reason has become a route and the fixture does not fill it",
        )
        self.assertEqual(len(_episodic_sections(card)), 2)

    def test_the_phi_guard_fires_when_an_identifier_reaches_the_card(self):
        """The guard has to be able to catch something, or reporting 0 says
        nothing. Episodic preferences are free text a clinician typed, which is
        the realistic route onto this card."""
        from treatmentrx.agent.memory import EpisodicItem
        from treatmentrx.demo_data import sample_ra_bundle
        from treatmentrx.feedback.audit import _identifiers_in_narrative
        from treatmentrx.orchestrator import TreatmentRxOrchestrator

        orchestrator = TreatmentRxOrchestrator()
        bundle = sample_ra_bundle()
        clean = orchestrator.run(bundle)
        self.assertFalse(_identifiers_in_narrative(clean))

        orchestrator.agent.memory.record(
            clean.patient_hash,
            EpisodicItem(
                stage=1,
                recommended_arm="TNF-inhibitor",
                clinician_action=None,
                override_reason=None,
                outcome_summary=None,
                preference=f"patient {clean.patient_hash} prefers oral therapy",
            ),
        )
        leaked = orchestrator.run(bundle)
        self.assertTrue(
            _identifiers_in_narrative(leaked),
            "the guard cannot see an identifier that reached the card",
        )


class EnsembleRegimeLabelTests(unittest.TestCase):
    """The card names the ensemble's regime, not the marginally heavier member."""

    def _estimate(self, name, regime):
        return RegimeEstimate(
            estimator=name,
            regime_type=regime,
            recommended_arm="a",
            q_values={"a": 0.8, "b": 0.7},
            policy_value=0.74,
            confidence_band=(0.7, 0.9),
        )

    def test_disagreeing_members_are_labelled_hybrid(self):
        """0.501 against 0.498 is not a basis for naming a regime."""
        from treatmentrx.decision.bma import BayesianModelAverager

        averaged = BayesianModelAverager().aggregate(
            [
                self._estimate("Q-Pooled", RegimeType.HYBRID),
                self._estimate("dWOLS-Shared", RegimeType.SPTR),
            ]
        )
        self.assertEqual(averaged.regime_type, RegimeType.HYBRID)

    def test_agreeing_members_keep_their_label(self):
        from treatmentrx.decision.bma import BayesianModelAverager

        averaged = BayesianModelAverager().aggregate(
            [
                self._estimate("Q-Pooled", RegimeType.SPTR),
                self._estimate("dWOLS-Shared", RegimeType.SPTR),
            ]
        )
        self.assertEqual(averaged.regime_type, RegimeType.SPTR)

    def test_the_deployed_card_does_not_claim_a_members_label(self):
        from treatmentrx.orchestrator import TreatmentRxOrchestrator

        audit = TreatmentRxOrchestrator().run(sample_ra_bundle()).audit_event
        weights = sorted(audit["model_weights"].values(), reverse=True)
        self.assertLess(weights[0] - weights[1], 0.05, "weights are no longer near-tied")
        self.assertEqual(audit["regime_type"], RegimeType.HYBRID.value)
