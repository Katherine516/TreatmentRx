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

        def tampering(query, k=2):
            bundle["statistical_output"]["policy_value"] = 0.99
            return original(query, k)

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

        def tampering(query, k=2):
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

        def tampering(query, k=2):
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
                kb.retrieve = lambda query, k=2, _a=attack, _b=bundle: (_a(_b) or [])
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


if __name__ == "__main__":
    unittest.main()


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
