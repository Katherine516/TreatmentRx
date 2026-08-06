from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from precisionrx_agent.layer1_ingestion.data_engineering import IPCWHandler, StageHistoryBuilder, VariableSelector, VisitAligner
from precisionrx_agent.layer1_ingestion.fhir import FHIRAdapter
from precisionrx_agent.layer1_ingestion.ra_data_contract import RADataContract
from precisionrx_agent.layer1_ingestion.timing import TimingModel
from precisionrx_agent.layer1_ingestion.switching import SwitchingCapture
from precisionrx_agent.layer1_ingestion.belief import BeliefStateFilter
from precisionrx_agent.layer1_ingestion.competing_risks import CompetingRiskBuilder
from precisionrx_agent.layer1_ingestion.leakage import LeakageError, LeakageTestSuite
from precisionrx_agent.layer2_encoder.baseline import GRUBaselineEncoder, HandcraftedFeatureEncoder
from precisionrx_agent.layer3_causal_dag.dag import CausalDAGRegistry
from precisionrx_agent.layer4_estimation import training
from precisionrx_agent.layer4_estimation.estimation import DWOLSSharedEstimator, QSharedEstimator, StageSpecificQEstimator
from precisionrx_agent.layer4_estimation.regime import AdaptiveRegimeSelector
from precisionrx_agent.layer4_estimation.actions import CompositeActionSpace
from precisionrx_agent.layer4_estimation.belief_aware import BeliefAwareAdjuster
from precisionrx_agent.layer4_estimation.competing_risk_outcomes import CompetingRiskEndpoint
from precisionrx_agent.layer4_estimation.explainability import ModelExplainer
from precisionrx_agent.layer4_estimation.goal_conditioned import GoalConditionedThresholds
from precisionrx_agent.layer5_bayesian_model.bma import BayesianModelAverager
from precisionrx_agent.layer6_uncertainty.uncertainty import UncertaintyDecomposer
from precisionrx_agent.layer8_safety.safety import SafetyGate
from precisionrx_agent.layer8_safety.feasible_set import FeasibleSet
from precisionrx_agent.layer9_memory_rag.memory import (
    EpisodicItem,
    EpisodicMemory,
    SemanticKnowledgeBase,
    apply_memory,
)
from precisionrx_agent.layer11_feedback.override_governance import OverrideRouter
from precisionrx_agent.layer10_llm_agents.explain import ContextBundleBuilder, RationaleGenerator
from precisionrx_agent.layer11_feedback.estimands import EstimandReporter
from precisionrx_agent.layer11_feedback.switching_aware_ope import SwitchingAwareOPE
from precisionrx_agent.layer11_feedback.validation_ladder import ValidationLadder
from precisionrx_agent.shared.models import (
    CareGoal,
    OverrideRecord,
    OverrideRouting,
    PatientRecord,
    Recommendation,
    StageRecord,
    ValidationRung,
)


class PrecisionRxAgent:
    """End-to-end prototype agent for multi-stage treatment regimes."""

    def __init__(self) -> None:
        self.fhir = FHIRAdapter()
        self.data_contract = RADataContract()
        self.stage_builder = StageHistoryBuilder()
        self.visit_aligner = VisitAligner()
        self.ipcw = IPCWHandler()
        self.variable_selector = VariableSelector()
        # v5.1 Layer 1 clinical-realism modules
        self.timing_model = TimingModel()
        self.switching_capture = SwitchingCapture()
        self.belief_filter = BeliefStateFilter()
        self.competing_risk = CompetingRiskBuilder()
        self.leakage_suite = LeakageTestSuite()
        self.handcrafted_encoder = HandcraftedFeatureEncoder()
        self.gru_encoder = GRUBaselineEncoder()
        self.dag_registry = CausalDAGRegistry()
        self.regime_selector = AdaptiveRegimeSelector()
        self.q_shared = QSharedEstimator()
        self.dwols_shared = DWOLSSharedEstimator()
        self.stage_specific = StageSpecificQEstimator()
        self.bma = BayesianModelAverager()
        # v5.1 estimation/decision additions
        self.competing_endpoint = CompetingRiskEndpoint()
        self.belief_adjuster = BeliefAwareAdjuster()
        self.goal_thresholds = GoalConditionedThresholds()
        self.action_space = CompositeActionSpace()
        self.explainer = ModelExplainer()
        self.uncertainty = UncertaintyDecomposer()
        self.safety = SafetyGate()
        self.feasible_set = FeasibleSet()
        # v5.1 memory + feedback
        self.memory = EpisodicMemory()
        self.knowledge_base = SemanticKnowledgeBase()
        self.estimands = EstimandReporter()
        self.ope = SwitchingAwareOPE()
        self.validation_ladder = ValidationLadder()
        self.override_router = OverrideRouter()
        self.context_builder = ContextBundleBuilder()
        self.rationale = RationaleGenerator()

    def _infer_care_goal(self, stages: list[StageRecord]) -> CareGoal:
        """v5.1 #6 — infer the treatment phase from trajectory state."""
        latest = stages[-1]
        alt = latest.features.get("alt")
        if isinstance(alt, (int, float)) and not isinstance(alt, bool) and float(alt) > 120:
            return CareGoal.TOXICITY_CONTROL
        belief = latest.belief
        if belief is not None and belief.activity <= 0.35:
            return CareGoal.MAINTENANCE
        if latest.outcome >= 0.7:
            return CareGoal.MAINTENANCE
        return CareGoal.INDUCTION

    def _set_goal(self, stage: StageRecord, care_goal: CareGoal) -> StageRecord:
        return StageRecord(**(stage.__dict__ | {"care_goal": care_goal}))

    def recommend_from_fhir(self, bundle: dict[str, Any]) -> Recommendation:
        patient = self.fhir.parse_bundle(bundle)
        return self.recommend(patient)

    def recommend(self, patient: PatientRecord) -> Recommendation:
        data_contract = self.data_contract.validate(patient)
        stages = self.stage_builder.build(patient)
        stages = self.visit_aligner.apply(stages, patient.encounters)
        stages = self.ipcw.apply(stages)
        # v5.1 Layer 1 clinical-realism annotations (order: switching before competing risks)
        stages = self.timing_model.apply(stages, patient.encounters)
        stages = self.switching_capture.apply(stages, patient)
        stages = self.belief_filter.apply(stages)
        stages = self.competing_risk.apply(stages)

        # v5.1 #10 — leakage / immortal-time guard (non-negotiable)
        leakage_report = self.leakage_suite.run(patient, stages)
        if not leakage_report.temporal_firewall_passed:
            raise LeakageError("; ".join(leakage_report.violations))

        care_goal = self._infer_care_goal(stages)
        stages = [self._set_goal(stage, care_goal) for stage in stages]

        tailoring_variables = self.variable_selector.select(stages)
        handcrafted_state = self.handcrafted_encoder.encode(stages)
        encoded_state = self.gru_encoder.encode(stages)

        assignment = self.regime_selector.select(stages)
        candidates = [
            self.q_shared.fit_predict(stages, assignment, tailoring_variables),
            self.dwols_shared.fit_predict(stages, assignment, tailoring_variables),
            self.stage_specific.fit_predict(stages, assignment, tailoring_variables),
        ]
        selected = self.bma.aggregate(candidates)

        # v5.1 #2/#5 — competing-risk endpoint + belief-aware adjustment of the chosen result
        incidence = self.competing_risk.cumulative_incidence(stages)
        selected = self.competing_endpoint.adjust(selected, stages, incidence)
        selected = self.belief_adjuster.adjust(selected, stages)

        # v5.1 #6 — goal-conditioned decision threshold
        goal_decision = self.goal_thresholds.decide(selected, care_goal)
        # v5.1 #4 — composite action space + composite-aware feasible set
        action_candidates = self.action_space.candidates(list(selected.q_values.keys()), stages)
        feasibility = self.feasible_set.filter(action_candidates, stages, patient.allergies)
        # v5.1 #9 — model-level explanations
        explanation = self.explainer.explain(selected, candidates, stages)
        # v5.1 #3 — estimands + switching-aware OPE
        estimand_results = self.estimands.report(stages, selected)
        ope_result = self.ope.evaluate(stages, selected)

        dag_validation = self.dag_registry.validate(patient, selected.recommended_action)
        uncertainty = self.uncertainty.decompose(stages, selected, candidates, encoded_state)
        # Calibration is a property of the *model*, measured on held-out patients
        # it was never fit to. Scoring a patient's own outcomes against themselves
        # would report a perfect ECE for every patient and make the deployment
        # gate below vacuous.
        calibration = training.holdout_calibration()
        safety = self.safety.evaluate(stages, selected, patient.allergies, dag_validation, data_contract)

        # v5.1 #8 — prospective validation ladder status (system starts at SILENT)
        validation = self.validation_ladder.assess(
            ValidationRung.SILENT,
            {"ope_stable": ope_result.effective_sample_size > 0, "calibration_passed": calibration.passed},
        )

        bundle = self.context_builder.build(
            patient,
            stages,
            selected,
            safety,
            data_contract=data_contract,
            dag_validation=dag_validation,
            encoded_state=encoded_state,
            uncertainty=uncertainty,
            calibration=calibration,
            care_goal=care_goal,
            explanation=explanation,
            feasible_actions=[action.label for action in feasibility.feasible],
            goal_decision=goal_decision.__dict__ | {"care_goal": care_goal.value},
        )
        # Memory shapes narrative & retrieval only — never a Q-value (enforced inside).
        bundle = apply_memory(bundle, self.memory, self.knowledge_base)
        clinician_rationale = self.rationale.generate_clinician(bundle, safety)
        patient_narrative = self.rationale.generate_patient(bundle, safety)

        audit_event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "patient_hash": bundle["patient_context"]["patient_id"],
            "selected_method": selected.method_name,
            "regime_type": selected.regime_type.value,
            "recommended_action": selected.recommended_action,
            "q_values": selected.q_values,
            "confidence_band": selected.confidence_band,
            "tailoring_variables": selected.top_tailoring_variables,
            "data_contract_passed": data_contract.passed,
            "dag_identified": dag_validation.identified,
            "dag_version": dag_validation.version,
            "handcrafted_encoder_dimension": len(handcrafted_state.vector),
            "encoder": encoded_state.encoder_name,
            "encoder_dimension": len(encoded_state.vector),
            "uncertainty": uncertainty.__dict__,
            "calibration": {
                "expected_calibration_error": calibration.expected_calibration_error,
                "passed": calibration.passed,
                "measured_on": "held-out cohort",
            },
            "estimator_scorecard": training.scorecard(),
            "safety_status": safety.status.value,
            "safety_findings": [finding.__dict__ for finding in safety.findings],
            "care_goal": care_goal.value,
            "goal_action": goal_decision.act,
            "competing_risk_incidence": incidence,
            "leakage_passed": leakage_report.passed,
            "feasible_action_count": len(feasibility.feasible),
            "removed_actions": feasibility.removed,
            "estimands": {est.estimand: est.policy_value for est in estimand_results},
            "ope": ope_result.__dict__,
            "validation_rung": validation.rung.value,
            "validation_gate_passed": validation.gate_passed,
        }

        return Recommendation(
            patient_hash=bundle["patient_context"]["patient_id"],
            status=safety.status,
            patient_context=bundle["patient_context"],
            statistical_output=bundle["statistical_output"],
            shared_parameter_psi=bundle["shared_parameter_psi"],
            safety=safety,
            clinician_rationale=clinician_rationale,
            patient_narrative=patient_narrative,
            audit_event=audit_event,
            explanation=explanation,
            estimands=estimand_results,
            validation=validation,
        )

    def record_encounter(
        self,
        recommendation: Recommendation,
        clinician_action: str | None = None,
        outcome_summary: str | None = None,
        preference: str | None = None,
    ) -> None:
        """Write a structured episodic-memory item (Tier 2) for longitudinal recall.

        Memory shapes only narrative/framing on future calls — never a Q-value.
        """
        self.memory.record(
            recommendation.patient_hash,
            EpisodicItem(
                stage=recommendation.patient_context.get("stage", 0),
                recommended_action=recommendation.statistical_output["recommended_action"],
                clinician_action=clinician_action,
                override_reason=None,
                outcome_summary=outcome_summary,
                preference=preference,
            ),
        )

    def submit_override(
        self,
        recommendation: Recommendation,
        clinician_action: str,
        reason_text: str,
        outcome_confirmed: bool | None = None,
    ) -> OverrideRouting:
        """v5.1 #7 — route a clinician override and log it to episodic memory.

        Overrides never directly retrain the policy; routing decides which review
        channel owns it and whether (only if outcome-validated) it may inform the
        misspecification analysis.
        """
        record = OverrideRecord(
            patient_hash=recommendation.patient_hash,
            recommended_action=recommendation.statistical_output["recommended_action"],
            clinician_action=clinician_action,
            reason_text=reason_text,
            outcome_confirmed_clinician=outcome_confirmed,
        )
        routing = self.override_router.route(record)
        self.memory.record(
            recommendation.patient_hash,
            EpisodicItem(
                stage=recommendation.patient_context.get("stage", 0),
                recommended_action=record.recommended_action,
                clinician_action=clinician_action,
                override_reason=reason_text,
                outcome_summary=None,
                preference=None,
            ),
        )
        return routing
