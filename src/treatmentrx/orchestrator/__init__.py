"""Sequencing only. No clinical logic lives here.

The orchestrator's single job is to run the six layers in an order that cannot
be rearranged: safety after decision, explanation after safety, feedback last
and non-blocking. It also assembles the audit event — the record of what was
decided, by which model version, under which safety findings.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from treatmentrx.agent import AgentLayer
from treatmentrx.contracts import Recommendation, VersionSet
from treatmentrx.data import DataLayer
from treatmentrx.decision import DecisionLayer
from treatmentrx.domain import PatientRecord
from treatmentrx.estimation import EstimationLayer, training
from treatmentrx.feedback import FeedbackLayer
from treatmentrx.safety import SafetyLayer

LAYER_ORDER = ("data", "estimation", "decision", "safety", "agent", "feedback")


class TreatmentRxOrchestrator:
    def __init__(self) -> None:
        self.data = DataLayer()
        self.estimation = EstimationLayer()
        self.decision = DecisionLayer()
        self.safety = SafetyLayer()
        self.agent = AgentLayer()
        self.feedback = FeedbackLayer()

    def run(
        self,
        request: dict[str, Any] | PatientRecord,
        versions: VersionSet | None = None,
    ) -> Recommendation:
        state = self.data.build_patient_state(request, versions or VersionSet())
        estimates = self.estimation.estimate(state)
        decision = self.decision.decide(state, estimates)
        safe = self.safety.apply(decision, state)
        context = self.agent.build_context(safe)
        recommendation = self.agent.run_agents(context, safe)
        receipt = self.feedback.enqueue(state, recommendation, safe)

        return replace(
            recommendation,
            estimands=receipt.estimands,
            validation=receipt.validation,
            audit_event=self._audit_event(state, decision, safe, recommendation, receipt),
            provenance=recommendation.provenance
            | {
                "orchestrator": "treatmentrx.orchestrator",
                "feedback": {
                    "observational_enqueued": receipt.observational_enqueued,
                    "ope_track_enqueued": receipt.ope_track_enqueued,
                    "retraining_allowed": receipt.retraining_allowed,
                    "message": receipt.message,
                },
                "layer_order": list(LAYER_ORDER),
            },
        )

    def record_encounter(self, recommendation: Recommendation, **kwargs) -> None:
        """Write an episodic-memory item for longitudinal continuity.

        Memory shapes narrative and retrieval on later calls, never a Q-value.
        """
        from treatmentrx.agent.memory import EpisodicItem

        self.agent.memory.record(
            recommendation.patient_hash,
            EpisodicItem(
                stage=recommendation.audit_event.get("stage", 0),
                recommended_arm=recommendation.recommended_arm or "none",
                clinician_action=kwargs.get("clinician_action"),
                override_reason=kwargs.get("override_reason"),
                outcome_summary=kwargs.get("outcome_summary"),
                preference=kwargs.get("preference"),
            ),
        )

    def submit_override(
        self,
        recommendation: Recommendation,
        clinician_action: str,
        reason_text: str,
        outcome_confirmed: bool | None = None,
    ):
        routing = self.feedback.submit_override(
            recommendation, clinician_action, reason_text, outcome_confirmed
        )
        self.record_encounter(
            recommendation, clinician_action=clinician_action, override_reason=reason_text
        )
        return routing

    def _audit_event(self, state, decision, safe, recommendation, receipt) -> dict[str, Any]:
        calibration = training.holdout_calibration()
        return {
            # The only wall-clock value the system produces.
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "patient_hash": state.patient_hash,
            "stage": state.stage,
            "care_goal": state.care_goal.value,
            "selected_method": decision.selected.estimator,
            "regime_type": decision.selected.regime_type.value,
            "recommended_arm": recommendation.recommended_arm,
            "q_values": decision.q_values,
            "model_weights": decision.model_weights,
            "confidence_band": list(decision.selected.confidence_band),
            "confidence_gap": decision.confidence_gap,
            "goal_action": decision.goal_decision.act,
            "tailoring_variables": decision.selected.top_tailoring_variables,
            "data_contract_passed": state.data_contract.passed,
            "dag_identified": state.dag_validation.identified,
            "dag_version": state.dag_validation.version,
            "encoder": state.encoded_state.encoder_name,
            "encoder_dimension": len(state.encoded_state.vector),
            "uncertainty": decision.uncertainty.__dict__,
            "calibration": {
                "expected_calibration_error": calibration.expected_calibration_error,
                "passed": calibration.passed,
                "measured_on": "held-out cohort",
            },
            "estimator_scorecard": training.scorecard(),
            "safety_status": safe.status.value,
            "safety_findings": [flag.__dict__ for flag in safe.safety_flags],
            "feasible_action_count": len(safe.feasible_actions),
            "removed_arms": safe.removed_arms,
            "competing_risk_incidence": state.competing_risk_incidence,
            "estimands": {result.estimand: result.policy_value for result in receipt.estimands},
            "ope": receipt.ope,
            "validation_rung": receipt.validation.rung.value if receipt.validation else None,
            "validation_gate_passed": receipt.validation.gate_passed if receipt.validation else None,
            "versions": state.versions.__dict__,
        }


__all__ = ["TreatmentRxOrchestrator"]
