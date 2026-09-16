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

# Import the concrete built-in workflow in the package's established order.  A
# number of statistical modules intentionally share training/evaluation types;
# initializing the built-in layers here prevents Python from entering those
# packages through a partially initialized leaf module.
from treatmentrx.agent import AgentLayer  # noqa: F401
from treatmentrx.contracts import Recommendation, VersionSet
from treatmentrx.data import DataLayer  # noqa: F401
from treatmentrx.data.fhir import FHIRAdapter
from treatmentrx.decision import DecisionLayer  # noqa: F401
from treatmentrx.diseases import DiseaseDefinition, DiseaseRegistry, DiseaseWorkflow
from treatmentrx.domain import PatientRecord
from treatmentrx.estimation import EstimationLayer, training  # noqa: F401
from treatmentrx.feedback import FeedbackLayer  # noqa: F401
from treatmentrx.safety import SafetyLayer  # noqa: F401
from treatmentrx.scientific import ScientificMode

LAYER_ORDER = ("data", "estimation", "decision", "safety", "agent", "feedback")


class TreatmentRxOrchestrator:
    def __init__(self, registry: DiseaseRegistry | None = None) -> None:
        self.registry = registry or DiseaseRegistry()
        self.fhir = FHIRAdapter()
        self._workflows: dict[str, DiseaseWorkflow] = {}

        # Compatibility aliases for callers that inspect the one currently
        # registered workflow.  Scoring itself always resolves through registry.
        supported = self.registry.supported_ids()
        if not supported:
            raise ValueError("TreatmentRxOrchestrator requires at least one disease workflow")
        default_id = (
            "rheumatoid_arthritis"
            if "rheumatoid_arthritis" in supported
            else supported[0]
        )
        default = self.registry.get(default_id)
        workflow = self._workflow(default)
        self.data = workflow.data
        self.estimation = workflow.estimation
        self.decision = workflow.decision
        self.safety = workflow.safety
        self.agent = workflow.agent
        self.feedback = workflow.feedback

    def run(
        self,
        request: dict[str, Any] | PatientRecord,
        versions: VersionSet | None = None,
        mode: ScientificMode = ScientificMode.DTR_RESEARCH,
    ) -> Recommendation:
        patient = request if isinstance(request, PatientRecord) else self.fhir.parse_bundle(request)
        definition = self.registry.resolve_diagnoses(patient.conditions or [patient.disease])
        if mode not in definition.operating_modes:
            raise ValueError(
                f"disease {definition.disease_id!r} does not support operating "
                f"mode {mode.value!r}"
            )
        estimand = definition.estimand_for(mode)
        # The source problem list remains on PatientRecord.conditions.  The
        # singular disease field is the explicitly selected workflow, not an
        # accidental function of Condition entry order.
        patient = replace(patient, disease=definition.display_name)
        workflow = self._workflow(definition)

        state = workflow.data.build_patient_state(
            patient,
            versions or VersionSet(),
            mode=mode,
            estimand_contract=estimand,
        )
        estimates = workflow.estimation.estimate(state)
        decision = workflow.decision.decide(state, estimates)
        safe = workflow.safety.apply(decision, state)
        context = workflow.agent.build_context(safe)
        recommendation = workflow.agent.run_agents(context, safe)
        receipt = workflow.feedback.enqueue(state, recommendation, safe)

        return replace(
            recommendation,
            estimands=receipt.estimands,
            validation=receipt.validation,
            audit_event=self._audit_event(
                state,
                decision,
                safe,
                recommendation,
                receipt,
                definition,
                workflow,
                mode,
                estimand,
            ),
            provenance=recommendation.provenance
            | {
                "orchestrator": "treatmentrx.orchestrator",
                "operating_mode": mode.value,
                "estimand_contract": estimand.as_dict(),
                "disease_definition": definition.capability(),
                "feedback": {
                    "observational_enqueued": receipt.observational_enqueued,
                    "ope_track_enqueued": receipt.ope_track_enqueued,
                    "full_system_track_enqueued": receipt.full_system_track_enqueued,
                    "policy_value_scopes": list(receipt.policy_value_scopes),
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

        disease_id = recommendation.provenance.get("disease_definition", {}).get(
            "disease_id", "rheumatoid_arthritis"
        )
        workflow = self._workflow(self.registry.get(disease_id))
        workflow.agent.memory.record(
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
        disease_id = recommendation.provenance.get("disease_definition", {}).get(
            "disease_id", "rheumatoid_arthritis"
        )
        workflow = self._workflow(self.registry.get(disease_id))
        routing = workflow.feedback.submit_override(
            recommendation, clinician_action, reason_text, outcome_confirmed
        )
        self.record_encounter(
            recommendation, clinician_action=clinician_action, override_reason=reason_text
        )
        return routing

    def capabilities(self) -> list[dict[str, object]]:
        """Every disease workflow this process is willing to score."""
        return self.registry.capabilities()

    def _workflow(self, definition: DiseaseDefinition) -> DiseaseWorkflow:
        workflow = self._workflows.get(definition.disease_id)
        if workflow is None:
            workflow = definition.workflow_factory()
            self._workflows[definition.disease_id] = workflow
        return workflow

    def _audit_event(
        self,
        state,
        decision,
        safe,
        recommendation,
        receipt,
        definition,
        workflow,
        mode,
        estimand,
    ) -> dict[str, Any]:
        calibration = workflow.training.holdout_calibration()
        return {
            # The only wall-clock value the system produces.
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "patient_hash": state.patient_hash,
            "operating_mode": mode.value,
            "estimand_contract": estimand.as_dict(),
            "stage": state.stage,
            "care_goal": state.care_goal.value,
            "selected_method": decision.selected.estimator,
            "regime_type": decision.selected.regime_type.value,
            "recommended_arm": recommendation.recommended_arm,
            "candidate_arms": list(decision.candidate_arms),
            "candidate_set_size": len(decision.candidate_arms),
            "candidate_contrasts": {
                arm: {
                    "difference": round(test.difference, 4),
                    "interval": [round(test.lower, 4), round(test.upper, 4)],
                    "alpha": test.alpha,
                    "excluded": test.robustly_distinguishable,
                }
                for arm, test in decision.candidate_contrasts.items()
            },
            "selection_inference": {
                "method": "Bonferroni simultaneous all-pairs contrasts",
                "familywise_alpha": 0.05,
                "unordered_pair_count": len(decision.q_values)
                * (len(decision.q_values) - 1)
                // 2,
                "per_pair_alpha": (
                    decision.contrast.alpha if decision.contrast is not None else None
                ),
            },
            "top_scored_arm": recommendation.top_scored_arm,
            "q_values": decision.q_values,
            "model_weights": decision.model_weights,
            # Which member's blip the explanation decomposes. The q_values are
            # averaged and psi is not, so the audit trail has to say which.
            "attribution_source": decision.selected.coefficients.get("attribution_source"),
            "confidence_band": list(decision.selected.confidence_band),
            "confidence_gap": decision.confidence_gap,
            "contrast": decision.contrast.as_dict() if decision.contrast else None,
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
            "estimator_scorecard": workflow.training.scorecard(),
            "safety_status": safe.status.value,
            "safety_findings": [flag.__dict__ for flag in safe.safety_flags],
            "feasible_action_count": len(safe.feasible_actions),
            "removed_arms": safe.removed_arms,
            "competing_risk_incidence": state.competing_risk_incidence,
            "estimands": {result.estimand: result.policy_value for result in receipt.estimands},
            "ope": receipt.ope,
            "policy_value_scopes": list(receipt.policy_value_scopes),
            "validation_rung": receipt.validation.rung.value if receipt.validation else None,
            "validation_gate_passed": receipt.validation.gate_passed if receipt.validation else None,
            "versions": state.versions.__dict__,
            "disease_definition": definition.capability(),
        }


__all__ = ["TreatmentRxOrchestrator"]
