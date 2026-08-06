from __future__ import annotations

import hashlib
from typing import Any

from precisionrx_agent.shared.models import (
    CalibrationReport,
    CareGoal,
    DAGValidationResult,
    DataContractReport,
    EncodedState,
    ExplanationBundle,
    MethodResult,
    PatientRecord,
    RecommendationStatus,
    SafetyResult,
    StageRecord,
    UncertaintyBundle,
)


class ContextBundleBuilder:
    """Creates a PHI-minimized bundle for downstream LLM/rationale generation."""

    def build(
        self,
        patient: PatientRecord,
        stages: list[StageRecord],
        result: MethodResult,
        safety: SafetyResult,
        data_contract: DataContractReport | None = None,
        dag_validation: DAGValidationResult | None = None,
        encoded_state: EncodedState | None = None,
        uncertainty: UncertaintyBundle | None = None,
        calibration: CalibrationReport | None = None,
        care_goal: CareGoal | None = None,
        explanation: ExplanationBundle | None = None,
        feasible_actions: list[str] | None = None,
        goal_decision: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        patient_hash = hashlib.sha256(patient.patient_id.encode("utf-8")).hexdigest()[:16]
        bundle = {
            "patient_context": {
                "patient_id": patient_hash,
                "disease": patient.disease,
                "stage": stages[-1].stage,
                "care_goal": care_goal.value if care_goal else None,
                "history_summary": self._history_summary(stages),
                "top_tailoring_vars": result.top_tailoring_variables,
            },
            "statistical_output": {
                "selected_method": result.method_name,
                "regime_type": result.regime_type.value,
                "recommended_action": result.recommended_action,
                "q_values": result.q_values,
                "confidence_band": list(result.confidence_band),
                "policy_value": result.policy_value,
                "policy_value_vs_alternatives": self._gap(result.q_values),
                "ood_flag": safety.ood_flag,
                "contraindication_flag": safety.contraindication_flag,
                "safety_status": safety.status.value,
            },
            "shared_parameter_psi": result.coefficients,
        }
        if feasible_actions is not None:
            bundle["feasible_actions"] = feasible_actions
        if goal_decision is not None:
            bundle["goal_decision"] = goal_decision
        if explanation is not None:
            bundle["explanation"] = {
                "attributions": [a.__dict__ for a in explanation.attributions],
                "why_not": [w.__dict__ for w in explanation.why_not],
                "counterfactuals": [c.__dict__ for c in explanation.counterfactuals],
                "assumption_sensitivity": explanation.sensitivity.__dict__,
            }
        if data_contract:
            bundle["data_contract"] = {
                "passed": data_contract.passed,
                "primary_endpoint": data_contract.primary_endpoint,
                "treatment_arms": data_contract.treatment_arms,
                "missing_variable_families": data_contract.missing_variable_families,
                "issues": [issue.__dict__ for issue in data_contract.issues],
            }
        if dag_validation:
            bundle["causal_dag"] = {
                "dag_name": dag_validation.dag_name,
                "version": dag_validation.version,
                "identified": dag_validation.identified,
                "adjustment_set": dag_validation.adjustment_set,
                "causal_path_text": dag_validation.causal_path_text,
                "blocked_reason": dag_validation.blocked_reason,
            }
        if encoded_state:
            bundle["encoded_state"] = {
                "encoder_name": encoded_state.encoder_name,
                "dimension": len(encoded_state.vector),
                "feature_map": encoded_state.feature_map,
            }
        if uncertainty:
            bundle["uncertainty"] = uncertainty.__dict__
        if calibration:
            bundle["calibration"] = {
                "expected_calibration_error": calibration.expected_calibration_error,
                "threshold": calibration.threshold,
                "passed": calibration.passed,
                "reliability_bins": calibration.reliability_bins,
            }
        return bundle

    def _history_summary(self, stages: list[StageRecord]) -> str:
        parts = []
        for stage in stages:
            duration = "ongoing" if stage.end_day is None else f"{max(stage.end_day - stage.start_day, 0)}d"
            response = stage.response or "response unknown"
            parts.append(f"{stage.treatment} {duration} -> {response}")
        return "; ".join(parts)

    def _gap(self, q_values: dict[str, float]) -> float:
        values = sorted(q_values.values(), reverse=True)
        if len(values) < 2:
            return 0.0
        return round(values[0] - values[1], 3)


class RationaleGenerator:
    """Offline deterministic explanation layer that enforces the architecture constraints."""

    def generate_clinician(self, bundle: dict[str, Any], safety: SafetyResult) -> str | None:
        output = bundle["statistical_output"]
        if safety.status == RecommendationStatus.BLOCKED:
            findings = " ".join(finding.message for finding in safety.findings)
            return f"Recommendation blocked pending clinical review. {findings}"

        action = output["recommended_action"]
        method = output["selected_method"]
        regime = output["regime_type"]
        q_values = output["q_values"]
        next_best = self._next_best(action, q_values)
        drivers = bundle["patient_context"]["top_tailoring_vars"] or ["stage history"]
        gap = output["policy_value_vs_alternatives"]
        caveat = ""
        if safety.status in {RecommendationStatus.EQUIPOISE, RecommendationStatus.REVIEW}:
            caveat = " Confidence is limited; clinician judgment or override should be considered."

        goal_intro = self._goal_framing(bundle["patient_context"].get("care_goal"), action)
        why_not = self._render_why_not(bundle, next_best, action)
        prior, evidence = self._render_memory(bundle)

        return (
            f"{goal_intro}Recommend {action} using {method} under a {regime} regime. "
            f"The recommendation is statistically derived, with an estimated Q-value gap of {gap:.3f} "
            f"over {next_best}.\n\n"
            f"The main tailoring drivers were {', '.join(drivers)}. In this prototype, shared parameters "
            f"mean the same treatment logic is applied across the patient's prior stages rather than fitting "
            f"an independent rule at each visit. This is most appropriate when longitudinal treatment effects "
            f"appear stable or when data are limited.\n\n"
            f"{why_not}{prior}{evidence}{caveat}"
        )

    # Memory and explanations are *rendered*, never invented (v5.1 division of labour).
    def _goal_framing(self, care_goal: str | None, action: str) -> str:
        framing = {
            "induction": "Goal is induction (drive disease activity down): ",
            "maintenance": "Goal is maintenance (hold remission, minimize burden): ",
            "tox_control": "Goal is toxicity control (back off, manage adverse effects): ",
            "qol": "Goal is quality of life (comfort and preference weighted): ",
        }
        return framing.get(care_goal or "", "")

    def _render_why_not(self, bundle: dict[str, Any], next_best: str, action: str) -> str:
        explanation = bundle.get("explanation")
        if not explanation or not explanation.get("why_not"):
            return f"Why not {next_best}? Its estimated Q-value was lower than {action} for the current stage context."
        top = explanation["why_not"][0]
        return (
            f"Why not {top['action']}? Q-gap {top['q_gap']:.3f}; {top['dominant_reason']}. "
            "(Full why-not table and blip attributions are available in the model explanation panel.)"
        )

    def _render_memory(self, bundle: dict[str, Any]) -> tuple[str, str]:
        memory = bundle.get("memory")
        if not memory:
            return "", ""
        prior = ""
        if memory.get("prior_context") and "No prior" not in memory["prior_context"]:
            prior = f"\n\nContinuity: {memory['prior_context']}"
        evidence = ""
        if memory.get("evidence"):
            evidence = "\n\nGuideline evidence: " + " ".join(memory["evidence"])
        return prior, evidence

    def generate_patient(self, bundle: dict[str, Any], safety: SafetyResult) -> str | None:
        output = bundle["statistical_output"]
        if safety.status == RecommendationStatus.BLOCKED:
            return "The care team needs to review a safety issue before a treatment suggestion is shown."
        return (
            f"The care team may consider {output['recommended_action']}. This suggestion is based on your "
            "recent treatment history and disease activity pattern. At the next visit, the team should monitor "
            "symptoms, lab response, side effects, and whether the plan still fits your goals."
        )

    def _next_best(self, action: str, q_values: dict[str, float]) -> str:
        ordered = sorted(q_values.items(), key=lambda item: item[1], reverse=True)
        for treatment, _ in ordered:
            if treatment != action:
                return treatment
        return "the next-best alternative"
