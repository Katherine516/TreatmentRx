from __future__ import annotations

from precisionrx_agent.layer1_ingestion.data_engineering import VariableSelector
from precisionrx_agent.layer4_estimation.dwols import DWOLSSharedEstimator
from precisionrx_agent.layer4_estimation.estimation import QSharedEstimator, StageSpecificQEstimator
from precisionrx_agent.layer4_estimation.regime import AdaptiveRegimeSelector
from precisionrx_agent.shared.models import RegimeAssignment, RegimeType, StageRecord
from treatmentrx.contracts import LayerDiagnostic, PatientState, RegimeEstimate


class EstimationLayer:
    """Layer 2: run treatment-regime estimators over the PatientState contract."""

    def __init__(self) -> None:
        self.variable_selector = VariableSelector()
        self.regime_selector = AdaptiveRegimeSelector()
        self.estimators = (
            QSharedEstimator(),
            DWOLSSharedEstimator(),
            StageSpecificQEstimator(),
        )

    def estimate(self, state: PatientState) -> list[RegimeEstimate]:
        stages = self._to_stage_records(state)
        tailoring_variables = self.variable_selector.select(stages)
        assignment = self.regime_selector.select(stages) if stages else self._fallback_assignment()
        estimates = []
        for estimator in self.estimators:
            result = estimator.fit_predict(stages, assignment, tailoring_variables)
            estimates.append(
                RegimeEstimate(
                    estimator=result.method_name,
                    q_values=result.q_values,
                    recommended_arm=result.recommended_action,
                    policy_value=result.policy_value,
                    confidence_band=result.confidence_band,
                    diagnostics=[
                        LayerDiagnostic(
                            name=f"estimator:{result.method_name}",
                            passed=True,
                            severity="info",
                            message=f"{result.method_name} completed with policy value {result.policy_value:.3f}",
                        )
                    ],
                    parameters=result.coefficients,
                )
            )
        return estimates

    def _to_stage_records(self, state: PatientState) -> list[StageRecord]:
        return [
            StageRecord(
                patient_id=state.patient_hash,
                disease=state.disease,
                stage=stage.stage,
                treatment=stage.treatment,
                start_day=stage.start_day,
                end_day=stage.end_day,
                features=stage.features,
                response=stage.response,
                outcome=stage.outcome,
                visit_weight=stage.visit_weight,
                censoring_weight=stage.censoring_weight,
            )
            for stage in state.stages
        ]

    def _fallback_assignment(self) -> RegimeAssignment:
        return RegimeAssignment(
            regime_type=RegimeType.SPTR,
            reason="fallback when no stages are available",
            shared_bic=0.0,
            stage_specific_bic=0.0,
        )
