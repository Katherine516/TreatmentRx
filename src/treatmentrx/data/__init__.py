from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any

from precisionrx_agent.layer1_ingestion.belief import BeliefStateFilter
from precisionrx_agent.layer1_ingestion.competing_risks import CompetingRiskBuilder
from precisionrx_agent.layer1_ingestion.data_engineering import IPCWHandler, StageHistoryBuilder, VariableSelector, VisitAligner
from precisionrx_agent.layer1_ingestion.fhir import FHIRAdapter
from precisionrx_agent.layer1_ingestion.leakage import LeakageTestSuite
from precisionrx_agent.layer1_ingestion.ra_data_contract import RADataContract
from precisionrx_agent.layer1_ingestion.switching import SwitchingCapture
from precisionrx_agent.layer1_ingestion.timing import TimingModel
from precisionrx_agent.layer2_encoder.baseline import GRUBaselineEncoder, HandcraftedFeatureEncoder
from precisionrx_agent.layer3_causal_dag.dag import CausalDAGRegistry
from precisionrx_agent.shared.models import StageRecord
from treatmentrx.contracts import LayerDiagnostic, PatientStage, PatientState, VersionSet


class DataLayer:
    """Layer 1: FHIR-like input to the typed PatientState contract."""

    def __init__(self) -> None:
        self.fhir = FHIRAdapter()
        self.contract = RADataContract()
        self.stage_builder = StageHistoryBuilder()
        self.timing = TimingModel()
        self.switching = SwitchingCapture()
        self.belief = BeliefStateFilter()
        self.competing_risks = CompetingRiskBuilder()
        self.visit_aligner = VisitAligner()
        self.ipcw = IPCWHandler()
        self.variables = VariableSelector()
        self.handcrafted = HandcraftedFeatureEncoder()
        self.encoder = GRUBaselineEncoder()
        self.dag = CausalDAGRegistry()
        self.leakage = LeakageTestSuite()

    def build_patient_state(self, bundle: dict[str, Any], versions: VersionSet | None = None) -> PatientState:
        versions = versions or VersionSet()
        patient = self.fhir.parse_bundle(bundle)
        data_report = self.contract.validate(patient)
        stages = self.stage_builder.build(patient)
        stages = self.timing.apply(stages, patient.encounters)
        stages = self.switching.apply(stages, patient)
        stages = self.belief.apply(stages)
        stages = self.competing_risks.apply(stages)
        stages = self.visit_aligner.apply(stages, patient.encounters)
        stages = self.ipcw.apply(stages)

        latest = stages[-1]
        dag_result = self.dag.validate(patient, treatment=latest.treatment)
        leakage_report = self.leakage.run(patient, stages)
        encoded = self.encoder.encode(stages)
        handcrafted = self.handcrafted.encode(stages)
        feature_names = [f"z_{idx}" for idx in range(len(encoded.vector))]
        features = encoded.vector

        diagnostics = [
            LayerDiagnostic("data_contract", data_report.passed, "error" if not data_report.passed else "info",
                            "RA data contract passed" if data_report.passed else "RA data contract failed"),
            LayerDiagnostic("causal_dag", dag_result.identified, "error" if not dag_result.identified else "info",
                            dag_result.blocked_reason or dag_result.causal_path_text),
            LayerDiagnostic("leakage", leakage_report.passed, "error" if not leakage_report.passed else "info",
                            "Temporal leakage suite passed" if leakage_report.passed else "; ".join(leakage_report.violations)),
            LayerDiagnostic("encoder", len(encoded.vector) == 256, "info", encoded.encoder_name),
            LayerDiagnostic("handcrafted_baseline", len(handcrafted.vector) > 0, "info", handcrafted.encoder_name),
        ]
        diagnostics.extend(
            LayerDiagnostic(f"data_contract:{issue.field}", issue.severity != "error", issue.severity, issue.message)
            for issue in data_report.issues
        )

        return PatientState(
            patient_hash=self._hash(patient.patient_id),
            disease=patient.disease,
            stage=latest.stage,
            features=features,
            feature_names=feature_names,
            adjustment_set=dag_result.adjustment_set,
            feasible_arms=sorted(self.contract.treatment_arms),
            history_summary=self._history_summary(stages),
            allergies=patient.allergies,
            stages=[self._contract_stage(stage) for stage in stages],
            diagnostics=diagnostics,
            versions=replace(versions, dag=dag_result.version),
            raw_patient=patient,
        )

    def _contract_stage(self, stage: StageRecord) -> PatientStage:
        return PatientStage(
            stage=stage.stage,
            treatment=stage.treatment,
            start_day=stage.start_day,
            end_day=stage.end_day,
            features=stage.features,
            outcome=stage.outcome,
            response=stage.response,
            visit_weight=stage.visit_weight,
            censoring_weight=stage.censoring_weight,
        )

    def _history_summary(self, stages: list[StageRecord]) -> str:
        parts = []
        for stage in stages:
            duration = "ongoing" if stage.end_day is None else f"{max(stage.end_day - stage.start_day, 0)}d"
            parts.append(f"stage {stage.stage}: {stage.treatment} {duration} -> {stage.response or 'unknown'}")
        return "; ".join(parts)

    def _hash(self, patient_id: str) -> str:
        return hashlib.sha256(patient_id.encode("utf-8")).hexdigest()[:16]
