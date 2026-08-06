"""Layer 1 — FHIR bundle to `PatientState`.

Everything the rest of the agent is allowed to condition on is built here, and
nothing downstream may add a covariate that did not pass through this layer.
The order matters clinically: switching is captured before competing risks (a
switch is one of the competing events), and the leakage suite runs last, after
every annotation exists, so it can see anything that leaked.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from treatmentrx.contracts import LayerDiagnostic, PatientState, VersionSet
from treatmentrx.data.belief import BeliefStateFilter
from treatmentrx.data.competing_risks import CompetingRiskBuilder
from treatmentrx.data.contract import RADataContract
from treatmentrx.data.dag import CausalDAGRegistry
from treatmentrx.data.encoders import GRUBaselineEncoder, HandcraftedFeatureEncoder
from treatmentrx.data.fhir import FHIRAdapter
from treatmentrx.data.leakage import LeakageError, LeakageTestSuite
from treatmentrx.data.stages import IPCWHandler, StageHistoryBuilder, VariableSelector, VisitAligner
from treatmentrx.data.switching import SwitchingCapture
from treatmentrx.data.timing import TimingModel
from treatmentrx.domain import CareGoal, PatientRecord, StageRecord

ALT_TOXICITY_THRESHOLD = 120.0
REMISSION_BELIEF = 0.35
REMISSION_OUTCOME = 0.7


class DataLayer:
    """Builds the patient state, and refuses to build one that leaks."""

    def __init__(self) -> None:
        self.fhir = FHIRAdapter()
        self.contract = RADataContract()
        self.stage_builder = StageHistoryBuilder()
        self.visit_aligner = VisitAligner()
        self.ipcw = IPCWHandler()
        self.variable_selector = VariableSelector()
        self.timing = TimingModel()
        self.switching = SwitchingCapture()
        self.belief = BeliefStateFilter()
        self.competing_risk = CompetingRiskBuilder()
        self.leakage = LeakageTestSuite()
        self.handcrafted_encoder = HandcraftedFeatureEncoder()
        self.encoder = GRUBaselineEncoder()
        self.dag = CausalDAGRegistry()

    def build_patient_state(
        self,
        request: dict[str, Any] | PatientRecord,
        versions: VersionSet | None = None,
    ) -> PatientState:
        versions = versions or VersionSet()
        patient = request if isinstance(request, PatientRecord) else self.fhir.parse_bundle(request)

        contract_report = self.contract.validate(patient)
        stages = self.stage_builder.build(patient)
        stages = self.visit_aligner.apply(stages, patient.encounters)
        stages = self.ipcw.apply(stages)
        stages = self.timing.apply(stages, patient.encounters)
        stages = self.switching.apply(stages, patient)
        stages = self.belief.apply(stages)
        stages = self.competing_risk.apply(stages)

        # Non-negotiable: a temporal-firewall violation raises out of the whole
        # pipeline. It is never downgraded to a diagnostic the caller can ignore.
        leakage_report = self.leakage.run(patient, stages)
        if not leakage_report.temporal_firewall_passed:
            raise LeakageError("; ".join(leakage_report.violations))

        care_goal = self.infer_care_goal(stages)
        stages = [replace(stage, care_goal=care_goal) for stage in stages]

        dag_result = self.dag.validate(patient, treatment=stages[-1].treatment)
        encoded = self.encoder.encode(stages)
        handcrafted = self.handcrafted_encoder.encode(stages)

        return PatientState(
            patient_hash=self.fhir.patient_hash(patient.patient_id),
            disease=patient.disease,
            stage=stages[-1].stage,
            care_goal=care_goal,
            features=handcrafted.vector,
            feature_names=sorted(handcrafted.feature_map),
            adjustment_set=dag_result.adjustment_set,
            feasible_arms=sorted(self.contract.treatment_arms),
            history_summary=self._history_summary(stages),
            allergies=patient.allergies,
            stages=stages,
            diagnostics=self._diagnostics(contract_report, dag_result, leakage_report),
            versions=replace(versions, dag=dag_result.version),
            data_contract=contract_report,
            dag_validation=dag_result,
            encoded_state=encoded,
            tailoring_variables=self.variable_selector.select(stages),
            competing_risk_incidence=self.competing_risk.cumulative_incidence(stages),
            raw_patient=patient,
        )

    def infer_care_goal(self, stages: list[StageRecord]) -> CareGoal:
        """Read the treatment phase off the trajectory.

        Toxicity control wins over everything: a patient with a failing liver is
        not in an induction conversation regardless of disease activity.
        """
        latest = stages[-1]
        alt = latest.features.get("alt")
        if isinstance(alt, (int, float)) and not isinstance(alt, bool) and float(alt) > ALT_TOXICITY_THRESHOLD:
            return CareGoal.TOXICITY_CONTROL
        if latest.belief is not None and latest.belief.activity <= REMISSION_BELIEF:
            return CareGoal.MAINTENANCE
        if latest.outcome >= REMISSION_OUTCOME:
            return CareGoal.MAINTENANCE
        return CareGoal.INDUCTION

    def _diagnostics(self, contract_report, dag_result, leakage_report) -> list[LayerDiagnostic]:
        diagnostics = [
            LayerDiagnostic(
                name=f"data_contract:{issue.field}",
                passed=False,
                severity="error" if issue.severity == "error" else "warning",
                message=issue.message,
            )
            for issue in contract_report.issues
        ]
        diagnostics.append(
            LayerDiagnostic(
                name="causal_identifiability",
                passed=dag_result.identified,
                severity="error" if not dag_result.identified else "info",
                message=dag_result.blocked_reason
                or f"{dag_result.dag_name} {dag_result.version} identifies the effect.",
            )
        )
        diagnostics.append(
            LayerDiagnostic(
                name="leakage_suite",
                passed=leakage_report.passed,
                severity="warning" if not leakage_report.passed else "info",
                message="; ".join(leakage_report.violations) or "No leakage detected.",
            )
        )
        return diagnostics

    def _history_summary(self, stages: list[StageRecord]) -> str:
        parts = []
        for stage in stages:
            duration = "ongoing" if stage.end_day is None else f"{max(stage.end_day - stage.start_day, 0)}d"
            parts.append(f"{stage.treatment} {duration} -> {stage.response or 'response unknown'}")
        return "; ".join(parts)


__all__ = ["DataLayer", "LeakageError"]
