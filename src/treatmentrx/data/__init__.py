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
from treatmentrx.arms import TREATMENT_ARMS
from treatmentrx.data.belief import BeliefStateFilter
from treatmentrx.data.competing_risks import CompetingRiskBuilder
from treatmentrx.data.contract import DataContractError, RADataContract
from treatmentrx.data.endpoints import Endpoint
from treatmentrx.data.dag import CausalDAGRegistry
from treatmentrx.data.encoders import HandcraftedFeatureEncoder
from treatmentrx.data.fhir import FHIRAdapter
from treatmentrx.data.leakage import LeakageError, LeakageTestSuite
from treatmentrx.data.stages import StageHistoryBuilder
from treatmentrx.data.switching import SwitchingCapture
from treatmentrx.data.timing import TimingModel
from treatmentrx.domain import CareGoal, PatientRecord, StageRecord
from treatmentrx.scientific import EstimandContract, ScientificMode, ra_dtr_estimand

ALT_TOXICITY_THRESHOLD = 120.0
REMISSION_BELIEF = 0.35
REMISSION_OUTCOME = 0.7


class DataLayer:
    """Builds the patient state, and refuses to build one that leaks.

    `endpoint` decides what a stage's outcome is — the reward everything
    downstream optimises. It defaults to the free-text mapping because that is
    what reproduces this repo's synthetic cohort; a real deployment passes
    `EULARResponseEndpoint()` or its own. See `data/endpoints.py`.
    """

    def __init__(self, endpoint: Endpoint | None = None) -> None:
        self.fhir = FHIRAdapter()
        self.contract = RADataContract()
        self.stage_builder = StageHistoryBuilder(endpoint)
        self.timing = TimingModel()
        self.switching = SwitchingCapture()
        self.belief = BeliefStateFilter()
        self.competing_risk = CompetingRiskBuilder()
        self.leakage = LeakageTestSuite()
        self.encoder = HandcraftedFeatureEncoder()
        self.dag = CausalDAGRegistry()

    def build_patient_state(
        self,
        request: dict[str, Any] | PatientRecord,
        versions: VersionSet | None = None,
        mode: ScientificMode = ScientificMode.DTR_RESEARCH,
        estimand_contract: EstimandContract | None = None,
    ) -> PatientState:
        versions = versions or VersionSet()
        estimand_contract = estimand_contract or ra_dtr_estimand(TREATMENT_ARMS)
        if estimand_contract.mode is not mode:
            raise ValueError(
                "patient-state operating mode does not match the estimand contract"
            )
        patient = request if isinstance(request, PatientRecord) else self.fhir.parse_bundle(request)
        patient_hash = self.fhir.patient_hash(patient.patient_id)

        contract_report = self.contract.validate(patient)
        # The contract runs first and its errors are fatal *here*, before any
        # module that assumes a usable record. A patient with no treatment
        # history or a non-RA diagnosis used to reach the stage builder and the
        # DAG registry and crash there with an untyped ValueError, even though
        # the contract had already identified exactly what was wrong.
        if not contract_report.passed:
            raise DataContractError(contract_report)

        stages = self.stage_builder.build(patient)
        # The raw identifier is needed only while the record is being assembled.
        # Replace it before the first cross-layer object is created.
        stages = [replace(stage, patient_id=patient_hash) for stage in stages]
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

        dag_result = self.dag.validate(patient, stages)
        # One encode, one object. There were two: a GRU-shaped wrapper whose
        # vector nothing read, and the handcrafted encoder it called internally —
        # so this ran twice per request and the expensive result was discarded.
        encoded = self.encoder.encode(stages)

        return PatientState(
            patient_hash=patient_hash,
            disease=patient.disease,
            stage=stages[-1].stage,
            care_goal=care_goal,
            features=encoded.vector,
            feature_names=sorted(encoded.feature_map),
            adjustment_set=dag_result.adjustment_set,
            feasible_arms=sorted(self.contract.treatment_arms),
            history_summary=self._history_summary(stages),
            allergies=patient.allergies,
            stages=stages,
            diagnostics=self._diagnostics(contract_report, dag_result, leakage_report, patient),
            versions=replace(versions, dag=dag_result.version),
            data_contract=contract_report,
            dag_validation=dag_result,
            encoded_state=encoded,
            competing_risk_incidence=self.competing_risk.cumulative_incidence(stages),
            operating_mode=mode,
            estimand_contract=estimand_contract,
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

    def _diagnostics(self, contract_report, dag_result, leakage_report, patient) -> list[LayerDiagnostic]:
        # The contract's severity is carried through rather than collapsed. A
        # recorded unit conversion is `info` — something happened and you should
        # be able to see it — and reporting that at the same level as a missing
        # variable family would make the warnings worth less.
        diagnostics = [
            LayerDiagnostic(
                name=f"data_contract:{issue.field}",
                passed=issue.severity == "info",
                severity=issue.severity if issue.severity in {"error", "info"} else "warning",
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
        # Reported for every patient, because it is a property of the model
        # rather than of the record: these are confounders the DAG asserts and
        # no estimator basis carries. Silence here is what let the identifiability
        # verdict read as "adjusted for" when it meant "mentioned somewhere".
        unmodelled = self.dag.unmodelled_confounders(patient)
        if unmodelled:
            diagnostics.append(
                LayerDiagnostic(
                    name="unmodelled_confounders",
                    passed=False,
                    severity="warning",
                    message=(
                        "The effect is not adjusted for "
                        + ", ".join(unmodelled)
                        + ": the DAG lists them as confounders and no estimator "
                        "basis carries them."
                    ),
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


__all__ = ["DataContractError", "DataLayer", "LeakageError"]
