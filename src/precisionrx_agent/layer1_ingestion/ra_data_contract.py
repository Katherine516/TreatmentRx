from __future__ import annotations

from dataclasses import dataclass

from precisionrx_agent.shared.arms import MANUAL_REVIEW, TREATMENT_ARMS, normalize_arm
from precisionrx_agent.shared.models import DataContractIssue, DataContractReport, PatientRecord


@dataclass(frozen=True)
class RAStudyConfig:
    """Active RA study definition used before training or inference."""

    primary_endpoint: str = "DAS28_response"
    minimum_treatment_events: int = 1
    minimum_encounters: int = 2
    required_variable_families: tuple[str, ...] = (
        "disease_activity",
        "inflammation",
        "serostatus",
        "safety_labs",
    )


class RADataContract:
    """Validates that a longitudinal patient record is usable for the RA MVP."""

    # The scoreable menu, straight from the shared vocabulary — the estimators,
    # the feasible-set filter and this contract must agree on arm names.
    treatment_arms = set(TREATMENT_ARMS)

    disease_activity_codes = {"das28", "cdai", "sdai", "tender_joint_count", "swollen_joint_count", "haq_di"}
    inflammation_codes = {"crp", "esr"}
    serostatus_codes = {"anti_ccp", "anti_ccp_positive", "rheumatoid_factor", "rf_positive"}
    safety_lab_codes = {"egfr", "alt", "ast", "pregnant", "pregnancy"}

    def __init__(self, config: RAStudyConfig | None = None) -> None:
        self.config = config or RAStudyConfig()

    def validate(self, patient: PatientRecord) -> DataContractReport:
        issues: list[DataContractIssue] = []

        if "rheumatoid" not in patient.disease.lower():
            issues.append(
                DataContractIssue(
                    field="disease",
                    severity="error",
                    message="RA MVP requires a rheumatoid arthritis diagnosis.",
                )
            )

        if len(patient.medications) < self.config.minimum_treatment_events:
            issues.append(
                DataContractIssue(
                    field="medications",
                    severity="error",
                    message="At least one treatment event is required for stage construction.",
                )
            )

        if len(patient.encounters) < self.config.minimum_encounters:
            issues.append(
                DataContractIssue(
                    field="encounters",
                    severity="warning",
                    message="Fewer than two encounters limits visit-intensity correction.",
                )
            )

        starts = [medication.start_day for medication in patient.medications]
        if starts != sorted(starts):
            issues.append(
                DataContractIssue(
                    field="medications.start_day",
                    severity="error",
                    message="Treatment events must be sortable into chronological stages.",
                )
            )

        families = self._available_families(patient)
        missing = [family for family in self.config.required_variable_families if family not in families]
        for family in missing:
            issues.append(
                DataContractIssue(
                    field=f"observations.{family}",
                    severity="warning",
                    message=f"Missing RA variable family: {family}.",
                )
            )

        observed_arms = [self.normalize_treatment_arm(medication.name) for medication in patient.medications]
        if all(arm == MANUAL_REVIEW for arm in observed_arms):
            issues.append(
                DataContractIssue(
                    field="medications.name",
                    severity="warning",
                    message="No medication could be mapped to a configured RA treatment arm.",
                )
            )

        return DataContractReport(
            disease="Rheumatoid Arthritis",
            primary_endpoint=self.config.primary_endpoint,
            treatment_arms=sorted(self.treatment_arms),
            issues=issues,
            missing_variable_families=missing,
        )

    def normalize_treatment_arm(self, medication_name: str) -> str:
        return normalize_arm(medication_name)

    def _available_families(self, patient: PatientRecord) -> set[str]:
        codes = {observation.code.lower().replace("-", "_").replace(" ", "_") for observation in patient.observations}
        families: set[str] = set()
        if codes & self.disease_activity_codes:
            families.add("disease_activity")
        if codes & self.inflammation_codes:
            families.add("inflammation")
        if codes & self.serostatus_codes:
            families.add("serostatus")
        if codes & self.safety_lab_codes:
            families.add("safety_labs")
        return families
