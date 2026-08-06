from __future__ import annotations

from dataclasses import dataclass

from treatmentrx.arms import MANUAL_REVIEW, TREATMENT_ARMS, normalize_arm
from treatmentrx.domain import DataContractIssue, DataContractReport, PatientRecord


class DataContractError(ValueError):
    """The record cannot support a treatment-regime estimate at all.

    Raised out of `DataLayer.build_patient_state` rather than allowing a
    downstream module to fail on data the contract already rejected. Carries the
    report so a caller can say *which* requirement failed.
    """

    def __init__(self, report: DataContractReport) -> None:
        self.report = report
        self.issues = [issue for issue in report.issues if issue.severity == "error"]
        super().__init__("; ".join(issue.message for issue in self.issues) or "data contract failed")


# Physiologically possible ranges. A value outside these is a corrupt record,
# not an unusual patient: DAS28 is bounded by its own formula, and a negative
# inflammatory marker does not exist. Silently modelling one changes the
# recommendation with no indication that anything was wrong.
PLAUSIBLE_RANGES = {
    "das28": (0.0, 10.0),
    "crp": (0.0, 500.0),
    "esr": (0.0, 200.0),
    "egfr": (0.0, 200.0),
    "alt": (0.0, 5000.0),
    "ast": (0.0, 5000.0),
    "haq_di": (0.0, 3.0),
}


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

        issues.extend(self._implausible_values(patient))
        issues.extend(self._unusable_values(patient))

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

    def _implausible_values(self, patient: PatientRecord) -> list[DataContractIssue]:
        """Range-check the clinical values the estimators condition on."""
        issues = []
        for observation in patient.observations:
            code = self._normalise(observation.code)
            bounds = PLAUSIBLE_RANGES.get(code)
            if bounds is None or isinstance(observation.value, bool):
                continue
            if not isinstance(observation.value, (int, float)):
                continue
            low, high = bounds
            if not low <= float(observation.value) <= high:
                issues.append(
                    DataContractIssue(
                        field=f"observations.{code}",
                        severity="error",
                        message=(
                            f"{code}={observation.value} is outside the physiologically "
                            f"possible range [{low:g}, {high:g}]."
                        ),
                    )
                )
        return issues

    def _unusable_values(self, patient: PatientRecord) -> list[DataContractIssue]:
        """Flag values that will silently fall back to a default.

        A DAS28 recorded as the string "high" is not missing — the record claims
        to carry it — but nothing downstream can read it, so the estimators use
        a default and nobody is told.
        """
        return [
            DataContractIssue(
                field=f"observations.{self._normalise(observation.code)}",
                severity="warning",
                message=(
                    f"{self._normalise(observation.code)} is recorded as a non-numeric value "
                    f"({observation.value!r}); estimators will fall back to a default."
                ),
            )
            for observation in patient.observations
            if self._normalise(observation.code) in PLAUSIBLE_RANGES
            and not isinstance(observation.value, (int, float))
        ]

    def _normalise(self, code: str) -> str:
        return code.lower().replace("-", "_").replace(" ", "_")

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
