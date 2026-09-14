from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RegimeType(str, Enum):
    SPTR = "SPTR"
    DTR = "DTR"
    HYBRID = "HYBRID"
    SURVIVAL_FOREST = "SURVIVAL_FOREST"


class RecommendationStatus(str, Enum):
    RECOMMEND = "recommend"
    BLOCKED = "blocked"
    EQUIPOISE = "equipoise"
    REVIEW = "review"


class CareGoal(str, Enum):
    """v5.1 #6 — the treatment goal/phase. Reward and thresholds condition on it."""

    INDUCTION = "induction"          # drive disease down fast
    MAINTENANCE = "maintenance"      # hold remission, minimize burden
    TOXICITY_CONTROL = "tox_control"  # back off, manage adverse effects
    QUALITY_OF_LIFE = "qol"          # palliative / preference-weighted


class ClinicalEventType(str, Enum):
    """v5.1 #2 — competing events, not just a single censoring flag."""

    ONGOING = "ongoing"
    RESPONSE = "response"
    PROGRESSION = "progression"
    DEATH = "death"
    DROPOUT = "dropout"
    SERIOUS_TOXICITY = "serious_toxicity"
    TREATMENT_SWITCH = "treatment_switch"


class ValidationRung(str, Enum):
    """v5.1 #8 — prospective validation ladder rungs."""

    SILENT = "silent"
    SHADOW = "shadow"
    ADVISORY = "advisory"
    PRAGMATIC_TRIAL = "pragmatic_trial"


class OverrideChannel(str, Enum):
    """v5.1 #7 — override governance routing channels."""

    USABILITY = "usability"
    SAFETY_REVIEW = "safety_review"
    GUIDELINE_CONFLICT = "guideline_conflict"
    POSSIBLE_MISSPECIFICATION = "possible_misspecification"


@dataclass(frozen=True)
class Observation:
    code: str
    value: float | str | bool
    unit: str | None = None
    days_from_baseline: int = 0


@dataclass(frozen=True)
class TreatmentEvent:
    name: str
    start_day: int
    dose: str | None = None
    stop_day: int | None = None
    response: str | None = None
    discontinuation_reason: str | None = None
    # What was actually dispensed or administered against this order, from
    # `MedicationDispense` / `MedicationAdministration`. `None` means the record
    # carries no such resource — which is different from "nothing was
    # dispensed", and the two must not be confused: an absent supply chain is a
    # missing measurement, an empty one is non-adherence.
    dispensed_name: str | None = None
    dispensed_days_supply: int | None = None


@dataclass(frozen=True)
class PatientRecord:
    patient_id: str
    disease: str
    demographics: dict[str, Any]
    conditions: list[str]
    allergies: list[str]
    medications: list[TreatmentEvent]
    observations: list[Observation]
    encounters: list[int]
    outcomes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TimingFeatures:
    """v5.1 #1 — irregular timing made first-class on the state."""

    time_since_last_treatment: int | None = None
    time_since_last_visit: int | None = None
    interval_to_next_decision: int | None = None
    response_window_days: int = 90
    delayed_toxicity_window_days: int = 180
    inverse_intensity_weight: float = 1.0


@dataclass(frozen=True)
class BeliefState:
    """v5.1 #5 — a filtered belief over latent disease activity, with uncertainty.

    `activity` is on the same 0..1 scale as a normalized disease-activity score;
    `uncertainty` is the standard deviation of that estimate.
    """

    activity: float
    uncertainty: float
    proxies_used: list[str] = field(default_factory=list)

    @property
    def confident(self) -> bool:
        return self.uncertainty <= 0.15


@dataclass(frozen=True)
class ClinicalEvent:
    """v5.1 #2 — outcome as a typed (type, time) event, not a scalar."""

    type: ClinicalEventType
    time: int | None = None


@dataclass(frozen=True)
class SwitchingRecord:
    """v5.1 #3 — realized vs assigned treatment plus adherence/rescue capture."""

    assigned: str
    realized: str
    adherence: float = 1.0
    switched: bool = False
    rescue_therapy: bool = False
    discontinuation_reason: str | None = None


@dataclass(frozen=True)
class CompositeAction:
    """v5.1 #4 — A_j is a structured object, not just an arm label."""

    drug: str
    dose: str | None = None
    route: str | None = None
    timing: str | None = None
    combination: str | None = None
    stop_continue: str = "continue"

    @property
    def label(self) -> str:
        parts = [self.drug]
        if self.combination:
            parts.append(f"+{self.combination}")
        if self.dose:
            parts.append(self.dose)
        if self.route:
            parts.append(self.route)
        if self.timing:
            parts.append(self.timing)
        return " ".join(parts)


@dataclass(frozen=True)
class StageRecord:
    patient_id: str
    disease: str
    stage: int
    treatment: str
    start_day: int
    end_day: int | None
    features: dict[str, float | str | bool]
    response: str | None
    outcome: float
    care_goal: CareGoal = CareGoal.INDUCTION
    timing: TimingFeatures | None = None
    belief: BeliefState | None = None
    event: ClinicalEvent | None = None
    switching: SwitchingRecord | None = None


@dataclass(frozen=True)
class DataContractIssue:
    field: str
    severity: str
    message: str


@dataclass(frozen=True)
class DataContractReport:
    disease: str
    primary_endpoint: str
    treatment_arms: list[str]
    issues: list[DataContractIssue]
    missing_variable_families: list[str]

    @property
    def passed(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)


@dataclass(frozen=True)
class DAGValidationResult:
    dag_name: str
    version: str
    identified: bool
    adjustment_set: list[str]
    causal_path_text: str
    blocked_reason: str | None = None


@dataclass(frozen=True)
class EncodedState:
    encoder_name: str
    vector: list[float]
    feature_map: dict[str, float]


@dataclass(frozen=True)
class Uncertainty:
    """The four uncertainty types the safety and UI layers report separately."""

    aleatoric: float
    epistemic: float
    model: float
    ood: float
    calibrated: bool
    flags: list[str]


@dataclass(frozen=True)
class CalibrationReport:
    expected_calibration_error: float
    threshold: float
    passed: bool
    reliability_bins: list[dict[str, float]]


@dataclass(frozen=True)
class SafetyFlag:
    """A single finding from the code-enforced safety layer.

    `severity` is "block" or "warn"; only the safety layer may emit a block and
    only the safety layer may clear one.
    """

    code: str
    severity: str
    message: str
    affected_arm: str | None = None


@dataclass(frozen=True)
class BlipAttribution:
    """v5.1 #9 — per-covariate contribution to a treatment's advantage (blip)."""

    action: str
    contributions: dict[str, float]
    total_advantage: float


@dataclass(frozen=True)
class WhyNotEntry:
    """v5.1 #9 — for a non-recommended action: the Q-gap and dominant reason."""

    action: str
    q_gap: float
    dominant_reason: str


@dataclass(frozen=True)
class CounterfactualProbe:
    """v5.1 #9 — would the recommendation flip under a covariate perturbation?"""

    covariate: str
    perturbation: str
    recommendation_changes: bool
    note: str


@dataclass(frozen=True)
class AssumptionSensitivity:
    """How strong unmeasured confounding would have to be to explain the contrast away.

    `e_value` is the VanderWeele-Ding bound on the *point estimate*;
    `e_value_for_interval` is the same bound on the confidence limit nearest the
    null, which is the one to quote — it answers "could confounding move this to
    no difference", and it is **1.0 exactly when the interval already contains
    zero**, because then nothing is needed.

    The previous version of this type carried a number that was none of that:
    `rr = q_values[0] / q_values[1]`, a ratio of two nearly-equal bounded means
    fed into the E-value formula. It never referenced confounding, and because
    the two Q-values are close by construction it returned 1.000 to 1.617 across
    60 patients, with 11 of them under 1.1. An E-value of 1.0 asserts that *no*
    unmeasured confounding is required — a strong claim, made by arithmetic on a
    quantity that could not support it. That is invariant 27's pattern.
    """

    e_value: float
    e_value_for_interval: float
    contrast: float
    outcome_sd: float
    note: str


@dataclass(frozen=True)
class ExplanationBundle:
    """Faithful, model-derived explanations. The agent renders, never invents."""

    attributions: list[BlipAttribution]
    why_not: list[WhyNotEntry]
    counterfactuals: list[CounterfactualProbe]
    sensitivity: AssumptionSensitivity


@dataclass(frozen=True)
class EstimandResult:
    """v5.1 #3 — ITT / per-protocol / as-treated reported side by side."""

    estimand: str  # "ITT" | "per_protocol" | "as_treated"
    policy_value: float
    n_effective: float
    note: str


@dataclass(frozen=True)
class OverrideRecord:
    """v5.1 #7 — a clinician override of the recommendation."""

    patient_hash: str
    recommended_arm: str
    clinician_action: str
    reason_text: str
    outcome_confirmed_clinician: bool | None = None


@dataclass(frozen=True)
class OverrideRouting:
    """v5.1 #7 — classified override + whether it may influence the model."""

    channel: OverrideChannel
    owner: str
    influences_model: bool
    rationale: str


@dataclass(frozen=True)
class ValidationStatus:
    """v5.1 #8 — current rung on the prospective-validation ladder + gate state."""

    rung: ValidationRung
    gate_description: str
    gate_passed: bool
    blockers: list[str]
    next_rung: ValidationRung | None
