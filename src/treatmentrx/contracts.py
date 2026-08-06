from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RecommendationStatus(str, Enum):
    RECOMMEND = "recommend"
    BLOCKED = "blocked"
    REVIEW = "review"
    EQUIPOISE = "equipoise"


@dataclass(frozen=True)
class VersionSet:
    model: str = "model-dev"
    dag: str = "RA_v1.0"
    reward: str = "reward-dev"
    thresholds: str = "thresholds-dev"
    prompts: str = "prompts-dev"
    knowledge_base: str = "ra-kb-dev"


@dataclass(frozen=True)
class LayerDiagnostic:
    name: str
    passed: bool
    severity: str
    message: str


@dataclass(frozen=True)
class PatientStage:
    stage: int
    treatment: str
    start_day: int
    end_day: int | None
    features: dict[str, float | str | bool]
    outcome: float
    response: str | None = None
    visit_weight: float = 1.0
    censoring_weight: float = 1.0


@dataclass(frozen=True)
class PatientState:
    """Layer 1 output: S_j in the v5 build spec."""

    patient_hash: str
    disease: str
    stage: int
    features: list[float]
    feature_names: list[str]
    adjustment_set: list[str]
    feasible_arms: list[str]
    history_summary: str
    allergies: list[str]
    stages: list[PatientStage]
    diagnostics: list[LayerDiagnostic]
    versions: VersionSet
    raw_patient: Any = None

    @property
    def diagnostics_passed(self) -> bool:
        return not any(d.severity == "error" and not d.passed for d in self.diagnostics)


@dataclass(frozen=True)
class RegimeEstimate:
    """Layer 2 output: one treatment-regime estimator result."""

    estimator: str
    q_values: dict[str, float]
    recommended_arm: str
    policy_value: float
    confidence_band: tuple[float, float]
    diagnostics: list[LayerDiagnostic] = field(default_factory=list)
    parameters: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Uncertainty:
    aleatoric: float
    epistemic: float
    model: float
    ood: float
    calibrated: bool
    flags: list[str]


@dataclass(frozen=True)
class Decision:
    """Layer 3 output: model-averaged treatment decision before safety."""

    recommended_arm: str
    q_values: dict[str, float]
    model_weights: dict[str, float]
    uncertainty: Uncertainty
    status: RecommendationStatus
    rationale: str
    estimates: list[RegimeEstimate]


@dataclass(frozen=True)
class SafetyFlag:
    code: str
    severity: str
    message: str
    affected_arm: str | None = None


@dataclass(frozen=True)
class SafeDecision:
    """Layer 4 output: decision after code-enforced feasibility and safety."""

    decision: Decision
    feasible_arms: list[str]
    removed_arms: dict[str, str]
    safety_flags: list[SafetyFlag]
    status: RecommendationStatus
    provenance: dict[str, Any]

    @property
    def hard_block(self) -> bool:
        return self.status == RecommendationStatus.BLOCKED


@dataclass(frozen=True)
class Citation:
    source: str
    text: str
    url: str | None = None


@dataclass(frozen=True)
class ContextBundle:
    patient: dict[str, Any]
    decision: dict[str, Any]
    safety: dict[str, Any]
    evidence: list[Citation]
    memory: dict[str, Any]
    versions: VersionSet


@dataclass(frozen=True)
class Recommendation:
    """Layer 5 output: schema-validated recommendation payload."""

    patient_hash: str
    status: RecommendationStatus
    clinician_card: str
    patient_summary: str
    recommended_arm: str | None
    q_values: dict[str, float]
    uncertainty: str
    evidence: list[Citation]
    safety_flags: list[SafetyFlag]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class FeedbackReceipt:
    """Layer 6 output: non-blocking feedback acknowledgement."""

    observational_enqueued: bool
    ope_track_enqueued: bool
    retraining_allowed: bool
    message: str
