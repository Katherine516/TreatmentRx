"""Typed objects that cross a layer boundary.

The split is deliberate: `domain.py` holds clinical objects (a patient, a stage,
a safety flag), this module holds the six *handoffs* between layers. If a type
is produced by one layer and consumed by the next, it belongs here.

    DataLayer       -> PatientState
    EstimationLayer -> RegimeEstimate[]
    DecisionLayer   -> Decision
    SafetyLayer     -> SafeDecision
    AgentLayer      -> ContextBundle -> Recommendation
    FeedbackLayer   -> FeedbackReceipt

There is exactly one type per handoff. An earlier build carried two parallel
sets of these (`MethodResult` alongside `RegimeEstimate`, `PatientStage`
alongside `StageRecord`) and the conversion between them silently dropped the
timing, belief, switching and competing-risk annotations before the estimators
ever saw them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from treatmentrx.domain import (
    CareGoal,
    DataContractReport,
    DAGValidationResult,
    EncodedState,
    EstimandResult,
    ExplanationBundle,
    RecommendationStatus,
    RegimeType,
    SafetyFlag,
    StageRecord,
    Uncertainty,
    ValidationStatus,
)
from treatmentrx.scientific import EstimandContract, ScientificMode


@dataclass(frozen=True)
class VersionSet:
    """Everything that must be pinned for a recommendation to be reproducible."""

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
class PatientState:
    """Layer 1 output: everything downstream is allowed to condition on.

    `stages` are full `StageRecord`s, carrying the timing, belief, competing-risk
    and switching annotations Layer 1 attached. Nothing between here and the
    estimators is permitted to strip them.
    """

    patient_hash: str
    disease: str
    stage: int
    care_goal: CareGoal
    features: list[float]
    feature_names: list[str]
    adjustment_set: list[str]
    feasible_arms: list[str]
    history_summary: str
    allergies: list[str]
    stages: list[StageRecord]
    diagnostics: list[LayerDiagnostic]
    versions: VersionSet
    data_contract: DataContractReport
    dag_validation: DAGValidationResult
    encoded_state: EncodedState
    # No `tailoring_variables` here. Which covariates the treatment effect varies
    # over is a property of the fitted blip, so it is decided in Layer 2 and
    # carried on `RegimeEstimate.top_tailoring_variables`. The Layer 1 version
    # ranked raw features by magnitude and could not see the boolean effect
    # modifiers at all — the same class of mistake as the old `visit_weight`.
    competing_risk_incidence: dict[str, float] = field(default_factory=dict)
    operating_mode: ScientificMode = ScientificMode.DTR_RESEARCH
    estimand_contract: EstimandContract | None = None

    @property
    def diagnostics_passed(self) -> bool:
        return not any(d.severity == "error" and not d.passed for d in self.diagnostics)

    @property
    def latest(self) -> StageRecord:
        return self.stages[-1]


@dataclass(frozen=True)
class RegimeEstimate:
    """Layer 2 output: one treatment-regime estimator's answer.

    `policy_value` is the estimator's held-out IPW policy value — a model-level
    score, not a transform of its own Q-values — so model averaging weights
    estimators by out-of-sample performance.
    """

    estimator: str
    regime_type: RegimeType
    recommended_arm: str
    q_values: dict[str, float]
    policy_value: float
    confidence_band: tuple[float, float]
    coefficients: dict[str, float] = field(default_factory=dict)
    top_tailoring_variables: list[str] = field(default_factory=list)
    diagnostics: list[LayerDiagnostic] = field(default_factory=list)
    estimand_fingerprint: str = ""


@dataclass(frozen=True)
class GoalDecision:
    """Whether the Q-gap clears the threshold this care goal demands."""

    act: bool
    threshold: float
    observed_gap: float
    care_goal: CareGoal
    rationale: str


@dataclass(frozen=True)
class Decision:
    """Layer 3 output: the model-averaged decision, before safety."""

    recommended_arm: str
    q_values: dict[str, float]
    model_weights: dict[str, float]
    uncertainty: Uncertainty
    status: RecommendationStatus
    rationale: str
    estimates: list[RegimeEstimate]
    selected: RegimeEstimate
    goal_decision: GoalDecision
    explanation: ExplanationBundle
    confidence_gap: float
    contrast: Any = None  # ContrastTest for the top arm vs the runner-up
    # Arms the data cannot separate from the leader, leader first. Size 1 means
    # one arm is defensible; larger means the agent can still rule the rest out.
    # This is what the agent has to say when it will not name a single arm — it
    # is not a recommendation, and nothing here has passed the safety layer yet.
    candidate_arms: tuple[str, ...] = ()
    candidate_contrasts: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SafeDecision:
    """Layer 4 output: the decision after code-enforced feasibility and safety."""

    decision: Decision
    feasible_arms: list[str]
    feasible_actions: list[str]
    removed_arms: dict[str, str]
    safety_flags: list[SafetyFlag]
    status: RecommendationStatus
    provenance: dict[str, Any]

    @property
    def hard_block(self) -> bool:
        return self.status == RecommendationStatus.BLOCKED

    @property
    def contraindicated(self) -> bool:
        return any(flag.severity == "block" for flag in self.safety_flags)


@dataclass(frozen=True)
class Citation:
    source: str
    text: str
    url: str | None = None


@dataclass(frozen=True)
class ContextBundle:
    """Layer 5 input: PHI-minimised, and the only thing an LLM layer may see."""

    patient: dict[str, Any]
    decision: dict[str, Any]
    safety: dict[str, Any]
    evidence: list[Citation]
    memory: dict[str, Any]
    versions: VersionSet


@dataclass(frozen=True)
class Recommendation:
    """Layer 5 output: the agent's complete, auditable answer."""

    patient_hash: str
    status: RecommendationStatus
    recommended_arm: str | None
    top_scored_arm: str
    q_values: dict[str, float]
    clinician_card: str
    patient_summary: str
    uncertainty: str
    evidence: list[Citation]
    safety_flags: list[SafetyFlag]
    provenance: dict[str, Any]
    explanation: ExplanationBundle | None = None
    estimands: list[EstimandResult] = field(default_factory=list)
    validation: ValidationStatus | None = None
    audit_event: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeedbackReceipt:
    """Layer 6 output: non-blocking acknowledgement, never a gate on Layer 5."""

    observational_enqueued: bool
    ope_track_enqueued: bool
    retraining_allowed: bool
    message: str
    full_system_track_enqueued: bool = False
    policy_value_scopes: tuple[str, ...] = ()
    estimands: list[EstimandResult] = field(default_factory=list)
    validation: ValidationStatus | None = None
    ope: dict[str, Any] = field(default_factory=dict)
