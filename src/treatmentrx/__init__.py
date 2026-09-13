"""TreatmentRx — a research-stage clinical decision-support agent for
sequential treatment decisions in rheumatoid arthritis.

Research scaffolding, not a medical device. The synthetic cohort is a test
fixture, not evidence.

    from treatmentrx import TreatmentRxOrchestrator
    from treatmentrx.demo_data import sample_ra_bundle

    recommendation = TreatmentRxOrchestrator().run(sample_ra_bundle())
"""

from treatmentrx.contracts import (
    ContextBundle,
    Decision,
    FeedbackReceipt,
    PatientState,
    Recommendation,
    RegimeEstimate,
    SafeDecision,
    VersionSet,
)
from treatmentrx.biomarkers import (
    BiomarkerDomain,
    BiomarkerDomainError,
    BiomarkerLeakageError,
    BiomarkerRegistry,
    BiomarkerRequest,
    FrozenPrognosticModel,
    PrognosticScore,
)
from treatmentrx.domain import RecommendationStatus
from treatmentrx.diseases import DiseaseDefinition, DiseaseRegistry, UnsupportedDiseaseError
from treatmentrx.orchestrator import TreatmentRxOrchestrator
from treatmentrx.scientific import (
    EstimandContract,
    EstimandContractError,
    EvaluationPartitionContract,
    OutcomeDirection,
    ScientificMode,
)
from treatmentrx.trials import (
    ContinuousTrialRecord,
    PrognosticAdjustmentContract,
    RandomizedTrialPrecisionAnalyzer,
    TrialPrecisionError,
)

__all__ = [
    "BiomarkerDomain",
    "BiomarkerDomainError",
    "BiomarkerLeakageError",
    "BiomarkerRegistry",
    "BiomarkerRequest",
    "ContinuousTrialRecord",
    "ContextBundle",
    "Decision",
    "DiseaseDefinition",
    "DiseaseRegistry",
    "EstimandContract",
    "EstimandContractError",
    "EvaluationPartitionContract",
    "FrozenPrognosticModel",
    "FeedbackReceipt",
    "PatientState",
    "OutcomeDirection",
    "PrognosticAdjustmentContract",
    "PrognosticScore",
    "Recommendation",
    "RecommendationStatus",
    "RegimeEstimate",
    "RandomizedTrialPrecisionAnalyzer",
    "SafeDecision",
    "ScientificMode",
    "TreatmentRxOrchestrator",
    "TrialPrecisionError",
    "UnsupportedDiseaseError",
    "VersionSet",
]
