"""v5 facade for the TreatmentRx six-layer architecture."""

from treatmentrx.contracts import (
    Citation,
    ContextBundle,
    Decision,
    FeedbackReceipt,
    LayerDiagnostic,
    PatientStage,
    PatientState,
    Recommendation,
    RecommendationStatus,
    RegimeEstimate,
    SafeDecision,
    SafetyFlag,
    Uncertainty,
    VersionSet,
)
from treatmentrx.orchestrator import TreatmentRxOrchestrator

__all__ = [
    "Citation",
    "ContextBundle",
    "Decision",
    "FeedbackReceipt",
    "LayerDiagnostic",
    "PatientStage",
    "PatientState",
    "Recommendation",
    "RecommendationStatus",
    "RegimeEstimate",
    "SafeDecision",
    "SafetyFlag",
    "TreatmentRxOrchestrator",
    "Uncertainty",
    "VersionSet",
]
