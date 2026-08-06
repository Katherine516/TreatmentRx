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
from treatmentrx.domain import RecommendationStatus
from treatmentrx.orchestrator import TreatmentRxOrchestrator

__all__ = [
    "ContextBundle",
    "Decision",
    "FeedbackReceipt",
    "PatientState",
    "Recommendation",
    "RecommendationStatus",
    "RegimeEstimate",
    "SafeDecision",
    "TreatmentRxOrchestrator",
    "VersionSet",
]
