from __future__ import annotations

from dataclasses import replace
from typing import Any

from treatmentrx.agent import AgentLayer
from treatmentrx.contracts import Recommendation, VersionSet
from treatmentrx.data import DataLayer
from treatmentrx.decision import DecisionLayer
from treatmentrx.estimation import EstimationLayer
from treatmentrx.feedback import FeedbackLayer
from treatmentrx.safety import SafetyLayer


class TreatmentRxOrchestrator:
    """Cross-cutting v5 orchestrator: sequencing only, no clinical logic."""

    def __init__(self) -> None:
        self.data = DataLayer()
        self.estimation = EstimationLayer()
        self.decision = DecisionLayer()
        self.safety = SafetyLayer()
        self.agent = AgentLayer()
        self.feedback = FeedbackLayer()

    def run(self, request: dict[str, Any], versions: VersionSet | None = None) -> Recommendation:
        versions = versions or VersionSet()

        state = self.data.build_patient_state(request, versions)
        estimates = self.estimation.estimate(state)
        decision = self.decision.decide(state, estimates)
        safe = self.safety.apply(decision, state)
        context = self.agent.build_context(safe)
        recommendation = self.agent.run_agents(context, safe)
        receipt = self.feedback.enqueue(state, recommendation)

        return replace(
            recommendation,
            provenance=recommendation.provenance
            | {
                "orchestrator": "treatmentrx.orchestrator",
                "feedback": receipt.__dict__,
                "layer_order": ["data", "estimation", "decision", "safety", "agent", "feedback"],
            },
        )
