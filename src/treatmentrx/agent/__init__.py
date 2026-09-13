"""Layer 5 — context assembly, memory/RAG, and the three explanation agents.

The division of labour is the point of this layer: it **renders** what Layers 3
and 4 decided and never adds to it. The three agents may not choose an arm,
invent a citation, soften a safety block, or make a causal claim beyond the
DAG-generated path text.

Memory reaches this layer and only this layer. `apply_memory` enforces, by
assertion rather than convention, that nothing it adds can move a statistical
quantity — memory shapes framing and retrieval, never a Q-value.
"""

from __future__ import annotations

from treatmentrx.agent.memory import EpisodicMemory, SemanticKnowledgeBase, apply_memory
from treatmentrx.agent.rationale import RationaleGenerator
from treatmentrx.contracts import Citation, ContextBundle, Recommendation, SafeDecision, VersionSet
from treatmentrx.domain import RecommendationStatus


class AgentLayer:
    def __init__(self) -> None:
        self.memory = EpisodicMemory()
        self.knowledge_base = SemanticKnowledgeBase()
        self.rationale = RationaleGenerator()

    def build_context(self, safe: SafeDecision) -> ContextBundle:
        decision = safe.decision
        patient_hash = safe.provenance.get("patient_hash", "unknown")

        bundle = {
            "patient_context": {
                "patient_id": patient_hash,
                "care_goal": safe.provenance.get("care_goal"),
                "history_summary": safe.provenance.get("history_summary", ""),
                "top_tailoring_vars": decision.selected.top_tailoring_variables,
            },
            # Everything under this key is off-limits to memory and to the agents.
            "statistical_output": {
                "selected_method": decision.selected.estimator,
                "regime_type": decision.selected.regime_type.value,
                "recommended_arm": decision.recommended_arm,
                "q_values": decision.q_values,
                "policy_value": decision.selected.policy_value,
                "confidence_band": list(decision.selected.confidence_band),
                "safety_status": safe.status.value,
            },
        }
        bundle = apply_memory(bundle, self.memory, self.knowledge_base)

        return ContextBundle(
            patient=bundle["patient_context"],
            decision={
                "recommended_arm": decision.recommended_arm,
                "q_values": decision.q_values,
                "model_weights": decision.model_weights,
                "rationale": decision.rationale,
                "confidence_gap": decision.confidence_gap,
                "goal_decision": decision.goal_decision,
                "explanation": decision.explanation,
                "uncertainty": decision.uncertainty,
            },
            safety={
                "status": safe.status.value,
                "flags": [flag.__dict__ for flag in safe.safety_flags],
                "feasible_arms": safe.feasible_arms,
                "feasible_actions": safe.feasible_actions,
            },
            evidence=[
                Citation(source=f"RA knowledge base {self.knowledge_base.version}", text=passage)
                for passage in bundle["memory"]["evidence"]
            ],
            memory=bundle["memory"],
            versions=VersionSet(**safe.provenance.get("versions", {})),
        )

    def run_agents(self, context: ContextBundle, safe: SafeDecision) -> Recommendation:
        safety_text = self._safety_agent(safe)
        if safe.hard_block:
            return self._blocked(context, safe, safety_text)

        guideline_text = self._guideline_agent(context)
        published_arm = (
            safe.decision.recommended_arm
            if safe.status is RecommendationStatus.RECOMMEND
            else None
        )
        return Recommendation(
            patient_hash=context.patient["patient_id"],
            status=safe.status,
            recommended_arm=published_arm,
            top_scored_arm=safe.decision.recommended_arm,
            q_values=safe.decision.q_values,
            clinician_card=self.rationale.clinician_card(context, safe, safety_text, guideline_text),
            patient_summary=self.rationale.patient_summary(context, safe),
            uncertainty=self.rationale.uncertainty_text(safe.decision.uncertainty),
            evidence=context.evidence,
            safety_flags=safe.safety_flags,
            explanation=safe.decision.explanation,
            provenance=safe.provenance | {"agent_layer": "treatmentrx.agent", "schema_validated": True},
        )

    def _safety_agent(self, safe: SafeDecision) -> str:
        """Formats safety findings verbatim. It has no authority to reword a block."""
        if not safe.safety_flags:
            return "No safety concern was identified by the safety layer."
        return "Safety review: " + "; ".join(
            f"{flag.severity}: {flag.message}" for flag in safe.safety_flags
        )

    def _guideline_agent(self, context: ContextBundle) -> str:
        if not context.evidence:
            return "No retrieved guideline evidence was available for this arm."
        return " ".join(citation.text for citation in context.evidence)

    def _blocked(self, context: ContextBundle, safe: SafeDecision, safety_text: str) -> Recommendation:
        return Recommendation(
            patient_hash=context.patient["patient_id"],
            status=RecommendationStatus.BLOCKED,
            recommended_arm=None,
            top_scored_arm=safe.decision.recommended_arm,
            q_values=safe.decision.q_values,
            clinician_card=self.rationale.blocked_card(safe, safety_text),
            patient_summary=(
                "The care team needs to review a safety or data-quality issue before a treatment "
                "suggestion is shown."
            ),
            uncertainty=self.rationale.uncertainty_text(safe.decision.uncertainty),
            evidence=context.evidence,
            safety_flags=safe.safety_flags,
            explanation=safe.decision.explanation,
            provenance=safe.provenance | {"agent_layer": "treatmentrx.agent", "schema_validated": True},
        )


__all__ = ["AgentLayer", "EpisodicMemory", "SemanticKnowledgeBase"]
