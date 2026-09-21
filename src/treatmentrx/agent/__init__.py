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
    """Layer 5 — renders a `SafeDecision` as a `Recommendation`. No LLM, no agents.

    Three method names claimed otherwise and are gone: `run_agents`,
    `_safety_agent` and `_guideline_agent`. What they named is a `"; ".join`
    over the safety flags and a lookup in a five-passage keyword index. This
    package has never contained a model call, and nothing here plans, chooses or
    acts — Layer 3 decides, Layer 4 may remove, and this composes what they
    produced into text.

    That matters more here than it would elsewhere, because refusing to let a
    name claim more than the arithmetic supports is what invariants 27, 45 and
    51 are each about. A method called `_safety_agent` that concatenates strings
    is the same defect in the one file a reader opens expecting to find an
    agent.

    The layer name stays. `agent/` is a layer in a decision-support *agent*,
    which is ordinary usage for a clinical DSS and is not a claim about how the
    text is produced; renaming the package would churn seventy invariants that
    say "Layer 5" to fix something the methods were already saying wrong.

    What it does hold that is worth reading: `apply_memory` deep-copies the
    statistical output and raises if anything numeric moved (invariant 5), and
    the safety text is formatted verbatim with no authority to reword a block
    (invariant 1).
    """

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

    def compose(self, context: ContextBundle, safe: SafeDecision) -> Recommendation:
        safety_text = self._safety_section(safe)
        if safe.hard_block:
            return self._blocked(context, safe, safety_text)

        evidence_text = self._evidence_section(context)
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
            clinician_card=self.rationale.clinician_card(context, safe, safety_text, evidence_text),
            patient_summary=self.rationale.patient_summary(context, safe),
            uncertainty=self.rationale.uncertainty_text(safe.decision.uncertainty),
            evidence=context.evidence,
            safety_flags=safe.safety_flags,
            explanation=safe.decision.explanation,
            provenance=safe.provenance | {"agent_layer": "treatmentrx.agent", "schema_validated": True},
        )

    def _safety_section(self, safe: SafeDecision) -> str:
        """Formats safety findings verbatim. It has no authority to reword a block."""
        if not safe.safety_flags:
            return "No safety concern was identified by the safety layer."
        return "Safety review: " + "; ".join(
            f"{flag.severity}: {flag.message}" for flag in safe.safety_flags
        )

    def _evidence_section(self, context: ContextBundle) -> str:
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
