from __future__ import annotations

from precisionrx_agent.layer9_memory_rag.memory import EpisodicMemory, SemanticKnowledgeBase
from treatmentrx.contracts import Citation, ContextBundle, Recommendation, RecommendationStatus, SafeDecision, VersionSet


class AgentLayer:
    """Layer 5: context assembly, RAG/memory, and schema-validated recommendation."""

    def __init__(self) -> None:
        self.memory = EpisodicMemory()
        self.knowledge_base = SemanticKnowledgeBase()

    def build_context(self, safe: SafeDecision) -> ContextBundle:
        patient_hash = safe.provenance.get("patient_hash", "unknown")
        arm = safe.decision.recommended_arm
        evidence = [
            Citation(source="RA knowledge base", text=passage, url=None)
            for passage in self.knowledge_base.retrieve(f"{arm} rheumatoid arthritis", k=3)
        ]
        memory = {
            "prior_context": self.memory.recent_summary(patient_hash),
            "framing_hints": self.memory.preferences(patient_hash),
            "influence": "narrative_and_retrieval_only",
        }
        return ContextBundle(
            patient={
                "patient_hash": patient_hash,
                "history_summary": safe.provenance.get("history_summary", ""),
            },
            decision={
                "recommended_arm": arm,
                "q_values": safe.decision.q_values,
                "model_weights": safe.decision.model_weights,
                "rationale": safe.decision.rationale,
            },
            safety={
                "status": safe.status.value,
                "flags": [flag.__dict__ for flag in safe.safety_flags],
                "feasible_arms": safe.feasible_arms,
            },
            evidence=evidence,
            memory=memory,
            versions=VersionSet(**safe.provenance.get("versions", {})),
        )

    def run_agents(self, context: ContextBundle, safe: SafeDecision) -> Recommendation:
        safety_text = self._safety_agent(context, safe)
        if safe.hard_block:
            return self._blocked_recommendation(context, safe, safety_text)

        guideline_text = self._guideline_agent(context)
        clinician_card = self._synthesis_agent(context, safe, safety_text, guideline_text)
        patient_summary = (
            f"The care team may consider {safe.decision.recommended_arm}. This recommendation should be reviewed "
            "with your clinician, including expected benefits, safety monitoring, and your preferences."
        )
        return Recommendation(
            patient_hash=context.patient["patient_hash"],
            status=safe.status,
            clinician_card=clinician_card,
            patient_summary=patient_summary,
            recommended_arm=safe.decision.recommended_arm,
            q_values=safe.decision.q_values,
            uncertainty=self._uncertainty_text(safe),
            evidence=context.evidence,
            safety_flags=safe.safety_flags,
            provenance=safe.provenance | {"agent_layer": "treatmentrx.agent", "schema_validated": True},
        )

    def _safety_agent(self, context: ContextBundle, safe: SafeDecision) -> str:
        if not safe.safety_flags:
            return "No hard safety block was identified."
        flags = "; ".join(f"{flag.severity}: {flag.message}" for flag in safe.safety_flags)
        return f"Safety review: {flags}"

    def _guideline_agent(self, context: ContextBundle) -> str:
        if not context.evidence:
            return "No retrieved guideline evidence was available for this arm."
        return " ".join(citation.text for citation in context.evidence)

    def _synthesis_agent(self, context: ContextBundle, safe: SafeDecision, safety_text: str, guideline_text: str) -> str:
        arm = safe.decision.recommended_arm
        q_values = safe.decision.q_values
        next_best = self._next_best(arm, q_values)
        gap = q_values.get(arm, 0.0) - q_values.get(next_best, 0.0)
        return (
            f"Recommendation: {arm}.\n\n"
            f"Model rationale: {safe.decision.rationale} The model-averaged Q-value gap versus {next_best} is {gap:.3f}. "
            f"Model weights were {safe.decision.model_weights}.\n\n"
            f"{safety_text}\n\n"
            f"Evidence summary: {guideline_text}\n\n"
            f"Uncertainty: {self._uncertainty_text(safe)}"
        )

    def _blocked_recommendation(self, context: ContextBundle, safe: SafeDecision, safety_text: str) -> Recommendation:
        return Recommendation(
            patient_hash=context.patient["patient_hash"],
            status=RecommendationStatus.BLOCKED,
            clinician_card=f"Recommendation blocked. {safety_text}",
            patient_summary="The care team needs to review a safety or data-quality issue before a treatment suggestion is shown.",
            recommended_arm=None,
            q_values=safe.decision.q_values,
            uncertainty=self._uncertainty_text(safe),
            evidence=context.evidence,
            safety_flags=safe.safety_flags,
            provenance=safe.provenance | {"agent_layer": "treatmentrx.agent", "schema_validated": True},
        )

    def _next_best(self, arm: str, q_values: dict[str, float]) -> str:
        for candidate, _ in sorted(q_values.items(), key=lambda item: item[1], reverse=True):
            if candidate != arm:
                return candidate
        return "manual-review"

    def _uncertainty_text(self, safe: SafeDecision) -> str:
        unc = safe.decision.uncertainty
        flags = ", ".join(unc.flags) if unc.flags else "none"
        return (
            f"aleatoric={unc.aleatoric:.3f}; epistemic={unc.epistemic:.3f}; "
            f"model={unc.model:.4f}; ood={unc.ood:.3f}; calibrated={unc.calibrated}; flags={flags}"
        )
