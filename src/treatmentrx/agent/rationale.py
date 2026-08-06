"""Deterministic rendering of a decision into clinician and patient language.

Offline and dependency-free on purpose: the same context bundle always produces
the same words, so an explanation can be diffed and audited. When a real LLM
replaces this, the constraints it must satisfy are exactly the ones this module
already obeys — every number, every why-not, and every citation is read from the
bundle, never composed.
"""

from __future__ import annotations

from treatmentrx.contracts import ContextBundle, SafeDecision
from treatmentrx.domain import Uncertainty

GOAL_FRAMING = {
    "induction": "Goal is induction (drive disease activity down)",
    "maintenance": "Goal is maintenance (hold remission, minimise burden)",
    "tox_control": "Goal is toxicity control (back off, manage adverse effects)",
    "qol": "Goal is quality of life (comfort and preference weighted)",
}


class RationaleGenerator:
    def clinician_card(
        self,
        context: ContextBundle,
        safe: SafeDecision,
        safety_text: str,
        guideline_text: str,
    ) -> str:
        decision = safe.decision
        arm = decision.recommended_arm
        goal = GOAL_FRAMING.get(context.patient.get("care_goal") or "", "Goal not classified")
        drivers = ", ".join(context.patient["top_tailoring_vars"] or ["stage history"])

        sections = [
            f"{goal}: recommend {arm}.",
            (
                f"Model: {decision.selected.estimator} under a {decision.selected.regime_type.value} regime, "
                f"model-averaged over {len(decision.estimates)} estimators "
                f"(weights {self._weights(decision.model_weights)}). "
                f"Q-gap over the next-best arm is {decision.confidence_gap:.3f}, against a "
                f"{decision.goal_decision.threshold:.2f} bar for this care goal."
            ),
            f"Tailoring drivers: {drivers}.",
            self._why_not(decision),
            self._attribution(decision),
            safety_text,
            f"Evidence: {guideline_text}",
            f"Uncertainty: {self.uncertainty_text(decision.uncertainty)}",
        ]
        sections.extend(self._memory_sections(context))
        return "\n\n".join(section for section in sections if section)

    def patient_summary(self, context: ContextBundle, safe: SafeDecision) -> str:
        arm = safe.decision.recommended_arm
        hedge = (
            " The care team should discuss this carefully, because the options are close."
            if not safe.decision.goal_decision.act
            else ""
        )
        return (
            f"The care team may consider {arm}. This suggestion comes from your recent treatment "
            f"history and disease activity pattern, and it is a starting point for a conversation, "
            f"not a decision on its own.{hedge} At the next visit the team should check symptoms, "
            "lab response, side effects, and whether the plan still fits your goals."
        )

    def uncertainty_text(self, uncertainty: Uncertainty) -> str:
        flags = ", ".join(uncertainty.flags) if uncertainty.flags else "none"
        return (
            f"aleatoric={uncertainty.aleatoric:.3f}; epistemic={uncertainty.epistemic:.3f}; "
            f"model={uncertainty.model:.4f}; ood={uncertainty.ood:.3f}; "
            f"calibrated={uncertainty.calibrated}; flags={flags}"
        )

    def _why_not(self, decision) -> str:
        entries = decision.explanation.why_not[:2] if decision.explanation else []
        if not entries:
            return ""
        rendered = "; ".join(
            f"{entry.action} (gap {entry.q_gap:.3f}) — {entry.dominant_reason}" for entry in entries
        )
        return f"Why not the alternatives: {rendered}."

    def _attribution(self, decision) -> str:
        """Report the blip decomposition, which sums exactly to the arm's advantage."""
        if not decision.explanation or not decision.explanation.attributions:
            return ""
        attribution = decision.explanation.attributions[0]
        if not attribution.contributions:
            return ""
        parts = ", ".join(f"{name} {value:+.3f}" for name, value in attribution.contributions.items())
        return (
            f"Estimated advantage of {attribution.action} over continuing current therapy is "
            f"{attribution.total_advantage:+.3f}, from {parts}."
        )

    def _memory_sections(self, context: ContextBundle) -> list[str]:
        memory = context.memory or {}
        sections = []
        prior = memory.get("prior_context", "")
        if prior and "No prior" not in prior:
            sections.append(f"Continuity: {prior}")
        if memory.get("framing_hints"):
            sections.append("Recorded patient preferences: " + "; ".join(memory["framing_hints"]))
        return sections

    def _weights(self, weights: dict[str, float]) -> str:
        if not weights:
            return "unavailable"
        return ", ".join(f"{name} {value:.2f}" for name, value in sorted(weights.items()))


__all__ = ["RationaleGenerator"]
