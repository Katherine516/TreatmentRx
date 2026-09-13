"""Deterministic rendering of a decision into clinician and patient language.

Offline and dependency-free on purpose: the same context bundle always produces
the same words, so an explanation can be diffed and audited. When a real LLM
replaces this, the constraints it must satisfy are exactly the ones this module
already obeys — every number, every why-not, and every citation is read from the
bundle, never composed.
"""

from __future__ import annotations

from treatmentrx.contracts import ContextBundle, SafeDecision
from treatmentrx.domain import RecommendationStatus, Uncertainty

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
            self._headline(goal, arm, safe),
            (
                f"Model: {decision.selected.estimator} under a {decision.selected.regime_type.value} regime, "
                f"model-averaged over {len(decision.estimates)} estimators "
                f"(weights {self._weights(decision.model_weights)}). "
                f"Q-gap over the next-best arm is {decision.confidence_gap:.3f}, against a "
                f"{decision.goal_decision.threshold:.2f} bar for this care goal."
            ),
            f"Tailoring drivers: {drivers}.",
            self._separation(decision),
            self._candidate_set(safe),
            self._basis_caveat(decision),
            self._why_not(decision),
            self._attribution(safe),
            safety_text,
            f"Evidence: {guideline_text}",
            f"Uncertainty: {self.uncertainty_text(decision.uncertainty)}",
        ]
        sections.extend(self._memory_sections(context))
        return "\n\n".join(section for section in sections if section)

    @staticmethod
    def _safety_review(safe: SafeDecision) -> bool:
        """Was REVIEW raised by a contraindication rather than by an out-of-
        distribution patient?

        REVIEW used to have exactly one cause, so both narratives named it in
        prose. Now that the safety layer can also raise it — when a
        contraindication lands on an arm that was never recommended — that prose
        is wrong for half the cases, and telling a patient their record is
        unusual when in fact a drug is contraindicated is a specific and
        avoidable falsehood.
        """
        return any(
            flag.code == "contraindication_without_recommendation"
            for flag in safe.safety_flags
        )

    def _headline(self, goal: str, arm: str, safe: SafeDecision) -> str:
        """Lead with what the system decided, not with the top-scored arm.

        The card opened "recommend {arm}" regardless of status, so an equipoise
        case — where the whole point is that the system is declining to pick —
        read as a recommendation in its first line and was qualified only several
        paragraphs later.
        """
        if safe.status is RecommendationStatus.EQUIPOISE:
            return (
                f"{goal}: EQUIPOISE — no arm is separated well enough to recommend. "
                f"{arm} scored highest; {safe.decision.rationale}"
            )
        if safe.status is RecommendationStatus.REVIEW:
            if self._safety_review(safe):
                return (
                    f"{goal}: MANUAL REVIEW — no arm was separated well enough to "
                    f"recommend, and {arm}, which scored highest, is not feasible for "
                    "this patient. Nothing has been substituted in its place."
                )
            return (
                f"{goal}: MANUAL REVIEW required, no recommendation issued. "
                f"{safe.decision.rationale}"
            )
        return f"{goal}: recommend {arm}."

    def patient_summary(self, context: ContextBundle, safe: SafeDecision) -> str:
        arm = safe.decision.recommended_arm
        if safe.status is RecommendationStatus.EQUIPOISE:
            return (
                "The model did not find enough separation between the leading treatment "
                "options to suggest one. The care team should compare the options using "
                "clinical judgement, safety considerations, and your preferences."
            )
        if safe.status is RecommendationStatus.REVIEW:
            if self._safety_review(safe):
                return (
                    "No treatment suggestion is being shown. The model could not tell "
                    "the leading options apart, and one of them is not safe to use in "
                    "your situation, so the care team is reviewing the remaining "
                    "options rather than the model picking one for you."
                )
            return (
                "No treatment suggestion is being shown because this health record falls "
                "outside the patient patterns on which the model was evaluated. The care "
                "team should review the case without relying on the model's ranking."
            )
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

    def _separation(self, decision) -> str:
        """State whether the data can actually resolve the top two arms.

        The verdict reads `robustly_distinguishable`, the same property the
        decision layer used. Reading `distinguishable` here instead let the card
        announce "separable at this sample size" on a case the layer had already
        called equipoise, because the separation did not survive widening the
        interval to the width coverage says it should have. A clinician reading
        both lines got two different answers to the same question — and the more
        confident of the two was the one the system had rejected.
        """
        contrast = decision.contrast
        if contrast is None:
            return ""
        confidence = int((1 - contrast.alpha) * 100)
        if contrast.robustly_distinguishable:
            verdict = "separable at this sample size"
        elif contrast.distinguishable:
            verdict = (
                "NOT separable — the interval excludes zero, but the separation does not "
                "survive widening the interval to the width measured coverage requires"
            )
        else:
            verdict = "NOT separable — the interval includes zero"
        caveat = f" {contrast.caveat}" if contrast.caveat else ""
        return (
            f"Separation: {contrast.arm} over {contrast.comparator} is "
            f"{contrast.difference:+.3f} (SE {contrast.standard_error:.3f}, {confidence}% CI "
            f"[{contrast.lower:+.3f}, {contrast.upper:+.3f}]) — {verdict}.{caveat}"
        )

    def blocked_card(self, safe: SafeDecision, safety_text: str) -> str:
        """What a reviewer needs, on the one status that demands a human.

        BLOCKED used to render as a single interpolated line — the block reason
        and nothing else. That is the worst information-per-need ratio in the
        system: the case that by definition escalates to a person gave that
        person the least to work with, while the candidate set, the contrast and
        the removals were all computed and discarded.

        What it must *not* become is a menu. Nothing here is offered, nothing is
        ranked, and the pre-safety set is labelled as superseded rather than
        presented as alternatives — a blocked case has no recommendation and this
        card must not read as though it does.
        """
        decision = safe.decision
        sections = [f"BLOCKED — no treatment is being suggested. {safety_text}"]

        if decision.candidate_arms:
            considered = ", ".join(decision.candidate_arms)
            plural = len(decision.candidate_arms) != 1
            sections.append(
                f"What the model had been considering, before safety: {considered}. "
                + (
                    "These arms were not separable from each other at this sample size. "
                    if plural
                    else "The model had separated this arm from the rest of the menu. "
                )
                + "This is context for the review, not a list of alternatives — the "
                "block stands and no arm here has been recommended."
            )

        if safe.removed_arms:
            sections.append(
                "Removed by the safety layer: "
                + "; ".join(f"{arm} ({reason})" for arm, reason in sorted(safe.removed_arms.items()))
                + "."
            )

        separation = self._separation(decision)
        if separation:
            sections.append(separation)

        sections.append(
            f"Uncertainty: {self.uncertainty_text(decision.uncertainty)}"
        )
        return "\n\n".join(sections)

    def _candidate_set(self, safe: SafeDecision) -> str:
        """What the agent can still say when it will not name one arm.

        The agent declines for most patients it sees, and until now that produced
        a status and nothing to act on — the clinician still has to prescribe, and
        was left choosing from the whole menu. The arms here are the ones the data
        cannot separate from the leader; every other arm has been ruled out by the
        same interval the separation line above reports. Measured over declined
        patients the set averages 2.5 of 6 arms, and choosing the worst of them
        rather than the worst of all six cuts worst-case regret by 83%.

        **Filtered by the safety layer, and that is not substitution.** Layer 4
        has already removed infeasible arms; a set printed for a clinician must
        not contain one of them. Nothing is re-ranked and nothing is promoted —
        removals are named, and an arm inside this set has not been recommended.
        """
        decision = safe.decision
        feasible = set(safe.feasible_arms)
        candidates = [arm for arm in decision.candidate_arms if arm in feasible]
        if len(decision.candidate_arms) < 2:
            return ""
        withheld = [arm for arm in decision.candidate_arms if arm not in feasible]
        excluded = len(decision.q_values) - len(decision.candidate_arms)

        if not candidates:
            body = (
                "Cannot separate: no feasible arm remains. The data could not "
                f"distinguish {len(decision.candidate_arms)} arms, and the safety "
                "layer removed all of them."
            )
        elif len(candidates) == 1:
            # One survivor is not a recommendation. It is what is left after
            # safety pruned a set the model could not rank, and saying so is the
            # difference between reporting a set and promoting an arm.
            body = (
                f"Only {candidates[0]} remains in contention. The data could not "
                f"separate it from {len(withheld)} other arm"
                f"{'s' if len(withheld) != 1 else ''} that the safety layer then "
                "removed, so it stands by elimination rather than by evidence — "
                "it has not been recommended."
            )
        else:
            body = (
                f"Cannot separate: {', '.join(candidates)}. The data does not "
                "distinguish these from each other at this sample size"
                + (
                    f"; the other {excluded} arm{'s' if excluded != 1 else ''} on the "
                    f"menu {'were' if excluded != 1 else 'was'} ruled out."
                    if excluded
                    else "."
                )
                + " This is not a recommendation — it is the narrowest defensible "
                "set, and the choice within it should turn on tolerability, route, "
                "comorbidity and patient preference rather than on these numbers."
            )

        lines = [body]
        if withheld:
            lines.append(
                "Removed from this set by the safety layer: "
                + "; ".join(
                    f"{arm} ({safe.removed_arms.get(arm, 'infeasible')})" for arm in withheld
                )
                + "."
            )
        return " ".join(lines)

    def _basis_caveat(self, decision) -> str:
        """Say it on the card when the model's own basis is in question.

        This is the one caveat that undercuts the separation line above it rather
        than qualifying it. If a covariate the blip basis omits tests as an effect
        modifier, the contrast reported is the covariate-averaged one and its
        interval does not cover — measured at 29% against a nominal 95% in
        `cli misspecification --omitted-modifier`. A reader who acts on the
        interval needs that on the same page, not in a command they never run.
        """
        omitted = [
            flag.split(":", 1)[1]
            for flag in decision.uncertainty.flags
            if flag.startswith("blip_basis_may_omit:")
        ]
        if not omitted:
            return ""
        return (
            "CAVEAT — the treatment-effect model may be missing an effect modifier: "
            + ", ".join(omitted)
            + ". The separation above is averaged over that covariate rather than "
            "specific to this patient, and its interval does not account for the "
            "difference. Treat the magnitude as indicative until the basis is refit."
        )

    def _why_not(self, decision) -> str:
        """Why the *ruled-out* arms were ruled out — never one still in contention.

        `WHY_NOT_REASONS` is hard-coded clinical prose hung on a model-derived
        Q-gap, and it used to be printed for the top two runners-up regardless of
        whether the model could actually separate them. Beside the candidate set
        that produced a flat contradiction: the card said "cannot separate: IL-6,
        rituximab, methotrexate" and then, two lines down, "why not rituximab —
        reserved for seropositive context; why not methotrexate — lower-yield
        after prior failure." The model had ruled out neither. Explaining away an
        arm the data cannot exclude is the card asserting a clinical judgement the
        model did not make, which is the one thing this layer must not do.

        Restricting it to arms outside the candidate set also makes the prose
        honest about its own role: those arms were excluded *statistically*, and
        the sentence attached is colour, not the reason.
        """
        entries = decision.explanation.why_not if decision.explanation else []
        in_contention = set(decision.candidate_arms)
        ruled_out = [entry for entry in entries if entry.action not in in_contention][:2]
        if not ruled_out:
            return ""
        rendered = "; ".join(
            f"{entry.action} (gap {entry.q_gap:.3f}) — {entry.dominant_reason}"
            for entry in ruled_out
        )
        return f"Why not the alternatives: {rendered}."

    def _attribution(self, safe: SafeDecision) -> str:
        """Report the blip decomposition, which sums exactly to the arm's advantage.

        The subject is the top-scored arm, which on a review card is sometimes the
        arm safety just removed. Stating its advantage unqualified reads as
        advocacy for something the patient must not receive, so when that is the
        case the line says so. It is kept rather than dropped because the reviewer
        needs to see *why* the model ranked a contraindicated arm first.
        """
        decision = safe.decision
        if not decision.explanation or not decision.explanation.attributions:
            return ""
        attribution = decision.explanation.attributions[0]
        if not attribution.contributions:
            return ""
        parts = ", ".join(f"{name} {value:+.3f}" for name, value in attribution.contributions.items())
        line = (
            f"Estimated advantage of {attribution.action} over continuing current therapy is "
            f"{attribution.total_advantage:+.3f}, from {parts}."
        )
        if attribution.action in safe.removed_arms:
            line += (
                f" This is why the model scored {attribution.action} highest; it has "
                "been removed by the safety layer and is not available for this "
                "patient."
            )
        return line

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
