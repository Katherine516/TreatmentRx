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
                f"Gap over the nearest arm is {decision.confidence_gap:.3f}, against a "
                f"{decision.goal_decision.threshold:.2f} bar for this care goal."
            ),
            f"Tailoring drivers: {drivers}.",
            self._separation(decision),
            self._candidate_set(safe),
            self._basis_caveat(decision),
            self._why_not(safe),
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
        # Only RECOMMEND reaches here — EQUIPOISE and REVIEW returned above, and
        # BLOCKED never calls this method (`AgentLayer.run_agents` branches on
        # `hard_block` first). This used to append " ...because the options are
        # close" when `goal_decision.act` was False, and that cannot happen:
        # `DecisionLayer._status` returns RECOMMEND only after `act` is True, so
        # the branch was unreachable by construction and measured 41/41 on the
        # audit cohort. It is removed rather than rewired. The 51 patients who
        # clear the care-goal bar but fail the interval condition are exactly the
        # ones for whom "the options are close" is true, and they are already
        # sent to the EQUIPOISE summary, which says it. A hedge that can only
        # fire where it is wrong is invariant 25 in patient-facing prose.
        return (
            f"The care team may consider {arm}. This suggestion comes from your recent treatment "
            f"history and disease activity pattern, and it is a starting point for a conversation, "
            "not a decision on its own. At the next visit the team should check symptoms, "
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
        """State whether the data can actually resolve the leader and its nearest arm.

        *Nearest by estimated contrast*, not by the display `q_values` — see
        `DecisionLayer._contrast`. It is not in general the second-highest-scoring
        arm, and it is not in general the least separable one either; the
        candidate set covers every arm and `_not_excluded` reports the gap.

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
        # The caveat travels with the line it undercuts. `blocked_card` carried
        # the separation interval and not the warning that the interval may be
        # centred on the wrong quantity, so the one reader who is by definition a
        # person got the number without its qualifier. It cannot fire on this
        # build — `deployment_readiness()["blip_basis_unflagged"]` is True and
        # `das28_squared` tests at 3.367 against a 3.669 threshold — so this is a
        # latent gap closed by inspection, not a measured one.
        caveat = self._basis_caveat(decision)
        if caveat:
            sections.append(caveat)

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
        patients the set averages 2.7 of 6 arms, and choosing the worst of them
        rather than the worst of all six cuts worst-case regret from 0.206 to
        0.047, a 77% reduction. (These read 2.5 and 83% until the all-pairs
        correction widened the set; a wider set is safer to be inside and less
        decisive to choose within.)

        **Filtered by the safety layer, and that is not substitution.** Layer 4
        has already removed infeasible arms; a set printed for a clinician must
        not contain one of them. Nothing is re-ranked and nothing is promoted —
        removals are named, and an arm inside this set has not been recommended.

        **The set is not empty on a recommendation, and the prose used not to
        know that.** Status is decided on the top-two contrast alone, so a lower-
        scoring arm with a wider interval can survive the exclusion test while
        the runner-up fails it: measured over 120 patients, **6 (5%)** were
        recommended with a two-arm set. Those cards read "recommend
        methotrexate-optimization" in the headline and, four paragraphs down,
        "Cannot separate: methotrexate-optimization, rituximab ... This is not a
        recommendation" — invariant 35's defect, in the block written to fix it.

        The information is right and only the framing was wrong, which the oracle
        settles: on all four of the clearest cases the leader and the surviving
        arm have a **true value of 1.0000 each** — genuinely tied at the optimum —
        while the comparator the separation line names is worth 0.96-0.99. The
        set is correct, the recommendation is correct (oracle arm, zero regret),
        and what was missing was a sentence saying how both can hold. So the
        block is kept and re-worded per status rather than suppressed: hiding it
        would make the card more confident than the evidence, which is the
        direction these notes warn about.
        """
        decision = safe.decision
        feasible = set(safe.feasible_arms)
        candidates = [arm for arm in decision.candidate_arms if arm in feasible]
        if len(decision.candidate_arms) < 2:
            return ""
        withheld = [arm for arm in decision.candidate_arms if arm not in feasible]
        excluded = len(decision.q_values) - len(decision.candidate_arms)

        if safe.status is RecommendationStatus.RECOMMEND:
            body = self._not_excluded(decision, candidates)
        elif not candidates:
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

        lines = [body] if body else []
        if withheld:
            lines.append(
                "Removed from this set by the safety layer: "
                + "; ".join(
                    f"{arm} ({safe.removed_arms.get(arm, 'infeasible')})" for arm in withheld
                )
                + "."
            )
        return " ".join(lines)

    @staticmethod
    def _not_excluded(decision, candidates: list[str]) -> str:
        """The same set, said in a way a recommendation can carry.

        Two claims, and they are about different pairs. The recommendation rests
        on the leader separating from the arm named in the separation line —
        the *nearest* arm by estimated contrast. These arms sit further off on
        that same estimate and were still not excluded, because their intervals
        are wider and contain zero: the smallest difference is not the smallest
        z. Saying so is not a hedge on the recommendation and not an invitation
        to override it — it is the rest of what the same intervals support, and a
        clinician who has to weigh route or tolerability is the reader who needs
        it.
        """
        others = [arm for arm in candidates if arm != decision.recommended_arm]
        if not others:
            return ""
        excluded = len(decision.q_values) - len(decision.candidate_arms)
        # Named when there is one. Without a contrast the decision reached
        # RECOMMEND on the care-goal bar alone, and there is no pair to name.
        separated_from = (
            f"{decision.contrast.comparator}, the nearest arm the data can compare it against"
            if decision.contrast
            else "the nearest arm the data can compare it against"
        )
        plural = len(others) != 1
        return (
            f"Not excluded: {', '.join(others)}. The recommendation above rests on "
            f"{decision.recommended_arm} separating from {separated_from}. "
            f"{'These arms sit' if plural else 'This arm sits'} further off on that "
            f"same estimate, but with a wider interval: "
            f"{'their' if plural else 'its'} own interval against "
            f"{decision.recommended_arm} contains zero, so the data does not rule "
            f"{'them' if plural else 'it'} out"
            + (
                f"; the other {excluded} arm{'s' if excluded != 1 else ''} on the menu "
                f"{'were' if excluded != 1 else 'was'} excluded."
                if excluded
                else "."
            )
            + " The recommendation stands on the comparison reported above — this is "
            "the rest of what the data leaves open, not a competing suggestion."
        )

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

    def _why_not(self, safe: SafeDecision) -> str:
        """Why the *statistically* ruled-out arms were ruled out — and only those.

        `WHY_NOT_REASONS` *was* hard-coded clinical prose hung on a model-derived
        gap — it is gone, and `WhyNotEntry.dominant_reason` is now that gap's own
        per-covariate decomposition (invariant 57). The filtering below is what
        survives of two earlier fixes, and both still matter, because deciding
        *which* arms may be explained at all is a separate question from whether
        the explanation is the model's.

        It used to be printed for the top two runners-up regardless of
        whether the model could actually separate them. Beside the candidate set
        that produced a flat contradiction: the card said "cannot separate: IL-6,
        rituximab, methotrexate" and then, two lines down, "why not rituximab —
        reserved for seropositive context; why not methotrexate — lower-yield
        after prior failure." The model had ruled out neither. Explaining away an
        arm the data cannot exclude is the card asserting a clinical judgement the
        model did not make, which is the one thing this layer must not do.

        **The same defect from the other side: an arm Layer 4 removed.** The
        filter read the candidate set alone, so a contraindicated arm — excluded
        by a rule, not by an interval — still collected a model reason. With
        pregnancy injected, **79 of 240** cards carried one, and the words were
        the wrong kind of wrong: "why not JAK-inhibitor — organ-function / safety
        profile reduces net benefit" for an arm the card elsewhere reports as
        contraindicated in pregnancy, and "why not methotrexate-optimization —
        csDMARD optimization is lower-yield after prior failure" for another. A
        contraindicated arm is not an option that lost on merit, and colour prose
        saying it narrowly lost reads as though it could be reconsidered.

        Nothing is lost by dropping them: the safety layer raises an
        `arm_removed` flag for every arm it removes, so each one is already named
        on this card with the reason that actually applies.

        Restricting it this way also keeps the block honest about its own scope:
        these arms were excluded *statistically*, so the decomposition attached
        is why the model ranked them below the leader — not a clinical argument
        against them, and not a safety finding, which is a different block.
        """
        decision = safe.decision
        entries = decision.explanation.why_not if decision.explanation else []
        # In contention, or removed by a rule rather than by an interval.
        unexplainable = set(decision.candidate_arms) | set(safe.removed_arms)
        ruled_out = [entry for entry in entries if entry.action not in unexplainable][:2]
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

        **It names its model, because this one is not the ensemble.** The line
        above it says the decision was model-averaged over two estimators, and a
        reader carries that down the card — but psi cannot be averaged across
        members that parameterise it differently, so `BayesianModelAverager`
        carries one member's coefficients and stamps which. Which one turns on a
        weight margin of 0.003 on the deployed fit, and the two members' blips
        for the same patient differ by up to 0.041 — the size of the contrast the
        decision reports. Attributing that to "the model" unqualified is the card
        claiming an ensemble quantity it does not have.
        """
        decision = safe.decision
        if not decision.explanation or not decision.explanation.attributions:
            return ""
        attribution = decision.explanation.attributions[0]
        if not attribution.contributions:
            return ""
        parts = ", ".join(f"{name} {value:+.3f}" for name, value in attribution.contributions.items())
        source = decision.selected.coefficients.get("attribution_source")
        attributed_to = f" ({source}'s blip, not the ensemble average)" if source else ""
        line = (
            f"Estimated advantage of {attribution.action} over continuing current therapy is "
            f"{attribution.total_advantage:+.3f}{attributed_to}, from {parts}."
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
