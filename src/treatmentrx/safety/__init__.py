"""Layer 4 — the code-enforced gate between the statistical engine and language.

Everything here is application code. An LLM layer may *render* a block; it may
never lift one, and no prompt reaches this module. The gate runs before any
narrative is generated, so a blocked recommendation never has an explanation
written for it in the first place.

Two levels of filtering, and both matter:

* **Composite actions** — feasibility is decided over `{drug, dose, route,
  timing, combination}` tuples, not arm labels, so "methotrexate 25mg PO" and
  "tocilizumab + MTX" can be judged on the combination and the dose rather than
  on the name of the arm they belong to.
* **Arms** — an arm survives only if at least one of its composite options does.

If the recommended arm is removed, the decision is **not** silently rewritten to
the runner-up. That behaviour previously turned an arm-naming mismatch into what
looked like a clinical judgement. The patient goes to review with the reason
stated instead.
"""

from __future__ import annotations

from treatmentrx.contracts import Decision, PatientState, SafeDecision
from treatmentrx.domain import RecommendationStatus, SafetyFlag
from treatmentrx.estimation.actions import CompositeActionSpace
from treatmentrx.safety.feasible_set import FeasibleSet
from treatmentrx.safety.rules import SafetyRules


class SafetyLayer:
    def __init__(self) -> None:
        self.action_space = CompositeActionSpace()
        self.feasible_set = FeasibleSet()
        self.rules = SafetyRules()

    def apply(self, decision: Decision, state: PatientState) -> SafeDecision:
        candidates = self.action_space.candidates(list(decision.q_values), state.stages)
        feasibility = self.feasible_set.filter(candidates, state.stages, state.allergies)

        surviving = {
            arm
            for arm, options in candidates.items()
            if any(option in feasibility.feasible for option in options)
        }
        feasible_arms = [arm for arm in decision.q_values if arm in surviving]
        removed_arms = self._removed_arms(candidates, surviving, feasibility)

        flags = self.rules.evaluate(state, decision)
        flags.extend(
            SafetyFlag("arm_removed", "warn", reason, arm) for arm, reason in removed_arms.items()
        )

        status, extra = self._status(decision, feasible_arms)
        flags.extend(extra)
        if any(flag.severity == "block" for flag in flags):
            status = RecommendationStatus.BLOCKED

        return SafeDecision(
            decision=decision,
            feasible_arms=feasible_arms,
            feasible_actions=[action.label for action in feasibility.feasible],
            removed_arms=removed_arms,
            safety_flags=flags,
            status=status,
            provenance={
                "safety_layer": "treatmentrx.safety",
                "patient_hash": state.patient_hash,
                "history_summary": state.history_summary,
                "care_goal": state.care_goal.value,
                "recommended_arm": decision.recommended_arm,
                "feasible_set_size": len(feasibility.feasible),
                "removed_arms": removed_arms,
                "removed_actions": feasibility.removed,
                "versions": state.versions.__dict__,
            },
        )

    def _removed_arms(self, candidates, surviving, feasibility) -> dict[str, str]:
        reasons = dict(feasibility.removed)
        removed: dict[str, str] = {}
        for arm, options in candidates.items():
            if arm in surviving:
                continue
            removed[arm] = "; ".join(
                sorted({reasons[option.label] for option in options if option.label in reasons})
            ) or f"no feasible {arm} option remains"
        return removed

    def _status(self, decision: Decision, feasible_arms: list[str]):
        if not feasible_arms:
            return (
                RecommendationStatus.BLOCKED,
                [SafetyFlag("empty_feasible_set", "block", "No safe treatment arm remains after filtering.")],
            )
        if decision.recommended_arm not in feasible_arms:
            surviving = [
                arm for arm in decision.candidate_arms if arm in set(feasible_arms)
            ]
            # Blocking on an arm that was never recommended.
            #
            # `recommended_arm` is the argmax, and on an equipoise decision Layer 3
            # has already declared it indistinguishable from the rest of the
            # candidate set — there is no recommendation for a contraindication to
            # strike down. Measured over injected scenarios, 12 of 34 blocked
            # cases were this: the patient was going to be told "these four cannot
            # be separated", one of the four turned out to be contraindicated, and
            # the case escalated as though a recommendation had been refused.
            #
            # Routing those to REVIEW is not the substitution invariant 2 forbids.
            # Nothing is re-ranked, nothing is promoted, and `recommended_arm`
            # stays absent — the set was computed before safety and safety may
            # only delete from it. Every other path keeps BLOCKED, because BLOCKED
            # is the only status that stops, and the bug that invariant is about
            # (a naming mismatch removing an arm) must still halt rather than
            # produce a tidy list of alternatives.
            if decision.status is RecommendationStatus.EQUIPOISE and surviving:
                return (
                    RecommendationStatus.REVIEW,
                    [
                        SafetyFlag(
                            "contraindication_without_recommendation",
                            "warn",
                            (
                                f"{decision.recommended_arm} scored highest but is not feasible "
                                "for this patient. No arm had been recommended — the data could "
                                f"not separate {len(decision.candidate_arms)} arms — so removing "
                                "it does not overturn a recommendation. No arm has been "
                                "substituted."
                            ),
                            # Deliberately no claim about the resulting status.
                            # This message said "routed to review, not blocked",
                            # and `apply()` can still upgrade to BLOCKED when a
                            # rule raises a block-severity flag — an allergy
                            # does. The card then read "blocked" in its heading
                            # and "not blocked" in its body. A flag reports what
                            # it observed; the status is decided elsewhere and
                            # says itself.
                            decision.recommended_arm,
                        )
                    ],
                )
            return (
                RecommendationStatus.BLOCKED,
                [
                    SafetyFlag(
                        "recommended_arm_infeasible",
                        "block",
                        (
                            f"{decision.recommended_arm} scored highest but is not feasible for this patient. "
                            "Routed to clinical review rather than substituting the next-best arm."
                        ),
                        decision.recommended_arm,
                    )
                ],
            )
        return decision.status, []


__all__ = ["SafetyLayer"]
