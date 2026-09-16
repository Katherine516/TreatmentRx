"""v5.1 #6 (L3) — Goal-conditioned thresholds.

The recommend/abstain/escalate threshold is conditioned on care_goal. The same
Q-gap means different things in induction vs QoL phases: in induction a small
gap can still warrant a confident recommendation; in QoL the bar for acting
(vs preserving comfort) is higher.
"""

from __future__ import annotations

from treatmentrx.contracts import GoalDecision, RegimeEstimate
from treatmentrx.domain import CareGoal


# Minimum Q-gap required to make a confident recommendation, per goal.
GOAL_ACTION_THRESHOLD = {
    CareGoal.INDUCTION: 0.02,
    CareGoal.MAINTENANCE: 0.05,
    CareGoal.TOXICITY_CONTROL: 0.08,
    CareGoal.QUALITY_OF_LIFE: 0.10,
}


class GoalConditionedThresholds:
    def decide(
        self,
        result: RegimeEstimate,
        care_goal: CareGoal,
        observed_gap: float | None = None,
    ) -> GoalDecision:
        """Is the top-two difference large enough to be worth acting on?

        `observed_gap` is the decision's own model-averaged contrast, passed in
        because deriving it here from `q_values` reads a **display** quantity.
        Those values are clamped to `[Q_FLOOR, Q_CEILING]` and rounded to three
        decimals, and both steps are many-to-one: measured over 120 patients,
        every one of the nine whose predicted response saturated the ceiling had
        a top-two gap of **exactly 0.0000** — arms the model does distinguish,
        reported to the clinician as identical and failing the action bar for an
        arithmetic reason.

        It also makes invariant 10 exact. The care goal sets how large a
        difference is worth acting on and the interval decides whether the data
        can resolve *a difference that size*; if the two are computed from
        different quantities, "that size" is ambiguous. They now read the same
        contrast.

        The fallback keeps working for callers with no contrast — `cli
        misspecification` and the subgroup sweep both construct estimates
        directly — and is the old behaviour, clamp and all.
        """
        if observed_gap is not None:
            # Signed, not absolute. The contrast is `leader - runner-up` and can
            # come out negative in the one honest case where the two serving
            # members disagree and the weighted vote picks the leader; a leader
            # that scores *below* its comparator should fail an action bar
            # directly rather than clear it on magnitude and be caught later by
            # the interval condition.
            q_gap = round(observed_gap, 4)
        else:
            ordered = sorted(result.q_values.values(), reverse=True)
            q_gap = round(ordered[0] - ordered[1], 4) if len(ordered) > 1 else 0.0
        threshold = GOAL_ACTION_THRESHOLD[care_goal]
        act = q_gap >= threshold
        if act:
            rationale = f"Q-gap {q_gap:.3f} clears the {care_goal.value} action bar ({threshold:.2f})."
        else:
            rationale = (
                f"Q-gap {q_gap:.3f} is below the {care_goal.value} action bar ({threshold:.2f}); "
                "prefer continuity / equipoise framing."
            )
        return GoalDecision(
            act=act,
            threshold=threshold,
            observed_gap=q_gap,
            care_goal=care_goal,
            rationale=rationale,
        )
