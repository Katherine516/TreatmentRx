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
    def decide(self, result: RegimeEstimate, care_goal: CareGoal) -> GoalDecision:
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
