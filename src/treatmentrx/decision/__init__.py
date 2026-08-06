"""Layer 3 — `RegimeEstimate`s to one `Decision`.

Order is not arbitrary. Model averaging comes first so everything after it acts
on a single estimate. The competing-risk endpoint and the belief adjustment then
correct that estimate for things the estimators do not see (a trajectory
dominated by death or dropout is worth less; an uncertain belief about latent
disease activity should widen the band). Only then is the Q-gap compared against
the threshold the care goal demands.

This layer decides; it does not gate. `status` here is a statistical reading
(equipoise, out-of-distribution) — a *clinical* block is Layer 4's alone.
"""

from __future__ import annotations

from treatmentrx.contracts import Decision, PatientState, RegimeEstimate
from treatmentrx.decision.bma import BayesianModelAverager
from treatmentrx.decision.uncertainty import UncertaintyDecomposer
from treatmentrx.domain import RecommendationStatus, Uncertainty
from treatmentrx.estimation import training
from treatmentrx.estimation.belief_aware import BeliefAwareAdjuster
from treatmentrx.estimation.competing_risk_outcomes import CompetingRiskEndpoint
from treatmentrx.estimation.explainability import ModelExplainer
from treatmentrx.estimation.dwols import DWOLSSharedEstimator
from treatmentrx.estimation.estimators import QSharedEstimator, StageSpecificQEstimator
from treatmentrx.estimation.goal_conditioned import GoalConditionedThresholds

OOD_REVIEW_THRESHOLD = 0.75


class DecisionLayer:
    def __init__(self) -> None:
        self.bma = BayesianModelAverager()
        self.competing_endpoint = CompetingRiskEndpoint()
        self.belief_adjuster = BeliefAwareAdjuster()
        self.goal_thresholds = GoalConditionedThresholds()
        self.uncertainty = UncertaintyDecomposer()
        self.explainer = ModelExplainer()
        self.estimators = (QSharedEstimator(), DWOLSSharedEstimator(), StageSpecificQEstimator())

    def decide(self, state: PatientState, estimates: list[RegimeEstimate]) -> Decision:
        if not estimates:
            return self._no_estimate_decision()

        selected = self.bma.aggregate(estimates)
        selected = self.competing_endpoint.adjust(selected, state.stages, state.competing_risk_incidence)
        selected = self.belief_adjuster.adjust(selected, state.stages)

        goal_decision = self.goal_thresholds.decide(selected, state.care_goal)
        contrast = self._contrast(state, selected)
        uncertainty = self.uncertainty.decompose(
            state.stages,
            selected,
            estimates,
            state.encoded_state,
            training.holdout_calibration(),
            contrast,
        )
        explanation = self.explainer.explain(selected, estimates, state.stages)

        status, rationale = self._status(state, uncertainty, goal_decision, contrast)
        return Decision(
            recommended_arm=selected.recommended_arm,
            q_values=selected.q_values,
            model_weights={
                name.split(":", 1)[1]: value
                for name, value in selected.coefficients.items()
                if name.startswith("bma_weight:")
            },
            uncertainty=uncertainty,
            status=status,
            rationale=rationale,
            estimates=estimates,
            selected=selected,
            goal_decision=goal_decision,
            explanation=explanation,
            confidence_gap=goal_decision.observed_gap,
            contrast=contrast,
        )

    def _contrast(self, state: PatientState, selected: RegimeEstimate):
        """Test the recommended arm against the runner-up.

        Each estimator that supports inference is asked, and the *widest*
        interval wins. If any of them cannot separate the two arms, the system
        does not claim separation — the conservative direction for a decision a
        clinician will act on.
        """
        ordered = sorted(selected.q_values, key=selected.q_values.get, reverse=True)
        if len(ordered) < 2:
            return None
        arm, comparator = ordered[0], ordered[1]
        tests = []
        for estimator in self.estimators:
            try:
                tests.append(estimator.contrast(state.stages, arm, comparator))
            except (KeyError, ValueError):
                continue
        if not tests:
            return None
        return max(tests, key=lambda test: test.standard_error)

    def _status(self, state, uncertainty: Uncertainty, goal_decision, contrast):
        if not state.diagnostics_passed:
            return (
                RecommendationStatus.BLOCKED,
                "Data or causal diagnostics failed before decision finalization.",
            )
        if uncertainty.ood >= OOD_REVIEW_THRESHOLD:
            return (
                RecommendationStatus.REVIEW,
                "Patient is outside the current training support; manual review is required.",
            )
        # Two independent conditions, and both must hold. The care goal sets how
        # large a difference is worth acting on; the interval decides whether the
        # data can resolve a difference that size at all. A gap that clears the
        # clinical bar but sits inside its own confidence interval is equipoise,
        # not a recommendation.
        if contrast is not None and not contrast.distinguishable:
            return (
                RecommendationStatus.EQUIPOISE,
                (
                    f"{contrast.arm} scores {contrast.difference:+.3f} over {contrast.comparator}, "
                    f"but the {int((1 - contrast.alpha) * 100)}% interval "
                    f"[{contrast.lower:+.3f}, {contrast.upper:+.3f}] includes zero: the data cannot "
                    "separate these arms for this patient."
                ),
            )
        if not goal_decision.act:
            return RecommendationStatus.EQUIPOISE, goal_decision.rationale
        return RecommendationStatus.RECOMMEND, goal_decision.rationale

    def _no_estimate_decision(self) -> Decision:
        raise ValueError(
            "DecisionLayer received no estimates; Layer 2 must fail loudly rather than "
            "hand the decision layer an empty set to guess from."
        )


__all__ = ["DecisionLayer"]
