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
from treatmentrx.estimation.inference import DEFAULT_ALPHA, Z_QUANTILE, ContrastTest
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

        model_weights = {
            name.split(":", 1)[1]: value
            for name, value in selected.coefficients.items()
            if name.startswith("bma_weight:")
        }
        goal_decision = self.goal_thresholds.decide(selected, state.care_goal)
        contrast = self._contrast(state, selected, model_weights)
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
            model_weights=model_weights,
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

    def _contrast(self, state: PatientState, selected: RegimeEstimate, weights: dict[str, float]):
        """Interval for the contrast the decision is actually made on.

        The decision uses the model-averaged Q-values, so the interval has to
        describe the *averaged* contrast. Taking the widest of the three
        estimators' intervals instead — the previous rule — was incoherent twice
        over: the reported difference came from one estimator while the decision
        came from the ensemble, and which estimator supplied it changed with the
        data, so the selection added its own sampling variability on top. Measured
        end to end that rule covered 78% against a nominal 95%, worse than any of
        its own components.

        The variance of a weighted average is bounded above by the weighted sum
        of the component standard deviations, with equality under perfect
        correlation. The three estimators are fit on the same patients and are
        strongly correlated, so the bound is close to tight rather than wasteful,
        and it errs in the safe direction for a decision a clinician will act on.
        """
        ordered = sorted(selected.q_values, key=selected.q_values.get, reverse=True)
        if len(ordered) < 2:
            return None
        arm, comparator = ordered[0], ordered[1]

        tests: dict[str, object] = {}
        for estimator in self.estimators:
            try:
                tests[estimator.method_name] = estimator.contrast(state.stages, arm, comparator)
            except (KeyError, ValueError):
                continue
        if not tests:
            return None

        total = sum(weights.get(name, 0.0) for name in tests)
        if total <= 0.0:  # no averaging weights available; fall back to equal ones
            weights = {name: 1.0 for name in tests}
            total = float(len(tests))

        difference = sum(weights.get(n, 0.0) * t.difference for n, t in tests.items()) / total
        standard_error = sum(weights.get(n, 0.0) * t.standard_error for n, t in tests.items()) / total
        margin = Z_QUANTILE[DEFAULT_ALPHA] * standard_error
        exact = all(getattr(t, "exact", False) for t in tests.values())
        return ContrastTest(
            arm=arm,
            comparator=comparator,
            difference=difference,
            standard_error=standard_error,
            lower=difference - margin,
            upper=difference + margin,
            alpha=DEFAULT_ALPHA,
            # The variance bound already errs wide — measured at 1.29x the
            # estimator's actual spread — so this interval must not then be
            # widened again by the sandwich-inflation guard.
            conservative=True,
            caveat=(
                ""
                if exact
                else (
                    "Model-averaged contrast; the variance is the weighted sum of the "
                    "component standard errors, an upper bound that is tight when they "
                    "are perfectly correlated."
                )
            ),
        )

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
        if contrast is not None and not contrast.robustly_distinguishable:
            if contrast.distinguishable:
                # The interval excludes zero, but only because the sandwich is
                # narrower than it should be. Measured, not hypothetical.
                return (
                    RecommendationStatus.EQUIPOISE,
                    (
                        f"{contrast.arm} scores {contrast.difference:+.3f} over "
                        f"{contrast.comparator} and the reported interval "
                        f"[{contrast.lower:+.3f}, {contrast.upper:+.3f}] excludes zero, but the "
                        "separation does not survive the interval being widened to the width "
                        "coverage says it should have. Treated as equipoise."
                    ),
                )
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
