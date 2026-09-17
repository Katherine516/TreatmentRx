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
from treatmentrx.estimation.features import model_features, stage_index
from treatmentrx.estimation.inference import (
    DEFAULT_ALPHA,
    ContrastTest,
    normal_critical_value,
)
from treatmentrx.estimation.belief_aware import BeliefAwareAdjuster
from treatmentrx.estimation.competing_risk_outcomes import CompetingRiskEndpoint
from treatmentrx.estimation.explainability import ModelExplainer
from treatmentrx.estimation.dwols import DWOLSSharedEstimator
from treatmentrx.estimation.estimators import PooledQEstimator
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
        # Must match `EstimationLayer.estimators`: the interval has to describe
        # the same ensemble the Q-values came from.
        self.estimators = (PooledQEstimator(), DWOLSSharedEstimator())

    def decide(self, state: PatientState, estimates: list[RegimeEstimate]) -> Decision:
        if not estimates:
            return self._no_estimate_decision()

        # Features go in so the ensemble ranks its tailoring drivers off the
        # averaged blip rather than the marginally heavier member's.
        selected = self.bma.aggregate(estimates, model_features(state.stages))
        selected = self.competing_endpoint.adjust(selected, state.stages, state.competing_risk_incidence)
        selected = self.belief_adjuster.adjust(selected, state.stages)

        model_weights = {
            name.split(":", 1)[1]: value
            for name, value in selected.coefficients.items()
            if name.startswith("bma_weight:")
        }
        # Every arm's interval against the leader, once. Three things read it:
        # the candidate set is its membership, `_contrast` is its minimum, and
        # the why-not gaps are its differences. Each of those used to derive its
        # own quantity — two of them from the clamped, rounded `q_values` — and
        # the card printed the three side by side.
        candidate_arms, candidate_contrasts = self._candidate_set(state, selected, model_weights)
        contrast = self._contrast(candidate_contrasts)
        # The care-goal bar judges the same difference the interval brackets, so
        # it is handed that difference rather than re-deriving one.
        goal_decision = self.goal_thresholds.decide(
            selected,
            state.care_goal,
            observed_gap=contrast.difference if contrast is not None else None,
        )
        uncertainty = self.uncertainty.decompose(
            state.stages,
            selected,
            estimates,
            training.holdout_calibration(),
            contrast,
        )
        explanation = self.explainer.explain(
            selected,
            estimates,
            state.stages,
            contrast=contrast,
            arm_contrasts=candidate_contrasts,
        )

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
            candidate_arms=candidate_arms,
            candidate_contrasts=candidate_contrasts,
        )

    @staticmethod
    def _contrast(arm_contrasts: dict):
        """Interval for the contrast the decision is actually made on.

        **The comparator is the arm the data says is closest, and it used to be
        whichever arm sorted first.** This is the last place invariant 46's
        display quantity reached a clinical output. The pair came from
        `_ranked()[1]` — the argmax over the *clamped, rounded* non-leader
        `q_values` — so for a patient whose predicted response saturates the
        ceiling, three arms sit at 0.99 and **dictionary order chose the arm the
        recommendation was justified against**. Measured over 120 patients: 5 had
        a tied comparator, and for 4 of them the arm dictionary order picked was
        separable (+0.051) while the genuinely closest arm was not (+0.018). The
        card announced a separation that held only for the pair nobody would have
        chosen on purpose.

        Selecting the minimum is a search over all five comparisons, which is
        exactly the family `_simultaneous_alpha` already corrects for — the
        leader is chosen by looking at every arm, so the honest family is the one
        the search ranged over (invariant 32). No level changes here.

        The care-goal bar in invariant 10 now judges the gap to the nearest
        competitor, which is the only thing "large enough to be worth acting on"
        can sensibly mean, and both of that invariant's conditions read the one
        interval. The pair is no longer computed here at all — `_candidate_set`
        already builds it for every arm, and computing the runner-up's twice was
        how the two could differ in the first place.

        **It does not make RECOMMEND mean "separated from every arm", and that
        would be the easy thing to assume.** The smallest *difference* is not the
        smallest z: an arm further away but less precisely estimated can be the
        one the data cannot exclude. Measured over 240 patients, 3 of 71
        recommendations are exactly that — TNF-inhibitor at +0.049 with SE 0.0144
        (z 3.41, separable) is the comparator, while IL-6 inhibitor at +0.050
        with SE 0.0176 (z 2.86) is not. Reporting the z-minimum instead would
        hand the care-goal bar the *larger* of two near-identical gaps, which is
        the permissive direction. The candidate set is what covers every arm, and
        `RationaleGenerator._not_excluded` is the card block that says so when
        the two differ.

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
        correlation. The estimators are fit on the same patients and are strongly
        correlated, so the bound is close to tight rather than wasteful, and it
        errs in the safe direction for a decision a clinician will act on.

        **Centring is the part that had to be fixed first.** Swept across the
        patient grid this interval once covered 74%, and the miss was not width —
        its SE ran 1.12-1.52x the actual spread everywhere. The average included a
        shared-blip estimator whose terminal contrast is a stage-pooled compromise
        biased +0.067, so the ensemble sat between two parameters. It now averages
        `training.SERVING_ENSEMBLE`, whose members estimate the same quantity, and
        covers **95.0% pooled with 93% at the worst patient, at SE/spread 1.04**.
        Those figures moved from 98%/97%/1.27 when `ArmFit.cross_covariance`
        started being kept: the interval lost width that was an error rather than
        a margin, so it is now at nominal rather than above it.

        **The figure is measured at the terminal decision**, which is the one
        stage where both serving members target the same quantity. Swept
        (`coverage.decision_rule_stage_sweep`) it holds at 96.7% for stage 1 —
        the only other stage Layer 1 can produce — and falls to 77.5% at stage 0,
        which is fitted and never served because `build_patient_state` appends
        the pending visit.
        """
        if not arm_contrasts:
            return None
        # Ties broken by name so the choice is deterministic. Unlike the rounded
        # display dict this sorts a continuous estimate, so a tie means two arms
        # the ensemble genuinely cannot order, not two it was not asked to.
        return min(arm_contrasts.values(), key=lambda test: (test.difference, test.comparator))

    @staticmethod
    def _ranked(selected: RegimeEstimate) -> list[str]:
        """Arms best-to-worst with the ensemble's own leader first.

        Sorting `q_values` is not enough, and the failure is not hypothetical.
        That dict is clamped to `[Q_FLOOR, Q_CEILING]` and rounded to 3dp for
        display; a patient whose predicted response saturates the ceiling has two
        arms collapse to 0.99, and a stable sort then returns them in dictionary
        order. The leader here and `selected.recommended_arm` — which BMA takes
        from the members' unclamped rankings — could disagree, and when they did
        the contrast was computed for the wrong pair *in the wrong direction*: the
        interval beside the recommendation said the runner-up was better.
        Measured over 120 patients, 6 of them.

        Choosing the leader once, in one place, is the fix.

        **Only position 0 is load-bearing now.** This used to supply the
        comparator as well, at position 1, and that is where dictionary order
        among clamped ties still reached the decision — see `_contrast`, which
        now takes it from the estimated contrasts instead. What is left is the
        leader, which matters, and the rest of the menu in display order, which
        is a presentation question: `_candidate_set` tests every one of them and
        the order it walks them in changes nothing it returns.
        """
        ordered = sorted(selected.q_values, key=selected.q_values.get, reverse=True)
        leader = selected.recommended_arm
        if leader in ordered:
            ordered.remove(leader)
            ordered.insert(0, leader)
        return ordered

    def _candidate_set(
        self, state: PatientState, selected: RegimeEstimate, weights: dict[str, float]
    ):
        """Every arm the data cannot separate from the leader, and why.

        **Why this exists.** The agent declines to name one arm for most patients
        it sees — about 66% at the training population, 54-89% across the sites in
        `cli transfer`, and 97% for seronegative patients. Until now that produced
        a status and a paragraph, and the clinician, who still has to prescribe
        something, got nothing to prescribe *with*. The information to do better
        was already computed and discarded: measured over declined patients, the
        arms that survive this test average 2.7 of 6, and choosing the worst of
        them instead of the worst of all six cuts worst-case regret from 0.206 to
        0.047, a 77% reduction. Replicated over 40 refits on the coverage grid the
        set contains the truly optimal arm in 240/240 draws at mean size 1.97.

        **What it is not.** It is not a way to recommend more often. The action
        bar is untouched, `status` is unchanged, and an arm inside this set has
        *not* been recommended — the set is what the agent can say when it cannot
        recommend. Collapsing it toward one arm to raise the recommend rate would
        be the tuning CLAUDE.md forbids, wearing a new name.

        The rule is the one Layer 3 already applies to the runner-up:
        `robustly_distinguishable`, on the same model-averaged interval. An arm is
        excluded only if the interval that *would* have been reported for it
        excludes zero.
        """
        arms = self._ranked(selected)
        if len(arms) < 2:
            return tuple(arms), {}

        leader = arms[0]
        alpha = self._simultaneous_alpha(len(arms))
        candidates = [leader]
        contrasts: dict[str, object] = {}
        for arm in arms[1:]:
            test = self._pair_contrast(
                state, leader, arm, weights, alpha=alpha
            )
            if test is None:
                # No interval means no evidence to exclude on. Keeping the arm is
                # the conservative direction: the set may only be too wide.
                candidates.append(arm)
                continue
            contrasts[arm] = test
            if not test.robustly_distinguishable:
                candidates.append(arm)
        return tuple(candidates), contrasts

    def _pair_contrast(
        self,
        state: PatientState,
        arm: str,
        comparator: str,
        weights: dict[str, float],
        alpha: float = DEFAULT_ALPHA,
    ):
        """The model-averaged interval for one ordered pair.

        Split out of `_contrast` so the candidate set can ask the same question
        of every arm. Nothing about the rule changes with the pair — which is the
        point: an arm is excluded by exactly the interval that would have been
        reported had it been the runner-up.
        """
        tests: dict[str, object] = {}
        for estimator in self.estimators:
            try:
                tests[estimator.method_name] = estimator.contrast(
                    state.stages, arm, comparator, alpha
                )
            except (KeyError, ValueError):
                continue
        if not tests:
            return None

        total = sum(weights.get(name, 0.0) for name in tests)
        if total <= 0.0:  # no averaging weights available; fall back to equal ones
            weights = {name: 1.0 for name in tests}
            total = float(len(tests))

        # If the estimators have been resampled together, their covariance is
        # measured and the bound is unnecessary.
        joint = self._joint_contrast(state, arm, comparator, weights, alpha)
        if joint is not None:
            return joint

        difference = sum(weights.get(n, 0.0) * t.difference for n, t in tests.items()) / total
        standard_error = sum(weights.get(n, 0.0) * t.standard_error for n, t in tests.items()) / total
        margin = normal_critical_value(alpha) * standard_error
        exact = all(getattr(t, "exact", False) for t in tests.values())
        return ContrastTest(
            arm=arm,
            comparator=comparator,
            difference=difference,
            standard_error=standard_error,
            lower=difference - margin,
            upper=difference + margin,
            alpha=alpha,
            # Exempt from the sandwich-inflation guard, and the reason has
            # changed. It used to be "the bound already errs wide", measured at
            # 1.15x the estimator's actual spread. With the dWOLS cross-arm
            # covariance kept, the bound runs at 1.04 — essentially exact — so
            # the argument is no longer that widening is redundant but that it
            # would make a calibrated interval wrong in the other direction.
            # `SANDWICH_INFLATION` was calibrated on a single estimator's
            # sandwich at 0.88 of its spread, which is not this quantity.
            conservative=True,
            caveat=(
                ""
                if exact
                else (
                    "Model-averaged contrast; the variance is the weighted sum of the "
                    "component standard errors, an upper bound that is tight when they "
                    "are perfectly correlated. The alpha level is Bonferroni-adjusted "
                    "over every unordered arm pair."
                )
            ),
        )

    def _joint_contrast(
        self, state: PatientState, arm: str, comparator: str, weights, alpha: float
    ):
        """Interval from the joint bootstrap, when one has been enabled.

        Falls back to None — and so to the conservative bound — whenever the
        draws are missing or the loadings cannot be built, because an interval
        that silently degrades is worse than one that is visibly wide.
        """
        bootstrap = training.joint_inference()
        if bootstrap is None:
            return None
        fit = training.fitted()
        features = model_features(state.stages)
        index = stage_index(state.stages, fit.pooled.n_stages)
        horizon = fit.pooled.remaining_stages(index)
        # Only the serving ensemble: the joint draws cover all three estimators,
        # but averaging the shared blip back in here would reintroduce exactly
        # the centring error that keeping it out of `self.estimators` removed.
        available = {
            training.Q_POOLED: lambda: fit.pooled.contrast_loading(
                arm, comparator, features, index
            ),
            training.DWOLS_SHARED: lambda: fit.dwols.contrast_loading(
                arm, comparator, features
            ),
        }
        loadings = {
            name: build() for name, build in available.items()
            if name in training.SERVING_ENSEMBLE
        }
        try:
            return bootstrap.contrast(
                loadings,
                weights,
                arm,
                comparator,
                alpha=alpha,
                scale_by=float(horizon),
            )
        except (ValueError, KeyError, IndexError):
            return None

    def _simultaneous_alpha(self, number_of_arms: int) -> float:
        """Bonferroni family-wise alpha over all unordered treatment pairs.

        The leader and comparator are selected from the same estimates used for
        inference.  A pointwise 95% interval does not account for that search.
        This conservative correction makes the emitted candidate set an
        all-pairs confidence set; joint bootstrap draws still supply covariance
        within each contrast when enabled.
        """
        from treatmentrx.estimation.inference import simultaneous_alpha

        return simultaneous_alpha(number_of_arms, DEFAULT_ALPHA)

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
