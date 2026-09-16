from __future__ import annotations

import math

from treatmentrx.contracts import RegimeEstimate
from treatmentrx.domain import RegimeType
from treatmentrx.estimation.dwols import DWOLS_METHOD
from treatmentrx.estimation.q_learning import (
    Q_POOLED_METHOD,
    Q_SHARED_METHOD,
    STAGE_SPECIFIC_METHOD,
)

BMA_ENSEMBLE = "BMA Ensemble"


class BayesianModelAverager:
    """Soft model averaging over the parallel treatment-regime estimators.

    Weights are a softmax over each estimator's held-out policy value minus a
    complexity penalty. Because those policy values are measured out of sample
    and currently sit within their own standard error of each other, the weights
    come out near-uniform — which is the honest reading of the evidence, not a
    defect. Sharpening them would manufacture a ranking the data does not
    support.
    """

    # Subtracted from the held-out policy value before the softmax, ordered by
    # how many effective parameters each fit spends: 38 for the shared blip, 98
    # for the partially pooled one but shrunk toward the shared level, 78
    # unshrunk for the stage-specific fit. Every *serving* estimator needs an
    # entry — falling through to the default silently penalised `Q-Pooled` more
    # than dWOLS for no stated reason.
    complexity_penalty = {
        Q_SHARED_METHOD: 0.02,
        DWOLS_METHOD: 0.03,
        Q_POOLED_METHOD: 0.035,
        STAGE_SPECIFIC_METHOD: 0.05,
    }

    def aggregate(self, results: list[RegimeEstimate]) -> RegimeEstimate:
        """Average the estimates, refusing to average different estimands.

        **What the fingerprint check is and is not.** It is a precondition on a
        public component: `BayesianModelAverager` can be called with any list of
        `RegimeEstimate`s, and averaging two that describe different targets
        produces a number that is neither. It is *not* a live guard on the
        serving path — `EstimationLayer.estimate` stamps every result with the
        same `state.estimand_contract.fingerprint`, from one source, so within a
        pipeline call the fingerprints are identical by construction and this can
        never fire. Saying so is the point: a check that cannot close is worse
        when it is mistaken for the thing keeping the ensemble honest.

        What actually keeps the ensemble honest is `EstimationLayer.estimators`
        matching `training.SERVING_ENSEMBLE`, which `tests/test_layers.py`
        asserts. That is the guard invariant 18 rests on. This one catches a
        hand-assembled list, which is a real way to misuse the class and a
        different failure.
        """
        if not results:
            raise ValueError("BMA requires at least one method result")
        fingerprints = {
            result.estimand_fingerprint
            for result in results
            if result.estimand_fingerprint
        }
        missing = sum(not result.estimand_fingerprint for result in results)
        if len(fingerprints) > 1 or (fingerprints and missing):
            raise ValueError(
                "model averaging requires one explicit, identical estimand "
                "fingerprint on every estimator"
            )

        weights = self._weights(results)
        treatment_arms = sorted({arm for result in results for arm in result.q_values})
        q_values = {
            arm: round(
                sum(weights[result.estimator] * result.q_values.get(arm, 0.0) for result in results),
                3,
            )
            for arm in treatment_arms
        }
        recommended = self._recommended(q_values, results, weights)
        model_variance = sum(
            weights[result.estimator] * (result.q_values.get(recommended, 0.0) - q_values[recommended]) ** 2
            for result in results
        )
        # The q_values above are averaged; the *coefficients* are not, and cannot
        # be — dWOLS's psi is a single-visit blip and Q-Pooled's is a stage psi
        # from a value-to-go fit, so summing them term by term would mix the two
        # scales invariant 9 keeps apart. One member's are carried instead, and
        # which one turns on a weight margin of **0.003** on the deployed fit
        # (0.4985 / 0.5015) while the two blips differ by up to **0.041** for the
        # same patient — as large as the contrast the decision reports.
        #
        # So the choice is recorded rather than left to be inferred. The card
        # renders this decomposition directly under a line saying the decision
        # was model-averaged over two estimators, and a reader is owed the name
        # of the one it is actually reading.
        dominant = max(results, key=lambda result: weights[result.estimator])
        low = sum(weights[result.estimator] * result.confidence_band[0] for result in results)
        high = sum(weights[result.estimator] * result.confidence_band[1] for result in results)

        coefficients = dominant.coefficients | {
            f"bma_weight:{name}": round(weight, 4) for name, weight in weights.items()
        }
        coefficients["model_disagreement_variance"] = round(model_variance, 6)
        coefficients["attribution_source"] = dominant.estimator

        return RegimeEstimate(
            estimator=BMA_ENSEMBLE,
            regime_type=self._regime_type(results),
            recommended_arm=recommended,
            q_values=q_values,
            policy_value=round(sum(weights[result.estimator] * result.policy_value for result in results), 3),
            confidence_band=(round(low, 3), round(high, 3)),
            coefficients=coefficients,
            top_tailoring_variables=dominant.top_tailoring_variables,
            estimand_fingerprint=next(iter(fingerprints), ""),
        )

    def _recommended(
        self,
        q_values: dict[str, float],
        results: list[RegimeEstimate],
        weights: dict[str, float],
    ) -> str:
        """The averaged argmax, with a clamped tie resolved by the members.

        `q_values` is a *display* quantity: each estimator clamps to
        `[Q_FLOOR, Q_CEILING]` and rounds to three decimals before this sees it,
        and both steps are many-to-one. A patient whose predicted response
        saturates the ceiling has two arms collapse to 0.99, `max` falls through
        to dictionary order, and the decision then reports a **negative** contrast
        for its own top pair — the interval beside the recommendation saying the
        runner-up is better. Measured over 120 patients that happened to **6 of
        them, 5%**, with 8.3% saturating the clamp on at least one arm.

        Each member already ranks correctly — `QLearningModel.recommend` and
        `DWOLSModel.recommend` read the unclamped value-to-go — so the
        information is not lost, only discarded here. On a tie the members vote,
        weighted by the same model-averaging weights used everywhere else. Ties
        that survive that are broken by name, which is arbitrary but deterministic
        and cannot depend on dictionary insertion order.
        """
        best = max(q_values.values())
        leaders = sorted(arm for arm, value in q_values.items() if value == best)
        if len(leaders) == 1:
            return leaders[0]
        votes = {arm: 0.0 for arm in leaders}
        for result in results:
            if result.recommended_arm in votes:
                votes[result.recommended_arm] += weights.get(result.estimator, 0.0)
        top = max(votes.values())
        return sorted(arm for arm, weight in votes.items() if weight == top)[0]

    def _regime_type(self, results: list[RegimeEstimate]) -> RegimeType:
        """What kind of regime the *ensemble* is, not whichever member weighed most.

        This used to read `dominant.regime_type`, where `dominant` is the
        max-weight member. With two near-uniform weights that is a coin flip: the
        demo patient's card said "under a SPTR regime" because dWOLS beat
        `Q-Pooled` 0.501 to 0.498. When the members disagree the honest label is
        the one that says so.
        """
        kinds = {result.regime_type for result in results}
        if len(kinds) == 1:
            return next(iter(kinds))
        return RegimeType.HYBRID

    def _weights(self, results: list[RegimeEstimate]) -> dict[str, float]:
        scores = {
            result.estimator: result.policy_value - self.complexity_penalty.get(result.estimator, 0.04)
            for result in results
        }
        max_score = max(scores.values())
        exp_scores = {name: math.exp(score - max_score) for name, score in scores.items()}
        total = sum(exp_scores.values())
        return {name: value / total for name, value in exp_scores.items()}
