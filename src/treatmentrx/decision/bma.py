from __future__ import annotations

import math

from treatmentrx.contracts import RegimeEstimate
from treatmentrx.estimation.dwols import DWOLS_METHOD
from treatmentrx.estimation.q_learning import Q_SHARED_METHOD, STAGE_SPECIFIC_METHOD

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

    complexity_penalty = {
        Q_SHARED_METHOD: 0.02,
        DWOLS_METHOD: 0.03,
        STAGE_SPECIFIC_METHOD: 0.05,
    }

    def aggregate(self, results: list[RegimeEstimate]) -> RegimeEstimate:
        if not results:
            raise ValueError("BMA requires at least one method result")

        weights = self._weights(results)
        treatment_arms = sorted({arm for result in results for arm in result.q_values})
        q_values = {
            arm: round(
                sum(weights[result.estimator] * result.q_values.get(arm, 0.0) for result in results),
                3,
            )
            for arm in treatment_arms
        }
        recommended = max(q_values, key=q_values.get)
        model_variance = sum(
            weights[result.estimator] * (result.q_values.get(recommended, 0.0) - q_values[recommended]) ** 2
            for result in results
        )
        dominant = max(results, key=lambda result: weights[result.estimator])
        low = sum(weights[result.estimator] * result.confidence_band[0] for result in results)
        high = sum(weights[result.estimator] * result.confidence_band[1] for result in results)

        coefficients = dominant.coefficients | {
            f"bma_weight:{name}": round(weight, 4) for name, weight in weights.items()
        }
        coefficients["model_disagreement_variance"] = round(model_variance, 6)

        return RegimeEstimate(
            estimator=BMA_ENSEMBLE,
            regime_type=dominant.regime_type,
            recommended_arm=recommended,
            q_values=q_values,
            policy_value=round(sum(weights[result.estimator] * result.policy_value for result in results), 3),
            confidence_band=(round(low, 3), round(high, 3)),
            coefficients=coefficients,
            top_tailoring_variables=dominant.top_tailoring_variables,
        )

    def _weights(self, results: list[RegimeEstimate]) -> dict[str, float]:
        scores = {
            result.estimator: result.policy_value - self.complexity_penalty.get(result.estimator, 0.04)
            for result in results
        }
        max_score = max(scores.values())
        exp_scores = {name: math.exp(score - max_score) for name, score in scores.items()}
        total = sum(exp_scores.values())
        return {name: value / total for name, value in exp_scores.items()}
