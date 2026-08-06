from __future__ import annotations

import math

from precisionrx_agent.shared.models import MethodResult


class BayesianModelAverager:
    """Layer 5 soft model averaging for parallel treatment-regime estimators."""

    complexity_penalty = {
        "Q-Shared + Penalized": 0.02,
        "dWOLS-Shared": 0.03,
        "Stage-Specific Q-learning": 0.05,
    }

    def aggregate(self, results: list[MethodResult]) -> MethodResult:
        if not results:
            raise ValueError("BMA requires at least one method result")

        weights = self._weights(results)
        treatment_arms = sorted({arm for result in results for arm in result.q_values})
        q_values = {
            arm: round(
                sum(weights[result.method_name] * result.q_values.get(arm, 0.0) for result in results),
                3,
            )
            for arm in treatment_arms
        }
        recommended = max(q_values, key=q_values.get)
        model_variance = sum(
            weights[result.method_name] * (result.q_values.get(recommended, 0.0) - q_values[recommended]) ** 2
            for result in results
        )
        dominant = max(results, key=lambda result: weights[result.method_name])
        low = sum(weights[result.method_name] * result.confidence_band[0] for result in results)
        high = sum(weights[result.method_name] * result.confidence_band[1] for result in results)

        coefficients = dominant.coefficients | {
            f"bma_weight:{name}": round(weight, 4) for name, weight in weights.items()
        }
        coefficients["model_disagreement_variance"] = round(model_variance, 6)

        return MethodResult(
            method_name="BMA Ensemble",
            regime_type=dominant.regime_type,
            recommended_action=recommended,
            q_values=q_values,
            policy_value=round(sum(weights[result.method_name] * result.policy_value for result in results), 3),
            confidence_band=(round(low, 3), round(high, 3)),
            coefficients=coefficients,
            top_tailoring_variables=dominant.top_tailoring_variables,
        )

    def _weights(self, results: list[MethodResult]) -> dict[str, float]:
        scores = {
            result.method_name: result.policy_value - self.complexity_penalty.get(result.method_name, 0.04)
            for result in results
        }
        max_score = max(scores.values())
        exp_scores = {name: math.exp(score - max_score) for name, score in scores.items()}
        total = sum(exp_scores.values())
        return {name: value / total for name, value in exp_scores.items()}
