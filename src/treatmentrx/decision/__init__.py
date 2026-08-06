from __future__ import annotations

import math

from treatmentrx.contracts import Decision, PatientState, RecommendationStatus, RegimeEstimate, Uncertainty


class DecisionLayer:
    """Layer 3: BMA, uncertainty, calibration status, and action thresholding."""

    complexity_penalty = {
        "Q-Shared + Penalized": 0.02,
        "dWOLS-Shared": 0.03,
        "Stage-Specific Q-learning": 0.05,
    }

    def decide(self, state: PatientState, estimates: list[RegimeEstimate]) -> Decision:
        if not estimates:
            uncertainty = Uncertainty(1.0, 1.0, 1.0, 1.0, False, ["no_estimates"])
            return Decision(
                recommended_arm="manual-review",
                q_values={"manual-review": 0.0},
                model_weights={},
                uncertainty=uncertainty,
                status=RecommendationStatus.BLOCKED,
                rationale="No estimator produced a valid treatment-regime estimate.",
                estimates=[],
            )

        weights = self._weights(estimates)
        arms = sorted({arm for estimate in estimates for arm in estimate.q_values})
        q_values = {
            arm: round(sum(weights[e.estimator] * e.q_values.get(arm, 0.0) for e in estimates), 3)
            for arm in arms
        }
        recommended = max(q_values, key=q_values.get)
        uncertainty = self._uncertainty(state, estimates, q_values[recommended])
        ordered = sorted(q_values.values(), reverse=True)
        gap = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]

        if not state.diagnostics_passed:
            status = RecommendationStatus.BLOCKED
            rationale = "Data or causal diagnostics failed before decision finalization."
        elif uncertainty.ood >= 0.75:
            status = RecommendationStatus.REVIEW
            rationale = "Patient is outside the current training support; manual review is required."
        elif gap < 0.04:
            status = RecommendationStatus.EQUIPOISE
            rationale = "Top candidate is close to the next-best arm; communicate equipoise."
        else:
            status = RecommendationStatus.RECOMMEND
            rationale = "Model-averaged Q-values support the selected arm."

        return Decision(
            recommended_arm=recommended,
            q_values=q_values,
            model_weights={name: round(weight, 4) for name, weight in weights.items()},
            uncertainty=uncertainty,
            status=status,
            rationale=rationale,
            estimates=estimates,
        )

    def _weights(self, estimates: list[RegimeEstimate]) -> dict[str, float]:
        scores = {
            estimate.estimator: estimate.policy_value - self.complexity_penalty.get(estimate.estimator, 0.04)
            for estimate in estimates
        }
        max_score = max(scores.values())
        exp_scores = {name: math.exp(score - max_score) for name, score in scores.items()}
        total = sum(exp_scores.values())
        return {name: value / total for name, value in exp_scores.items()}

    def _uncertainty(self, state: PatientState, estimates: list[RegimeEstimate], selected_q: float) -> Uncertainty:
        values = [estimate.q_values.get(max(estimate.q_values, key=estimate.q_values.get), 0.0) for estimate in estimates]
        mean = sum(values) / len(values)
        model_var = sum((value - mean) ** 2 for value in values) / len(values)
        outcomes = [stage.outcome for stage in state.stages]
        outcome_mean = sum(outcomes) / len(outcomes) if outcomes else 0.5
        aleatoric = math.sqrt(sum((outcome - outcome_mean) ** 2 for outcome in outcomes) / len(outcomes)) if outcomes else 0.25
        epistemic = 1 / math.sqrt(max(len(state.stages), 1))
        ood = self._ood_score(state)
        calibrated = 0.05 <= selected_q <= 0.95
        flags: list[str] = []
        if epistemic > 0.5:
            flags.append("limited_history")
        if model_var > 0.0025:
            flags.append("model_disagreement")
        if ood >= 0.75:
            flags.append("ood_review")
        if not calibrated:
            flags.append("calibration_edge")
        return Uncertainty(
            aleatoric=round(min(aleatoric, 1.0), 4),
            epistemic=round(min(epistemic, 1.0), 4),
            model=round(model_var, 6),
            ood=round(ood, 4),
            calibrated=calibrated,
            flags=flags,
        )

    def _ood_score(self, state: PatientState) -> float:
        latest = state.stages[-1] if state.stages else None
        if latest is None:
            return 1.0
        das28 = self._feature(latest.features, "das28", 4.0)
        crp = self._feature(latest.features, "crp", 8.0)
        egfr = self._feature(latest.features, "egfr", 90.0)
        score = 0.0
        score += max(das28 - 7.0, 0) / 3
        score += max(crp - 80.0, 0) / 80
        score += max(30 - egfr, 0) / 30
        return min(score, 1.0)

    def _feature(self, features: dict[str, object], key: str, default: float) -> float:
        value = features.get(key, default)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default
