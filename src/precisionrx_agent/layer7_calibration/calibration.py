from __future__ import annotations

from precisionrx_agent.shared.models import CalibrationReport


class CalibrationEvaluator:
    """Reliability summary for retrospective validation and deployment gates."""

    def evaluate(
        self,
        predicted_probabilities: list[float],
        observed_outcomes: list[float],
        bins: int = 5,
        threshold: float = 0.05,
    ) -> CalibrationReport:
        if len(predicted_probabilities) != len(observed_outcomes):
            raise ValueError("Predictions and outcomes must have the same length")
        if not predicted_probabilities:
            return CalibrationReport(0.0, threshold, True, [])

        reliability_bins: list[dict[str, float]] = []
        ece = 0.0
        n = len(predicted_probabilities)
        for index in range(bins):
            lower = index / bins
            upper = (index + 1) / bins
            if index == bins - 1:
                selected = [
                    (prediction, outcome)
                    for prediction, outcome in zip(predicted_probabilities, observed_outcomes)
                    if lower <= prediction <= upper
                ]
            else:
                selected = [
                    (prediction, outcome)
                    for prediction, outcome in zip(predicted_probabilities, observed_outcomes)
                    if lower <= prediction < upper
                ]
            if not selected:
                reliability_bins.append(
                    {"lower": round(lower, 3), "upper": round(upper, 3), "count": 0, "confidence": 0.0, "accuracy": 0.0}
                )
                continue
            confidence = sum(prediction for prediction, _ in selected) / len(selected)
            accuracy = sum(outcome for _, outcome in selected) / len(selected)
            ece += (len(selected) / n) * abs(confidence - accuracy)
            reliability_bins.append(
                {
                    "lower": round(lower, 3),
                    "upper": round(upper, 3),
                    "count": float(len(selected)),
                    "confidence": round(confidence, 4),
                    "accuracy": round(accuracy, 4),
                }
            )

        rounded = round(ece, 4)
        return CalibrationReport(
            expected_calibration_error=rounded,
            threshold=threshold,
            passed=rounded <= threshold,
            reliability_bins=reliability_bins,
        )
