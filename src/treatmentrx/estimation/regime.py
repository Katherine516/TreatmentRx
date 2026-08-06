from __future__ import annotations

from treatmentrx.domain import RegimeAssignment, RegimeType, StageRecord


PHASE_STRUCTURED_TERMS = {
    "cancer",
    "oncology",
    "leukemia",
    "lymphoma",
    "major depression",
    "induction",
    "maintenance",
}

INDEFINITE_HORIZON_TERMS = {
    "chronic pain",
    "palliative",
    "frailty",
}

CHRONIC_STABLE_TERMS = {
    "rheumatoid arthritis",
    "hypertension",
    "type 2 diabetes",
    "diabetes",
    "hiv",
}


class AdaptiveRegimeSelector:
    """Routes patients to SPTR, DTR, hybrid, or indefinite-horizon fallback."""

    def select(self, stages: list[StageRecord]) -> RegimeAssignment:
        if not stages:
            raise ValueError("Cannot select a regime without stage records")

        disease = stages[0].disease.lower()
        shared_bic, stage_specific_bic = self._bic_proxy(stages)
        bic_gap = shared_bic - stage_specific_bic

        if any(term in disease for term in INDEFINITE_HORIZON_TERMS) or self._highly_irregular(stages):
            return RegimeAssignment(
                RegimeType.SURVIVAL_FOREST,
                "Irregular or continuous-time treatment pattern favors indefinite-horizon fallback.",
                shared_bic,
                stage_specific_bic,
            )

        if any(term in disease for term in PHASE_STRUCTURED_TERMS) and bic_gap > 4:
            return RegimeAssignment(
                RegimeType.DTR,
                "Clinical phase structure plus stage-specific BIC advantage favors DTR.",
                shared_bic,
                stage_specific_bic,
            )

        if any(term in disease for term in CHRONIC_STABLE_TERMS) and len(stages) <= 4:
            return RegimeAssignment(
                RegimeType.SPTR,
                "Chronic disease with limited longitudinal stages favors shared treatment logic.",
                shared_bic,
                stage_specific_bic,
            )

        if abs(bic_gap) <= 2 and len(stages) >= 3:
            return RegimeAssignment(
                RegimeType.HYBRID,
                "BIC scores are close, suggesting partial sharing with stage residuals.",
                shared_bic,
                stage_specific_bic,
            )

        if shared_bic <= stage_specific_bic or len(stages) < 4:
            return RegimeAssignment(
                RegimeType.SPTR,
                "Shared treatment logic is favored or data are limited, so parameter sharing improves stability.",
                shared_bic,
                stage_specific_bic,
            )

        return RegimeAssignment(
            RegimeType.DTR,
            "Stage-specific model has a clear BIC advantage.",
            shared_bic,
            stage_specific_bic,
        )

    def _bic_proxy(self, stages: list[StageRecord]) -> tuple[float, float]:
        outcomes = [stage.outcome for stage in stages]
        mean_outcome = sum(outcomes) / len(outcomes)
        shared_error = sum((outcome - mean_outcome) ** 2 for outcome in outcomes)
        stage_error = sum((outcome - stage.outcome) ** 2 for stage, outcome in zip(stages, outcomes))
        n = max(len(stages), 2)
        shared_params = 2
        stage_params = max(len(stages), 2)
        shared_bic = n * shared_error + shared_params * n.bit_length()
        stage_specific_bic = n * stage_error + stage_params * n.bit_length()
        return round(shared_bic, 4), round(stage_specific_bic, 4)

    def _highly_irregular(self, stages: list[StageRecord]) -> bool:
        gaps = [
            next_stage.start_day - stage.start_day
            for stage, next_stage in zip(stages, stages[1:])
            if next_stage.start_day > stage.start_day
        ]
        if len(gaps) < 3:
            return False
        mean_gap = sum(gaps) / len(gaps)
        if mean_gap == 0:
            return False
        max_deviation = max(abs(gap - mean_gap) for gap in gaps)
        return max_deviation / mean_gap > 1.25
