from __future__ import annotations

import math

from precisionrx_agent.shared.models import EncodedState, MethodResult, StageRecord, UncertaintyBundle


class UncertaintyDecomposer:
    """Computes the four uncertainty values used by the safety and UI layers."""

    def decompose(
        self,
        stages: list[StageRecord],
        selected: MethodResult,
        candidates: list[MethodResult],
        encoded_state: EncodedState | None = None,
    ) -> UncertaintyBundle:
        aleatoric = self._aleatoric(stages)
        epistemic = self._epistemic(stages)
        model = float(selected.coefficients.get("model_disagreement_variance", self._model_variance(candidates)))
        ood = self._ood_score(encoded_state, stages)

        flags: list[str] = []
        if aleatoric >= 0.18:
            flags.append("high_aleatoric")
        if epistemic >= 0.2:
            flags.append("high_epistemic")
        if model >= 0.0025:
            flags.append("model_disagreement")
        if ood >= 0.75:
            flags.append("ood_review")

        return UncertaintyBundle(
            aleatoric=round(aleatoric, 4),
            epistemic=round(epistemic, 4),
            model=round(model, 6),
            ood=round(ood, 4),
            flags=flags,
        )

    def _aleatoric(self, stages: list[StageRecord]) -> float:
        if len(stages) < 2:
            return 0.2
        mean = sum(stage.outcome for stage in stages) / len(stages)
        variance = sum((stage.outcome - mean) ** 2 for stage in stages) / len(stages)
        return min(math.sqrt(variance), 1.0)

    def _epistemic(self, stages: list[StageRecord]) -> float:
        effective_n = sum(1 / max(stage.censoring_weight * stage.visit_weight, 0.1) for stage in stages)
        return min(1 / math.sqrt(max(effective_n, 1)), 1.0)

    def _model_variance(self, candidates: list[MethodResult]) -> float:
        if not candidates:
            return 0.0
        values = [candidate.policy_value for candidate in candidates]
        mean = sum(values) / len(values)
        return sum((value - mean) ** 2 for value in values) / len(values)

    def _ood_score(self, encoded_state: EncodedState | None, stages: list[StageRecord]) -> float:
        latest = stages[-1]
        das28 = self._feature(latest, "das28", 4.0)
        crp = self._feature(latest, "crp", 8.0)
        egfr = self._feature(latest, "egfr", 90.0)
        clinical_score = 0.0
        clinical_score += max(das28 - 7.0, 0) / 3
        clinical_score += max(crp - 80.0, 0) / 80
        clinical_score += max(30.0 - egfr, 0) / 30
        if encoded_state:
            vector_energy = math.sqrt(sum(value * value for value in encoded_state.vector[:32]) / 32)
            clinical_score += max(vector_energy - 0.85, 0)
        return min(clinical_score, 1.0)

    def _feature(self, stage: StageRecord, key: str, default: float) -> float:
        value = stage.features.get(key, default)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default
