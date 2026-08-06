from __future__ import annotations

import math

from treatmentrx.contracts import RegimeEstimate
from treatmentrx.domain import CalibrationReport, EncodedState, StageRecord, Uncertainty


# An estimate whose parameter uncertainty alone spans this much of the response
# scale is fragile regardless of how large the gap looks.
EPISTEMIC_FLAG = 0.05
ALEATORIC_FLAG = 0.18
# Fewer than this many decision points is a thin trajectory to condition on.
SHORT_HISTORY_STAGES = 3
# No contrast available (a single-arm menu): claim nothing about precision.
UNKNOWN_EPISTEMIC = 1.0


class UncertaintyDecomposer:
    """The four uncertainty types, kept separate because they have different fixes.

    Aleatoric noise cannot be reduced by more data; epistemic uncertainty can;
    model disagreement means the estimators contradict each other; and an
    out-of-distribution patient means none of the three numbers can be trusted.
    Collapsing them into one score would hide which of those is happening.
    """

    def decompose(
        self,
        stages: list[StageRecord],
        selected: RegimeEstimate,
        candidates: list[RegimeEstimate],
        encoded_state: EncodedState | None = None,
        calibration: CalibrationReport | None = None,
        contrast=None,
    ) -> Uncertainty:
        aleatoric = self._aleatoric(stages)
        epistemic = self._epistemic(contrast)
        model = float(selected.coefficients.get("model_disagreement_variance", self._model_variance(candidates)))
        ood = self._ood_score(encoded_state, stages)
        # Calibration is a model-level property measured on held-out patients;
        # it is not something a single patient's history can establish.
        calibrated = calibration.passed if calibration is not None else False

        flags: list[str] = []
        if aleatoric >= ALEATORIC_FLAG:
            flags.append("high_aleatoric")
        if epistemic >= EPISTEMIC_FLAG:
            flags.append("high_epistemic")
        if len(stages) < SHORT_HISTORY_STAGES:
            flags.append("limited_history")
        if model >= 0.0025:
            flags.append("model_disagreement")
        if ood >= 0.75:
            flags.append("ood_review")
        if not calibrated:
            flags.append("uncalibrated_model")

        return Uncertainty(
            aleatoric=round(aleatoric, 4),
            epistemic=round(epistemic, 4),
            model=round(model, 6),
            ood=round(ood, 4),
            calibrated=calibrated,
            flags=flags,
        )

    def _aleatoric(self, stages: list[StageRecord]) -> float:
        if len(stages) < 2:
            return 0.2
        mean = sum(stage.outcome for stage in stages) / len(stages)
        variance = sum((stage.outcome - mean) ** 2 for stage in stages) / len(stages)
        return min(math.sqrt(variance), 1.0)

    def _epistemic(self, contrast) -> float:
        """Parameter uncertainty: the standard error of the decision.

        This used to be `1/sqrt(visits this patient has had)`, which measures
        how short the patient's history is, not how well the model's parameters
        are determined — the two are unrelated, and the estimators now carry a
        real cluster-robust standard error for exactly this quantity. Short
        histories are still reported, under `limited_history`, where they belong.
        """
        if contrast is None:
            return UNKNOWN_EPISTEMIC
        return min(contrast.standard_error, 1.0)

    def _model_variance(self, candidates: list[RegimeEstimate]) -> float:
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
