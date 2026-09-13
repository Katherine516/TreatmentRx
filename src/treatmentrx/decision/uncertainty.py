from __future__ import annotations

import math

from treatmentrx.contracts import RegimeEstimate
from treatmentrx.domain import CalibrationReport, StageRecord, Uncertainty


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
        calibration: CalibrationReport | None = None,
        contrast=None,
    ) -> Uncertainty:
        aleatoric = self._aleatoric(stages)
        epistemic = self._epistemic(contrast)
        model = float(selected.coefficients.get("model_disagreement_variance", self._model_variance(candidates)))
        ood = self._ood_score(stages)
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
        # Belief uncertainty is reported here rather than folded into the
        # confidence band. A recommendation resting on an uncertain read of the
        # latent disease activity should be flagged as such — but it is a
        # different kind of not-knowing from parameter uncertainty, and adding it
        # to a standard error produces a number that is neither.
        if not self._belief_is_confident(stages):
            flags.append("uncertain_disease_activity_belief")
        if model >= 0.0025:
            flags.append("model_disagreement")
        if ood >= 0.75:
            flags.append("ood_review")
        if not calibrated:
            flags.append("uncalibrated_model")
        # Model-level, and identical for every patient — which is the point.
        # A basis that omits a real effect modifier makes this patient's contrast
        # the covariate-averaged one, by an amount the interval cannot show.
        flags.extend(self._basis_flag())

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

    def _basis_flag(self) -> list[str]:
        """A model-level caveat that has to reach the patient-level output.

        If a candidate covariate tests as an effect modifier the basis omits,
        *this patient's* contrast is the covariate-averaged one rather than
        theirs, by an amount the interval cannot show. That is exactly the kind
        of thing a reader needs on the card and not in a CLI they never run.
        """
        from treatmentrx.estimation import training

        flagged = training.basis_specification().get("flagged", [])
        return [f"blip_basis_may_omit:{name}" for name in flagged]

    def _belief_is_confident(self, stages: list[StageRecord]) -> bool:
        belief = stages[-1].belief if stages else None
        return belief is None or belief.confident

    def _model_variance(self, candidates: list[RegimeEstimate]) -> float:
        if not candidates:
            return 0.0
        values = [candidate.policy_value for candidate in candidates]
        mean = sum(values) / len(values)
        return sum((value - mean) ** 2 for value in values) / len(values)

    def _ood_score(self, stages: list[StageRecord]) -> float:
        """How far outside the training cohort this patient sits, on three axes.

        Three one-sided range checks, each normalised so that crossing its own
        threshold alone contributes 1.0 and trips the review gate. They are crude
        — a real detector would measure distance in the covariate space rather
        than clipping three margins — but each one *can* fire on a record the
        data contract admits, which is the property that matters for a gate:

            das28 >= 9.25   (contract admits up to 10.0)
            crp   >= 140    (contract admits up to 500)
            egfr  <= 7.5    (contract admits down to 0)

        **A fourth term used to sit here and could not fire.** It added
        `max(rms(encoded_state.vector[:32]) - 0.85, 0)`. The encoder is a
        deterministic summariser of nine features, each normalised into [0, 1]
        and then tiled to fill the vector, so its root-mean-square is bounded
        well below the threshold by construction: measured over 121 patients it
        ran 0.374 to **0.664**, never once reaching 0.85. It contributed exactly
        zero to every score this system has ever produced.

        Removing it costs nothing and settles a question it was posing badly.
        "Vector energy" was never a distributional statistic — the vector is a
        fixed function of the same clinical features the three terms already
        read, so its magnitude carries no information about being *out of
        distribution* that they do not. A trained encoder would be a different
        argument; this one is not trained, and the docstring says so.
        """
        latest = stages[-1]
        das28 = self._feature(latest, "das28", 4.0)
        crp = self._feature(latest, "crp", 8.0)
        egfr = self._feature(latest, "egfr", 90.0)
        score = 0.0
        score += max(das28 - 7.0, 0) / 3
        score += max(crp - 80.0, 0) / 80
        score += max(30.0 - egfr, 0) / 30
        return min(score, 1.0)

    def _feature(self, stage: StageRecord, key: str, default: float) -> float:
        value = stage.features.get(key, default)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default
