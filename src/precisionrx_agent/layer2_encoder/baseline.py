from __future__ import annotations

import math

from precisionrx_agent.shared.models import EncodedState, StageRecord


class HandcraftedFeatureEncoder:
    """Transparent baseline encoder for early RA policy validation."""

    encoder_name = "handcrafted-ra-baseline"

    def encode(self, stages: list[StageRecord], dimension: int = 32) -> EncodedState:
        if not stages:
            raise ValueError("Cannot encode an empty stage history")

        latest = stages[-1]
        features = {
            "stage_count": float(len(stages)),
            "das28": self._feature(latest, "das28", 4.0),
            "crp": self._feature(latest, "crp", 8.0),
            "haq_di": self._feature(latest, "haq_di", 1.0),
            "egfr": self._feature(latest, "egfr", 90.0),
            "alt": self._feature(latest, "alt", 25.0),
            "anti_ccp": 1.0 if latest.features.get("anti_ccp") or latest.features.get("anti_ccp_positive") else 0.0,
            "prior_tnf_exposure": 1.0 if any("tnf" in stage.treatment.lower() for stage in stages) else 0.0,
            "latest_outcome": float(latest.outcome),
        }
        vector = self._project(features, dimension)
        return EncodedState(encoder_name=self.encoder_name, vector=vector, feature_map=features)

    def _feature(self, stage: StageRecord, key: str, default: float) -> float:
        value = stage.features.get(key, default)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default

    def _project(self, features: dict[str, float], dimension: int) -> list[float]:
        normalized = [
            features["stage_count"] / 6,
            features["das28"] / 10,
            features["crp"] / 100,
            features["haq_di"] / 3,
            features["egfr"] / 120,
            features["alt"] / 120,
            features["anti_ccp"],
            features["prior_tnf_exposure"],
            features["latest_outcome"],
        ]
        vector: list[float] = []
        while len(vector) < dimension:
            for value in normalized:
                if len(vector) >= dimension:
                    break
                vector.append(round(max(min(value, 1.0), -1.0), 6))
        return vector


class GRUBaselineEncoder:
    """Dependency-free GRU-compatible baseline interface.

    This is a deterministic sequence summarizer, not a trained neural GRU. It
    preserves the planned `z_t` interface so the statistical layer can be tested
    before PyTorch pretraining is introduced.
    """

    encoder_name = "gru-compatible-baseline"

    def __init__(self, hidden_size: int = 256) -> None:
        self.hidden_size = hidden_size
        self.handcrafted = HandcraftedFeatureEncoder()

    def encode(self, stages: list[StageRecord]) -> EncodedState:
        base = self.handcrafted.encode(stages, dimension=min(32, self.hidden_size))
        hidden = [0.0 for _ in range(self.hidden_size)]
        for index, value in enumerate(base.vector):
            hidden[index] = value

        for stage in stages:
            stage_signal = (stage.outcome * stage.visit_weight * stage.censoring_weight) / max(stage.stage, 1)
            for index in range(len(base.vector), self.hidden_size):
                previous = hidden[index]
                hidden[index] = math.tanh(0.88 * previous + 0.12 * stage_signal + ((index % 7) - 3) * 0.002)

        return EncodedState(
            encoder_name=self.encoder_name,
            vector=[round(value, 6) for value in hidden],
            feature_map=base.feature_map,
        )
