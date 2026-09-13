from __future__ import annotations

import math

from treatmentrx.domain import EncodedState, StageRecord


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


# The hand-rolled bias cycles with this period, which is what makes the
# recurrent half collapse to a handful of distinct trajectories.
_BIAS_PERIOD = 7


class GRUBaselineEncoder:
    """Dependency-free GRU-compatible baseline interface.

    This is a deterministic sequence summarizer, not a trained neural GRU. It
    preserves the planned `z_t` interface so the statistical layer can be tested
    before PyTorch pretraining is introduced.

    **Nothing reads the vector.** It had one consumer — the out-of-distribution
    score sliced `vector[:32]` and added `max(rms - 0.85, 0)` — and that term
    could not fire: the handcrafted prefix is nine features normalised into
    [0, 1] and tiled, so its root-mean-square is bounded well below the
    threshold by construction, and over 121 patients it ran 0.374 to 0.664. The
    term contributed zero to every score ever produced and has been removed.

    What survives is the *shape* of the planned `z_t` interface and the
    `feature_map`, which the audit event reports. Read the vector as a
    placeholder holding a seat, not as a state representation: it is a fixed
    function of nine clinical features, and its 224-entry recurrent tail
    collapses to `_BIAS_PERIOD` distinct values per patient by construction.
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

        # Every hidden unit past the handcrafted prefix starts at zero and is
        # driven by the same recurrence, differing only by a bias that depends on
        # `index % 7`. There are therefore exactly `_BIAS_PERIOD` distinct
        # trajectories, not `hidden_size - 32` of them: solve those and broadcast.
        # Bit-identical to updating each unit in turn, at a thirty-second of the
        # arithmetic.
        recurrent = range(len(base.vector), self.hidden_size)
        tracks = [0.0] * _BIAS_PERIOD
        for stage in stages:
            stage_signal = stage.outcome / max(stage.stage, 1)
            for residue in range(_BIAS_PERIOD):
                tracks[residue] = math.tanh(
                    0.88 * tracks[residue]
                    + 0.12 * stage_signal
                    + (residue - 3) * 0.002
                )
        for index in recurrent:
            hidden[index] = tracks[index % _BIAS_PERIOD]

        return EncodedState(
            encoder_name=self.encoder_name,
            vector=[round(value, 6) for value in hidden],
            feature_map=base.feature_map,
        )
