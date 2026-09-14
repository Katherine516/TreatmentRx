from __future__ import annotations

import math

from treatmentrx.domain import EncodedState, StageRecord


class HandcraftedFeatureEncoder:
    """Transparent baseline encoder for early RA policy validation.

    Nine clinical features, each normalised into [0, 1] and tiled to the
    requested width. `vector` is `PatientState.features` and `feature_map` names
    them, so both are read by things that matter.

    **A `GRUBaselineEncoder` used to wrap this one** and is gone. It ran a
    256-unit recurrence on every request and produced a vector nothing read: its
    single consumer, the out-of-distribution term that sliced `vector[:32]`, was
    removed once measurement showed it could never cross its threshold (invariant
    37), and after that the only surviving use of its output was its own name and
    length in one audit line. Its `feature_map` was a copy of this encoder's, and
    its 224-entry tail held exactly `7` distinct values by construction. It cost
    177us of the 211us this layer spent encoding, 29% of `build_patient_state`.

    The planned `z_t` interface is preserved by `EncodedState` itself, which this
    encoder already returns — the wrapper was holding a seat that the thing it
    wrapped was already holding.
    """

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
