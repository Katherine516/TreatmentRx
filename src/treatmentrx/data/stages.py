from __future__ import annotations

import math
from collections import Counter

from treatmentrx.domain import Observation, PatientRecord, StageRecord


class StageHistoryBuilder:
    """Builds H_j-like stage records from treatment initiations."""

    def build(self, patient: PatientRecord) -> list[StageRecord]:
        if not patient.medications:
            raise ValueError("Patient record has no treatment history")

        stages: list[StageRecord] = []
        for index, treatment in enumerate(patient.medications):
            next_start = (
                patient.medications[index + 1].start_day
                if index + 1 < len(patient.medications)
                else None
            )
            features = self._features_until(patient.observations, treatment.start_day)
            outcome = self._stage_outcome(treatment.response, patient.outcomes)
            stages.append(
                StageRecord(
                    patient_id=patient.patient_id,
                    disease=patient.disease,
                    stage=index + 1,
                    treatment=treatment.name,
                    start_day=treatment.start_day,
                    end_day=next_start,
                    features=features,
                    response=treatment.response,
                    outcome=outcome,
                )
            )
        return stages

    def _features_until(self, observations: list[Observation], day: int) -> dict[str, float | str | bool]:
        features: dict[str, float | str | bool] = {}
        for observation in observations:
            if observation.days_from_baseline <= day:
                features[self._feature_name(observation.code)] = observation.value
        return features

    def _stage_outcome(self, response: str | None, outcomes: dict[str, object]) -> float:
        if response:
            normalized = response.lower()
            if "remission" in normalized or "good" in normalized:
                return 1.0
            if "partial" in normalized:
                return 0.55
            if "inadequate" in normalized or "failure" in normalized:
                return 0.15
        value = outcomes.get("default_stage_outcome", 0.5)
        return float(value)

    def _feature_name(self, code: str) -> str:
        return code.lower().replace(" ", "_").replace("-", "_")


class VariableSelector:
    """Identifies candidate tailoring variables from numeric feature variance."""

    def select(self, stages: list[StageRecord], limit: int = 5) -> list[str]:
        numeric_values: dict[str, list[float]] = {}
        for stage in stages:
            for key, value in stage.features.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric_values.setdefault(key, []).append(float(value))

        scored: list[tuple[float, str]] = []
        for key, values in numeric_values.items():
            if not values:
                continue
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            latest = values[-1]
            score = math.sqrt(variance) + abs(latest)
            scored.append((score, key))

        if scored:
            return [key for _, key in sorted(scored, reverse=True)[:limit]]

        counts = Counter(key for stage in stages for key in stage.features)
        return [key for key, _ in counts.most_common(limit)]
