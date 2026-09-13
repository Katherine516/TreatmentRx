from __future__ import annotations

import math
from collections import Counter

from treatmentrx.arms import is_concomitant, is_current_decision
from treatmentrx.data.endpoints import DEFAULT_ENDPOINT, UNKNOWN, Endpoint
from treatmentrx.domain import Observation, PatientRecord, StageRecord


def split_medications(patient: PatientRecord):
    """Separate the DMARD line sequence from medications given alongside it.

    One definition, two callers: the stage builder needs the line events to
    number the decision points, and `SwitchingCapture` needs the concomitant
    ones to flag rescue therapy on the stage they overlap. Deriving it twice is
    how the two would disagree about which stage a steroid belongs to.
    """
    line_events = [m for m in patient.medications if not is_concomitant(m.name)]
    concomitant = [m for m in patient.medications if is_concomitant(m.name)]
    return line_events, concomitant


class StageHistoryBuilder:
    """Builds H_j-like stage records from treatment initiations.

    The `endpoint` decides what a stage's outcome *is* — the reward the whole
    system optimises. It is a constructor argument rather than a hard-coded
    keyword match because that choice belongs to whoever knows what the study is
    measuring; see `data/endpoints.py` for why the default is the text one on
    this synthetic cohort.
    """

    def __init__(self, endpoint: Endpoint | None = None) -> None:
        self.endpoint = endpoint or DEFAULT_ENDPOINT

    def build(self, patient: PatientRecord) -> list[StageRecord]:
        if not patient.medications:
            raise ValueError("Patient record has no treatment history")

        line_events, _ = split_medications(patient)
        if not line_events:
            raise ValueError(
                "Patient record has no treatment-line events: every medication is "
                "concomitant, so there is no decision sequence to build"
            )

        stages: list[StageRecord] = []
        for index, treatment in enumerate(line_events):
            next_start = (
                line_events[index + 1].start_day
                if index + 1 < len(line_events)
                else treatment.stop_day
            )
            features = self._features_until(patient.observations, treatment.start_day)
            outcome = (
                UNKNOWN
                if is_current_decision(treatment.name) and next_start is None
                else self.endpoint.score(
                    treatment.response,
                    patient.observations,
                    treatment.start_day,
                    next_start,
                    default=float(patient.outcomes.get("default_stage_outcome", UNKNOWN)),
                )
            )
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

    def _feature_name(self, code: str) -> str:
        return code.lower().replace(" ", "_").replace("-", "_")


# `VariableSelector` used to live here, scoring raw features by
# `sqrt(variance) + abs(latest)` and handing the result to Layer 2 as the
# patient's "tailoring variables". It is gone for the same reason
# `StageRecord.visit_weight` is: it named a statistical role it did not have.
#
# Unstandardised, the score ranked by unit size — `egfr` (82) above `crp` (28)
# above `das28` (5.2) — and it skipped booleans entirely, so neither of the two
# effect modifiers in the blip basis could ever be selected. A tailoring variable
# is one the treatment effect varies over, which is a property of the *fitted
# blip*, not of a feature's raw spread, and Layer 1 has no model. The selection
# now happens where the model is: `estimation.features.top_tailoring_variables`.
