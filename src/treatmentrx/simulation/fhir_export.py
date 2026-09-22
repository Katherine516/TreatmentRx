"""Render a simulated trajectory as a FHIR-like bundle.

This closes the loop. The cohort was only ever estimator training data, entering
the system below Layer 1, so every ingestion module — stage construction, the
timing model, switching capture, competing-risk typing — had no ground truth to
be checked against. It was tested on one hand-written demo patient.

Exporting trajectories as bundles means a simulated patient whose timing,
switching and terminal event are *known by construction* can be pushed through
the real `DataLayer` and the recovered annotations compared against what was
generated. `tests/test_data_layer.py` does exactly that.

The mapping is deliberately lossy in the same ways real data is: the bundle
carries observations and medication records, not the latent parameters. Layer 1
has to reconstruct the stage structure from dates and drug names, which is the
part worth testing.
"""

from __future__ import annotations

from typing import Any

from treatmentrx.simulation.ra_cohort import (
    CohortShift,
    CohortTrajectory,
    generate_ra_cohort,
)

# How the generator's terminal events read in a medication record.
_DISCONTINUATION_TEXT = {
    "serious_toxicity": "serious toxicity",
    "progression": "loss of response",
    "dropout": "patient withdrew",
}

# Response labels, so Layer 1 has the same free text a real record would carry.
_GOOD_RESPONSE = 0.65
_PARTIAL_RESPONSE = 0.5


def _response_text(outcome: float) -> str:
    if outcome >= _GOOD_RESPONSE:
        return "good response"
    if outcome >= _PARTIAL_RESPONSE:
        return "partial response"
    return "inadequate response"


def _observations(day: int, features: dict[str, float]) -> list[dict[str, Any]]:
    numeric = (
        ("DAS28", features["das28"], "score"),
        ("CRP", features["crp"], "mg/L"),
        ("eGFR", features["egfr"], "mL/min"),
        ("ALT", features["alt"], "U/L"),
    )
    entries = [
        {
            "resource": {
                "resourceType": "Observation",
                "code": {"text": code},
                "valueQuantity": {"value": round(value, 2), "unit": unit},
                "effectiveDay": day,
            }
        }
        for code, value, unit in numeric
    ]
    entries.append(
        {
            "resource": {
                "resourceType": "Observation",
                "code": {"text": "anti_CCP"},
                "valueBoolean": bool(features["anti_ccp"]),
                "effectiveDay": day,
            }
        }
    )
    return entries


def trajectory_to_bundle(
    trajectory: CohortTrajectory, through_stage: int | None = None
) -> dict[str, Any]:
    """Render one simulated trajectory as an ingestible FHIR-like bundle.

    An open "current decision point" is appended so the bundle represents a
    patient standing at a decision, which is what the agent is asked about.

    `through_stage=j` truncates the history to the stages **before** `j` and
    puts that open decision point at stage `j`'s own day, carrying stage `j`'s
    own covariates — the state as it stood when the clinician chose, with the
    arm they chose withheld. That is what lets a historical decision be
    re-scored, which nothing here could do: the default bundle always ends in a
    synthetic open point extrapolated past the last visit, so `stages[-1]`
    carries the sentinel `'current decision point'` and there is no clinician
    choice anywhere in the record to compare an answer against.

    `through_stage=None` reproduces the original bundle byte for byte.
    """
    entries: list[dict[str, Any]] = [
        {
            "resource": {
                "resourceType": "Patient",
                "id": f"sim-{trajectory.patient_index:05d}",
                "gender": "unknown",
                "birthDate": "1970-01-01",
            }
        },
        {"resource": {"resourceType": "Condition", "code": {"text": "Rheumatoid Arthritis"}}},
    ]

    stages = trajectory.stages
    if through_stage is None:
        history = stages
        last = stages[-1]
        decision_day = last.day + (last.interval_days or 90)
        decision_features = last.features
        # Only a trajectory rendered to its end can carry why it ended.
        ended_here = trajectory.censored
    else:
        if not 1 <= through_stage < len(stages):
            raise ValueError(
                f"through_stage must leave at least one stage of history and one "
                f"to ask about: got {through_stage} for {len(stages)} stages"
            )
        history = stages[:through_stage]
        decision_day = stages[through_stage].day
        decision_features = stages[through_stage].features
        ended_here = False

    for position, stage in enumerate(history):
        following = history[position + 1] if position + 1 < len(history) else None
        entries.append({"resource": {"resourceType": "Encounter", "day": stage.day}})
        entries.extend(_observations(stage.day, stage.features))

        record: dict[str, Any] = {
            "resourceType": "MedicationRequest",
            "medicationCodeableConcept": {"text": stage.arm},
            "authoredOnDay": stage.day,
            "response": _response_text(stage.outcome),
        }
        if following is not None:
            record["stopDay"] = following.day
        elif through_stage is not None:
            # The truncated history runs up to the decision being asked about.
            record["stopDay"] = decision_day
        elif ended_here:
            # The trajectory ended here; say why, the way a record would.
            record["stopDay"] = stage.day + (stage.interval_days or 90)
            record["discontinuationReason"] = _DISCONTINUATION_TEXT.get(
                trajectory.censoring_reason or "", "patient withdrew"
            )
        entries.append({"resource": record})

    # The open decision the agent is being asked about.
    entries.append({"resource": {"resourceType": "Encounter", "day": decision_day}})
    entries.extend(_observations(decision_day, decision_features))
    entries.append(
        {
            "resource": {
                "resourceType": "MedicationRequest",
                "medicationCodeableConcept": {"text": "current decision point"},
                "authoredOnDay": decision_day,
                "response": "response unknown",
            }
        }
    )
    return {"resourceType": "Bundle", "type": "collection", "entry": entries}


def simulated_bundles(
    n: int = 20, seed: int = 991, shift: CohortShift | None = None
) -> list[dict[str, Any]]:
    """A batch of ingestible patients, for exercising the pipeline end to end.

    `shift` describes a structurally different site (`feedback/transfer.py`).
    Omitted, the batch is byte-identical to what it has always been.
    """
    return [
        trajectory_to_bundle(trajectory)
        for trajectory in generate_ra_cohort(n, seed, shift=shift)
    ]


__all__ = ["simulated_bundles", "trajectory_to_bundle"]
