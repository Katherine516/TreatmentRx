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

from treatmentrx.simulation.ra_cohort import CohortTrajectory, generate_ra_cohort

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


def trajectory_to_bundle(trajectory: CohortTrajectory) -> dict[str, Any]:
    """Render one simulated trajectory as an ingestible FHIR-like bundle.

    An open "current decision point" is appended so the bundle represents a
    patient standing at a decision, which is what the agent is asked about.
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
    for position, stage in enumerate(stages):
        following = stages[position + 1] if position + 1 < len(stages) else None
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
        elif trajectory.censored:
            # The trajectory ended here; say why, the way a record would.
            record["stopDay"] = stage.day + (stage.interval_days or 90)
            record["discontinuationReason"] = _DISCONTINUATION_TEXT.get(
                trajectory.censoring_reason or "", "patient withdrew"
            )
        entries.append({"resource": record})

    # The open decision the agent is being asked about.
    last = stages[-1]
    decision_day = last.day + (last.interval_days or 90)
    entries.append({"resource": {"resourceType": "Encounter", "day": decision_day}})
    entries.extend(_observations(decision_day, last.features))
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


def simulated_bundles(n: int = 20, seed: int = 991) -> list[dict[str, Any]]:
    """A batch of ingestible patients, for exercising the pipeline end to end."""
    return [trajectory_to_bundle(trajectory) for trajectory in generate_ra_cohort(n, seed)]


__all__ = ["simulated_bundles", "trajectory_to_bundle"]
