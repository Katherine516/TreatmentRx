from __future__ import annotations

from typing import Any


def sample_ra_bundle() -> dict[str, Any]:
    """Synthetic RA patient inspired by the architecture document."""

    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "patient-demo-001",
                    "gender": "female",
                    "birthDate": "1976-04-12",
                }
            },
            {
                "resource": {
                    "resourceType": "Condition",
                    "code": {"text": "Rheumatoid Arthritis"},
                }
            },
            {"resource": {"resourceType": "Encounter", "day": 0}},
            {"resource": {"resourceType": "Encounter", "day": 90}},
            {"resource": {"resourceType": "Encounter", "day": 210}},
            {"resource": {"resourceType": "Encounter", "day": 365}},
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"text": "DAS28"},
                    "valueQuantity": {"value": 5.2, "unit": "score"},
                    "effectiveDay": 365,
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"text": "CRP"},
                    "valueQuantity": {"value": 28, "unit": "mg/L"},
                    "effectiveDay": 365,
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"text": "HAQ-DI"},
                    "valueQuantity": {"value": 1.4, "unit": "score"},
                    "effectiveDay": 365,
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"text": "anti_CCP"},
                    "valueBoolean": True,
                    "effectiveDay": 0,
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"text": "eGFR"},
                    "valueQuantity": {"value": 82, "unit": "mL/min"},
                    "effectiveDay": 365,
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "medicationCodeableConcept": {"text": "methotrexate"},
                    "authoredOnDay": 0,
                    "stopDay": 240,
                    "dose": "20mg weekly",
                    "response": "partial response",
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "medicationCodeableConcept": {"text": "TNF-inhibitor adalimumab"},
                    "authoredOnDay": 240,
                    "stopDay": 365,
                    "dose": "40mg every other week",
                    "response": "inadequate response",
                    "discontinuationReason": "loss of response",
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "medicationCodeableConcept": {"text": "current decision point"},
                    "authoredOnDay": 365,
                    "response": "response unknown",
                }
            },
        ],
    }
