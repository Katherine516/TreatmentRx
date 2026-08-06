from __future__ import annotations

import hashlib

from typing import Any

from treatmentrx.domain import Observation, PatientRecord, TreatmentEvent


class FHIRAdapter:
    """Parses a small useful subset of HL7 FHIR R4 Bundle resources.

    The adapter accepts real FHIR-ish dictionaries but intentionally normalizes
    into our internal dataclasses. Full FHIR coverage belongs in a later adapter.
    """

    def patient_hash(self, patient_id: str) -> str:
        """Stable pseudonym. The raw identifier never leaves this layer."""
        return hashlib.sha256(patient_id.encode("utf-8")).hexdigest()[:16]

    def parse_bundle(self, bundle: dict[str, Any]) -> PatientRecord:
        entries = [entry.get("resource", {}) for entry in bundle.get("entry", [])]
        patient = self._first(entries, "Patient")
        if not patient:
            raise ValueError("FHIR bundle is missing a Patient resource")

        patient_id = str(patient.get("id", "unknown"))
        demographics = {
            "birthDate": patient.get("birthDate"),
            "gender": patient.get("gender"),
        }

        conditions = [
            self._display(resource.get("code", {}))
            for resource in entries
            if resource.get("resourceType") == "Condition"
        ]
        observations = [
            self._parse_observation(resource)
            for resource in entries
            if resource.get("resourceType") == "Observation"
        ]
        medications = [
            self._parse_medication(resource)
            for resource in entries
            if resource.get("resourceType") in {"MedicationRequest", "MedicationStatement"}
        ]
        encounters = [
            int(resource.get("period", {}).get("startDay", resource.get("day", 0)))
            for resource in entries
            if resource.get("resourceType") == "Encounter"
        ]
        allergies = [
            self._display(resource.get("code", {}))
            for resource in entries
            if resource.get("resourceType") == "AllergyIntolerance"
        ]

        disease = conditions[0] if conditions else bundle.get("disease", "Unknown disease")
        outcomes = bundle.get("outcomes", {})
        return PatientRecord(
            patient_id=patient_id,
            disease=disease,
            demographics=demographics,
            conditions=conditions,
            allergies=allergies,
            medications=sorted(medications, key=lambda item: item.start_day),
            observations=sorted(observations, key=lambda item: item.days_from_baseline),
            encounters=sorted(encounters),
            outcomes=outcomes,
        )

    def _first(self, entries: list[dict[str, Any]], resource_type: str) -> dict[str, Any] | None:
        return next((entry for entry in entries if entry.get("resourceType") == resource_type), None)

    def _parse_observation(self, resource: dict[str, Any]) -> Observation:
        quantity = resource.get("valueQuantity", {})
        value: float | str | bool = quantity.get("value", resource.get("valueString", resource.get("valueBoolean", 0)))
        if isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        return Observation(
            code=self._display(resource.get("code", {})),
            value=value,
            unit=quantity.get("unit"),
            days_from_baseline=int(resource.get("effectiveDay", resource.get("day", 0))),
        )

    def _parse_medication(self, resource: dict[str, Any]) -> TreatmentEvent:
        medication = resource.get("medicationCodeableConcept", resource.get("medication", {}))
        return TreatmentEvent(
            name=self._display(medication),
            start_day=int(resource.get("authoredOnDay", resource.get("startDay", 0))),
            dose=resource.get("dose"),
            stop_day=resource.get("stopDay"),
            response=resource.get("response"),
            discontinuation_reason=resource.get("discontinuationReason"),
        )

    def _display(self, codeable: dict[str, Any]) -> str:
        if "text" in codeable:
            return str(codeable["text"])
        coding = codeable.get("coding", [])
        if coding:
            return str(coding[0].get("display", coding[0].get("code", "unknown")))
        return "unknown"
