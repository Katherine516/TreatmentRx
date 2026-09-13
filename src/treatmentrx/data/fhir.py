from __future__ import annotations

import hashlib

from typing import Any

from treatmentrx.data import units
from treatmentrx.domain import Observation, PatientRecord, TreatmentEvent


def normalise_observation_code(code: str) -> str:
    """The one spelling of an observation code the rest of Layer 1 uses.

    Lives here because the adapter is the first thing to see a code; the data
    contract and the stage builder both normalise the same way, and having three
    copies of `lower().replace(...)` is how they drift apart.
    """
    return code.lower().replace("-", "_").replace(" ", "_")


class FHIRAdapter:
    """Parses a small useful subset of HL7 FHIR R4 Bundle resources.

    The adapter accepts real FHIR-ish dictionaries but intentionally normalizes
    into our internal dataclasses. Full FHIR coverage belongs in a later adapter.
    """

    def patient_hash(self, patient_id: str) -> str:
        """Stable pseudonym. The raw identifier never leaves this layer."""
        return hashlib.sha256(patient_id.encode("utf-8")).hexdigest()[:16]

    def parse_bundle(self, bundle: dict[str, Any]) -> PatientRecord:
        raw_entries = bundle.get("entry", [])
        entries = [entry.get("resource", {}) for entry in raw_entries]
        patient_entries = [
            entry for entry in raw_entries
            if entry.get("resource", {}).get("resourceType") == "Patient"
        ]
        if len(patient_entries) != 1:
            raise ValueError(
                "FHIR bundle must contain exactly one Patient resource "
                f"(found {len(patient_entries)})"
            )
        patient_entry = patient_entries[0]
        patient = patient_entry["resource"]

        patient_id = str(patient.get("id", "unknown"))
        self._validate_subject_scope(entries, patient_id, patient_entry.get("fullUrl"))
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
        supply = self._supply_records(entries)
        medications = [
            self._parse_medication(resource, supply)
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

    def _validate_subject_scope(
        self,
        entries: list[dict[str, Any]],
        patient_id: str,
        patient_full_url: str | None,
    ) -> None:
        """Reject a Bundle that mixes resources from different patients.

        The prototype accepts unscoped resources in a one-patient collection, as
        its fixtures do, but whenever a FHIR subject/patient reference is present
        it must name the one Patient resource selected above.
        """
        aliases = {patient_id, f"Patient/{patient_id}"}
        if patient_full_url:
            aliases.add(str(patient_full_url))

        for resource in entries:
            if resource.get("resourceType") == "Patient":
                continue
            reference = self._subject_reference(resource)
            if reference is None:
                continue
            if reference in aliases or reference.endswith(f"/Patient/{patient_id}"):
                continue
            raise ValueError(
                f"{resource.get('resourceType', 'Resource')} references {reference!r}, "
                f"not the bundle patient Patient/{patient_id}"
            )

    def _subject_reference(self, resource: dict[str, Any]) -> str | None:
        for field in ("subject", "patient"):
            value = resource.get(field)
            if isinstance(value, str):
                return value
            if isinstance(value, dict) and value.get("reference"):
                return str(value["reference"])
        return None

    def _first(self, entries: list[dict[str, Any]], resource_type: str) -> dict[str, Any] | None:
        return next((entry for entry in entries if entry.get("resourceType") == resource_type), None)

    def _parse_observation(self, resource: dict[str, Any]) -> Observation:
        quantity = resource.get("valueQuantity", {})
        value: float | str | bool = quantity.get("value", resource.get("valueString", resource.get("valueBoolean", 0)))
        if isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        code = self._display(resource.get("code", {}))
        unit = quantity.get("unit")
        # Converted here, where the raw resource is; *reported* by the contract,
        # which is the gate. The `unit` field keeps what the record said so the
        # contract can say what was assumed or changed — the value below is
        # always in `units.CANONICAL_UNITS`.
        value, _ = units.convert(normalise_observation_code(code), value, unit)
        return Observation(
            code=code,
            value=value,
            unit=unit,
            days_from_baseline=int(resource.get("effectiveDay", resource.get("day", 0))),
        )

    def _supply_records(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Dispense and administration resources, sorted by the day they happened.

        These are what distinguish *assigned* from *realized* treatment. Without
        them `SwitchingRecord.realized` can only echo the order, and the
        ITT / per-protocol / as-treated split has no input to work from.
        """
        records = []
        for resource in entries:
            kind = resource.get("resourceType")
            if kind not in {"MedicationDispense", "MedicationAdministration"}:
                continue
            medication = resource.get(
                "medicationCodeableConcept", resource.get("medication", {})
            )
            day = resource.get(
                "whenHandedOverDay",
                resource.get("effectiveDay", resource.get("day", 0)),
            )
            records.append(
                {
                    "name": self._display(medication),
                    "day": int(day),
                    "days_supply": resource.get("daysSupply"),
                }
            )
        return sorted(records, key=lambda record: record["day"])

    def _parse_medication(
        self, resource: dict[str, Any], supply: list[dict[str, Any]] | None = None
    ) -> TreatmentEvent:
        medication = resource.get("medicationCodeableConcept", resource.get("medication", {}))
        start_day = int(resource.get("authoredOnDay", resource.get("startDay", 0)))
        stop_day = resource.get("stopDay")
        dispensed_name, days_supply = self._realized(supply or [], start_day, stop_day)
        return TreatmentEvent(
            name=self._display(medication),
            start_day=start_day,
            dose=resource.get("dose"),
            stop_day=stop_day,
            response=resource.get("response"),
            discontinuation_reason=resource.get("discontinuationReason"),
            dispensed_name=dispensed_name,
            dispensed_days_supply=days_supply,
        )

    def _realized(
        self, supply: list[dict[str, Any]], start_day: int, stop_day: int | None
    ) -> tuple[str | None, int | None]:
        """What was dispensed inside this order's window, and for how many days.

        The window is `[start_day, stop_day)`; an open order takes everything
        from its start onward. Multiple dispenses inside one window are summed
        for days supply and the first one names the molecule, which is the
        common case of a repeat prescription filled several times.
        """
        inside = [
            record
            for record in supply
            if record["day"] >= start_day and (stop_day is None or record["day"] < stop_day)
        ]
        if not inside:
            return None, None
        total = sum(
            int(record["days_supply"])
            for record in inside
            if record.get("days_supply") is not None
        )
        return inside[0]["name"], (total or None)

    def _display(self, codeable: dict[str, Any]) -> str:
        if "text" in codeable:
            return str(codeable["text"])
        coding = codeable.get("coding", [])
        if coding:
            return str(coding[0].get("display", coding[0].get("code", "unknown")))
        return "unknown"
