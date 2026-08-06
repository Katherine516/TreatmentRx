"""v5.1 #3 — Switching & rescue therapy capture.

Records non-adherence, switching, dose escalation, and rescue therapy as
structured events on the trajectory, tagging each stage with the *realized*
treatment vs the *assigned* one — the raw material for ITT / per-protocol /
as-treated estimands (consumed in Layer 6).
"""

from __future__ import annotations

from treatmentrx.domain import PatientRecord, StageRecord, SwitchingRecord


RESCUE_TOKENS = ("rescue", "steroid", "prednisone", "bridge")


class SwitchingCapture:
    """Derives a SwitchingRecord per stage from the realized treatment sequence."""

    def apply(self, stages: list[StageRecord], patient: PatientRecord) -> list[StageRecord]:
        annotated: list[StageRecord] = []
        for index, stage in enumerate(stages):
            assigned = stage.treatment
            realized = stage.treatment
            switched = False
            reason = None

            medication = patient.medications[index] if index < len(patient.medications) else None
            if medication is not None:
                reason = medication.discontinuation_reason
                if reason:
                    switched = True

            # An early discontinuation before the next planned decision is a switch.
            if (
                index + 1 < len(stages)
                and stage.end_day is not None
                and reason is None
                and stage.response
                and ("inadequate" in stage.response.lower() or "failure" in stage.response.lower())
            ):
                switched = True
                reason = reason or "loss of response"

            switching = SwitchingRecord(
                assigned=assigned,
                realized=realized,
                adherence=self._adherence(stage),
                switched=switched,
                rescue_therapy=any(token in realized.lower() for token in RESCUE_TOKENS),
                discontinuation_reason=reason,
            )
            annotated.append(self._replace(stage, switching=switching))
        return annotated

    def _adherence(self, stage: StageRecord) -> float:
        value = stage.features.get("adherence")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return round(min(max(float(value), 0.0), 1.0), 3)
        # Default: full adherence unless an explicit non-adherence signal exists.
        if stage.response and "non-adher" in stage.response.lower():
            return 0.5
        return 1.0

    def _replace(self, stage: StageRecord, **updates: object) -> StageRecord:
        return StageRecord(**(stage.__dict__ | updates))
