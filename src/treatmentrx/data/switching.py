"""v5.1 #3 — Switching & rescue therapy capture.

Records non-adherence, switching, dose escalation, and rescue therapy as
structured events on the trajectory, tagging each stage with the *realized*
treatment vs the *assigned* one — the raw material for ITT / per-protocol /
as-treated estimands (consumed in Layer 6).
"""

from __future__ import annotations

from treatmentrx.arms import is_rescue, normalize_arm
from treatmentrx.data.stages import split_medications
from treatmentrx.domain import PatientRecord, StageRecord, SwitchingRecord


class SwitchingCapture:
    """Derives a SwitchingRecord per stage from the realized treatment sequence."""

    def apply(self, stages: list[StageRecord], patient: PatientRecord) -> list[StageRecord]:
        line_events, concomitant = split_medications(patient)
        annotated: list[StageRecord] = []
        for index, stage in enumerate(stages):
            assigned = stage.treatment
            realized = stage.treatment
            switched = False
            reason = None

            medication = line_events[index] if index < len(line_events) else None
            if medication is not None:
                reason = medication.discontinuation_reason
                if reason:
                    switched = True
                # `realized` used to be a copy of `assigned` unconditionally,
                # which made the ITT / per-protocol / as-treated split a
                # distinction with no input. It now comes from the dispense or
                # administration record when the bundle carries one — and only
                # then, because an absent supply chain is a missing measurement
                # rather than evidence that nothing was supplied.
                if medication.dispensed_name:
                    realized = medication.dispensed_name
                    if normalize_arm(realized) != normalize_arm(assigned):
                        switched = True
                        reason = reason or (
                            f"dispensed {normalize_arm(realized)} against an order "
                            f"for {normalize_arm(assigned)}"
                        )

            # The definitional signal, and the one that was missing: the arm
            # changed. Detection used to rely entirely on a free-text
            # discontinuation reason or the word "inadequate" in the response, so
            # a patient moved from methotrexate to a TNF inhibitor with a good
            # response and no reason recorded was not counted as having switched
            # at all — roughly half of them in the simulated cohort.
            previous = stages[index - 1] if index > 0 else None
            if previous is not None and normalize_arm(previous.treatment) != normalize_arm(stage.treatment):
                switched = True
                reason = reason or f"changed from {normalize_arm(previous.treatment)}"

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
                adherence=self._adherence(stage, medication),
                switched=switched,
                rescue_therapy=self._rescued(stage, concomitant),
                discontinuation_reason=reason,
            )
            annotated.append(self._replace(stage, switching=switching))
        return annotated

    def _rescued(self, stage: StageRecord, concomitant) -> bool:
        """Was a rescue medication given during this stage's window?

        This used to search the *arm name* for "rescue", "steroid", "prednisone"
        or "bridge" — none of which any canonical arm contains, so the flag was
        permanently False. Rescue therapy is by definition something given
        alongside the line, so it has to be looked for in the concomitant
        records and matched against the window it overlaps.
        """
        return any(
            is_rescue(medication.name)
            and stage.start_day <= medication.start_day
            and (stage.end_day is None or medication.start_day < stage.end_day)
            for medication in concomitant
        )

    def _adherence(self, stage: StageRecord, medication=None) -> float:
        """Proportion of days covered, when the record can support one.

        Order of preference: an explicit `adherence` observation, then days
        supplied over days the order was open — the standard PDC — then the free
        text, then the 1.0 default. The default is the weakest of the four and
        the only one available today on a bundle with no dispense records, which
        is worth knowing before reading an adherence-weighted number.
        """
        value = stage.features.get("adherence")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return round(min(max(float(value), 0.0), 1.0), 3)
        covered = self._days_covered(stage, medication)
        if covered is not None:
            return covered
        # Default: full adherence unless an explicit non-adherence signal exists.
        if stage.response and "non-adher" in stage.response.lower():
            return 0.5
        return 1.0

    def _days_covered(self, stage: StageRecord, medication) -> float | None:
        """Days supplied over days the order was open, capped at 1.0."""
        if medication is None or medication.dispensed_days_supply is None:
            return None
        if stage.end_day is None:
            return None
        window = stage.end_day - stage.start_day
        if window <= 0:
            return None
        return round(min(medication.dispensed_days_supply / window, 1.0), 3)

    def _replace(self, stage: StageRecord, **updates: object) -> StageRecord:
        return StageRecord(**(stage.__dict__ | updates))
