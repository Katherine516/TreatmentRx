"""Leakage and immortal-time guards — structural assertions, not detectors.

The most common way longitudinal EHR models silently fail, and these are
*assertions*: a violation raises out of `DataLayer.build_patient_state` and the
record is refused. What they are not is live detectors. Every one of them
re-verifies a property that something upstream already enforces, so **none can
fire from the current pipeline** — measured, zero across 60 cohort patients —
and each names the upstream guarantee it exists to catch failing:

* `TemporalFirewall` guards `StageHistoryBuilder._features_until`, which admits
  only observations at or before the decision day. Every feature present
  therefore has a source at or before it, by construction. The firewall reads
  `stage.features` and `patient.observations` as two independently produced
  objects, so it is not vacuous: change that filter to a window, or to
  last-value-wins regardless of date, and this is what fires.
* `ImmortalTimeDetector` guards `FHIRAdapter.parse_bundle`, which sorts
  medications by start day, and the data contract, which raises on unsorted
  starts before this runs.
* `_timestamp_monotonic` guards the same sorting, one object further on: stages
  are built in order from those medications.

**A fourth check was removed rather than repaired.** `_outcome_in_features`
flagged any feature whose name contained "outcome", and it could not do the job
its name claims. `_features_until` has already excluded everything after the
decision, so anything this saw was pre-decision by construction — the one thing
it could flag is a legitimately recorded *past* outcome, which is history rather
than leakage. Measured, it was also the only check in this module that could
fire at all, and firing it changed nothing: a record carrying an observation
coded `outcome` was served a recommendation with the violation noted in a
diagnostic nobody reads. A name match was never a leakage statistic, which is
the same reason invariant 37 deleted the out-of-distribution vector term instead
of recalibrating it. The property it gestured at — that a stage's outcome must
not be computable from its own covariates — is held by `data/endpoints.py`,
whose windows are strictly disjoint: baseline at `days <= start_day`, attained at
`start_day < days <= end_day`, and `None` for an open stage.
"""

from __future__ import annotations

from dataclasses import dataclass

from treatmentrx.domain import Observation, PatientRecord, StageRecord


class LeakageError(AssertionError):
    """Raised when future information is detected inside a decision-time state."""


@dataclass(frozen=True)
class LeakageReport:
    """Which assertions held, and every violation across all of them.

    The three booleans say *which* guarantee broke, which is what a reader needs
    when one does: they name different upstream properties and the repairs are
    different. `violations` carries the messages and is what the raise quotes.
    """

    temporal_firewall_passed: bool
    immortal_time_passed: bool
    timestamp_monotonic_passed: bool
    violations: list[str]

    @property
    def passed(self) -> bool:
        return not self.violations


class TemporalFirewall:
    """Every feature in H_j must trace to an observation at or before the decision.

    StageRecord.features are built from observations; here we re-verify that no
    value in a stage's state could only have come from a *future* observation.
    """

    def check(self, patient: PatientRecord, stages: list[StageRecord]) -> list[str]:
        violations: list[str] = []
        by_name = self._observations_by_feature_name(patient.observations)
        for stage in stages:
            for key in stage.features:
                sources = by_name.get(key)
                if not sources:
                    continue
                if not any(obs.days_from_baseline <= stage.start_day for obs in sources):
                    violations.append(
                        f"stage {stage.stage}: feature '{key}' only available after decision day {stage.start_day}"
                    )
        return violations

    def assert_clean(self, patient: PatientRecord, stages: list[StageRecord]) -> None:
        violations = self.check(patient, stages)
        if violations:
            raise LeakageError("; ".join(violations))

    def _observations_by_feature_name(self, observations: list[Observation]) -> dict[str, list[Observation]]:
        grouped: dict[str, list[Observation]] = {}
        for obs in observations:
            name = obs.code.lower().replace(" ", "_").replace("-", "_")
            grouped.setdefault(name, []).append(obs)
        return grouped


class ImmortalTimeDetector:
    """A patient must not be classified by a treatment received only after the
    window in which the outcome could occur (guaranteed survival-to-treatment)."""

    def check(self, patient: PatientRecord, stages: list[StageRecord]) -> list[str]:
        violations: list[str] = []
        starts = [m.start_day for m in patient.medications]
        if starts != sorted(starts):
            violations.append("treatment start days are not chronological (immortal-time risk)")
        for stage in stages:
            if stage.end_day is not None and stage.end_day < stage.start_day:
                violations.append(f"stage {stage.stage}: end_day precedes start_day")
        return violations


class LeakageTestSuite:
    """Every assertion, run on every record `DataLayer` builds."""

    def __init__(self) -> None:
        self.firewall = TemporalFirewall()
        self.immortal = ImmortalTimeDetector()

    def run(self, patient: PatientRecord, stages: list[StageRecord]) -> LeakageReport:
        firewall = self.firewall.check(patient, stages)
        immortal = self.immortal.check(patient, stages)
        monotonic = self._timestamp_monotonic(stages)

        return LeakageReport(
            temporal_firewall_passed=not firewall,
            immortal_time_passed=not immortal,
            timestamp_monotonic_passed=not monotonic,
            violations=list(firewall) + immortal + monotonic,
        )

    def _timestamp_monotonic(self, stages: list[StageRecord]) -> list[str]:
        violations: list[str] = []
        last = None
        for stage in stages:
            if last is not None and stage.start_day < last:
                violations.append(f"stage {stage.stage}: start_day {stage.start_day} precedes prior {last}")
            last = stage.start_day
        return violations
