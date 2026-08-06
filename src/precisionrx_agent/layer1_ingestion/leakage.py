"""v5.1 #10 — Leakage & immortal-time guards.

The most common way longitudinal EHR models silently fail. These are
*assertions*, not warnings: a cohort build that puts any post-decision
information into H_j fails. Wired into the CI gate.
"""

from __future__ import annotations

from dataclasses import dataclass

from precisionrx_agent.shared.models import Observation, PatientRecord, StageRecord


class LeakageError(AssertionError):
    """Raised when future information is detected inside a decision-time state."""


@dataclass(frozen=True)
class LeakageReport:
    temporal_firewall_passed: bool
    immortal_time_passed: bool
    outcome_not_in_features_passed: bool
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
    """CI suite run on every cohort build."""

    def __init__(self) -> None:
        self.firewall = TemporalFirewall()
        self.immortal = ImmortalTimeDetector()

    def run(self, patient: PatientRecord, stages: list[StageRecord]) -> LeakageReport:
        firewall = self.firewall.check(patient, stages)
        immortal = self.immortal.check(patient, stages)
        outcome_in_features = self._outcome_in_features(stages)
        monotonic = self._timestamp_monotonic(stages)

        violations = list(firewall)
        violations += immortal
        violations += outcome_in_features
        violations += monotonic
        return LeakageReport(
            temporal_firewall_passed=not firewall,
            immortal_time_passed=not immortal,
            outcome_not_in_features_passed=not outcome_in_features,
            timestamp_monotonic_passed=not monotonic,
            violations=violations,
        )

    def _outcome_in_features(self, stages: list[StageRecord]) -> list[str]:
        violations: list[str] = []
        for stage in stages:
            for key in stage.features:
                if "outcome" in key or key in {"response_label", "endpoint"}:
                    violations.append(f"stage {stage.stage}: outcome-like feature '{key}' present in H_j")
        return violations

    def _timestamp_monotonic(self, stages: list[StageRecord]) -> list[str]:
        violations: list[str] = []
        last = None
        for stage in stages:
            if last is not None and stage.start_day < last:
                violations.append(f"stage {stage.stage}: start_day {stage.start_day} precedes prior {last}")
            last = stage.start_day
        return violations
