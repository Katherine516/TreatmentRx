"""v5.1 #8 — Prospective validation ladder.

Silent -> shadow -> advisory -> pragmatic trial, with a gate between every rung.
Skipping rungs is how clinical AI tools get deployed and then quietly harm
people. Each rung exposes failure modes the previous one cannot.
"""

from __future__ import annotations

from typing import Any

from treatmentrx.domain import ValidationRung, ValidationStatus


_NEXT = {
    ValidationRung.SILENT: ValidationRung.SHADOW,
    ValidationRung.SHADOW: ValidationRung.ADVISORY,
    ValidationRung.ADVISORY: ValidationRung.PRAGMATIC_TRIAL,
    ValidationRung.PRAGMATIC_TRIAL: None,
}

_GATES = {
    ValidationRung.SILENT: "OPE + calibration stable on live data",
    ValidationRung.SHADOW: "no safety events; concordance reasonable",
    ValidationRung.ADVISORY: "clinician-utility positive; fairness clean; IRB approval",
    ValidationRung.PRAGMATIC_TRIAL: "the real endpoint — measured patient benefit",
}


class ValidationLadder:
    def assess(self, rung: ValidationRung, metrics: dict[str, Any]) -> ValidationStatus:
        blockers = self._blockers(rung, metrics)
        return ValidationStatus(
            rung=rung,
            gate_description=_GATES[rung],
            gate_passed=not blockers,
            blockers=blockers,
            next_rung=_NEXT[rung],
        )

    def _blockers(self, rung: ValidationRung, m: dict[str, Any]) -> list[str]:
        blockers: list[str] = []
        if rung is ValidationRung.SILENT:
            if not m.get("ope_stable", False):
                blockers.append("OPE not stable on live data")
            if not m.get("calibration_passed", False):
                blockers.append("calibration not passing")
        elif rung is ValidationRung.SHADOW:
            if m.get("safety_events", 0) > 0:
                blockers.append("safety events observed in shadow")
            if m.get("concordance", 0.0) < 0.5:
                blockers.append("clinician concordance too low")
        elif rung is ValidationRung.ADVISORY:
            if m.get("clinician_utility", 0.0) <= 0.0:
                blockers.append("clinician utility not positive")
            if not m.get("fairness_clean", False):
                blockers.append("fairness checks not clean")
            if not m.get("irb_approved", False):
                blockers.append("IRB approval missing")
        elif rung is ValidationRung.PRAGMATIC_TRIAL:
            if not m.get("trial_endpoint_met", False):
                blockers.append("primary patient-benefit endpoint not met")
        return blockers
