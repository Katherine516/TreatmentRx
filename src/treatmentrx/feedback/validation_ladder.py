"""v5.1 #8 — Prospective validation ladder.

Silent -> shadow -> advisory -> pragmatic trial, with a gate between every rung.
Skipping rungs is how clinical AI tools get deployed and then quietly harm
people. Each rung exposes failure modes the previous one cannot.

**The gate must be able to close.** The SILENT gate used to read
`ope_stable = effective_sample_size > 0` off the *patient's* switching-aware OPE.
That quantity is at least 1 for anyone with a single visit, so the gate passed for
every patient ever scored — a rung-advancement check that could not fail, in the
module whose entire purpose is that advancement should be hard. It was also the
wrong level: whether a *model* is ready to leave silent mode is not something one
patient's trajectory can establish.

The criteria below are model-level and measured on held-out data
(`training.deployment_readiness`), plus one flag that is structural rather than
statistical: this build has never seen live data, so the "stable **on live
data**" gate cannot be satisfied by a held-out split of the same simulated
generating process the model was fit on. That blocker is permanent here, in the
same way `retraining_allowed=False` is, and it is stated rather than assumed.
"""

from __future__ import annotations

from typing import Any

from treatmentrx.domain import ValidationRung, ValidationStatus

# A held-out IPW value backed by fewer effective observations than this is too
# noisy to gate a deployment decision on. Matches the threshold at which
# `offline_evaluation.evaluate_policy` already annotates the score as
# high-variance, so the two do not drift apart.
MIN_OPE_EFFECTIVE_SAMPLE = 30.0


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


def _criteria(metrics: dict[str, Any], *checks) -> list[str]:
    """Blockers for criteria that must be *positively measured* to pass.

    `metrics.get(key, default)` conflates "measured and failing" with "nobody
    measured this", and the two need different words in front of a reader
    deciding whether to advance a rung. Both directions were wrong here:

    * `fairness_clean` defaulted to False, so every assessment blocked with
      "fairness checks not clean" — which reads as a failed check. Nothing in
      this codebase computes fairness at all. `cli subgroups` stratifies clinical
      covariates and says in its own docstring that it is not a fairness audit;
      the synthetic cohort has no protected attributes to audit.
    * `safety_events` defaulted to 0, which *passes*. An unmeasured safety
      criterion that reads as satisfied is fail-open, and on the one rung whose
      entire purpose is catching harm before it reaches anyone. Same class of
      defect as the override router's unmatched-reason default.

    An absent key is now its own blocker, worded as the absence it is.
    """
    blockers: list[str] = []
    for key, label, passes in checks:
        if key not in metrics:
            blockers.append(f"{label}: not measured (no `{key}` in the readiness metrics)")
        elif not passes(metrics[key]):
            blockers.append(f"{label}: measured and not met ({key}={metrics[key]!r})")
    return blockers


# The criteria each rung above SILENT gates on, declared once because two
# things read them: `_blockers`, which turns them into gate messages, and
# `instrumentation_coverage`, which asks how many are supplied at all. Inferring
# the second from the first does not work — a criterion that is measured *and
# passing* produces no blocker, so a blocker count cannot see it, and a rung with
# one criterion met and one missing read as wholly uninstrumented.
_RUNG_CRITERIA: dict[ValidationRung, tuple] = {
    ValidationRung.SHADOW: (
        ("safety_events", "no safety events in shadow", lambda v: v == 0),
        ("concordance", "clinician concordance at least 0.5", lambda v: v >= 0.5),
    ),
    ValidationRung.ADVISORY: (
        ("clinician_utility", "clinician utility positive", lambda v: v > 0.0),
        ("fairness_clean", "fairness checks clean", bool),
        ("irb_approved", "IRB approval", bool),
    ),
    ValidationRung.PRAGMATIC_TRIAL: (
        ("trial_endpoint_met", "primary patient-benefit endpoint met", bool),
    ),
}

# SILENT's messages are bespoke rather than built by `_criteria`, so its keys are
# named here for the coverage report alone. `tests/test_layers.py` asserts they
# are the ones `deployment_readiness()` actually supplies, because a list that
# drifts from the gate it describes is invariant 33's defect.
_SILENT_KEYS = (
    "ope_effective_sample_size",
    "sequential_ope_effective_sample_size",
    "ope_improvement_lower",
    "calibration_passed",
    "blip_basis_unflagged",
    "live_data",
)


def rung_criteria_keys(rung: ValidationRung) -> tuple[str, ...]:
    """The readiness keys a rung gates on."""
    if rung is ValidationRung.SILENT:
        return _SILENT_KEYS
    return tuple(key for key, _label, _test in _RUNG_CRITERIA.get(rung, ()))


def instrumentation_coverage(metrics: dict[str, Any]) -> dict[str, Any]:
    """Which rungs can be assessed at all, and which are only declarations.

    The ladder describes four rungs and `training.deployment_readiness()`
    supplies criteria for one. So a reader who sees `rung: silent` learns the
    gate is shut and not that the three gates above it have nothing behind them
    — a model that cleared SILENT would stall at SHADOW immediately, not on a
    failure but on the absence of any measurement.

    Derived by running the real gate rather than listing keys, so it cannot
    drift from the thing it describes: a criterion counts as unmeasured exactly
    when `_criteria` says so. That is invariant 33's lesson about a card figure
    parting company with its study, applied before it happens rather than after.
    """
    rungs: dict[str, Any] = {}
    instrumented = 0
    for rung in ValidationRung:
        status = ValidationLadder().assess(rung, metrics)
        keys = rung_criteria_keys(rung)
        supplied = [key for key in keys if key in metrics]
        has_any = bool(supplied)
        instrumented += has_any
        rungs[rung.value] = {
            "gate": status.gate_description,
            "gate_passed": status.gate_passed,
            "blockers": len(status.blockers),
            "criteria": len(keys),
            "criteria_measured": len(supplied),
            "criteria_not_measured": len(keys) - len(supplied),
            "missing": sorted(set(keys) - set(supplied)),
            "instrumented": has_any,
        }
    return {
        "rungs": rungs,
        "rungs_instrumented": instrumented,
        "rungs_total": len(rungs),
        "note": (
            "A rung with no instrumentation is a declaration, not a gate. The "
            "two SHADOW criteria are built and measured elsewhere — "
            "`cli audit` reports `safety_guarantee_violations` and "
            "`retrospective_concordance` — and are deliberately NOT fed here: "
            "`safety_events` means no patient was harmed while running beside "
            "clinicians, and `concordance` means agreement with clinicians, "
            "and a simulation supplies neither. Feeding them would let a gate "
            "read satisfied on simulated patients, which is the rung-skip this "
            "ladder exists to prevent."
        ),
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
            # Bespoke messages: every key here is populated by
            # `training.deployment_readiness()`, so absence is not a case that
            # arises, and the numbers are worth quoting.
            blockers.extend(self._silent_blockers(m))
        else:
            blockers.extend(_criteria(m, *_RUNG_CRITERIA.get(rung, ())))
        return blockers

    def _silent_blockers(self, m: dict[str, Any]) -> list[str]:
        """Everything that must hold before the model may run in shadow mode.

        Each condition is model-level and each one can actually fail. `live_data`
        is the structural one: a held-out split of the cohort the model was fit on
        is not live data, however well it scores.
        """
        blockers: list[str] = []

        ess = m.get("ope_effective_sample_size")
        if ess is None:
            blockers.append("held-out OPE effective sample size not measured")
        elif ess < MIN_OPE_EFFECTIVE_SAMPLE:
            blockers.append(
                f"held-out OPE effective sample size {ess:.0f} is below "
                f"{MIN_OPE_EFFECTIVE_SAMPLE:.0f}"
            )

        # The criterion above is *per-decision*: it keeps every stage-row where
        # the policy agreed with the arm given, weighted by that stage's own
        # propensity. What leaves silent mode is a three-stage regime, and the
        # regime's value requires agreement at every prior decision and the
        # cumulative propensity product — 44 rows at an effective sample of 14.6
        # against the per-decision 88 and 75.5. Gating a sequential deployment on
        # the per-decision number is the easier question wearing the harder
        # one's name.
        #
        # The doubly-robust estimate does not retire this gate. It augments with
        # the fitted Q-function, keeps all 120 trajectories and lands within half
        # a standard error of the known truth — so the regime's value is
        # identified *under the outcome model*. The IPW estimate is what could
        # falsify that model, and at this effective sample it has no power to.
        # A doubly-robust estimate whose IPW check cannot fail is a model-based
        # estimate wearing a robustness label, so the blocker names both.
        sequential = m.get("sequential_ope_effective_sample_size")
        if sequential is None:
            blockers.append(
                "the regime's own (sequential) OPE effective sample size is not measured"
            )
        elif sequential < MIN_OPE_EFFECTIVE_SAMPLE:
            surviving = m.get("regime_consistent_trajectories") or []
            tail = (
                f"; {surviving[-1]} of the holdout's trajectories follow the "
                f"regime to the end"
                if surviving
                else ""
            )
            doubly_robust = m.get("sequential_dr_value")
            interval = m.get("sequential_dr_interval")
            if doubly_robust is not None and interval:
                known = (
                    f" The doubly-robust estimate is {doubly_robust:.3f} "
                    f"({interval[0]:.3f} to {interval[1]:.3f}) over "
                    f"{m.get('sequential_dr_trajectories')} trajectories, so the "
                    f"value is identified under the outcome model — but the "
                    f"assumption-light check has no power to falsify that model."
                )
            else:
                known = ""
            blockers.append(
                f"the regime's value is not identified without leaning on the "
                f"outcome model: sequential IPW effective sample size "
                f"{sequential:.0f} is below {MIN_OPE_EFFECTIVE_SAMPLE:.0f}{tail}."
                f"{known}"
            )

        lower = m.get("ope_improvement_lower")
        if lower is None:
            blockers.append("improvement over the behaviour policy has no interval")
        elif lower <= 0.0:
            blockers.append(
                f"improvement over the behaviour policy is not separated from zero "
                f"(lower bound {lower:+.4f})"
            )

        if not m.get("calibration_passed", False):
            blockers.append("calibration not passing")

        # A flagged basis is a model-level blocker, not a patient-level one, and
        # it belongs here rather than in a caveat: `cli misspecification
        # --omitted-modifier` measures an omitted effect modifier costing up to
        # 65 points of interval coverage, and the interval narrows rather than
        # widens as it happens. A model whose contrasts are biased by an unknown
        # amount has no business advancing a rung.
        # `m.get(key, True)` was the shape here, so an **absent** key read as
        # "the basis is fine" — invariant 25's defect, in the one branch of this
        # module its fix did not reach, and defaulting the unsafe way. The
        # specification test runs on every fit so the key is always supplied
        # today and this could not arise; what makes it worth repairing is that
        # `calibration_passed` and `live_data` two lines either side both
        # default to *blocking*, so this was the only criterion here whose
        # silence was taken for a pass. Unmeasured and failing get different
        # words, which is what a reader deciding whether to advance needs.
        unflagged = m.get("blip_basis_unflagged")
        if unflagged is None:
            blockers.append(
                "the blip-basis specification test was not run, so whether the "
                "reported contrasts are biased by an omitted effect modifier is "
                "unknown"
            )
        elif not unflagged:
            flagged = ", ".join(m.get("flagged_modifiers", [])) or "a candidate covariate"
            blockers.append(
                f"the blip basis looks misspecified ({flagged} tests as an effect "
                f"modifier); the reported contrasts are covariate-averaged and their "
                f"intervals do not cover"
            )

        if not m.get("live_data", False):
            blockers.append(
                "no live data: the held-out estimate comes from the same generating "
                "process the model was fit on, so it cannot establish stability on "
                "live data"
            )
        return blockers
