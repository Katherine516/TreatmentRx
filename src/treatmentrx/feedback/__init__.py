"""Layer 6 — feedback, estimands, off-policy evaluation, and deployment gating.

Everything here is *non-blocking*. Layer 5 has already produced the
recommendation by the time this runs; nothing in this layer may change it. What
it does instead is decide what the system is allowed to *learn*, and how far up
the validation ladder it is allowed to move.

Three separations that matter:

* **Track A vs Track B.** Observational records are logged for audit; only
  outcome-bearing records enter the off-policy evaluation track. Neither retrains
  anything on its own.
* **Estimands are reported side by side.** ITT, per-protocol and as-treated
  answer different clinical questions, so the system reports all three rather
  than silently picking the flattering one.
* **Overrides do not retrain the model.** A clinician disagreeing is routed to a
  review channel; only an override whose *outcome* was later confirmed may
  inform the misspecification analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from treatmentrx.contracts import FeedbackReceipt, PatientState, Recommendation, SafeDecision
from treatmentrx.domain import (
    OverrideRecord,
    OverrideRouting,
    RecommendationStatus,
    ValidationRung,
)
from treatmentrx.estimation import training
from treatmentrx.feedback.estimands import EstimandReporter
from treatmentrx.feedback.override_governance import OverrideRouter
from treatmentrx.feedback.switching_aware_ope import SwitchingAwareOPE
from treatmentrx.feedback.validation_ladder import ValidationLadder

# The system starts silent and stays there until a human moves it.
CURRENT_RUNG = ValidationRung.SILENT


@dataclass
class FeedbackLayer:
    observational_track: list[dict[str, object]] = field(default_factory=list)
    ope_track: list[dict[str, object]] = field(default_factory=list)
    estimands: EstimandReporter = field(default_factory=EstimandReporter)
    ope: SwitchingAwareOPE = field(default_factory=SwitchingAwareOPE)
    ladder: ValidationLadder = field(default_factory=ValidationLadder)
    override_router: OverrideRouter = field(default_factory=OverrideRouter)

    def enqueue(
        self,
        state: PatientState,
        recommendation: Recommendation,
        safe: SafeDecision | None = None,
    ) -> FeedbackReceipt:
        self.observational_track.append(
            {
                "patient_hash": state.patient_hash,
                "stage": state.stage,
                "recommended_arm": recommendation.recommended_arm,
                "status": recommendation.status.value,
                "note": "Track A observational record; not a direct retraining target.",
            }
        )

        ope_allowed = recommendation.status in {
            RecommendationStatus.RECOMMEND,
            RecommendationStatus.EQUIPOISE,
        }
        if ope_allowed:
            self.ope_track.append(
                {
                    "patient_hash": state.patient_hash,
                    "stage": state.stage,
                    "policy_arm": recommendation.recommended_arm,
                    "note": "Track B candidate for later DR-OPE once outcomes are observed.",
                }
            )

        estimand_results = []
        ope_result = None
        if safe is not None:
            estimand_results = self.estimands.report(state.stages, safe.decision.selected)
            ope_result = self.ope.evaluate(state.stages, safe.decision.selected)

        calibration = training.holdout_calibration()
        validation = self.ladder.assess(
            CURRENT_RUNG,
            {
                "ope_stable": ope_result.effective_sample_size > 0 if ope_result else False,
                "calibration_passed": calibration.passed,
            },
        )

        return FeedbackReceipt(
            observational_enqueued=True,
            ope_track_enqueued=ope_allowed,
            # Never true in this build, and deliberately so: automated retraining
            # is gated behind the validation ladder, not behind a code path.
            retraining_allowed=False,
            message=(
                "Feedback recorded. Retraining remains disabled until outcome validation, "
                "calibration checks, and OPE gates pass."
            ),
            estimands=estimand_results,
            validation=validation,
            ope=ope_result.__dict__ if ope_result else {},
        )

    def submit_override(
        self,
        recommendation: Recommendation,
        clinician_action: str,
        reason_text: str,
        outcome_confirmed: bool | None = None,
    ) -> OverrideRouting:
        """Route a clinician override to a review channel.

        The override never reaches the estimators directly. Routing decides which
        channel owns it and whether — only when the outcome was confirmed — it may
        inform the misspecification analysis.
        """
        return self.override_router.route(
            OverrideRecord(
                patient_hash=recommendation.patient_hash,
                recommended_arm=recommendation.recommended_arm or "none",
                clinician_action=clinician_action,
                reason_text=reason_text,
                outcome_confirmed_clinician=outcome_confirmed,
            )
        )


__all__ = ["FeedbackLayer"]
