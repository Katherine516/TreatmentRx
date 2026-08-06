from __future__ import annotations

from dataclasses import dataclass, field

from treatmentrx.contracts import FeedbackReceipt, PatientState, Recommendation, RecommendationStatus


@dataclass
class FeedbackLayer:
    """Layer 6: observational feedback and OPE/retraining governance."""

    observational_track: list[dict[str, object]] = field(default_factory=list)
    ope_track: list[dict[str, object]] = field(default_factory=list)

    def enqueue(self, state: PatientState, recommendation: Recommendation) -> FeedbackReceipt:
        observational_record = {
            "patient_hash": state.patient_hash,
            "stage": state.stage,
            "recommended_arm": recommendation.recommended_arm,
            "status": recommendation.status.value,
            "note": "Track A observational record; not a direct retraining target.",
        }
        self.observational_track.append(observational_record)

        ope_allowed = recommendation.status in {RecommendationStatus.RECOMMEND, RecommendationStatus.EQUIPOISE}
        if ope_allowed:
            self.ope_track.append(
                {
                    "patient_hash": state.patient_hash,
                    "stage": state.stage,
                    "policy_arm": recommendation.recommended_arm,
                    "note": "Track B candidate for later DR-OPE once outcomes are observed.",
                }
            )

        return FeedbackReceipt(
            observational_enqueued=True,
            ope_track_enqueued=ope_allowed,
            retraining_allowed=False,
            message=(
                "Feedback recorded. Retraining remains disabled until outcome validation, calibration checks, "
                "and OPE gates pass."
            ),
        )
