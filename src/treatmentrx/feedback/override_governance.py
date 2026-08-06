"""v5.1 #7 (L6) — Override governance.

Overrides are a rich signal and a dangerous training target. The rule: overrides
never directly retrain the causal policy; they route to four review channels, and
only outcome-validated overrides influence the model at all.
"""

from __future__ import annotations

from treatmentrx.domain import OverrideChannel, OverrideRecord, OverrideRouting


CHANNEL_OWNERS = {
    OverrideChannel.USABILITY: "product / UX",
    OverrideChannel.SAFETY_REVIEW: "clinical safety officer",
    OverrideChannel.GUIDELINE_CONFLICT: "guideline committee",
    OverrideChannel.POSSIBLE_MISSPECIFICATION: "modeling team",
}


class OverrideRouter:
    """Classifies an override by its captured reason into one of four channels."""

    def route(self, record: OverrideRecord) -> OverrideRouting:
        reason = record.reason_text.lower()
        if any(t in reason for t in ("unclear", "confusing", "couldn't tell", "hard to read", "ui")):
            channel = OverrideChannel.USABILITY
        elif any(t in reason for t in ("risk", "unsafe", "contraindicat", "danger", "adverse")):
            channel = OverrideChannel.SAFETY_REVIEW
        elif any(t in reason for t in ("guideline", "protocol", "policy", "formulary")):
            channel = OverrideChannel.GUIDELINE_CONFLICT
        else:
            channel = OverrideChannel.POSSIBLE_MISSPECIFICATION

        # Only outcome-validated possible-misspecification overrides may influence the model.
        influences = (
            channel is OverrideChannel.POSSIBLE_MISSPECIFICATION
            and record.outcome_confirmed_clinician is True
        )
        return OverrideRouting(
            channel=channel,
            owner=CHANNEL_OWNERS[channel],
            influences_model=influences,
            rationale=self._rationale(channel, influences),
        )

    def _rationale(self, channel: OverrideChannel, influences: bool) -> str:
        if influences:
            return "Outcome-validated misspecification signal: eligible for the misspecification analysis."
        if channel is OverrideChannel.POSSIBLE_MISSPECIFICATION:
            return "Possible misspecification, but outcome not yet validated: monitoring only, no model update."
        return f"Routed to {channel.value}: informs review/usability, never the causal policy."


class OverrideValidator:
    """An override becomes a modeling signal only after its outcome confirms it."""

    def is_modeling_signal(self, record: OverrideRecord) -> bool:
        return record.outcome_confirmed_clinician is True
