"""v5.1 #7 (L6) — Override governance.

Overrides are a rich signal and a dangerous training target. The rule: overrides
never directly retrain the causal policy; they route to four review channels, and
only outcome-validated overrides influence the model at all.
"""

from __future__ import annotations

import re

from treatmentrx.domain import OverrideChannel, OverrideRecord, OverrideRouting


CHANNEL_OWNERS = {
    OverrideChannel.USABILITY: "product / UX",
    OverrideChannel.SAFETY_REVIEW: "clinical safety officer",
    OverrideChannel.GUIDELINE_CONFLICT: "guideline committee",
    OverrideChannel.POSSIBLE_MISSPECIFICATION: "modeling team",
}


# Tokens that positively identify a channel. Keyword matching is a placeholder
# for a real reason taxonomy — a production build captures a structured reason
# code at the point of override rather than parsing free text.
CHANNEL_TOKENS = (
    (OverrideChannel.USABILITY, ("unclear", "confusing", "couldn't tell", "hard to read", "ui")),
    (OverrideChannel.SAFETY_REVIEW, ("risk", "unsafe", "contraindicat", "danger", "adverse")),
    (OverrideChannel.GUIDELINE_CONFLICT, ("guideline", "protocol", "policy", "formulary")),
    (
        OverrideChannel.POSSIBLE_MISSPECIFICATION,
        ("model", "wrong", "misspecif", "estimate", "prediction", "disagree"),
    ),
)

# Where an override goes when nothing matches. It used to fall through to
# POSSIBLE_MISSPECIFICATION — the *only* channel that can reach the model — so
# every reason the keyword list did not recognise ("patient preference",
# "insurance denied", a typo) was routed by default to the one place with model
# influence. A governance component's unknown case belongs with a human, and a
# clinician overriding for a reason the taxonomy cannot classify is exactly the
# case a safety officer should see.
UNCLASSIFIED_CHANNEL = OverrideChannel.SAFETY_REVIEW

# Tokens shorter than this match whole words only. "ui" is a substring of
# "guideline", "requires", "build" and "quality", so plain `in` sent every
# override that mentioned a guideline to the usability channel and left
# GUIDELINE_CONFLICT unreachable by its own name. Longer tokens stay as prefixes
# so "contraindicat" still catches "contraindicated".
_WHOLE_WORD_BELOW = 4
_WORD = re.compile(r"[a-z']+")


class OverrideRouter:
    """Classifies an override by its captured reason into one of four channels."""

    def route(self, record: OverrideRecord) -> OverrideRouting:
        reason = record.reason_text.lower()
        channel = self.classify(reason)
        classified = channel is not None
        channel = channel or UNCLASSIFIED_CHANNEL

        # Two keys, and both are required: the override has to be *positively*
        # identified as a possible-misspecification signal, and its outcome has
        # to have been confirmed. Neither alone lets an override reach the model.
        influences = (
            classified
            and channel is OverrideChannel.POSSIBLE_MISSPECIFICATION
            and record.outcome_confirmed_clinician is True
        )
        return OverrideRouting(
            channel=channel,
            owner=CHANNEL_OWNERS[channel],
            influences_model=influences,
            rationale=self._rationale(channel, influences, classified),
        )

    def classify(self, reason: str) -> OverrideChannel | None:
        """The channel this reason names, or None when nothing matches."""
        words = set(_WORD.findall(reason))
        for channel, tokens in CHANNEL_TOKENS:
            if any(self._matches(token, reason, words) for token in tokens):
                return channel
        return None

    def _matches(self, token: str, reason: str, words: set[str]) -> bool:
        if " " in token:
            return token in reason
        if len(token) < _WHOLE_WORD_BELOW:
            return token in words
        return any(word.startswith(token) for word in words)

    def _rationale(self, channel: OverrideChannel, influences: bool, classified: bool) -> str:
        if influences:
            return "Outcome-validated misspecification signal: eligible for the misspecification analysis."
        if not classified:
            return (
                f"Reason text matched no channel; routed to {channel.value} for a human to "
                "classify. An unrecognised override never influences the causal policy."
            )
        if channel is OverrideChannel.POSSIBLE_MISSPECIFICATION:
            return "Possible misspecification, but outcome not yet validated: monitoring only, no model update."
        return f"Routed to {channel.value}: informs review/usability, never the causal policy."


class OverrideValidator:
    """An override becomes a modeling signal only after its outcome confirms it."""

    def is_modeling_signal(self, record: OverrideRecord) -> bool:
        return record.outcome_confirmed_clinician is True
