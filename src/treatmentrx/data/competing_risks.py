"""v5.1 #2 — Competing risks / survival.

Replaces single-event censoring with a competing-events model: progression,
death, dropout, serious toxicity, and treatment switch are distinct events.
The outcome becomes a typed (type, time) event, not a scalar — so Layer 2
estimators can tell "died" from "switched" from "censored" (IPCW alone
conflates them).
"""

from __future__ import annotations

from treatmentrx.domain import ClinicalEvent, ClinicalEventType, StageRecord


class CompetingRiskBuilder:
    """Classifies each stage's terminating event from its response/discontinuation."""

    def apply(self, stages: list[StageRecord]) -> list[StageRecord]:
        annotated: list[StageRecord] = []
        for index, stage in enumerate(stages):
            is_last = index + 1 == len(stages)
            event = self._classify(stage, is_last)
            annotated.append(self._replace(stage, event=event))
        return annotated

    def _classify(self, stage: StageRecord, is_last: bool) -> ClinicalEvent:
        text = " ".join(
            str(part) for part in (stage.response, stage.switching.discontinuation_reason if stage.switching else None)
        ).lower()
        time = stage.end_day

        if "death" in text or "died" in text:
            return ClinicalEvent(ClinicalEventType.DEATH, time)
        if "progress" in text or "failure" in text or "inadequate" in text:
            return ClinicalEvent(ClinicalEventType.PROGRESSION, time)
        if "toxicity" in text or "adverse" in text or "intoleran" in text:
            return ClinicalEvent(ClinicalEventType.SERIOUS_TOXICITY, time)
        if "dropout" in text or "lost to follow" in text:
            return ClinicalEvent(ClinicalEventType.DROPOUT, time)
        if stage.switching and stage.switching.switched:
            return ClinicalEvent(ClinicalEventType.TREATMENT_SWITCH, time)
        if "remission" in text or "good" in text or "response" in text:
            return ClinicalEvent(ClinicalEventType.RESPONSE, time)
        if is_last:
            return ClinicalEvent(ClinicalEventType.ONGOING, None)
        return ClinicalEvent(ClinicalEventType.ONGOING, time)

    def cumulative_incidence(self, stages: list[StageRecord]) -> dict[str, float]:
        """Naive cause-specific cumulative incidence over the trajectory.

        Placeholder for Aalen-Johansen CIFs — keeps the contract so Layer 2 can
        consume per-cause incidence without a rewrite later.
        """
        counts: dict[str, int] = {}
        terminating = [s.event for s in stages if s.event and s.event.type is not ClinicalEventType.ONGOING]
        n = max(len(stages), 1)
        for event in terminating:
            counts[event.type.value] = counts.get(event.type.value, 0) + 1
        return {cause: round(count / n, 4) for cause, count in counts.items()}

    def _replace(self, stage: StageRecord, **updates: object) -> StageRecord:
        return StageRecord(**(stage.__dict__ | updates))
