"""v5.1 #5 (L2) — Belief-aware estimation.

Estimators condition on PatientState.belief and propagate its uncertainty into
the epistemic component: a confident recommendation built on an *uncertain*
belief is correctly flagged uncertain downstream.
"""

from __future__ import annotations

from dataclasses import replace

from treatmentrx.contracts import RegimeEstimate
from treatmentrx.domain import StageRecord


class BeliefAwareAdjuster:
    """Widens the confidence band in proportion to belief uncertainty."""

    def adjust(self, result: RegimeEstimate, stages: list[StageRecord]) -> RegimeEstimate:
        belief = stages[-1].belief
        if belief is None:
            return result
        low, high = result.confidence_band
        widen = round(belief.uncertainty * 0.5, 3)
        band = (round(max(low - widen, 0.0), 3), round(min(high + widen, 1.0), 3))
        coefficients = result.coefficients | {
            "belief_activity": belief.activity,
            "belief_uncertainty": belief.uncertainty,
        }
        return replace(result, confidence_band=band, coefficients=coefficients)
