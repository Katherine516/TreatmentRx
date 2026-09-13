"""v5.1 #5 (L2) — Belief-aware estimation.

Latent disease activity is never observed directly, so `PatientState` carries a
filtered belief with its own uncertainty. This module routes that uncertainty to
the place that reports it, and — deliberately — keeps it out of the place that
does not.

**It no longer widens the confidence band.** It used to add
`0.5 * belief.uncertainty` to each end, and on the demo patient that turned a
measured (0.781, 0.837) into (0.593, 1.000): seven times wider, clipped at the
ceiling, and dominated by a hand-chosen multiplier rather than by anything
estimated. The band is the cluster-robust interval for the recommended arm's
blip; that is a statement about how well the parameters are determined, and
adding a filter's self-reported spread to it makes the number mean neither thing.

Two further reasons not to trust that widening. `BeliefStateFilter` uses the
`PROXIES` *mixing weights* as precisions, so `belief.uncertainty` is close to a
fixed function of which proxies happen to be present rather than of how noisy
they are — it lands near 0.29-0.38 for almost every patient. And the band is a
Q-value interval, while the equipoise decision reads the contrast; widening the
band therefore changed the audit trail and the narrative without changing any
decision, which is the worst of both.

The belief still reaches the output. It is attached to the coefficients, where
the explanation layer and the audit event can read it, and `Uncertainty` already
carries a separate epistemic channel for exactly this kind of quantity.
"""

from __future__ import annotations

from dataclasses import replace

from treatmentrx.contracts import RegimeEstimate
from treatmentrx.domain import StageRecord


class BeliefAwareAdjuster:
    """Attaches belief state to the estimate without moving a measured number."""

    def adjust(self, result: RegimeEstimate, stages: list[StageRecord]) -> RegimeEstimate:
        belief = stages[-1].belief
        if belief is None:
            return result
        return replace(
            result,
            coefficients=result.coefficients
            | {
                "belief_activity": belief.activity,
                "belief_uncertainty": belief.uncertainty,
                "belief_confident": 1.0 if belief.confident else 0.0,
            },
        )
