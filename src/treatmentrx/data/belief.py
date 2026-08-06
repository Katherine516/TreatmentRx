"""v5.1 #5 — Partial observability.

Latent disease activity (true inflammatory burden) is never directly observed.
This produces a *belief* over the hidden state from observed proxies — a
filtered estimate b(latent | H_j) with its own uncertainty. The state carries
the belief, not a false point value.

`POMDPInterface` is a deliberate seam: today the belief is a filter feeding the
existing estimators; later it can become an explicit belief-state policy without
a rewrite.
"""

from __future__ import annotations

from treatmentrx.domain import BeliefState, StageRecord


# Proxy -> (normalizer, weight). Observed proxies are noisy reads of the latent
# inflammatory burden; we combine them into a 0..1 activity estimate.
PROXIES = {
    "das28": (10.0, 0.5),
    "crp": (100.0, 0.25),
    "esr": (100.0, 0.15),
    "haq_di": (3.0, 0.10),
}


class BeliefStateFilter:
    """A simple precision-weighted filter over the latent activity scale (0..1)."""

    def apply(self, stages: list[StageRecord]) -> list[StageRecord]:
        annotated: list[StageRecord] = []
        prior_activity = 0.5
        prior_uncertainty = 0.3
        for stage in stages:
            belief = self._filter(stage, prior_activity, prior_uncertainty)
            prior_activity, prior_uncertainty = belief.activity, belief.uncertainty
            annotated.append(self._replace(stage, belief=belief))
        return annotated

    def _filter(self, stage: StageRecord, prior_activity: float, prior_uncertainty: float) -> BeliefState:
        observations: list[tuple[float, float]] = []
        used: list[str] = []
        for proxy, (norm, weight) in PROXIES.items():
            value = stage.features.get(proxy)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                observations.append((min(max(float(value) / norm, 0.0), 1.0), weight))
                used.append(proxy)

        if not observations:
            # No fresh proxy: belief decays toward the prior, uncertainty grows.
            return BeliefState(round(prior_activity, 4), round(min(prior_uncertainty + 0.05, 0.5), 4), used)

        obs_mean = sum(v * w for v, w in observations) / sum(w for _, w in observations)
        obs_precision = sum(w for _, w in observations)
        prior_precision = 1.0 / max(prior_uncertainty ** 2, 1e-6)
        # Bayesian update of a Gaussian-ish activity estimate.
        posterior = (prior_precision * prior_activity + obs_precision * obs_mean) / (prior_precision + obs_precision)
        uncertainty = (1.0 / (prior_precision + obs_precision)) ** 0.5
        return BeliefState(round(posterior, 4), round(min(uncertainty, 0.5), 4), used)

    def _replace(self, stage: StageRecord, **updates: object) -> StageRecord:
        return StageRecord(**(stage.__dict__ | updates))


class POMDPInterface:
    """Seam for a future explicit belief-state policy.

    Today this just exposes the current belief as the policy input. The contract
    exists now so swapping in a real POMDP solver is not a rewrite.
    """

    def belief_input(self, stages: list[StageRecord]) -> dict[str, float]:
        latest = stages[-1]
        belief = latest.belief
        if belief is None:
            return {"activity": float(latest.outcome), "uncertainty": 0.3}
        return {"activity": belief.activity, "uncertainty": belief.uncertainty}
