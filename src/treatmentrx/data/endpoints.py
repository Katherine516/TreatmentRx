"""What counts as a good outcome — made a choice instead of a keyword match.

`StageHistoryBuilder` used to score a stage by looking for "remission", "partial"
and "inadequate" in a free-text response field. That is the reward the whole
system optimises, defined by string matching on prose a clinician typed. It is
the single most consequential placeholder in Layer 1: change the mapping and
every Q-value, every contrast and every recommendation moves, and nothing about
the record says which mapping was used.

Two endpoints, and picking one is a deployment decision rather than a default:

* :class:`ResponseTextEndpoint` — the keyword mapping, kept because it is what
  reproduces this repo's synthetic cohort (see below) and because plenty of real
  extracts carry nothing better. It is a placeholder and says so.
* :class:`EULARResponseEndpoint` — the real thing: EULAR response from the DAS28
  change across the stage and the level attained, which is a published,
  auditable definition computed from measurements.

**They disagree on the simulated cohort, by construction, and that is not a
bug.** `simulation/ra_cohort` draws `outcome` as a synthetic 0..1 response
probability and then moves DAS28 by `3.0 * (outcome - 0.5)`. A patient with
outcome 0.55 therefore improves by 0.15 DAS28 points, which EULAR correctly calls
no response. The generator's scale and the clinical scale are simply different
scales; reconciling them would mean re-tuning the simulation to flatter a
particular endpoint, which is the one thing this repo's notes say not to do. So
the text endpoint stays the default *for the synthetic cohort*, and a real
deployment sets `EULARResponseEndpoint` — or its own, which is the point of the
seam existing.
"""

from __future__ import annotations

from dataclasses import dataclass

from treatmentrx.domain import Observation

# The reward scale, shared by both endpoints so they stay comparable: a good
# response, a moderate one, and a failure. The values are the ones the estimators
# were fit against and are not free to move independently of the cohort.
GOOD = 1.0
MODERATE = 0.55
NONE = 0.15
UNKNOWN = 0.5

# EULAR response, from the 1996 criteria: improvement in DAS28 across the stage,
# read against the level attained at the end of it.
LARGE_IMPROVEMENT = 1.2
MINIMAL_IMPROVEMENT = 0.6
LOW_ACTIVITY = 3.2
MODERATE_ACTIVITY = 5.1


class Endpoint:
    """How one stage's outcome is scored. One method, deliberately."""

    name = "endpoint"
    measured = False

    def score(
        self,
        response: str | None,
        observations: list[Observation],
        start_day: int,
        end_day: int | None,
        default: float = UNKNOWN,
    ) -> float:
        raise NotImplementedError


class ResponseTextEndpoint(Endpoint):
    """Keyword match on the free-text response. A placeholder, and labelled one.

    Kept as the default because it is what the FHIR exporter writes and therefore
    what reproduces the generator's outcomes end to end. On a real extract it is
    only as good as the vocabulary whoever typed the note happened to use.
    """

    name = "response_text"
    measured = False

    def score(self, response, observations, start_day, end_day, default=UNKNOWN) -> float:
        if response:
            normalized = response.lower()
            if "remission" in normalized or "good" in normalized:
                return GOOD
            if "partial" in normalized or "moderate" in normalized:
                return MODERATE
            if "inadequate" in normalized or "failure" in normalized or "none" in normalized:
                return NONE
        return default


@dataclass(frozen=True)
class EULARResponseEndpoint(Endpoint):
    """EULAR response, computed from the DAS28 the record actually carries.

    Needs two measurements: one at or before the decision, and one inside the
    stage that follows it. When either is missing the stage is not scoreable from
    measurements and the endpoint falls back to `fallback` rather than inventing
    a number — a missing endpoint and a failed one are different facts.

        attained DAS28   improvement > 1.2   0.6 < improvement <= 1.2   <= 0.6
        <= 3.2           good                moderate                   none
        3.2 .. 5.1       moderate            moderate                   none
        > 5.1            moderate            none                       none
    """

    name = "eular_response"
    measured = True
    fallback: Endpoint = ResponseTextEndpoint()

    def score(self, response, observations, start_day, end_day, default=UNKNOWN) -> float:
        baseline = _das28_at_or_before(observations, start_day)
        attained = _das28_within(observations, start_day, end_day)
        if baseline is None or attained is None:
            return self.fallback.score(response, observations, start_day, end_day, default)
        return eular_response(baseline, attained)


def eular_response(baseline: float, attained: float) -> float:
    """The EULAR table, as a reward on the shared scale."""
    improvement = baseline - attained
    if improvement > LARGE_IMPROVEMENT:
        return GOOD if attained <= LOW_ACTIVITY else MODERATE
    if improvement > MINIMAL_IMPROVEMENT:
        return MODERATE if attained <= MODERATE_ACTIVITY else NONE
    return NONE


def _das28_at_or_before(observations: list[Observation], day: int) -> float | None:
    """The most recent DAS28 the patient had when the decision was made."""
    return _latest(
        [
            observation
            for observation in observations
            if _is_das28(observation) and observation.days_from_baseline <= day
        ]
    )


def _das28_within(
    observations: list[Observation], start_day: int, end_day: int | None
) -> float | None:
    """The last DAS28 inside the stage — what the treatment achieved.

    An open stage (`end_day is None`) has no attained value: the patient is
    standing at the decision and the outcome has not happened. Scoring one would
    be reading the future.
    """
    if end_day is None:
        return None
    return _latest(
        [
            observation
            for observation in observations
            if _is_das28(observation) and start_day < observation.days_from_baseline <= end_day
        ]
    )


def _is_das28(observation: Observation) -> bool:
    return observation.code.lower().replace("-", "_").replace(" ", "_") == "das28"


def _latest(observations: list[Observation]) -> float | None:
    numeric = [
        observation
        for observation in observations
        if isinstance(observation.value, (int, float)) and not isinstance(observation.value, bool)
    ]
    if not numeric:
        return None
    return float(max(numeric, key=lambda observation: observation.days_from_baseline).value)


# What Layer 1 uses unless told otherwise. See the module docstring for why this
# is the text endpoint and not the measured one.
DEFAULT_ENDPOINT: Endpoint = ResponseTextEndpoint()


__all__ = [
    "DEFAULT_ENDPOINT",
    "GOOD",
    "MODERATE",
    "NONE",
    "UNKNOWN",
    "Endpoint",
    "EULARResponseEndpoint",
    "ResponseTextEndpoint",
    "eular_response",
]
