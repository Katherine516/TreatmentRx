"""Deterministic synthetic RA cohort with *known* per-arm blip functions.

A causal estimator is meaningless without data whose truth we know. This module
is the ground truth the Layer 4 estimators are fit on and validated against.

Design choices that make it a real test rather than a demo:

* **Multi-arm.** Every treatment arm has its own blip function
  ``tau_a(X) = psi_a . h(X)``, so the optimal arm genuinely varies by patient
  (seropositive patients favour rituximab, high-activity patients favour IL-6,
  prior-TNF failures lose most of the TNF benefit).
* **Confounded assignment.** The behaviour policy is a softmax over the same
  covariates that drive the blips, so naive arm means are biased and only a
  propensity-aware or outcome-model-based estimator recovers ``psi``.
* **Two stages with a delayed effect.** Hepatotoxic arms raise ALT, and the
  treatment-free model penalises high ALT at the next stage. A myopic policy
  over-prescribes them; only backward induction sees the delayed cost.
* **Positivity holds.** Every arm keeps non-trivial probability for every
  patient, so inverse-probability weights stay finite.

The true parameters are exported so tests can assert *recovery*, and
:func:`rollout_value` evaluates any policy against the generating process
itself — the oracle benchmark that held-out IPW estimates are compared to.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

# The arm vocabulary is clinical, not simulated: it is shared with the data
# contract and the safety filter. The reference arm carries a zero blip by
# construction, so every other arm's blip is identified against it.
from treatmentrx.arms import REFERENCE_ARM, TREATMENT_ARMS

# Blip basis h(X); psi vectors below are in this order.
BLIP_BASIS = ("intercept", "das28_std", "anti_ccp", "prior_tnf")

# True blip parameters: tau_a(X) = psi_a . h(X). Same at every stage — this is
# the "shared parameter" premise the SPTR estimators are built on.
TRUE_BLIPS: dict[str, tuple[float, float, float, float]] = {
    "continue-current": (0.00, 0.00, 0.00, 0.00),
    "methotrexate-optimization": (0.06, -0.04, 0.00, -0.03),
    "TNF-inhibitor": (0.12, 0.02, 0.01, -0.18),
    "IL-6 inhibitor": (0.15, 0.05, 0.00, 0.02),
    "JAK-inhibitor": (0.13, 0.04, -0.01, 0.03),
    "rituximab": (0.05, 0.01, 0.14, 0.06),
}

# Arms that raise ALT, whose cost only shows up at the *next* stage.
HEPATOTOXIC_ARMS = frozenset({"methotrexate-optimization", "JAK-inhibitor"})
_ALT_RISE = 30.0
_ALT_DECAY = 3.0
# Cost per unit of ALT above the reference ceiling, paid at the *following*
# stage. Sized to be comparable to a blip, so a myopic policy measurably loses.
_ALT_PENALTY = 0.005

# Behaviour policy: clinicians escalate with disease activity, avoid TNF after a
# prior TNF failure, and reach for rituximab in seropositive patients. Same
# drivers as the blips, so the assignment is confounded.
_ASSIGNMENT_SCORE: dict[str, tuple[float, float, float, float]] = {
    "continue-current": (0.30, -0.70, 0.00, -0.20),
    "methotrexate-optimization": (0.20, -0.40, 0.00, -0.30),
    "TNF-inhibitor": (0.40, 0.25, 0.00, -0.90),
    "IL-6 inhibitor": (0.10, 0.60, 0.00, 0.35),
    "JAK-inhibitor": (0.00, 0.55, -0.15, 0.45),
    "rituximab": (-0.30, 0.15, 0.70, 0.30),
}

_DAS28_MEAN = 5.5
_DAS28_SD = 1.5
_OUTCOME_NOISE_SD = 0.05

# Curvature in the treatment-free response to disease activity. Zero by default,
# which makes the estimators' linear basis correctly specified — and that is
# precisely why inverse-probability weighting is nearly a no-op on the default
# cohort: selection on a covariate the outcome model already conditions on
# correctly does not bias it.
#
# Turning this on misspecifies the *nuisance* function while leaving the blips
# linear and unchanged, so the estimand stays exactly the same and the only thing
# that varies is whether the outcome model can represent the surface it is fit
# on. That is the regime where re-balancing the covariate distribution by
# weighting earns its keep, and `cli misspecification` measures how much.
DEFAULT_CURVATURE = 0.0

DEFAULT_STAGES = 3

# --- Dropout ---------------------------------------------------------------
# Informative by construction: patients drop out when they are toxic or not
# responding, both of which are consequences of the arm they were given. Naive
# complete-case analysis is therefore biased, and only inverse-probability-of-
# censoring weighting recovers the blip. That is the ground truth
# `estimation/censoring.py` is validated against.
_DROPOUT_INTERCEPT = -1.9
_DROPOUT_ALT = 0.030  # per unit of ALT above the reference ceiling
_DROPOUT_RESPONSE = -3.0  # good responders stay

# Infusion therapies are burdensome, so patients abandon them unless they are
# clearly working. That makes retention *differentially* response-dependent by
# arm, which is what biases a blip contrast: among survivors on a high-burden
# arm, outcomes are more positively selected than among survivors on an oral
# one. Weighting by the estimated censoring probability is what removes it.
HIGH_BURDEN_ARMS = frozenset({"IL-6 inhibitor", "rituximab"})
_DROPOUT_BURDEN = 0.7  # baseline attrition on a burdensome arm
_DROPOUT_BURDEN_RESPONSE = -5.0  # ...but only if it is not working

# --- Visit timing ----------------------------------------------------------
# Sicker patients are seen sooner, so observation intensity is confounded with
# disease severity in exactly the way inverse-intensity weights exist to fix.
_BASE_INTERVAL_DAYS = 120.0
_INTERVAL_PER_SD = -22.0
_INTERVAL_NOISE = 15.0
_MIN_INTERVAL_DAYS = 28


@dataclass(frozen=True)
class CohortStage:
    """One decision point: covariates, the arm taken, and what followed."""

    stage: int
    day: int
    features: dict[str, float]
    arm: str
    propensity: float  # P(observed arm | features) under the behaviour policy
    outcome: float
    # P(this patient is still under observation at this stage). Stage 1 is 1.0
    # by construction; later stages carry the accumulated survival probability.
    uncensored_probability: float = 1.0
    interval_days: int | None = None  # days since the previous decision
    event: str = "ongoing"


@dataclass(frozen=True)
class CohortTrajectory:
    patient_index: int
    stages: tuple[CohortStage, ...]
    censored: bool = False
    censoring_reason: str | None = None

    @property
    def total_reward(self) -> float:
        return sum(stage.outcome for stage in self.stages)

    @property
    def n_observed(self) -> int:
        return len(self.stages)

    @property
    def terminal_event(self) -> str:
        return self.stages[-1].event if self.stages else "ongoing"


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def das28_std(das28: float) -> float:
    return (das28 - _DAS28_MEAN) / _DAS28_SD


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def blip_basis(features: dict[str, float]) -> list[float]:
    """h(X) — the covariates the treatment effect is allowed to vary over."""
    return [1.0, das28_std(features["das28"]), features["anti_ccp"], features["prior_tnf"]]


def true_blip(arm: str, features: dict[str, float]) -> float:
    """tau_a(X): the causal advantage of `arm` over the reference arm."""
    psi = TRUE_BLIPS.get(arm)
    if psi is None:
        return 0.0
    return sum(p * b for p, b in zip(psi, blip_basis(features)))


def treatment_free_value(features: dict[str, float], curvature: float = DEFAULT_CURVATURE) -> float:
    """f(X): expected outcome under the reference arm.

    Depends on the same confounders that drive assignment, plus ALT — which is
    how a stage-1 hepatotoxic choice is punished at stage 2.

    `curvature` adds a quadratic term in disease activity that the estimators'
    linear basis cannot represent. It changes only this nuisance function; the
    blips stay linear, so the estimand is untouched.
    """
    severity = das28_std(features["das28"])
    return (
        0.50
        - 0.11 * severity
        + curvature * severity * severity
        + 0.09 * features["anti_ccp"]
        - 0.06 * features["prior_tnf"]
        - 0.0012 * (features["crp"] - 30.0)
        - _ALT_PENALTY * max(features["alt"] - 40.0, 0.0)
    )


def expected_outcome(features: dict[str, float], arm: str) -> float:
    """E[Y | X, A] — noise-free, used by the oracle policy evaluator."""
    return clamp(treatment_free_value(features) + true_blip(arm, features), 0.0, 1.0)


def optimal_arm(features: dict[str, float]) -> str:
    """The myopically optimal arm for these covariates (ignores delayed cost)."""
    return max(TREATMENT_ARMS, key=lambda arm: true_blip(arm, features))


def assignment_probabilities(features: dict[str, float]) -> dict[str, float]:
    """The behaviour policy pi_b(a | X) that generated the observational data."""
    basis = blip_basis(features)
    scores = {
        arm: sum(k * b for k, b in zip(kappa, basis))
        for arm, kappa in _ASSIGNMENT_SCORE.items()
    }
    largest = max(scores.values())
    exponentiated = {arm: math.exp(score - largest) for arm, score in scores.items()}
    total = sum(exponentiated.values())
    return {arm: value / total for arm, value in exponentiated.items()}


def sample_baseline_features(rng: random.Random) -> dict[str, float]:
    return {
        "das28": rng.uniform(3.0, 8.0),
        "crp": rng.uniform(5.0, 60.0),
        "anti_ccp": 1.0 if rng.random() < 0.6 else 0.0,
        "prior_tnf": 1.0 if rng.random() < 0.35 else 0.0,
        "egfr": rng.uniform(45.0, 110.0),
        "alt": rng.uniform(12.0, 45.0),
    }


def transition(features: dict[str, float], arm: str, outcome: float, rng: random.Random) -> dict[str, float]:
    """X_{j+1} | X_j, A_j, Y_j — responders improve; hepatotoxic arms raise ALT."""
    response = outcome - 0.5
    alt_shift = _ALT_RISE if arm in HEPATOTOXIC_ARMS else -_ALT_DECAY
    return {
        "das28": clamp(features["das28"] - 3.0 * response + rng.gauss(0.0, 0.3), 1.5, 9.0),
        "crp": clamp(features["crp"] - 30.0 * response + rng.gauss(0.0, 4.0), 2.0, 120.0),
        "anti_ccp": features["anti_ccp"],
        "prior_tnf": 1.0 if features["prior_tnf"] or arm == "TNF-inhibitor" else 0.0,
        "egfr": clamp(features["egfr"] + rng.gauss(0.0, 2.0), 15.0, 130.0),
        "alt": clamp(features["alt"] + alt_shift + rng.gauss(0.0, 3.0), 5.0, 200.0),
    }


def _sample_arm(features: dict[str, float], rng: random.Random) -> tuple[str, float]:
    probabilities = assignment_probabilities(features)
    draw = rng.random()
    cumulative = 0.0
    for arm, probability in probabilities.items():
        cumulative += probability
        if draw <= cumulative:
            return arm, probability
    last = TREATMENT_ARMS[-1]
    return last, probabilities[last]


def generate_ra_cohort(
    n: int = 400,
    seed: int = 7,
    stages: int = DEFAULT_STAGES,
    dropout: bool = True,
    curvature: float = DEFAULT_CURVATURE,
) -> list[CohortTrajectory]:
    """Generate `n` confounded multi-stage trajectories under the behaviour policy.

    Trajectories end early when the patient drops out. Because dropout depends on
    toxicity and response — both consequences of the arm given — the censoring is
    informative, and the resulting cohort is what `estimation/censoring.py` is
    validated against. Pass `dropout=False` for the complete-data comparison.
    """
    rng = random.Random(seed)
    cohort: list[CohortTrajectory] = []
    for index in range(n):
        features = sample_baseline_features(rng)
        trajectory: list[CohortStage] = []
        day = 0
        interval: int | None = None
        survival = 1.0
        censored = False
        reason: str | None = None

        for stage in range(1, stages + 1):
            arm, propensity = _sample_arm(features, rng)
            outcome = clamp(
                treatment_free_value(features, curvature)
                + true_blip(arm, features)
                + rng.gauss(0.0, _OUTCOME_NOISE_SD),
                0.0,
                1.0,
            )
            leaving = dropout and stage < stages and rng.random() < dropout_probability(features, outcome, arm)
            if leaving:
                reason = dropout_reason(features, outcome)
            trajectory.append(
                CohortStage(
                    stage=stage,
                    day=day,
                    features=dict(features),
                    arm=arm,
                    propensity=round(propensity, 6),
                    outcome=round(outcome, 4),
                    uncensored_probability=round(survival, 6),
                    interval_days=interval,
                    event=reason or ("response" if outcome >= 0.7 else "ongoing"),
                )
            )
            if leaving:
                censored = True
                break
            if stage < stages:
                survival *= 1.0 - dropout_probability(features, outcome, arm)
                interval = visit_interval(features, rng)
                day += interval
                features = transition(features, arm, outcome, rng)

        cohort.append(
            CohortTrajectory(
                patient_index=index,
                stages=tuple(trajectory),
                censored=censored,
                censoring_reason=reason,
            )
        )
    return cohort


def train_test_split(
    cohort: list[CohortTrajectory],
    holdout_fraction: float = 0.3,
) -> tuple[list[CohortTrajectory], list[CohortTrajectory]]:
    """Deterministic split — trajectories are already in random order."""
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be strictly between 0 and 1")
    cut = int(len(cohort) * (1.0 - holdout_fraction))
    return cohort[:cut], cohort[cut:]


def rollout_value(
    policy,
    n: int = 2000,
    seed: int = 101,
    stages: int = DEFAULT_STAGES,
    dropout: bool = True,
) -> float:
    """Oracle policy value: expected total reward under the generating process.

    `policy(features, stage_index) -> arm`, with `stage_index` zero-based. Only
    available in simulation — it is the yardstick the observational IPW estimates
    in `feedback/offline_evaluation.py` are checked against.

    Dropout is simulated, so a policy that drives toxicity is penalised twice:
    once through the delayed ALT cost, and again through the visits it loses. A
    policy is worth what the patient actually accrues, not what they would have
    accrued had they stayed.
    """
    rng = random.Random(seed)
    total = 0.0
    for _ in range(n):
        features = sample_baseline_features(rng)
        for stage_index in range(stages):
            arm = policy(features, stage_index)
            outcome = clamp(
                treatment_free_value(features) + true_blip(arm, features) + rng.gauss(0.0, _OUTCOME_NOISE_SD),
                0.0,
                1.0,
            )
            total += outcome
            if stage_index >= stages - 1:
                break
            if dropout and rng.random() < dropout_probability(features, outcome, arm):
                break
            features = transition(features, arm, outcome, rng)
    return round(total / n, 4)


def rollout_retention(
    policy,
    n: int = 2000,
    seed: int = 101,
    stages: int = DEFAULT_STAGES,
) -> float:
    """Mean number of decision points a policy keeps the patient in care for."""
    rng = random.Random(seed)
    observed = 0
    for _ in range(n):
        features = sample_baseline_features(rng)
        for stage_index in range(stages):
            arm = policy(features, stage_index)
            outcome = clamp(
                treatment_free_value(features) + true_blip(arm, features) + rng.gauss(0.0, _OUTCOME_NOISE_SD),
                0.0,
                1.0,
            )
            observed += 1
            if stage_index >= stages - 1:
                break
            if rng.random() < dropout_probability(features, outcome, arm):
                break
            features = transition(features, arm, outcome, rng)
    return round(observed / n, 4)


def dropout_probability(features: dict[str, float], outcome: float, arm: str = REFERENCE_ARM) -> float:
    """P(patient leaves the study before the next decision point).

    Depends on toxicity, on response, and — for burdensome infusion arms — on
    response *more steeply*. All three are consequences of the arm given, so the
    censoring is informative and differential, and complete-case analysis is
    biased in a way that differs between arms.
    """
    burden = arm in HIGH_BURDEN_ARMS
    logit = (
        _DROPOUT_INTERCEPT
        + _DROPOUT_ALT * max(features["alt"] - 40.0, 0.0)
        + (_DROPOUT_RESPONSE + (_DROPOUT_BURDEN_RESPONSE if burden else 0.0)) * (outcome - 0.5)
        + (_DROPOUT_BURDEN if burden else 0.0)
    )
    return _logistic(logit)


def dropout_reason(features: dict[str, float], outcome: float) -> str:
    """Which competing event ended the trajectory."""
    if features["alt"] > 90.0:
        return "serious_toxicity"
    if outcome < 0.45:
        return "progression"
    return "dropout"


def visit_interval(features: dict[str, float], rng: random.Random) -> int:
    """Days until the next decision point — shorter for more active disease."""
    mean = _BASE_INTERVAL_DAYS + _INTERVAL_PER_SD * das28_std(features["das28"])
    return max(_MIN_INTERVAL_DAYS, int(round(rng.gauss(mean, _INTERVAL_NOISE))))


def behaviour_policy(features: dict[str, float], stage_index: int) -> str:
    """Sample from pi_b — the observed clinician policy, for benchmarking."""
    arm, _ = _sample_arm(features, random.Random(hash((round(features["das28"], 3), stage_index)) & 0xFFFF))
    return arm


def myopic_optimal_policy(features: dict[str, float], stage_index: int) -> str:
    """The best arm for *this* stage only — ignores the delayed ALT cost."""
    return optimal_arm(features)
