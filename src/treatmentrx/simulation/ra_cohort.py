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


# How strongly the true treatment effect varies over a covariate the estimators'
# blip basis does not contain. `curvature` bends the *nuisance* surface, which
# double robustness is supposed to survive — measured, dWOLS's `|A - pi|` weight
# removes 27-39% of the bias it creates (invariant 75) — while this bends the
# **estimand** directly, which nothing survives. The contrast is real but not as
# clean as it reads: the clamp in `expected_outcome` means curvature moves the
# realised estimand too, for up to a quarter of arm-pairs. It is the failure mode a real deployment actually has, since
# the true effect modifiers will not be exactly `anti_ccp` and `prior_tnf`.
#
# CRP is the natural choice: it is measured, it is in the treatment-free basis,
# and it is deliberately *not* in `BLIP_BASIS`. So an estimator can adjust for it
# as a confounder and still be unable to represent it as an effect modifier.
DEFAULT_BLIP_MODIFIER = 0.0

# Per-arm sensitivity to the omitted modifier. Signed, so it is not a uniform
# shift the intercept could absorb: the inflammatory arms benefit more in high
# CRP, the csDMARD less.
_OMITTED_MODIFIER_WEIGHT = {
    "methotrexate-optimization": -1.0,
    "TNF-inhibitor": 0.4,
    "IL-6 inhibitor": 1.0,
    "JAK-inhibitor": 0.5,
    "rituximab": -0.3,
}


def true_blip(
    arm: str,
    features: dict[str, float],
    blip_modifier: float = DEFAULT_BLIP_MODIFIER,
) -> float:
    """tau_a(X): the causal advantage of `arm` over the reference arm.

    With `blip_modifier` non-zero the effect also varies over standardised CRP,
    which `blip_basis` does not carry — so no amount of data lets the estimators
    recover it, and the contrast they report is the CRP-averaged one. That is a
    bias in the estimand rather than in the nuisance model, and it is what
    `feedback/misspecification.omitted_modifier_report` measures.
    """
    psi = TRUE_BLIPS.get(arm)
    if psi is None:
        return 0.0
    value = sum(p * b for p, b in zip(psi, blip_basis(features)))
    if blip_modifier:
        value += (
            blip_modifier
            * _OMITTED_MODIFIER_WEIGHT.get(arm, 0.0)
            * (features["crp"] - 30.0)
            / 25.0
        )
    return value


def treatment_free_value(features: dict[str, float], curvature: float = DEFAULT_CURVATURE) -> float:
    """f(X): expected outcome under the reference arm.

    Depends on the same confounders that drive assignment, plus ALT — which is
    how a stage-1 hepatotoxic choice is punished at stage 2.

    `curvature` adds a quadratic term in disease activity that the estimators'
    linear basis cannot represent. It changes only this function and
    `true_blip` never reads it, so the **declared** blips stay linear.

    **The realised estimand does move, and this docstring used to deny it.**
    `expected_outcome` clamps `treatment_free_value + true_blip` to [0, 1] as a
    sum, so once `curvature * severity^2` drives the baseline against a bound
    both arms saturate together and the contrast between them collapses.
    Measured on real stage covariates (3 cohorts of 400):

    | curvature | rows clamped | arm-pairs whose blip drifts | worst drift |
    | --- | --- | --- | --- |
    | 0.00 | 3.8% | 1.1% | 0.0836 |
    | 0.05 | 18.9% | 13.4% | 0.2294 |
    | 0.10 | 26.0% | 21.3% | 0.2336 |
    | 0.15 | 29.1% | 25.0% | 0.2357 |

    That is the whole of `misspecification.DEFAULT_CURVATURES`, so the study
    scoring fitted parameters against `TRUE_BLIPS` is comparing them to a
    quantity a quarter of the cohort no longer has. See invariant 75: the
    study's conclusion survives being rescored against the contrast the cohort
    actually has, but the apparent crossover at 0.15 does not.

    The clamp is **not** removed. It is what keeps the outcome a bounded
    response, every other result in the repo is measured through it, and
    re-tuning the generator to make a study read better is the thing this repo
    keeps warning against. What is fixed is the claim.
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


def expected_outcome(
    features: dict[str, float],
    arm: str,
    curvature: float = DEFAULT_CURVATURE,
    blip_modifier: float = DEFAULT_BLIP_MODIFIER,
) -> float:
    """E[Y | X, A] — noise-free, used by the oracle policy evaluator."""
    return clamp(
        treatment_free_value(features, curvature) + true_blip(arm, features, blip_modifier),
        0.0,
        1.0,
    )


def optimal_arm(features: dict[str, float]) -> str:
    """The myopically optimal arm for these covariates (ignores delayed cost).

    This is *not* the optimal policy for the sequential problem and must not be
    used as an oracle for one — it over-prescribes the hepatotoxic arms whose
    cost lands at the next stage, and it ignores that the burdensome arms lose
    patients to dropout. `oracle_arm` is the backward-induction optimum; measured
    over 4000 rollouts this myopic rule scores below every fitted estimator's
    policy, so scoring the agent's regret against it rewards the wrong behaviour.
    """
    return max(TREATMENT_ARMS, key=lambda arm: true_blip(arm, features))


def oracle_action_value(
    features: dict[str, float],
    arm: str,
    stage_index: int = 0,
    stages: int = DEFAULT_STAGES,
) -> float:
    """True value-to-go of taking `arm` now and playing optimally afterwards.

    Backward induction over the *generating process*, with the transition taken
    at its conditional mean rather than integrated by Monte Carlo. The recursion
    mirrors `rollout_value`: reward accrues only while the patient is still in
    care, so the continuation is weighted by the probability they return. A
    policy that drives toxicity or burden is charged for the visits it costs,
    which is the whole reason the myopic blip argmax is not the oracle.

    **It is a near-optimal reference, not a proven upper bound.** Certainty
    equivalence moves the `max` inside the expectation, so the value it computes
    sits below the true optimum by a Jensen gap. Measured over eight seeds at
    n=3000 rollouts each:

        oracle (this function)  2.1570      myopic blip argmax  2.1367
        Q-Shared + Penalized    2.1553      dWOLS-Shared        2.1319

    It clears the myopic rule by +0.0203 on every seed, which is what makes it a
    usable regret reference. It clears the fitted Q-Shared policy by only +0.0016
    against a paired standard deviation of 0.0040 — that policy is at the
    certainty-equivalent optimum to within the noise, and on individual seeds it
    can come out ahead. Read a small negative regret as "indistinguishable from
    optimal", not as a bug.
    """
    outcome = expected_outcome(features, arm)
    if stage_index >= stages - 1:
        return outcome
    survival = 1.0 - dropout_probability(features, outcome, arm)
    following = transition(features, arm, outcome)
    return outcome + survival * oracle_value(following, stage_index + 1, stages)


def oracle_value(
    features: dict[str, float], stage_index: int = 0, stages: int = DEFAULT_STAGES
) -> float:
    """V*(X, j) under the generating process."""
    return max(
        oracle_action_value(features, arm, stage_index, stages) for arm in TREATMENT_ARMS
    )


def oracle_arm(
    features: dict[str, float], stage_index: int = 0, stages: int = DEFAULT_STAGES
) -> str:
    """The sequentially optimal arm: the one the agent's regret is measured against."""
    return max(
        TREATMENT_ARMS,
        key=lambda arm: oracle_action_value(features, arm, stage_index, stages),
    )


def oracle_policy(stages: int = DEFAULT_STAGES):
    """`policy(features, stage_index) -> arm` for the backward-induction optimum."""

    def policy(features: dict[str, float], stage_index: int) -> str:
        return oracle_arm(features, stage_index, stages)

    return policy


@dataclass(frozen=True)
class CohortShift:
    """A structurally different site, with the *same* treatment effects.

    The estimand is deliberately untouched: `TRUE_BLIPS` is the same function of
    the same covariates, and it stays representable in `BLIP_BASIS`. What differs
    is everything around it — who walks through the door, how clinicians
    prescribe, and who stops coming back. That separation is the point. Bending
    the estimand is already measured by `blip_modifier`, and nothing survives it;
    the open question is whether a pipeline that is *correctly specified* still
    works when the population and the practice change, which is the shift a real
    deployment actually meets.

    `blip_scale` is the exception and is off by default: it multiplies every true
    effect, so the parameters fitted at the training site are genuinely wrong at
    the evaluation site. It exists to bound the other results, not to be mixed
    with them.
    """

    name: str = "unshifted"
    # Case mix — a different population walking through the door.
    das28_range: tuple[float, float] = (3.0, 8.0)
    crp_range: tuple[float, float] = (5.0, 60.0)
    anti_ccp_rate: float = 0.6
    prior_tnf_rate: float = 0.35
    # Practice — clinicians here prefer different arms, by an additive tilt on
    # each arm's assignment intercept.
    assignment_tilt: tuple[tuple[str, float], ...] = ()
    # Retention — more or less attrition, on the logit scale.
    dropout_shift: float = 0.0
    # The bound case. 1.0 leaves the estimand alone.
    blip_scale: float = 1.0

    @property
    def shifts_the_estimand(self) -> bool:
        return self.blip_scale != 1.0


UNSHIFTED = CohortShift()


def assignment_probabilities(
    features: dict[str, float], shift: CohortShift | None = None
) -> dict[str, float]:
    """The behaviour policy pi_b(a | X) that generated the observational data."""
    basis = blip_basis(features)
    tilt = dict(shift.assignment_tilt) if shift is not None else {}
    scores = {
        arm: sum(k * b for k, b in zip(kappa, basis)) + tilt.get(arm, 0.0)
        for arm, kappa in _ASSIGNMENT_SCORE.items()
    }
    largest = max(scores.values())
    exponentiated = {arm: math.exp(score - largest) for arm, score in scores.items()}
    total = sum(exponentiated.values())
    return {arm: value / total for arm, value in exponentiated.items()}


def sample_baseline_features(
    rng: random.Random, shift: CohortShift | None = None
) -> dict[str, float]:
    """Baseline X. With `shift=None` the draw order and values are unchanged.

    Determinism matters more than tidiness here: every seeded result in the repo
    depends on this consuming exactly six numbers from `rng` in this order, so a
    shift changes the *parameters* of each draw and never the sequence.
    """
    s = shift if shift is not None else UNSHIFTED
    return {
        "das28": rng.uniform(*s.das28_range),
        "crp": rng.uniform(*s.crp_range),
        "anti_ccp": 1.0 if rng.random() < s.anti_ccp_rate else 0.0,
        "prior_tnf": 1.0 if rng.random() < s.prior_tnf_rate else 0.0,
        "egfr": rng.uniform(45.0, 110.0),
        "alt": rng.uniform(12.0, 45.0),
    }


def transition(
    features: dict[str, float],
    arm: str,
    outcome: float,
    rng: random.Random | None = None,
) -> dict[str, float]:
    """X_{j+1} | X_j, A_j, Y_j — responders improve; hepatotoxic arms raise ALT.

    `rng=None` gives the noise-free (conditional-mean) transition, which is what
    the oracle's backward induction integrates over. One definition of the
    dynamics, used by both the generator and the oracle, so they cannot drift.
    """
    noise = (lambda sd: rng.gauss(0.0, sd)) if rng is not None else (lambda sd: 0.0)
    response = outcome - 0.5
    alt_shift = _ALT_RISE if arm in HEPATOTOXIC_ARMS else -_ALT_DECAY
    return {
        "das28": clamp(features["das28"] - 3.0 * response + noise(0.3), 1.5, 9.0),
        "crp": clamp(features["crp"] - 30.0 * response + noise(4.0), 2.0, 120.0),
        "anti_ccp": features["anti_ccp"],
        "prior_tnf": 1.0 if features["prior_tnf"] or arm == "TNF-inhibitor" else 0.0,
        "egfr": clamp(features["egfr"] + noise(2.0), 15.0, 130.0),
        "alt": clamp(features["alt"] + alt_shift + noise(3.0), 5.0, 200.0),
    }


def _sample_arm(
    features: dict[str, float], rng: random.Random, shift: CohortShift | None = None
) -> tuple[str, float]:
    probabilities = assignment_probabilities(features, shift)
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
    blip_modifier: float = DEFAULT_BLIP_MODIFIER,
    shift: CohortShift | None = None,
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
        features = sample_baseline_features(rng, shift)
        trajectory: list[CohortStage] = []
        day = 0
        interval: int | None = None
        survival = 1.0
        censored = False
        reason: str | None = None

        for stage in range(1, stages + 1):
            arm, propensity = _sample_arm(features, rng, shift)
            scale = shift.blip_scale if shift is not None else 1.0
            outcome = clamp(
                treatment_free_value(features, curvature)
                + scale * true_blip(arm, features, blip_modifier)
                + rng.gauss(0.0, _OUTCOME_NOISE_SD),
                0.0,
                1.0,
            )
            leaving = (
                dropout
                and stage < stages
                and rng.random() < dropout_probability(features, outcome, arm, shift)
            )
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
    shift: CohortShift | None = None,
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
    scale = shift.blip_scale if shift is not None else 1.0
    total = 0.0
    for _ in range(n):
        features = sample_baseline_features(rng, shift)
        for stage_index in range(stages):
            arm = policy(features, stage_index)
            outcome = clamp(
                treatment_free_value(features)
                + scale * true_blip(arm, features)
                + rng.gauss(0.0, _OUTCOME_NOISE_SD),
                0.0,
                1.0,
            )
            total += outcome
            if stage_index >= stages - 1:
                break
            if dropout and rng.random() < dropout_probability(features, outcome, arm, shift):
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


def dropout_probability(
    features: dict[str, float],
    outcome: float,
    arm: str = REFERENCE_ARM,
    shift: CohortShift | None = None,
) -> float:
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
        + (shift.dropout_shift if shift is not None else 0.0)
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
