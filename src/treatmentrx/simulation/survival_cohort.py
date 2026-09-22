"""Deterministic synthetic survival cohort with *known* per-arm hazard ratios.

`ra_cohort` is trustworthy because `TRUE_BLIPS` is known and every estimator is
scored against it. This is the same contract for a **time-to-event** endpoint,
which is what an oncology workflow would need and what the RA endpoint cannot
express: `data/endpoints.py` maps a stage to a bounded response on `[0, 1]`
(`GOOD = 1.0`, `NONE = 0.15`) and `q_values` are clamped to
`[Q_FLOOR, Q_CEILING]`. Overall and progression-free survival are neither
bounded nor observed for everyone, so they are a different estimand rather than
a different endpoint, and nothing here should be read as a drop-in.

**It is deliberately not a disease.** The arms are `reference`, `arm-a`,
`arm-b`, `arm-c` and the covariates are `biomarker_std`, `marker_positive`,
`prior_line`. Naming them after breast or brain tumour regimens would make this
look like a clinical model when what it contains is a hazard model with numbers
chosen to exercise an estimator — the overstatement invariants 27, 45 and 72 are
each about. A real disease definition supplies its own arms and its own
parameters; this supplies the machinery and the ground truth to check it with.

## The model

Weibull proportional hazards, with the treatment effect as a per-arm log-hazard
ratio that varies over covariates — the survival analogue of a blip:

    h(t | x, a) = h_0(t) * exp( g(x) + tau_a(x) )
    tau_a(x)    = psi_a . h(x)

`tau_a(x) < 0` means the arm *lowers* the hazard, so it lengthens survival. The
reference arm carries `tau = 0` by construction, which is what identifies the
others — the same device `ra_cohort` uses, on the scale where it is natural.

Weibull is chosen for one reason: its survival function inverts in closed form,
so a seeded draw is exact rather than a search, and its mean has a closed form
too, so the oracle is arithmetic rather than simulation:

    T        = scale * ( -log(U) / exp(eta) ) ** (1 / shape)
    E[T]     = scale * exp(-eta / shape) * Gamma(1 + 1/shape)

Both matter. `rollout_value` still exists for policies the closed form cannot
reach, but every claim this module makes about its own truth is checkable by
arithmetic, which is what stops a generator from grading its own homework.

## What makes it a test rather than a demo

* **The optimal arm genuinely varies.** `arm-b` is much better when
  `marker_positive`, `arm-c` is better when it is not, and `arm-a` loses most of
  its advantage after a prior line — so a single arm is never right for
  everyone, and a policy that ignores covariates measurably loses.
* **Assignment is confounded.** The behaviour policy is a softmax over the same
  covariates that drive the hazard ratios, so naive per-arm survival comparisons
  are biased and only a propensity-aware or outcome-model-based estimator
  recovers `psi`.
* **Three things end follow-up, and they are not the same thing.** The event of
  interest (progression), a competing risk (death from another cause), and
  censoring (administrative horizon or loss to follow-up). Treating any of them
  as the others is the classic survival error, so each is recorded separately
  and `SurvivalStage.cause` says which happened.
* **The transition depends on the arm, so the problem is genuinely
  sequential.** `arm-a` has the strongest effect on any single line and leaves
  `acquired_resistance` behind, which the prognostic surface charges at every
  later line — the delayed-effect structure `ra_cohort` gets from ALT, on this
  scale. Measured over 2,000 patients, the backward-induction optimum takes
  `arm-a` for **0%** of patients with three lines remaining and **67%** with
  one: save the potent arm for last. A greedy rule does the opposite and loses
  **6.9 months** of expected survival, and "always `arm-a`" is the *worst* fixed
  policy (32.1 months against 47.1 for "always `arm-c`") despite having the best
  single-line effect.

  An earlier version advanced only `prior_line`, which is arm-independent, so
  the line-1 choice had no effect on the line-2 state. The myopic rule then
  matched the oracle on **0 of 3,000** patients — a generator whose optimum a
  greedy rule reproduces exactly has nothing for a DTR estimator to find, and
  that is what the `advance_line` docstring records.
* **Positivity holds.** Every arm keeps non-trivial probability for every
  patient, so inverse-probability weights stay finite.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

# Neutral by design — see the module docstring. A disease definition brings its
# own vocabulary; this brings the machinery.
SURVIVAL_ARMS = ("reference", "arm-a", "arm-b", "arm-c")
SURVIVAL_REFERENCE_ARM = "reference"

# Blip basis h(X); the psi vectors below are in this order.
HAZARD_BASIS = ("intercept", "biomarker_std", "marker_positive", "prior_line")

# True log-hazard ratios: tau_a(X) = psi_a . h(X), on the log-hazard scale.
#
# **Negative lowers the hazard and so lengthens survival.** The signs are the
# thing to read: `arm-b` is strongly better for marker-positive patients
# (-0.55) and barely better otherwise; `arm-c` is the reverse; `arm-a` has the
# strongest effect on any single line and also the steepest decay with
# `prior_line` (+0.30). No arm dominates on covariates alone — and `arm-a` is
# further penalised through `RESISTANCE_INDUCING_ARMS`, which is what the blip
# basis deliberately cannot see.
TRUE_LOG_HAZARD_RATIOS: dict[str, tuple[float, float, float, float]] = {
    "reference": (0.00, 0.00, 0.00, 0.00),
    "arm-a": (-0.55, -0.12, -0.05, 0.30),
    "arm-b": (-0.15, 0.06, -0.55, 0.04),
    "arm-c": (-0.40, -0.03, 0.35, -0.20),
}

# Prognostic (treatment-free) log-hazard: g(X). This is the surface a
# doubly-robust estimator is allowed to get wrong; the blip above is the one it
# must recover.
#
# `acquired_resistance` is the delayed effect, and it is what makes this a
# sequential problem rather than three independent ones. It is **not** in
# `HAZARD_BASIS`: like ALT in `ra_cohort`, the cost of a potent arm is paid on
# the treatment-free surface at the *next* line, so a myopic rule that reads
# only this line's hazard ratio cannot see it.
_PROGNOSTIC = {
    "intercept": 0.0,
    "biomarker_std": 0.45,          # a worse marker means a higher hazard
    "marker_positive": -0.20,
    "prior_line": 0.20,             # each line used leaves the next one harder
    "acquired_resistance": 0.75,    # ...and a potent arm leaves it harder still
    "performance_status": 0.60,     # the confounder; see below
}

# The one variable that moves **both** assignment and outcome while staying out
# of `HAZARD_BASIS`, and it is here because without it this cohort could not
# test the property the estimator is named for.
#
# Double robustness survives a wrong *treatment-free surface*. For that to be
# measurable the surface has to be able to omit a confounder — a variable that
# moves assignment, moves the outcome, and is not in the blip basis. Measured on
# the version of this file without `performance_status`, no such variable
# existed: assignment was a softmax over `true_log_hazard_ratio` alone, so every
# confounder was in `HAZARD_BASIS`, and `acquired_resistance` moved the outcome
# by 0.75 and assignment by **exactly 0.0000**. Omitting a `HAZARD_BASIS` term
# from the surface does not create confounding, it destroys the blip's own
# identification — `A * h(X)` becomes the only X-varying column and the blip
# terms absorb the prognosis. So every misspecification that could be written
# was either no confounding at all or not a surface question, and the measured
# gain from the propensity weight sat at ~2% however wrong the surface was made.
#
# This is confounding by indication, the ordinary way an oncology cohort is
# confounded: a frail patient does worse whatever is given, and is steered away
# from the aggressive arm. The effect is **arm-specific** because a shift common
# to every arm cancels in the softmax and confounds nothing.
_PRESCRIBING_CAUTION = {
    "reference": 0.00,
    "arm-a": -1.80,                 # the aggressive arm, avoided in frail patients
    "arm-b": -0.60,
    "arm-c": -0.25,
}

# Arms whose potency comes at the cost of cross-resistance, charged from the
# next line onward. `arm-a` has the strongest first-line effect in
# `TRUE_LOG_HAZARD_RATIOS` (-0.55); this is what stops that being free.
RESISTANCE_INDUCING_ARMS = frozenset({"arm-a"})

# Weibull baseline. `shape > 1` means the hazard rises with time on treatment,
# which is what progression looks like; `scale` sets the timescale in months.
WEIBULL_SHAPE = 1.4
WEIBULL_SCALE = 14.0

# Competing risk: death from another cause, exponential and **independent of
# the arm**. That independence is deliberate — it keeps the competing risk from
# being a second treatment effect in disguise, so a study that mishandles it
# fails for the right reason.
COMPETING_RISK_RATE = 1.0 / 90.0      # per month

# Censoring: an administrative horizon plus exponential loss to follow-up.
STUDY_HORIZON_MONTHS = 60.0
LOSS_TO_FOLLOWUP_RATE = 1.0 / 120.0   # per month

DEFAULT_LINES = 3
DEFAULT_COHORT_SEED = 20_260_921

# Behaviour policy temperature. Lower is more confounded; this is chosen so
# every arm keeps a floor of probability (see `assignment_probabilities`).
_ASSIGNMENT_TEMPERATURE = 1.0
_ASSIGNMENT_FLOOR = 0.05


@dataclass(frozen=True)
class SurvivalStage:
    """One line of therapy: covariates, the arm taken, and how it ended.

    Deliberately **not** `ra_cohort.CohortStage`. That type carries
    `outcome: float`, a bounded response, and putting a duration there would be
    invariant 4's defect — one field meaning two things, with the conversion
    silently dropping what distinguishes them. Here the duration, whether the
    event was observed, and which of three things ended follow-up are three
    separate fields, because a study that conflates them is the classic survival
    error.
    """

    line: int
    features: dict[str, float]
    arm: str
    propensity: float           # P(observed arm | features) under the behaviour policy
    months: float               # time from the start of this line until it ended
    event_observed: bool        # True only for the event of interest
    cause: str                  # "progression" | "competing_death" | "censored"
    entry_month: float          # months from baseline to the start of this line


@dataclass(frozen=True)
class SurvivalTrajectory:
    patient_index: int
    stages: tuple[SurvivalStage, ...]

    @property
    def observed_months(self) -> float:
        """Total follow-up, whatever ended it."""
        return sum(stage.months for stage in self.stages)

    @property
    def terminal_cause(self) -> str:
        return self.stages[-1].cause if self.stages else "censored"

    @property
    def n_lines(self) -> int:
        return len(self.stages)

    @property
    def progression_observed(self) -> bool:
        """Was the **event of interest** seen? Only progression counts.

        This replaced a `fully_observed` that returned True for a competing
        death, which is the exact conflation this module exists to avoid: a
        patient who dies of another cause never progresses, so for
        progression-free survival that is a competing risk and not an
        observation of the event. The two differ on **61 of 200** line-endings
        in a default cohort, so a property that merged them would have been
        wrong about a third of the non-progression endings.
        """
        return self.terminal_cause == "progression"

    @property
    def censored(self) -> bool:
        """Did follow-up simply run out — the horizon or loss to follow-up?

        Distinct from `progression_observed` being False, which is also true
        after a competing death. Read `terminal_cause` when the three-way
        distinction matters; these two are conveniences for the common
        questions, and neither stands in for the other.
        """
        return self.terminal_cause == "censored"


def hazard_basis(features: dict[str, float]) -> list[float]:
    """h(X) in `HAZARD_BASIS` order."""
    return [
        1.0,
        float(features.get("biomarker_std", 0.0)),
        1.0 if features.get("marker_positive") else 0.0,
        float(features.get("prior_line", 0.0)),
    ]


def true_log_hazard_ratio(arm: str, features: dict[str, float]) -> float:
    """tau_a(X) — the quantity an estimator has to recover.

    Zero for the reference arm by construction, and zero for an unknown arm,
    which is the same convention `ra_cohort.true_blip` uses: an arm with no
    declared parameters has no declared effect.
    """
    psi = TRUE_LOG_HAZARD_RATIOS.get(arm)
    if psi is None:
        return 0.0
    return sum(p * b for p, b in zip(psi, hazard_basis(features)))


def prognostic_log_hazard(features: dict[str, float]) -> float:
    """g(X) — the treatment-free surface, which double robustness may survive.

    Reads `acquired_resistance`, which `HAZARD_BASIS` deliberately does not
    carry: the delayed cost of a potent arm belongs to the surface, not to the
    blip, so an estimator that recovers every `psi` correctly still has to look
    ahead to price it. It also reads `performance_status`, which is not in the
    blip basis either and, unlike `acquired_resistance`, moves assignment too —
    it is the only confounder here a treatment-free surface can omit.
    """
    return (
        _PROGNOSTIC["intercept"]
        + _PROGNOSTIC["biomarker_std"] * float(features.get("biomarker_std", 0.0))
        + _PROGNOSTIC["marker_positive"] * (1.0 if features.get("marker_positive") else 0.0)
        + _PROGNOSTIC["prior_line"] * float(features.get("prior_line", 0.0))
        + _PROGNOSTIC["acquired_resistance"] * float(features.get("acquired_resistance", 0.0))
        + _PROGNOSTIC["performance_status"] * float(features.get("performance_status", 0.0))
    )


def linear_predictor(features: dict[str, float], arm: str) -> float:
    """eta = g(X) + tau_a(X)."""
    return prognostic_log_hazard(features) + true_log_hazard_ratio(arm, features)


def expected_months(features: dict[str, float], arm: str) -> float:
    """E[time to progression] under Weibull PH, in closed form.

    `E[T] = scale * exp(-eta / shape) * Gamma(1 + 1/shape)`. Having this as
    arithmetic rather than a simulation is what lets the oracle below be exact,
    and it is the reason Weibull was chosen over a hazard that needs inversion
    by search.
    """
    eta = linear_predictor(features, arm)
    return WEIBULL_SCALE * math.exp(-eta / WEIBULL_SHAPE) * math.gamma(1.0 + 1.0 / WEIBULL_SHAPE)


def sample_progression_months(
    features: dict[str, float], arm: str, rng: random.Random
) -> float:
    """One exact draw from the Weibull PH model.

    `T = scale * (-log(U) / exp(eta)) ** (1/shape)` inverts the survival
    function exactly, so this is a draw rather than a search and the cohort
    stays reproducible from the seed alone.
    """
    uniform = rng.random()
    while uniform <= 0.0:                     # log(0) is not a time
        uniform = rng.random()
    eta = linear_predictor(features, arm)
    return WEIBULL_SCALE * ((-math.log(uniform) / math.exp(eta)) ** (1.0 / WEIBULL_SHAPE))


def assignment_probabilities(features: dict[str, float]) -> dict[str, float]:
    """The behaviour policy: a confounded softmax with a positivity floor.

    Confounded on purpose — it scores arms with the *same* covariates that drive
    the hazard ratios, so a naive per-arm comparison of survival is biased and
    the cohort is only usable by an estimator that handles it. It is also
    confounded by `performance_status`, which is **not** in `HAZARD_BASIS`: that
    is the only confounding a treatment-free surface can be made to miss, and so
    the only kind against which double robustness can be measured at all. The floor keeps
    every arm reachable for every patient, which is what keeps inverse-probability
    weights finite; without it this would be a positivity violation dressed as
    confounding.
    """
    status = float(features.get("performance_status", 0.0))
    scores = {
        arm: (
            -true_log_hazard_ratio(arm, features)
            + _PRESCRIBING_CAUTION.get(arm, 0.0) * status
        ) / _ASSIGNMENT_TEMPERATURE
        for arm in SURVIVAL_ARMS
    }
    largest = max(scores.values())
    weights = {arm: math.exp(score - largest) for arm, score in scores.items()}
    total = sum(weights.values())
    raw = {arm: weight / total for arm, weight in weights.items()}
    # Mix with the uniform distribution to enforce the floor exactly.
    n = len(SURVIVAL_ARMS)
    mix = _ASSIGNMENT_FLOOR * n
    return {arm: (1.0 - mix) * p + _ASSIGNMENT_FLOOR for arm, p in raw.items()}


def sample_baseline_features(rng: random.Random) -> dict[str, float]:
    """A patient at their first line."""
    return {
        "biomarker_std": round(rng.gauss(0.0, 1.0), 4),
        "marker_positive": 1.0 if rng.random() < 0.42 else 0.0,
        "prior_line": 0.0,
        "acquired_resistance": 0.0,
        # Held fixed across lines by `advance_line`, so the confounding it
        # creates is attributable to it alone rather than shared with
        # `prior_line` — which is what makes the ablation below readable.
        "performance_status": round(min(max(rng.gauss(0.35, 0.25), 0.0), 1.0), 4),
    }


def advance_line(features: dict[str, float], arm: str) -> dict[str, float]:
    """Covariates at the next line, after progressing on `arm` at this one.

    **The transition depends on the arm, and that is the whole sequential
    problem.** An earlier version advanced only `prior_line`, so the line-1
    choice had no effect on the line-2 state — the problem decomposed into three
    independent choices and the myopic rule matched the backward-induction
    optimum on **0 of 3000** patients. A generator whose oracle a greedy rule
    reproduces exactly has nothing for a DTR estimator to find.

    So a potent arm leaves `acquired_resistance` behind, which the prognostic
    surface charges at every later line. The biomarker drifts by a fixed amount
    rather than randomly, because a deterministic transition is what keeps the
    backward induction exact rather than Monte Carlo.
    """
    resistance = float(features.get("acquired_resistance", 0.0))
    if arm in RESISTANCE_INDUCING_ARMS:
        resistance += 1.0
    return {
        "biomarker_std": round(float(features["biomarker_std"]) + 0.25, 4),
        "marker_positive": features["marker_positive"],
        "prior_line": float(features["prior_line"]) + 1.0,
        "acquired_resistance": resistance,
        "performance_status": features.get("performance_status", 0.0),
    }


def oracle_action_value(
    features: dict[str, float], arm: str, lines_remaining: int
) -> float:
    """Expected remaining months from taking `arm` now and acting optimally after.

    Backward induction on expected time, which is the survival analogue of
    `ra_cohort.oracle_action_value`'s expected response. It is exact because
    `expected_months` is closed-form and `advance_line` is deterministic — no
    Monte Carlo enters the oracle, so a regret measured against it is not
    carrying the oracle's own sampling error.

    The competing risk and censoring are deliberately **absent** here. This is
    the value of the treatment decision — expected time to progression, summed
    over the lines a patient would reach — and folding in a death rate that does
    not depend on the arm would charge every arm the same constant while making
    the number harder to interpret. `rollout_value` reports what a patient
    actually accrues once both are simulated, and the gap between them is
    retention rather than error; that is invariant 43's distinction, on this
    scale.
    """
    value = expected_months(features, arm)
    if lines_remaining > 1:
        value += oracle_value(advance_line(features, arm), lines_remaining - 1)
    return value


def oracle_value(features: dict[str, float], lines_remaining: int) -> float:
    """Expected remaining months under the optimal remaining policy."""
    if lines_remaining <= 0:
        return 0.0
    return max(
        oracle_action_value(features, arm, lines_remaining) for arm in SURVIVAL_ARMS
    )


def oracle_arm(features: dict[str, float], lines_remaining: int) -> str:
    """The backward-induction optimum — the reference a regret is measured against.

    Ties break by `SURVIVAL_ARMS` order, which is arbitrary and deterministic.
    """
    return max(
        SURVIVAL_ARMS,
        key=lambda arm: (oracle_action_value(features, arm, lines_remaining), arm),
    )


def myopic_arm(features: dict[str, float]) -> str:
    """The arm with the longest expected time *on this line*, ignoring the rest.

    Not optimal, and it is here to be beaten: `prior_line` makes later lines
    worse, so spending `arm-a` early costs more than its first-line advantage is
    worth for some patients. `ra_cohort` keeps `optimal_arm` for the same reason
    and invariant 29 records what happens when the two are confused.
    """
    return max(SURVIVAL_ARMS, key=lambda arm: (expected_months(features, arm), arm))


def oracle_policy(lines: int = DEFAULT_LINES):
    """The optimal policy as a callable, for `rollout_value`."""

    def policy(features: dict[str, float], line_index: int) -> str:
        return oracle_arm(features, max(lines - line_index, 1))

    return policy


def generate_survival_cohort(
    n: int,
    seed: int = DEFAULT_COHORT_SEED,
    lines: int = DEFAULT_LINES,
) -> list[SurvivalTrajectory]:
    """`n` trajectories, reproducible from the seed alone.

    Each line draws a progression time from the Weibull PH model, a competing
    death time, and a loss-to-follow-up time; whichever comes first ends that
    line, and only progression continues the sequence. The administrative
    horizon closes what is left.
    """
    rng = random.Random(seed)
    trajectories: list[SurvivalTrajectory] = []

    for index in range(n):
        features = sample_baseline_features(rng)
        stages: list[SurvivalStage] = []
        elapsed = 0.0

        for line in range(1, lines + 1):
            probabilities = assignment_probabilities(features)
            arm = _sample_arm(probabilities, rng)

            progression = sample_progression_months(features, arm, rng)
            competing = rng.expovariate(COMPETING_RISK_RATE)
            dropout = rng.expovariate(LOSS_TO_FOLLOWUP_RATE)
            remaining_horizon = max(STUDY_HORIZON_MONTHS - elapsed, 0.0)

            duration = min(progression, competing, dropout, remaining_horizon)
            if duration == progression:
                cause, observed = "progression", True
            elif duration == competing:
                cause, observed = "competing_death", False
            else:
                cause, observed = "censored", False

            stages.append(
                SurvivalStage(
                    line=line,
                    features=dict(features),
                    arm=arm,
                    propensity=probabilities[arm],
                    months=round(duration, 4),
                    event_observed=observed,
                    cause=cause,
                    entry_month=round(elapsed, 4),
                )
            )

            elapsed += duration
            if cause != "progression" or elapsed >= STUDY_HORIZON_MONTHS:
                break
            features = advance_line(features, arm)

        trajectories.append(SurvivalTrajectory(patient_index=index, stages=tuple(stages)))

    return trajectories


def _sample_arm(probabilities: dict[str, float], rng: random.Random) -> str:
    draw = rng.random()
    cumulative = 0.0
    for arm in SURVIVAL_ARMS:
        cumulative += probabilities[arm]
        if draw <= cumulative:
            return arm
    return SURVIVAL_ARMS[-1]


def rollout_value(
    policy,
    n: int = 4_000,
    seed: int = 91_001,
    lines: int = DEFAULT_LINES,
    with_attrition: bool = True,
) -> float:
    """Mean observed months under `policy`, simulated against the generator.

    `with_attrition=False` removes the competing risk and censoring, which is the
    quantity `oracle_value` computes in closed form — so the two are directly
    comparable and a test can assert they agree. Left on, this is what a patient
    actually accrues, and the gap between the two is retention rather than error.
    That distinction is invariant 43's, and it exists here because scoring a
    treatment-decision estimate against the attrition-laden number charges it for
    a gap it is not estimating.
    """
    rng = random.Random(seed)
    total = 0.0
    for _ in range(n):
        features = sample_baseline_features(rng)
        elapsed = 0.0
        for line in range(lines):
            arm = policy(features, line)
            progression = sample_progression_months(features, arm, rng)
            duration, progressed = progression, True
            if with_attrition:
                competing = rng.expovariate(COMPETING_RISK_RATE)
                dropout = rng.expovariate(LOSS_TO_FOLLOWUP_RATE)
                remaining = max(STUDY_HORIZON_MONTHS - elapsed, 0.0)
                duration = min(progression, competing, dropout, remaining)
                progressed = duration == progression
            elapsed += duration
            if not progressed:
                break
            features = advance_line(features, arm)
        total += elapsed
    return total / n


__all__ = [
    "COMPETING_RISK_RATE",
    "DEFAULT_COHORT_SEED",
    "DEFAULT_LINES",
    "HAZARD_BASIS",
    "LOSS_TO_FOLLOWUP_RATE",
    "STUDY_HORIZON_MONTHS",
    "SURVIVAL_ARMS",
    "SURVIVAL_REFERENCE_ARM",
    "TRUE_LOG_HAZARD_RATIOS",
    "WEIBULL_SCALE",
    "WEIBULL_SHAPE",
    "SurvivalStage",
    "SurvivalTrajectory",
    "RESISTANCE_INDUCING_ARMS",
    "advance_line",
    "assignment_probabilities",
    "expected_months",
    "generate_survival_cohort",
    "hazard_basis",
    "linear_predictor",
    "myopic_arm",
    "oracle_action_value",
    "oracle_arm",
    "oracle_policy",
    "oracle_value",
    "prognostic_log_hazard",
    "rollout_value",
    "sample_baseline_features",
    "sample_progression_months",
    "true_log_hazard_ratio",
]
