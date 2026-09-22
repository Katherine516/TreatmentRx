"""Doubly-robust blip estimation for a time-to-event endpoint.

`dwols.py` recovers `psi` for a bounded response by weighting each row by
`|A - propensity|` and running a weighted least squares. This is the same
estimator on the log-hazard scale, and it is a port rather than new numerics
because of one fact about the generating model:

    T       = scale * ( -log U / exp(eta) ) ** (1/k)
    log T   = log(scale) - eta/k + (1/k) * log(-log U)

**`log T` is linear in the covariates.** The error `(1/k) log(-log U)` is a
Gumbel with mean `-gamma/k` and variance `pi^2 / (6 k^2)`, and neither depends on
`eta`. So a weighted least squares on `log T` estimates `-eta/k` — the same
machinery, the same `weighted_least_squares`, the same `sandwich_covariance`,
against an estimand on a different scale.

Two things follow, and both are why this is worth doing rather than reaching for
a Cox partial likelihood:

* The **shape is identified from the residuals**. The Gumbel spread is
  `pi / (k * sqrt(6))`, so `k = pi / (sd * sqrt(6))` and the log-hazard ratio is
  `tau = -k * coefficient`. Nothing has to be assumed or supplied.
* Everything `dwols.py` earned applies unchanged — the cross-arm covariance of
  invariant 38, the bread reuse of invariant 66, the Cholesky path of invariant
  68. A partial likelihood would need its own Newton iteration in the module
  invariant 23 says to keep exact, for an estimand this reaches directly.

## What it assumes, stated because the assumptions are the product

* **Weibull proportional hazards.** The linearity above is a property of that
  family, not of survival data. On a generating process with a different
  baseline the log-time model is misspecified, and unlike the nuisance surface
  that is not something double robustness survives — it is the
  `--omitted-modifier` situation of `cli misspecification`, one level down.
  `tests/test_survival_dwols.py` measures recovery against a cohort that *is*
  Weibull, so the claim is "recovers `psi` when the family is right", which is
  weaker than it looks and is the honest one.
* **Cause-specific hazard.** Progression is the event; a competing death and an
  administrative censoring are *both* treated as censoring. That is the standard
  cause-specific analysis and it answers "what is the hazard of progression
  among those still at risk". It is **not** the subdistribution hazard, which
  answers a different question about cumulative incidence, and reporting one
  under the other's name is the error `survival_cohort` records three separate
  causes to make visible.
* **Censoring is independent of the covariates — and in this generator it is
  not.** The IPCW weights come from a Kaplan-Meier fit of the censoring
  distribution with **no covariates**. Two of the three ways follow-up ends do
  satisfy that: loss to follow-up and the competing risk are exponential with
  constant rates, independent of the arm by construction. The third does not.
  Administrative censoring is `horizon - entry_month`, and entry month is the
  sum of the earlier lines' durations, which depend on the covariates and on the
  arms taken — so from line 2 onward a patient's censoring time is informative
  about exactly what is being estimated.

  An earlier version of this paragraph claimed the independence was "right for
  the generator". It is not, and the cost is the largest single bias here.
  Measured over 20 seeds at n=3000, with a correct surface and the generator's
  own propensity, as total |bias| over the 12 blip parameters:

  | | total absolute bias | worst |
  | --- | --- | --- |
  | no censoring at all | **0.106** | 0.027 |
  | deployed, IPCW on | 0.307 | 0.058 |
  | deployed, IPCW **off** | **0.515** | 0.124 |

  So the marginal Kaplan-Meier correction removes about **40%** of what
  censoring costs and cannot remove the rest.

  **A covariate-dependent censoring model is not the repair**, which an earlier
  version of this paragraph asserted without checking. Fitting the censoring
  curve within strata of the *remaining horizon* — which is observable, and is
  the quantity the dependence runs through — moves the bias 0.3073 to **0.3033**
  with five strata and to 0.3085 with ten. It buys about 2% of the 0.20 that
  censoring costs.

  There are **two** things in the residual and neither is a censoring curve.

  *Within a line, the complete case is truncated and reweighting cannot undo
  it.* `_rows_for` keeps only rows that progressed, and a patient whose
  progression time exceeds their remaining horizon is dropped — which is more
  likely the lower their hazard, so the kept rows are short-time-selected in a
  covariate-dependent way. Inverse weighting repairs *random* censoring by
  upweighting comparable survivors; it has nothing to upweight when the
  truncation is administrative. Measured over 12 seeds at n=3000, scoring the
  three parameters a single line can identify, censoring costs **+0.0822** on
  **line 1 alone** — where every patient enters at month 0 and the horizon is
  the same 60 months for all. So this is not an entry-time effect.

  *Across lines, the risk set itself is selected.* A slow progressor reaches the
  horizon before ever starting line 2, so later lines over-represent fast
  progressors — and fast means high `biomarker_std`, a blip-basis term. Mean
  `biomarker_std` among the rows that exist, 8 cohorts of 3000:

  | line | rows, censored | rows, uncensored | biomarker shift |
  | --- | --- | --- | --- |
  | 1 | 24,000 | 24,000 | **+0.0003** |
  | 2 | 18,096 | 24,000 | **+0.0805** |
  | 3 | 14,573 | 24,000 | **+0.1564** |

  Line 1's covariates are unshifted because every patient has a line 1 — which
  is consistent with the paragraph above rather than in tension with it: the
  first says line 1's observed *times* are truncated, the second that its
  *patients* are not selected. Censoring costs more at line 3 (+0.1357) than at
  line 1 (+0.0822), which is the two stacking.

  **The first of those is now what ships, and the second turned out not to be a
  repair at all.** `use_buckley_james=True` is the default: censored rows are
  kept and imputed rather than discarded and reweighted, so there is no
  censoring weight on the deployed path. Measured over 20 refits at n=2000:

  | | complete case + IPCW | Buckley-James |
  | --- | --- | --- |
  | total absolute bias | 0.4023 | **0.2978** |
  | shape error | 0.0104 | **0.0030** |
  | worst SE / actual spread | 0.91 | **1.03** |

  The at-risk weight that paragraph also promised — `1 / P(reach line j)` from
  the baseline covariates — makes the bias **worse**, 0.2122 to 0.2220 on the
  measurement that set this up. There is a reason: given `X_j` the line-j
  outcome is independent of how the patient got there, so selection on X alone
  does not bias a correctly specified regression and the weight only adds
  variance. The covariate shift across lines is real and harmless. Stratifying
  the residual Kaplan-Meier on the remaining horizon buys 0.6% and is also not
  taken.

  What remains: Buckley-James closes about 40% of the gap to the no-censoring
  floor rather than all of it, because its own assumption — censoring
  independent of the residual given the covariates — is not exactly true when
  the remaining horizon depends on history the covariates do not carry.

The propensity is **fitted, never read off the generator**, for the reason
`dwols.py` gives: the true assignment probability does not exist in real data,
and an estimator that needs it is not an estimator.

## Is it doubly robust? Measured, and the answer has a caveat

"Doubly robust" is the first word in this docstring, so it is measured rather
than inherited from the method it ports. The test omits `performance_status`
from the treatment-free surface: that is the one covariate in the generator
which moves assignment *and* the outcome while staying out of `HAZARD_BASIS`,
so omitting it is confounding rather than a loss of identification. Six seeds
at n=3000, scored as total blip error:

| treatment-free surface | no weight | fitted propensity | the true propensity |
| --- | --- | --- | --- |
| correct | 0.675 | 0.661 | 0.671 |
| omits the confounder | 0.712 | 0.699 | **0.671** |

With a correct propensity the wrong surface lands on **0.671**, the same number
the correctly-specified surface gives, so the damage is fully recovered — and
the **fitted** propensity recovers 36% of it, the gap being the price of
`_propensities` being a linear probability model standing in for a softmax
rather than anything about the method.

**Read that as directional, not as a clean demonstration.** The damage is 0.037
on a base of 0.675, which is smaller than this estimator's own baseline bias
(the censoring table above), and scored as a *bias* over eight seeds rather than
as absolute error the omitted-confounder-with-true-propensity arm comes out
**below** the correctly-specified one — 0.308 against 0.326 — which is noise
rather than a wrong surface beating a right one. So the rescue is consistent
with double robustness and is not on its own strong evidence for it. The firmer
evidence is the mechanism, which is checkable without any of this arithmetic:
`|A - pi|` satisfies `pi w(1,X) = (1-pi) w(0,X)` pointwise, and it measurably
balances the covariates — worst imbalance **0.0725 unweighted to 0.0157
weighted**, a 4.6x reduction. That is what `tests/test_survival_dwols.py` pins.

Two things this does **not** establish, both worth stating because the numbers
above are small. The damage being repaired is 0.037 on a base of 0.675, which
is well under this estimator's own baseline bias (see the censoring table
above), so it is a measurement made once here rather than a property pinned by
a test — `tests/test_survival_dwols.py` pins the **mechanism** instead, that
`|A - pi|` balances the covariates, because that is what the rescue is made of
and it is stable at a cohort size this suite can afford. And it says nothing
about the other branch: a correct surface with a wrong propensity is not tested
here, because every confounder in the generator enters assignment through one
softmax and there is no natural way to bend it.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

from treatmentrx.estimation import linalg
from treatmentrx.estimation.inference import sandwich_covariance
from treatmentrx.simulation.survival_cohort import (
    HAZARD_BASIS,
    SURVIVAL_ARMS,
    SURVIVAL_REFERENCE_ARM,
    SurvivalTrajectory,
    hazard_basis,
)

SURVIVAL_METHOD = "Weibull-AFT-dWOLS"

# The nuisance surface the estimator conditions on. Two of its terms are not in
# `HAZARD_BASIS`: `acquired_resistance`, the delayed cost, which moves the
# outcome only; and `performance_status`, which moves assignment as well and is
# therefore the one term whose omission is genuine confounding. Dropping that
# one is the misspecification double robustness is supposed to survive, and it
# is the ablation the tests run — see the module docstring.
SURVIVAL_TREATMENT_FREE_BASIS = (
    "intercept",
    "biomarker_std",
    "marker_positive",
    "prior_line",
    "acquired_resistance",
    "performance_status",
)

_RIDGE = 1e-6
# Buckley-James iteration. It converges quickly here — measured, every arm fit
# on the deployed cohort settles inside ten passes — and the cap exists so a
# cohort that cycles fails slowly rather than forever.
_BJ_MAX_ITERATIONS = 25
_BJ_TOLERANCE = 1e-7
# Buckley-James's asymptotic variance is **not** the least-squares sandwich: the
# imputation carries uncertainty the sandwich cannot see, because it treats an
# imputed response as though it had been observed. Measured rather than assumed,
# the way `inference.SANDWICH_INFLATION` was — 20 refits at n=2000, reported SE
# against the estimator's actual spread across those refits, at one probe
# covariate point per arm:
#
#   arm-a 0.98   arm-b 0.76   arm-c 0.84   mean 0.86
#
# So an honest interval is about 1/0.76 = 1.32x wider at the worst arm. Set at
# the conservative end, as `SANDWICH_INFLATION` is, because an interval that is
# too narrow somewhere is not repaired by being right on average.
_BJ_SE_INFLATION = 1.35
_PROPENSITY_FLOOR = 0.05
_PROPENSITY_CEILING = 0.95
# Gumbel spread to Weibull shape: sd = pi / (k sqrt(6)).
_GUMBEL_SD_FACTOR = math.pi / math.sqrt(6.0)


def survival_treatment_free_basis(features: dict[str, float]) -> list[float]:
    """f(X) in `SURVIVAL_TREATMENT_FREE_BASIS` order."""
    return [
        1.0,
        float(features.get("biomarker_std", 0.0)),
        1.0 if features.get("marker_positive") else 0.0,
        float(features.get("prior_line", 0.0)),
        float(features.get("acquired_resistance", 0.0)),
        float(features.get("performance_status", 0.0)),
    ]


@dataclass(frozen=True)
class _Row:
    features: dict[str, float]
    assignment: float       # 1.0 for the target arm, 0.0 for the reference
    log_months: float
    censor_weight: float
    cluster: int
    event_observed: bool = True   # False for a row that never progressed


def kaplan_meier_censoring(
    trajectories: list[SurvivalTrajectory],
) -> list[tuple[float, float]]:
    """S_C(t) for the *censoring* distribution, as (time, survival) steps.

    The roles are swapped from the usual estimate: an observation is an "event"
    here when follow-up ended **without** progression, and a progression is what
    censors that. Estimating the censoring distribution is the only way to
    reweight a complete-case analysis back to the population that would have
    been observed, and getting the roles the wrong way round produces a curve
    that looks entirely plausible.
    """
    rows = sorted(
        ((stage.months, stage.cause) for t in trajectories for stage in t.stages),
        key=lambda row: row[0],
    )
    total = len(rows)
    if not total:
        return []
    survival = 1.0
    curve: list[tuple[float, float]] = []
    at_risk = total
    index = 0
    while index < total:
        time = rows[index][0]
        tied = 0
        censoring_events = 0
        while index + tied < total and rows[index + tied][0] == time:
            if rows[index + tied][1] != "progression":
                censoring_events += 1
            tied += 1
        if censoring_events and at_risk > 0:
            survival *= 1.0 - censoring_events / at_risk
        curve.append((time, survival))
        at_risk -= tied
        index += tied
    return curve


def _residual_km_jumps(
    residuals: list[float], observed: list[bool]
) -> list[tuple[float, float]]:
    """Kaplan-Meier of the *residual* distribution, as (residual, jump) pairs.

    Buckley-James needs `E[e | e > e_i]`, and the residual distribution is
    exactly what is left unspecified by an AFT fit — so it is estimated rather
    than assumed Gumbel. Assuming it would make this a parametric Weibull fit by
    a different route, and the point of the semiparametric form is that the
    imputation does not inherit the baseline's misspecification.
    """
    order = sorted(range(len(residuals)), key=lambda i: residuals[i])
    total = len(order)
    survival = 1.0
    at_risk = total
    jumps: list[tuple[float, float]] = []
    index = 0
    while index < total:
        value = residuals[order[index]]
        tied = events = 0
        while index + tied < total and residuals[order[index + tied]] == value:
            events += 1 if observed[order[index + tied]] else 0
            tied += 1
        if events and at_risk > 0:
            previous = survival
            survival *= 1.0 - events / at_risk
            jumps.append((value, previous - survival))
        at_risk -= tied
        index += tied
    return jumps


def _censoring_survival(curve: list[tuple[float, float]], time: float) -> float:
    """S_C just before `time`, which is what the IPCW weight divides by."""
    survival = 1.0
    for step_time, step_survival in curve:
        if step_time >= time:
            break
        survival = step_survival
    return survival


@dataclass
class SurvivalArmFit:
    """One arm against the reference, on the log-time scale.

    Mirrors `dwols.ArmFit`: the same one-vs-reference design, the same
    `|A - propensity|` weight, the same sandwich. What differs is the response
    (`log T` rather than a bounded outcome), the censoring weight, and that the
    fitted blip coefficient is `-tau / k` rather than `tau`.
    """

    arm: str
    aft_coefficients: list[float]     # on the log-time scale
    shape: float                      # Weibull k, from the residual spread
    covariance: list[list[float]]
    clusters: list[int]
    n_rows: int
    _bread: list[list[float]]
    imputed: bool = False             # fitted by Buckley-James

    @property
    def psi(self) -> list[float]:
        """The log-hazard ratio parameters, `tau = -k * coefficient`.

        The sign is the thing to get right: a longer time is a *lower* hazard,
        so a positive log-time coefficient is a negative log-hazard ratio.
        """
        n_free = len(SURVIVAL_TREATMENT_FREE_BASIS)
        return [-self.shape * value for value in self.aft_coefficients[n_free:]]

    def log_hazard_ratio(self, features: dict[str, float]) -> float:
        """tau_a(X) — directly comparable to
        `survival_cohort.true_log_hazard_ratio`."""
        return linalg.dot(self.psi, hazard_basis(features))

    def standard_error(self, features: dict[str, float]) -> float:
        """Cluster-robust SE of `log_hazard_ratio`, carried through the scale.

        `tau = -k * coefficient`, so the SE scales by `k`. The shape is itself
        estimated and that uncertainty is **not** propagated — stated rather
        than hidden, because at these sample sizes the residual spread is far
        better determined than the coefficients and folding it in would imply a
        precision this does not have.

        Under Buckley-James the sandwich is widened by `_BJ_SE_INFLATION`,
        because the imputation's uncertainty is invisible to a variance computed
        as though every response had been observed. That constant is measured;
        see its comment. Without it this reads 0.76 of the estimator's actual
        spread at the worst arm, which is an interval a quarter too narrow on a
        published quantity.
        """
        if not self.covariance:
            return 0.0
        n_free = len(SURVIVAL_TREATMENT_FREE_BASIS)
        loading = [0.0] * len(self.covariance)
        for offset, value in enumerate(hazard_basis(features)):
            loading[n_free + offset] = value
        variance = max(linalg.quadratic_form(loading, self.covariance), 0.0)
        inflation = _BJ_SE_INFLATION if self.imputed else 1.0
        return inflation * self.shape * math.sqrt(variance)


class SurvivalBlipModel:
    """Per-arm log-hazard ratios from a censored survival cohort.

    Fit once per cohort, one `SurvivalArmFit` per non-reference arm, exactly as
    `DWOLSModel` holds one `ArmFit` per arm.
    """

    def __init__(
        self,
        trajectories: list[SurvivalTrajectory],
        arms: tuple[str, ...] = SURVIVAL_ARMS,
        reference: str = SURVIVAL_REFERENCE_ARM,
        use_buckley_james: bool = True,
    ) -> None:
        if not trajectories:
            raise ValueError("Cannot fit a survival blip model on an empty cohort")
        self.arms = arms
        self.reference = reference
        # `False` reproduces the complete-case-plus-IPCW fit this file shipped
        # first, kept as a measured comparator the way `ra_cohort`'s
        # `treat_censored_as_terminal` is. Nothing should set it but a study.
        self.use_buckley_james = use_buckley_james
        self.censoring_curve = kaplan_meier_censoring(trajectories)
        self.fits: dict[str, SurvivalArmFit] = {}
        for arm in arms:
            if arm == reference:
                continue
            rows = self._rows_for(trajectories, arm)
            fit = self._fit(arm, rows)
            if fit is not None:
                self.fits[arm] = fit

    def _rows_for(self, trajectories, arm: str) -> list[_Row]:
        """Rows on `arm` or the reference.

        Under Buckley-James every row is kept, carrying whether its event was
        observed, and `censor_weight` is 1.0 — censoring is handled by the
        imputation rather than by a weight, and applying both would count it
        twice. Under the complete-case comparator only progressions are kept and
        each carries `1 / S_C(t)`, which is what this file shipped first.
        """
        rows: list[_Row] = []
        for trajectory in trajectories:
            for stage in trajectory.stages:
                if stage.arm not in (arm, self.reference) or stage.months <= 0.0:
                    continue
                observed = stage.cause == "progression"
                if self.use_buckley_james:
                    weight = 1.0
                else:
                    if not observed:
                        continue
                    survival = _censoring_survival(self.censoring_curve, stage.months)
                    if survival <= 0.0:
                        continue
                    weight = 1.0 / survival
                rows.append(
                    _Row(
                        features=stage.features,
                        assignment=1.0 if stage.arm == arm else 0.0,
                        log_months=math.log(stage.months),
                        censor_weight=weight,
                        cluster=trajectory.patient_index,
                        event_observed=observed,
                    )
                )
        return rows

    def _propensities(self, rows: list[_Row]) -> list[float]:
        """P(target arm | X) within this one-vs-reference subset.

        Fitted, never read off the generator — the reason `dwols.py` gives, and
        the reason `estimation/propensity.py` exists. A linear probability model
        then clipped: it is only used to form weights, and misspecifying it is
        what double robustness is supposed to survive.
        """
        design = [hazard_basis(row.features) for row in rows]
        treated = [row.assignment for row in rows]
        gamma = linalg.weighted_least_squares(
            design, treated, [1.0] * len(rows), ridge=_RIDGE
        )
        return [
            min(max(linalg.dot(row, gamma), _PROPENSITY_FLOOR), _PROPENSITY_CEILING)
            for row in design
        ]

    def _fit(self, arm: str, rows: list[_Row]) -> SurvivalArmFit | None:
        n_free = len(SURVIVAL_TREATMENT_FREE_BASIS)
        n_features = n_free + len(HAZARD_BASIS)
        if len(rows) <= n_features:
            return None

        propensities = self._propensities(rows)
        design: list[list[float]] = []
        targets: list[float] = []
        weights: list[float] = []
        clusters: list[int] = []
        for row, propensity in zip(rows, propensities):
            free = survival_treatment_free_basis(row.features)
            blip = [row.assignment * value for value in hazard_basis(row.features)]
            design.append(free + blip)
            targets.append(row.log_months)
            # dWOLS weight times the censoring weight. The first is what
            # carries the double robustness — measured in the module docstring,
            # where a correct propensity makes a wrong surface cost nothing —
            # and the second is what makes a complete case stand in for the
            # patients who were lost, which it does only partially, for the
            # reason that docstring's censoring table gives.
            weights.append(abs(row.assignment - propensity) * row.censor_weight)
            clusters.append(row.cluster)

        if sum(weights) <= 0.0:
            return None
        observed = [row.event_observed for row in rows]
        beta = linalg.weighted_least_squares(design, targets, weights, ridge=_RIDGE)

        if self.use_buckley_james:
            beta, targets = self._buckley_james(design, targets, weights, observed, beta)

        residuals = [y - linalg.dot(row, beta) for row, y in zip(design, targets)]

        # The Weibull shape, from the Gumbel spread of the residuals.
        if self.use_buckley_james:
            # From the residual Kaplan-Meier on the **recorded** times, not the
            # imputed ones. The shape describes the error distribution, and a
            # censoring indicator only means something against the time actually
            # observed; scoring it against the imputed residuals moves every row
            # that was censored and reads 1.343 against a true 1.4, where this
            # reads 1.406. The sandwich below is the other half of that split —
            # it describes the estimating equation that was solved, so it uses
            # the imputed residuals.
            recorded = [
                row.log_months - linalg.dot(x, beta) for x, row in zip(design, rows)
            ]
            jumps = _residual_km_jumps(recorded, observed)
            mass = sum(jump for _value, jump in jumps)
            if mass <= 0.0:
                return None
            mean = sum(value * jump for value, jump in jumps) / mass
            variance = sum(
                jump * (value - mean) ** 2 for value, jump in jumps
            ) / mass
        else:
            total_weight = sum(weights)
            mean = sum(w * r for w, r in zip(weights, residuals)) / total_weight
            variance = sum(
                w * (r - mean) ** 2 for w, r in zip(weights, residuals)
            ) / total_weight
        spread = math.sqrt(max(variance, 1e-12))
        shape = _GUMBEL_SD_FACTOR / spread

        sparse = [
            [(i, value) for i, value in enumerate(row) if value != 0.0] for row in design
        ]
        normal = linalg.sparse_normal_matrix(sparse, weights, n_features, _RIDGE)
        bread = linalg.inverse(normal)
        covariance = sandwich_covariance(
            sparse, residuals, weights, clusters, bread, n_features
        )
        return SurvivalArmFit(
            arm=arm,
            aft_coefficients=beta,
            shape=shape,
            covariance=covariance,
            clusters=clusters,
            n_rows=len(rows),
            _bread=bread,
            imputed=self.use_buckley_james,
        )

    def _buckley_james(
        self,
        design: list[list[float]],
        targets: list[float],
        weights: list[float],
        observed: list[bool],
        beta: list[float],
    ) -> tuple[list[float], list[float]]:
        """Iteratively impute the censored rows, then refit.

        A row whose follow-up ended without progression is not missing at
        random: it is right-censored, and what is known about it is that the
        event came *later* than the time recorded. Buckley-James replaces its
        response by `x'beta + E[e | e > e_i]`, with the residual distribution
        estimated by Kaplan-Meier, and refits until the coefficients settle.

        This reuses `weighted_least_squares` and adds no optimiser to the module
        invariant 23 says to keep exact. A parametric Weibull AFT likelihood
        would be more efficient and would need Newton with a Hessian — that is
        the trade, and it is the reason for this choice rather than an oversight.

        The conditional expectations come from **suffix sums over the jumps**
        rather than a scan per row. The obvious form is a tail scan inside the
        row loop, which is quadratic and measured 10x slower than the
        complete-case fit at n=2000 — enough to matter to a test suite this file
        has to live in.
        """
        for _ in range(_BJ_MAX_ITERATIONS):
            residuals = [y - linalg.dot(row, beta) for row, y in zip(design, targets)]
            imputed = self._impute(design, targets, observed, beta, residuals)
            updated = linalg.weighted_least_squares(
                design, imputed, weights, ridge=_RIDGE
            )
            shift = max(abs(a - b) for a, b in zip(updated, beta))
            beta = updated
            if shift < _BJ_TOLERANCE:
                break
        residuals = [y - linalg.dot(row, beta) for row, y in zip(design, targets)]
        return beta, self._impute(design, targets, observed, beta, residuals)

    @staticmethod
    def _impute(
        design: list[list[float]],
        targets: list[float],
        observed: list[bool],
        beta: list[float],
        residuals: list[float],
    ) -> list[float]:
        """`x'beta + E[e | e > e_i]` for every censored row, `y` for the rest."""
        jumps = _residual_km_jumps(residuals, observed)
        values = [value for value, _jump in jumps]
        # Suffix sums, so each row's tail is two lookups rather than a scan.
        tail_mass = [0.0] * (len(jumps) + 1)
        tail_weighted = [0.0] * (len(jumps) + 1)
        for index in range(len(jumps) - 1, -1, -1):
            value, jump = jumps[index]
            tail_mass[index] = tail_mass[index + 1] + jump
            tail_weighted[index] = tail_weighted[index + 1] + value * jump

        imputed: list[float] = []
        for index, (value, seen) in enumerate(zip(targets, observed)):
            if seen:
                imputed.append(value)
                continue
            start = bisect.bisect_right(values, residuals[index])
            remaining = tail_mass[start]
            if remaining <= 0.0:
                # Nothing observed beyond this residual, so the tail is not
                # identified and the row keeps its recorded time. That is a
                # lower bound, so it biases **downward** — the standard
                # Buckley-James edge case, stated rather than hidden.
                imputed.append(value)
                continue
            imputed.append(
                linalg.dot(design[index], beta) + tail_weighted[start] / remaining
            )
        return imputed

    def log_hazard_ratio(self, arm: str, features: dict[str, float]) -> float:
        """Estimated tau_a(X); zero for the reference and for an unknown arm,
        the same convention the generator uses."""
        fit = self.fits.get(arm)
        return fit.log_hazard_ratio(features) if fit else 0.0

    def standard_error(self, arm: str, features: dict[str, float]) -> float:
        fit = self.fits.get(arm)
        return fit.standard_error(features) if fit else 0.0

    def blip_parameters(self, arm: str) -> dict[str, float]:
        """psi for one arm, keyed by `HAZARD_BASIS` — what a recovery test
        compares against `TRUE_LOG_HAZARD_RATIOS`."""
        fit = self.fits.get(arm)
        if fit is None:
            return {name: 0.0 for name in HAZARD_BASIS}
        return {name: value for name, value in zip(HAZARD_BASIS, fit.psi)}

    def recommend(self, features: dict[str, float]) -> str:
        """The arm with the **lowest** estimated hazard.

        Lowest, not highest: this scale runs the other way from `q_values`, and
        an argmax here would recommend the worst arm while looking entirely
        reasonable. That direction is the single easiest thing to get wrong when
        porting a decision rule onto a hazard, so it is stated here and asserted
        in `tests/test_survival_dwols.py` rather than left to the reader.
        """
        return min(
            self.arms,
            key=lambda arm: (self.log_hazard_ratio(arm, features), arm),
        )

    @property
    def shape(self) -> float:
        """Mean estimated Weibull shape across the arm fits."""
        if not self.fits:
            return float("nan")
        return sum(fit.shape for fit in self.fits.values()) / len(self.fits)


__all__ = [
    "SURVIVAL_METHOD",
    "SURVIVAL_TREATMENT_FREE_BASIS",
    "SurvivalArmFit",
    "SurvivalBlipModel",
    "kaplan_meier_censoring",
    "survival_treatment_free_basis",
]
