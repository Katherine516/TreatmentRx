"""Do the confidence intervals actually cover?

Everything else this system reports about uncertainty — the standard errors, the
equipoise rule, the bootstrap — rests on one unverified claim: that a nominal 95%
interval contains the true contrast 95% of the time. Nothing so far tests it. A
standard error can shrink correctly with sqrt(n), be reported on the right scale,
and still systematically miss.

The study is a straightforward Monte Carlo. Regenerate the cohort under a fresh
seed, refit, build the interval for a *fixed* reference patient's contrast, and
count how often it contains the known truth.

**Which estimand.** Coverage is only interpretable against the quantity the
estimator targets. At a stage-specific terminal block that is exactly the true
single-visit blip contrast, so that is the headline configuration. The shared
blip targets a stage-averaged quantity instead — it deliberately absorbs the
delayed effects — so measuring it against the single-visit truth mixes bias with
interval width. It is reported anyway, because the size of that gap is the price
of parameter sharing and worth seeing.

**Which patients.** Coverage at one covariate point is not coverage. The
intervals this system reports are per-patient, and the blip basis makes both the
estimand and its standard error vary over the covariate space — a seropositive
patient's rituximab contrast is a different quantity, estimated from a different
effective sample, than a seronegative TNF-naive patient's. The study therefore
sweeps a grid spanning the two effect modifiers (`anti_ccp`, `prior_tnf`) and the
range of disease activity, contrasts each patient's own true top-two arms, and
reports both the per-patient results and the pooled figure. Refitting is the
expensive part and it is shared: one cohort and one fit per replication serves
every patient in the grid, so the sweep costs barely more than the single point
it replaces.

**Which stage.** The same argument applies a second time and was not made for
years: the stage index moves the estimand exactly the way the covariates do, and
every study in this module pinned it at the terminal block — nine call sites
computing `n_stages - 1`. `decision_rule_stage_sweep` sweeps it. The terminal
block is the one stage where both serving estimators target the same quantity
(there is no future left, so a value-to-go blip *is* a single-visit blip), so it
is the most flattering stage to measure at and it was the only one measured.
Swept, stage 1 comes in at nominal and stage 0 does not:

    stage       coverage   SE/spread   worst patient
    0             77.5%      0.64          37.5%
    1             96.7%      1.07          95.0%
    terminal      95.0%      1.04          92.5%

Stage 0 is unreachable through the pipeline (`SERVED_STAGE_INDICES`), which is
why it has never cost anything — but it is unreachable because of how Layer 1
numbers stages, not because of anything the estimator does.

**What the sweep found, and it is not small.** The single reference patient was
the best case for two of the three methods, at 60 replications and n=280:

    method                              pooled   demo patient   worst patient
    sandwich, stage-specific              91%        93%            85%
    sandwich, shared blip                 29%        75%             0%
    decision rule, correlation bound      74%       100%            45%

The shared blip's collapse is *not* a defect: it lands inside
`estimand_range` — the span of the true contrast over the decision points it
pools — for five of six patients. It estimates a value-to-go contrast, and a
hepatotoxic arm's delayed ALT cost is 0.165 per following stage, several times
larger than most single-visit contrasts in the grid. Scoring it against the
single-visit truth measures the estimand gap, not the estimator.

The decision rule is the finding that matters. Its intervals are wide enough
(SE/SD 1.12 to 1.52 across the grid) and its coverage still falls to 45%, because
the miss is **bias**: it averages one estimator targeting value-to-go with two
targeting the single visit, so the ensemble is centred between two different
quantities. The demo patient hid this because it is the one patient where the
two estimands nearly coincide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from treatmentrx.estimation.q_learning import QLearningModel
from treatmentrx.estimation.inference import DEFAULT_ALPHA
from treatmentrx.simulation.ra_cohort import TREATMENT_ARMS, generate_ra_cohort, true_blip

NOMINAL = 0.95

# The size the deployed models are actually fit on: COHORT_SIZE 400 less the 0.3
# holdout. Coverage depends on n, so measuring it at a different n than the one
# in use answers a question nobody asked.
DEPLOYED_N = 280

# Fixed reference patient: seropositive with a prior TNF failure, the demo
# patient's signature. Holding the covariates fixed keeps the estimand fixed
# across replications, which is what makes the coverage count meaningful.
REFERENCE_FEATURES = {
    "das28": 5.2,
    "crp": 28.0,
    "anti_ccp": 1.0,
    "prior_tnf": 1.0,
    "egfr": 82.0,
    "alt": 25.0,
}
REFERENCE_ARM = "rituximab"
REFERENCE_COMPARATOR = "IL-6 inhibitor"

# The stage indices a served patient can actually land on.
#
# `DataLayer.build_patient_state` appends the *pending* visit to the observed
# history, so a patient with one recorded visit arrives carrying two stages and
# `features.stage_index` returns 1. Index 0 is therefore unreachable through the
# pipeline: it is fitted, it is consumed by backward induction, and no
# recommendation is ever scored with it. Measured over 240 audit bundles the
# split is 22 at stage 1 and 218 at the terminal stage, none at stage 0.
#
# `decision_rule_stage_sweep` reports every stage anyway and marks which are
# served, because "unreachable today" is a property of Layer 1's stage
# bookkeeping rather than of the estimator, and it is exactly the kind of fact
# that stops being true quietly.
SERVED_STAGE_INDICES = (1, 2)


def _patient(das28: float, anti_ccp: float, prior_tnf: float) -> dict[str, float]:
    return {
        "das28": das28,
        "crp": 28.0,
        "anti_ccp": anti_ccp,
        "prior_tnf": prior_tnf,
        "egfr": 82.0,
        "alt": 25.0,
    }


# The grid spans both effect modifiers and the disease-activity range, because
# those are what the blip basis is a function of. `crp`, `egfr` and `alt` are
# held fixed: they enter the treatment-free model, which cancels out of a
# contrast, so varying them would add cost without changing the estimand.
PATIENT_GRID: dict[str, dict[str, float]] = {
    "seropositive, prior TNF failure (demo)": REFERENCE_FEATURES,
    "seropositive, TNF-naive": _patient(5.2, 1.0, 0.0),
    "seronegative, prior TNF failure": _patient(5.2, 0.0, 1.0),
    "seronegative, TNF-naive": _patient(5.2, 0.0, 0.0),
    "low activity, seropositive": _patient(3.2, 1.0, 0.0),
    "high activity, seronegative": _patient(7.4, 0.0, 0.0),
}


def top_two(features: dict[str, float]) -> tuple[str, str]:
    """The patient's true best and runner-up arms — the contrast Layer 3 reports.

    Fixing one arm pair across the grid would measure a quantity that is near
    zero, or even the wrong sign, for patients the pair does not suit. The
    decision layer always contrasts the top two, so that is what is measured.
    """
    ranked = sorted(TREATMENT_ARMS, key=lambda arm: true_blip(arm, features), reverse=True)
    return ranked[0], ranked[1]


def value_to_go_contrast(
    features: dict[str, float], arm: str, comparator: str, stage_index: int
) -> float:
    """True contrast including everything that happens after this decision.

    The single-visit blip difference is the estimand only at a terminal block.
    One decision earlier, choosing a hepatotoxic arm costs `_ALT_PENALTY` times
    the ALT it raises, at the *next* visit — 0.165 per following stage, which is
    larger than most of the single-visit contrasts in the grid. Reporting this
    alongside the blip truth is what makes it possible to tell an estimator that
    is *wrong* from one that is answering a different question.
    """
    from treatmentrx.simulation.ra_cohort import oracle_action_value

    return oracle_action_value(features, arm, stage_index) - oracle_action_value(
        features, comparator, stage_index
    )


def reported_scale_contrast(
    features: dict[str, float],
    arm: str,
    comparator: str,
    stage_index: int,
    n_stages: int,
) -> float:
    """The true contrast on the scale the decision layer actually reports it.

    `QLearningModel.sandwich_contrast` divides the value-to-go contrast by the
    number of remaining stages, so that its numbers sit on the same
    per-remaining-visit scale as `q_values` — and as dWOLS's single-visit blip,
    which carries no horizon at all. At a terminal block the horizon is 1 and
    this reduces to `value_to_go_contrast`, which in turn reduces to the
    single-visit blip difference, which is why the terminal study can use the
    blip truth directly.

    Away from the terminal block the division matters and getting it wrong looks
    like a finding. Scored against the *undivided* value-to-go, the swept rule
    reads 0% at stage 0 and 13% at stage 1 against 95% at the terminal — which
    is arithmetic, not an estimator. Scored on this scale it reads 77.5% and
    96.7%, and only the first of those is a defect.
    """
    horizon = max(n_stages - stage_index, 1)
    return value_to_go_contrast(features, arm, comparator, stage_index) / horizon


def estimand_range(features: dict[str, float], arm: str, comparator: str) -> tuple[float, float]:
    """Span of the true contrast across the decision points a shared blip pools.

    A shared-blip fit estimates one psi from every stage's rows at once, so the
    quantity it targets is a weighted blend across this span, not the terminal
    value. An estimate inside the span is answering a different question; an
    estimate outside it is wrong.
    """
    from treatmentrx.simulation.ra_cohort import DEFAULT_STAGES

    values = [
        value_to_go_contrast(features, arm, comparator, index)
        for index in range(DEFAULT_STAGES)
    ]
    return (min(values), max(values))


@dataclass(frozen=True)
class CoverageResult:
    method: str
    replications: int
    covered: int
    mean_width: float
    mean_estimate: float
    truth: float
    mean_standard_error: float = 0.0
    empirical_sd: float = 0.0
    patient: str = ""
    contrast: str = ""
    # Per-patient results, when this row pools a grid sweep.
    per_patient: tuple[CoverageResult, ...] = field(default=())

    @property
    def coverage(self) -> float:
        return self.covered / self.replications if self.replications else 0.0

    @property
    def monte_carlo_error(self) -> float:
        """SE of the coverage estimate itself, so it is not over-read."""
        p = self.coverage
        return math.sqrt(p * (1.0 - p) / self.replications) if self.replications else 0.0

    @property
    def bias(self) -> float:
        return self.mean_estimate - self.truth

    @property
    def se_to_sd_ratio(self) -> float:
        """Reported standard error over the actual spread of the estimates.

        The direct measurement of whether a standard error is honest, and far
        more stable at small replication counts than the coverage count itself:
        coverage is a proportion of a few dozen Bernoulli draws, this is a ratio
        of two means. A value below 1 means the interval is too narrow no matter
        what the coverage tally happens to land on.
        """
        return self.mean_standard_error / self.empirical_sd if self.empirical_sd else 0.0

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "method": self.method,
            "replications": self.replications,
            "nominal": NOMINAL,
            "coverage": round(self.coverage, 4),
            "monte_carlo_error": round(self.monte_carlo_error, 4),
            "mean_interval_width": round(self.mean_width, 4),
            "bias": round(self.bias, 4),
            "truth": round(self.truth, 4),
            "mean_standard_error": round(self.mean_standard_error, 4),
            "empirical_sd": round(self.empirical_sd, 4),
            "se_to_sd_ratio": round(self.se_to_sd_ratio, 3),
        }
        if self.patient:
            payload["patient"] = self.patient
            payload["contrast"] = self.contrast
        if self.per_patient:
            payload["per_patient"] = [result.as_dict() for result in self.per_patient]
            payload["worst_patient_coverage"] = round(
                min(result.coverage for result in self.per_patient), 4
            )
            payload["worst_patient_se_to_sd_ratio"] = round(
                min(result.se_to_sd_ratio for result in self.per_patient), 3
            )
        return payload


def reference_truth(arm: str = REFERENCE_ARM, comparator: str = REFERENCE_COMPARATOR) -> float:
    """The known single-visit contrast for the reference patient."""
    return true_blip(arm, REFERENCE_FEATURES) - true_blip(comparator, REFERENCE_FEATURES)


class _Tally:
    """Running coverage counts for one patient's contrast."""

    def __init__(
        self,
        patient: str,
        features: dict[str, float],
        truth: float | None = None,
    ) -> None:
        self.patient = patient
        self.features = features
        self.arm, self.comparator = top_two(features)
        # The single-visit blip difference is the estimand at a terminal block
        # and nowhere else, so a swept stage passes its own truth in rather than
        # inheriting this one. `None` keeps every existing caller unchanged.
        self.truth = (
            true_blip(self.arm, features) - true_blip(self.comparator, features)
            if truth is None
            else truth
        )
        self.covered = 0
        self.widths: list[float] = []
        self.estimates: list[float] = []
        self.errors: list[float] = []

    def record(self, contrast) -> None:
        if contrast.lower <= self.truth <= contrast.upper:
            self.covered += 1
        self.widths.append(contrast.upper - contrast.lower)
        self.estimates.append(contrast.difference)
        self.errors.append(contrast.standard_error)

    def result(self, method: str) -> CoverageResult:
        return _result(
            method,
            len(self.estimates),
            self.covered,
            self.widths,
            self.estimates,
            self.errors,
            self.truth,
            patient=self.patient,
            contrast=f"{self.arm} vs {self.comparator}",
        )


def sandwich_coverage(
    replications: int = 120,
    n: int = DEPLOYED_N,
    share_blip: bool = False,
    base_seed: int = 9_000,
    patients: dict[str, dict[str, float]] | None = None,
) -> CoverageResult:
    """Coverage of the cluster-robust interval at the terminal stage.

    Sweeps `patients` (the whole grid by default) off a single fit per
    replication and returns the pooled result, with the per-patient rows attached.
    """
    patients = PATIENT_GRID if patients is None else patients
    method = "sandwich (stage-specific)" if not share_blip else "sandwich (shared blip)"
    tallies = [_Tally(name, features) for name, features in patients.items()]

    for replication in range(replications):
        cohort = generate_ra_cohort(n, seed=base_seed + replication)
        model = QLearningModel(cohort, share_blip=share_blip)
        terminal = model.n_stages - 1
        for tally in tallies:
            tally.record(
                model.sandwich_contrast(
                    tally.arm, tally.comparator, tally.features, terminal
                )
            )

    return _pool(method, tallies)


def _pool(method: str, tallies: list[_Tally]) -> CoverageResult:
    """Pool a grid sweep into one row, keeping the per-patient rows attached.

    The pooled SE/SD ratio is a ratio of pooled means rather than a mean of
    ratios, so a patient whose contrast happens to be small cannot dominate it.
    Coverage pools as a plain count, which is what it is.
    """
    per_patient = tuple(tally.result(method) for tally in tallies)
    covered = sum(tally.covered for tally in tallies)
    replications = sum(len(tally.estimates) for tally in tallies)
    widths = [width for tally in tallies for width in tally.widths]
    errors = [error for tally in tallies for error in tally.errors]
    # Each patient's estimates are centred on their own truth before pooling:
    # the grid deliberately spans different estimands, so a raw pooled spread
    # would measure the grid rather than the estimator.
    centred = [
        estimate - tally.truth for tally in tallies for estimate in tally.estimates
    ]
    mean_bias = sum(centred) / len(centred)
    variance = sum((value - mean_bias) ** 2 for value in centred) / max(len(centred) - 1, 1)
    return CoverageResult(
        method=method,
        replications=replications,
        covered=covered,
        mean_width=sum(widths) / len(widths),
        mean_estimate=mean_bias,
        truth=0.0,
        mean_standard_error=sum(errors) / len(errors),
        empirical_sd=math.sqrt(variance),
        patient=f"pooled over {len(tallies)} patients",
        per_patient=per_patient,
    )


def bootstrap_coverage(
    replications: int = 20,
    n: int = DEPLOYED_N,
    bootstrap_replicates: int = 25,
    share_blip: bool = False,
    base_seed: int = 9_500,
    patients: dict[str, dict[str, float]] | None = None,
) -> CoverageResult:
    """Coverage of the m-out-of-n percentile interval.

    Costs one full refit per bootstrap replicate per replication, so the default
    is deliberately small and the Monte Carlo error is reported alongside. The
    patient grid rides along free: the draws are per-cohort and each patient's
    interval is a dot product against them.
    """
    patients = PATIENT_GRID if patients is None else patients
    method = (
        "m-out-of-n bootstrap (stage-specific)"
        if not share_blip
        else "m-out-of-n bootstrap (shared blip)"
    )
    tallies = [_Tally(name, features) for name, features in patients.items()]

    for replication in range(replications):
        cohort = generate_ra_cohort(n, seed=base_seed + replication)
        model = QLearningModel(cohort, share_blip=share_blip)
        model.fit_bootstrap(cohort, replicates=bootstrap_replicates, seed=base_seed + replication)
        terminal = model.n_stages - 1
        for tally in tallies:
            tally.record(
                model.bootstrap_contrast(
                    tally.arm, tally.comparator, tally.features, terminal
                )
            )

    return _pool(method, tallies)


def _result(
    method, replications, covered, widths, estimates, errors, truth, patient="", contrast=""
) -> CoverageResult:
    mean = sum(estimates) / len(estimates)
    variance = sum((value - mean) ** 2 for value in estimates) / max(len(estimates) - 1, 1)
    return CoverageResult(
        method=method,
        replications=replications,
        covered=covered,
        mean_width=sum(widths) / len(widths),
        mean_estimate=mean,
        truth=truth,
        mean_standard_error=sum(errors) / len(errors),
        empirical_sd=math.sqrt(variance),
        patient=patient,
        contrast=contrast,
    )


def decision_rule_coverage(
    replications: int = 60,
    n: int = DEPLOYED_N,
    base_seed: int = 9_700,
    patients: dict[str, dict[str, float]] | None = None,
) -> CoverageResult:
    """Coverage of the interval the decision layer actually uses.

    The individual-estimator studies validate components. Layer 3 uses neither of
    them directly: it centres on the model-averaged contrast and bounds the
    variance by the weighted sum of the component standard errors. That rule has
    its own sampling behaviour — a bound over three correlated estimators is not
    the same as any one of them — so it is measured here rather than assumed to
    inherit the components' coverage.
    """
    patients = PATIENT_GRID if patients is None else patients
    tallies = [_Tally(name, features) for name, features in patients.items()]

    for replication in range(replications):
        contrasts_for = _serving_contrasts(n, base_seed + replication)
        for tally in tallies:
            components = contrasts_for(tally)
            tally.record(
                _averaged_contrast(list(components.values()), tally.arm, tally.comparator)
            )

    return _pool("decision rule (correlation bound)", tallies)


def _serving_contrasts(n: int, seed: int, stage_index: int | None = None):
    """Each serving estimator's contrast for a patient, on one fresh cohort.

    Both this and `_joint_draws` derive their membership from
    `_serving_models`, so neither can quietly measure a different ensemble than
    the one that serves. They previously each kept a local map of estimator
    names filtered against `SERVING_ENSEMBLE`; when the serving Q-learning model
    changed, both maps stopped containing it and both studies silently collapsed
    to dWOLS alone while still reporting an ensemble figure.

    `stage_index` defaults to the terminal block, which is what every study here
    measured until `decision_rule_stage_sweep` was added. Only the Q-learning
    member takes a stage at all: dWOLS pools every stage's rows into one psi per
    arm and `DWOLSSharedEstimator.fit_predict` never passes it an index, so
    asking it for a stage-specific contrast would measure a model that does not
    exist.
    """
    models = _serving_models(generate_ra_cohort(n, seed=seed))
    n_stages = max(model.n_stages for model in models.values())
    default = n_stages - 1 if stage_index is None else stage_index

    def contrasts_for(tally, at: int | None = None):
        """`at` overrides the stage for this call, on the *same* fit.

        The stage sweep needs every stage off one refit, for the same reason the
        patient grid does: the fit is the expensive part and it does not depend
        on where it is evaluated. Refitting per stage would have tripled the
        study's cost for three identical ensembles.
        """
        index = default if at is None else at
        return {
            name: (
                model.sandwich_contrast(tally.arm, tally.comparator, tally.features, index)
                if isinstance(model, QLearningModel)
                else _dwols_contrast(model, tally.arm, tally.comparator, tally.features)
            )
            for name, model in models.items()
        }

    contrasts_for.n_stages = n_stages
    return contrasts_for


def decision_rule_stage_sweep(
    replications: int = 40,
    n: int = DEPLOYED_N,
    base_seed: int = 9_700,
    patients: dict[str, dict[str, float]] | None = None,
) -> dict[str, object]:
    """Coverage of the decision rule at every stage, not only the terminal one.

    **Why this exists.** `PATIENT_GRID` was introduced because coverage at one
    covariate point is not coverage — the estimand and its standard error both
    move over the covariate space, so a single patient measures a single patient.
    The stage index is a second axis with exactly the same property, and it was
    pinned at the terminal block by every study in this module: nine call sites
    computing `n_stages - 1`. The argument that justified sweeping the first axis
    was never applied to the second.

    **What the sweep found.** At n=280 over 40 refits of the six-patient grid:

        stage       coverage   SE/spread   bias      worst patient
        0             77.5%      0.64     -0.009        37.5%
        1             96.7%      1.07     +0.000        95.0%
        terminal      95.0%      1.04     +0.000        92.5%

    Stage 1 is at nominal and the terminal block is the figure already reported.
    Stage 0 is not: its interval runs about a third too narrow. The ensemble is
    not badly *centred* anywhere — each member's bias against its own estimand
    stays under 0.011 at every stage, and the per-remaining-visit rescaling in
    `sandwich_contrast` is what keeps a value-to-go blip and a single-visit one
    comparable — so this is a width failure, not the centring failure that
    dropping the shared blip fixed.

    **Why it has never bitten**, and why that is not a reason to leave it:
    `SERVED_STAGE_INDICES` is `(1, 2)`. Layer 1 appends the pending visit, so
    stage 0 is fitted, consumed by backward induction, and never used to score
    anyone. The row is reported with `served: False` rather than dropped,
    because the thing keeping it out of reach is Layer 1's stage bookkeeping and
    not a property of the estimator.
    """
    patients = PATIENT_GRID if patients is None else patients

    probe = _serving_contrasts(n, base_seed)
    n_stages = probe.n_stages

    tallies = {
        stage: [
            _Tally(
                name,
                features,
                truth=reported_scale_contrast(
                    features, *top_two(features), stage, n_stages
                ),
            )
            for name, features in patients.items()
        ]
        for stage in range(n_stages)
    }

    for replication in range(replications):
        # One refit serves every stage *and* every patient, the way it already
        # serves every patient in `decision_rule_coverage`. The fit does not
        # depend on the stage it is evaluated at.
        contrasts_for = _serving_contrasts(n, base_seed + replication)
        for stage in range(n_stages):
            for tally in tallies[stage]:
                tally.record(
                    _averaged_contrast(
                        list(contrasts_for(tally, stage).values()),
                        tally.arm,
                        tally.comparator,
                    )
                )

    rows = []
    for stage in range(n_stages):
        label = "terminal" if stage == n_stages - 1 else f"stage {stage}"
        result = _pool(f"decision rule, {label}", tallies[stage]).as_dict()
        result["stage_index"] = stage
        result["served"] = stage in SERVED_STAGE_INDICES
        rows.append(result)

    unserved_misses = [
        row for row in rows if not row["served"] and row["coverage"] < NOMINAL - 0.05
    ]
    return {
        "replications": replications,
        "n": n,
        "n_stages": n_stages,
        "served_stage_indices": list(SERVED_STAGE_INDICES),
        "stages": rows,
        "note": (
            "The truth at each stage is the value-to-go contrast divided by the "
            "remaining horizon, which is the scale `sandwich_contrast` reports "
            "on. Stages outside `served_stage_indices` are fitted but never used "
            "to score a patient: Layer 1 appends the pending visit, so "
            "`stage_index` is at least 1 for anyone the pipeline sees."
            + (
                ""
                if not unserved_misses
                else " Unserved stages below nominal: "
                + ", ".join(str(row["stage_index"]) for row in unserved_misses)
                + " — latent, not live."
            )
        ),
    }


# Where the percentile stops being an artefact of the draw count.
# `joint_replicate_sweep` measures coverage climbing 87% -> 93% as the draws go
# 25 -> 200 *while the interval widens*, which is a quantile stabilising rather
# than the method changing. Below ~100 the reported figure says more about the
# draw count than the estimator, so the default sits above it.
JOINT_BOOTSTRAP_REPLICATES = 120


def joint_rule_coverage(
    replications: int = 20,
    n: int = DEPLOYED_N,
    bootstrap_replicates: int = JOINT_BOOTSTRAP_REPLICATES,
    base_seed: int = 9_900,
    patients: dict[str, dict[str, float]] | None = None,
) -> CoverageResult:
    """Coverage of the model-averaged interval once the covariance is measured.

    The bound-based rule reaches nominal by erring wide. Replacing the bound with
    a joint resampling of all three estimators should keep coverage while
    narrowing the interval — which is only an improvement if the coverage
    actually survives, so it is measured. The joint draws are shared across the
    patient grid: the resampling is per-cohort, and a contrast is a dot product
    against it, so the whole grid costs one bootstrap.
    """
    patients = PATIENT_GRID if patients is None else patients
    tallies = [_Tally(name, features) for name, features in patients.items()]

    for replication in range(replications):
        booted, loadings_for = _joint_draws(
            n, base_seed + replication, bootstrap_replicates
        )
        for tally in tallies:
            loadings = loadings_for(tally)
            tally.record(
                booted.contrast(
                    loadings,
                    {name: 1.0 for name in loadings},
                    tally.arm,
                    tally.comparator,
                )
            )

    return _pool("decision rule (joint bootstrap)", tallies)


def _serving_models(cohort):
    """The models `training.SERVING_ENSEMBLE` names, fit on this cohort.

    Keyed by the *method name* rather than by a local label. The previous version
    kept its own `{"shared", "stage_specific", "dwols"}` map and filtered it
    against `SERVING_ENSEMBLE`; when the serving Q-learning model became
    `Q-Pooled` that map no longer contained it, so the filter silently collapsed
    to dWOLS alone and this study measured a single estimator while reporting an
    ensemble. Deriving the membership from one place is what stops that.
    """
    from treatmentrx.estimation import training
    from treatmentrx.estimation.dwols import DWOLSModel
    from treatmentrx.estimation.q_learning import DEFAULT_POOLING_RIDGE

    builders = {
        training.Q_POOLED: lambda: QLearningModel(
            cohort, share_blip=True, pooling_ridge=DEFAULT_POOLING_RIDGE
        ),
        training.Q_SHARED: lambda: QLearningModel(cohort, share_blip=True),
        training.STAGE_SPECIFIC: lambda: QLearningModel(cohort, share_blip=False),
        training.DWOLS_SHARED: lambda: DWOLSModel(cohort),
    }
    missing = [name for name in training.SERVING_ENSEMBLE if name not in builders]
    if missing:
        raise ValueError(f"no builder for serving estimator(s): {missing}")
    return {name: builders[name]() for name in training.SERVING_ENSEMBLE}


def _joint_draws(n: int, seed: int, replicates: int):
    """Fit the serving ensemble on a fresh cohort and resample it jointly.

    Returns the draws plus a callable that builds a tally's loadings, so callers
    can evaluate any number of patients — or any prefix of the draws — without
    paying for the refits again.
    """
    from treatmentrx.estimation import training
    from treatmentrx.estimation.inference import joint_bootstrap

    cohort = generate_ra_cohort(n, seed=seed)
    models = _serving_models(cohort)
    terminal = max(model.n_stages for model in models.values()) - 1

    def refit_all(sample):
        return {name: model.refit(sample) for name, model in models.items()}

    # `non_regularity` sets the resample size and only a Q-learning model can
    # measure it — it is a property of the `max` in the pseudo-outcome, which
    # dWOLS does not have. With no Q-learning member the design is regular and
    # the ordinary bootstrap applies.
    anchor = next(
        (m for m in models.values() if isinstance(m, QLearningModel)), None
    )
    booted = joint_bootstrap(
        refit_all,
        cohort,
        {name: model.flat_parameters() for name, model in models.items()},
        anchor.non_regularity(cohort) if anchor is not None else 0.0,
        replicates=replicates,
        seed=seed,
    )

    def loadings_for(tally):
        # dWOLS has no stage index: its blip is one vector per arm. Both classes
        # carry `n_stages`, so the type is the discriminator, not the attribute.
        return {
            name: (
                model.contrast_loading(tally.arm, tally.comparator, tally.features, terminal)
                if isinstance(model, QLearningModel)
                else model.contrast_loading(tally.arm, tally.comparator, tally.features)
            )
            for name, model in models.items()
        }

    return booted, loadings_for


def joint_replicate_sweep(
    replicate_counts: tuple[int, ...] = (25, 50, 100, 200),
    replications: int = 20,
    n: int = DEPLOYED_N,
    base_seed: int = 9_900,
    patients: dict[str, dict[str, float]] | None = None,
) -> dict[int, CoverageResult]:
    """Is the joint interval's shortfall a real one, or too few draws?

    A percentile interval at alpha=0.05 needs the 2.5th and 97.5th quantiles, and
    from 25 draws those are essentially the minimum and maximum — an interval
    built that way is as much a statement about the draw count as about the
    estimator. This separates the two by taking *nested prefixes of one set of
    draws*, so every replicate count is evaluated on the same resamples and the
    refits are paid once rather than once per count.
    """
    patients = PATIENT_GRID if patients is None else patients
    counts = tuple(sorted(replicate_counts))
    tallies = {
        count: [_Tally(name, features) for name, features in patients.items()]
        for count in counts
    }

    for replication in range(replications):
        booted, loadings_for = _joint_draws(n, base_seed + replication, counts[-1])
        for count in counts:
            prefix = replace(booted, draws=booted.draws[:count])
            for tally in tallies[count]:
                loadings = loadings_for(tally)
                tally.record(
                    prefix.contrast(
                        loadings,
                        {name: 1.0 for name in loadings},
                        tally.arm,
                        tally.comparator,
                    )
                )

    return {
        count: _pool(f"joint bootstrap ({count} replicates)", tallies[count])
        for count in counts
    }


def _averaged_contrast(tests: list, arm: str, comparator: str, alpha: float = 0.05):
    """The decision layer's rule: centre on the average, bound the variance above.

    Equal weights here because the model-averaging weights come out near-uniform
    on this cohort; the point being measured is the averaging, not the weighting.
    """
    from treatmentrx.estimation.inference import ContrastTest

    from treatmentrx.estimation.inference import normal_critical_value

    difference = sum(test.difference for test in tests) / len(tests)
    standard_error = sum(test.standard_error for test in tests) / len(tests)
    margin = normal_critical_value(alpha) * standard_error
    return ContrastTest(
        arm=arm,
        comparator=comparator,
        difference=difference,
        standard_error=standard_error,
        lower=difference - margin,
        upper=difference + margin,
        alpha=alpha,
        # Matches `DecisionLayer._pair_contrast`. Coverage reads only the bounds,
        # so this flag never moved a number here — but a study that claims to
        # measure the decision layer's rule and then disagrees with it about
        # whether a contrast separates is a divergence waiting to be found by
        # someone who trusts the label. `candidate_set_coverage` below does read
        # the verdict.
        conservative=True,
        caveat="model-averaged",
    )


def candidate_set_coverage(
    replications: int = 40,
    n: int = DEPLOYED_N,
    base_seed: int = 9_700,
    patients: dict[str, dict[str, float]] | None = None,
) -> dict[str, object]:
    """Does the candidate set contain the truly optimal arm?

    The set is what Layer 3 reports when it will not name one arm, so the
    property that makes it safe is not its size but whether the right answer is
    inside it. Measured on one fit that fraction reads 100% over 97 declined
    patients, and invariant 31 is exactly why that number cannot be quoted:
    every patient's set is built from the same parameters, so a single unlucky
    fit misses for everybody at once.

    So this replicates over fits, like every other coverage study here. Each
    replication refits the serving ensemble on a fresh cohort, rebuilds each
    grid patient's set under the rule Layer 3 applies, and asks whether the arm
    that is truly optimal at the terminal stage survived.
    """
    from treatmentrx.simulation.ra_cohort import TREATMENT_ARMS, oracle_action_value, oracle_arm

    grid = PATIENT_GRID if patients is None else patients
    contained = {name: 0 for name in grid}
    sizes = {name: [] for name in grid}
    regret_set = {name: [] for name in grid}
    regret_all = {name: [] for name in grid}

    for replication in range(replications):
        models = _serving_models(generate_ra_cohort(n, seed=base_seed + replication))
        terminal = max(model.n_stages for model in models.values()) - 1
        for name, features in grid.items():
            estimates = {
                arm: sum(
                    _model_q(model, features, terminal).get(arm, 0.0)
                    for model in models.values()
                )
                for arm in TREATMENT_ARMS
            }
            leader = max(estimates, key=estimates.get)
            candidates = [leader]
            # The same family-wise level Layer 3 applies. The leader and the
            # comparator are selected from the estimates used for inference, so
            # a pointwise 95% interval does not account for that search — and a
            # study that measures the set at 0.05 while the agent emits it at
            # 0.05/15 is reporting a rule nobody deploys. That divergence is
            # invisible in the output: the numbers stay plausible.
            from treatmentrx.estimation.inference import simultaneous_alpha

            alpha = simultaneous_alpha(len(TREATMENT_ARMS))
            for arm in TREATMENT_ARMS:
                if arm == leader:
                    continue
                tests = [
                    _pair_test(model, leader, arm, features, terminal)
                    for model in models.values()
                ]
                contrast = _averaged_contrast(tests, leader, arm, alpha)
                if not contrast.robustly_distinguishable:
                    candidates.append(arm)

            truth = oracle_arm(features, terminal)
            contained[name] += truth in candidates
            sizes[name].append(len(candidates))
            best = oracle_action_value(features, truth, terminal)
            regret_set[name].append(
                best - min(oracle_action_value(features, a, terminal) for a in candidates)
            )
            regret_all[name].append(
                best - min(oracle_action_value(features, a, terminal) for a in TREATMENT_ARMS)
            )

    total = replications * len(grid)
    flat_set = [v for rows in regret_set.values() for v in rows]
    flat_all = [v for rows in regret_all.values() for v in rows]
    return {
        "replications": replications,
        "cohort_size": n,
        "patients": len(grid),
        "contains_optimal_arm": round(sum(contained.values()) / total, 4),
        "mean_set_size": round(
            sum(v for rows in sizes.values() for v in rows) / total, 3
        ),
        "arms_on_the_menu": len(TREATMENT_ARMS),
        "worst_case_regret_whole_menu": round(sum(flat_all) / len(flat_all), 4),
        "worst_case_regret_candidate_set": round(sum(flat_set) / len(flat_set), 4),
        "regret_reduction": round(1 - sum(flat_set) / sum(flat_all), 4)
        if sum(flat_all)
        else None,
        "per_patient": {
            name: {
                "contains_optimal_arm": round(contained[name] / replications, 4),
                "mean_set_size": round(sum(sizes[name]) / replications, 3),
            }
            for name in grid
        },
    }


def multiplicity_sweep(
    divisors: tuple[int, ...] = (1, 5, 15),
    replications: int = 12,
    patients: int = 40,
    n: int = DEPLOYED_N,
    base_seed: int = 5_000,
    patient_seed: int = 90_000,
) -> dict[str, object]:
    """What the family-wise correction buys, and what it costs.

    `inference.simultaneous_alpha` divides by the number of unordered pairs — 15
    for six arms — on the argument that the leader is *selected* from the same
    estimates used for inference. That argument is sound and the divisor was
    still chosen rather than measured, which is the one thing invariant 24 does
    not allow a constant here to be.

    Three levels are worth separating. `1` is pointwise: no correction at all.
    `5` corrects for the comparisons the procedure actually performs — the leader
    against each of the other arms, which is the only family it ever reports on.
    `15` corrects for every unordered pair, including the ten the system never
    forms.

    Measured on a *fresh* patient sample per replication rather than the
    six-patient grid: on the grid all three levels contain the true best arm in
    240 of 240 draws, so the grid cannot tell them apart, and reporting it as
    agreement would be reporting the limits of the fixture.
    """
    from treatmentrx.simulation.ra_cohort import oracle_action_value, oracle_arm

    levels = {d: {"contained": 0, "sizes": [], "regret": [], "declined": 0} for d in divisors}
    scored = 0

    for replication in range(replications):
        models = _serving_models(generate_ra_cohort(n, seed=base_seed + replication))
        terminal = max(model.n_stages for model in models.values()) - 1
        cohort = generate_ra_cohort(patients, seed=patient_seed + replication)

        for trajectory in cohort:
            features = trajectory.stages[-1].features
            scored += 1
            estimates = {
                arm: sum(
                    _model_q(model, features, terminal).get(arm, 0.0)
                    for model in models.values()
                )
                for arm in TREATMENT_ARMS
            }
            leader = max(estimates, key=estimates.get)
            tests = {
                arm: [
                    _pair_test(model, leader, arm, features, terminal)
                    for model in models.values()
                ]
                for arm in TREATMENT_ARMS
                if arm != leader
            }
            truth = oracle_arm(features, terminal)
            best = oracle_action_value(features, truth, terminal)

            for divisor, row in levels.items():
                alpha = DEFAULT_ALPHA / divisor
                candidates = [leader] + [
                    arm
                    for arm, component in tests.items()
                    if not _averaged_contrast(
                        component, leader, arm, alpha
                    ).robustly_distinguishable
                ]
                row["contained"] += truth in candidates
                row["sizes"].append(len(candidates))
                row["declined"] += len(candidates) > 1
                row["regret"].append(
                    best
                    - min(oracle_action_value(features, arm, terminal) for arm in candidates)
                )

    deployed = len(TREATMENT_ARMS) * (len(TREATMENT_ARMS) - 1) // 2
    return {
        "patient_draws": scored,
        "replications": replications,
        "cohort_size": n,
        "deployed_divisor": deployed,
        "levels": [
            {
                "divisor": divisor,
                "alpha": round(DEFAULT_ALPHA / divisor, 5),
                "family": _FAMILY_LABEL.get(divisor, f"{divisor} comparisons"),
                "contains_optimal_arm": round(row["contained"] / scored, 4),
                "misses": scored - row["contained"],
                "mean_set_size": round(sum(row["sizes"]) / scored, 3),
                "worst_in_set_regret": round(sum(row["regret"]) / scored, 4),
                "decline_rate": round(row["declined"] / scored, 4),
            }
            for divisor, row in sorted(levels.items())
        ],
        "note": (
            "Containment sits above 99% at every level, and that is not "
            "over-coverage against the nominal 95%: set containment and interval "
            "coverage are different properties. The set always holds the leader "
            "and the leader is the true best arm about 91% of the time, so a high "
            "containment rate here is structural rather than a margin — reading "
            "it as over-coverage was the error in an earlier version of this "
            "note. The interval is the calibrated object and sits at 95.0% with "
            "SE/spread 1.04. Read the misses column against the decline column "
            "before changing the divisor."
        ),
    }


_FAMILY_LABEL = {
    1: "pointwise (no correction)",
    5: "leader vs each other arm — the comparisons actually made",
    15: "all unordered pairs — the deployed divisor",
}


def _model_q(model, features: dict[str, float], stage_index: int) -> dict[str, float]:
    return (
        model.q_values(features, stage_index)
        if isinstance(model, QLearningModel)
        else model.q_values(features)
    )


def _pair_test(model, arm: str, comparator: str, features: dict[str, float], stage_index: int):
    return (
        model.sandwich_contrast(arm, comparator, features, stage_index)
        if isinstance(model, QLearningModel)
        else _dwols_contrast(model, arm, comparator, features)
    )


def _dwols_contrast(model, arm: str, comparator: str, features: dict[str, float]):
    """The dWOLS contrast, covariance kept — the same rule the facade deploys.

    This used to add the two arms' variances as if independent, mirroring what
    the facade did at the time. Both now use `contrast_standard_error`, and they
    have to stay in step: a coverage study measuring a wider interval than the
    agent emits reports a rule nobody deploys, which is the failure invariant 19
    is about.
    """
    from treatmentrx.estimation.inference import ContrastTest

    difference = model.blip(arm, features) - model.blip(comparator, features)
    standard_error = model.contrast_standard_error(arm, comparator, features)
    margin = 1.96 * standard_error
    return ContrastTest(
        arm=arm,
        comparator=comparator,
        difference=difference,
        standard_error=standard_error,
        lower=difference - margin,
        upper=difference + margin,
        alpha=0.05,
        caveat="",
    )


def verdict(results: list[CoverageResult]) -> str:
    """Read the coverage numbers back in plain terms.

    Pooled rows also report their worst patient. A method can pool to nominal
    while missing badly somewhere in the covariate space, and the patient in
    front of a clinician is one point, not the average of six.
    """
    lines = []
    for result in results:
        lines.extend(_one_verdict(result))
        if result.per_patient:
            worst = min(result.per_patient, key=lambda row: row.coverage)
            lines.append(
                f"Worst patient for {result.method}: {worst.patient} "
                f"({worst.contrast}) at {worst.coverage:.0%}."
            )
    return " ".join(lines)


def _one_verdict(result: CoverageResult) -> list[str]:
    low = result.coverage < NOMINAL - 2 * result.monte_carlo_error
    high = result.coverage > NOMINAL + 2 * result.monte_carlo_error
    narrow = result.se_to_sd_ratio and result.se_to_sd_ratio < 0.95
    if low or narrow:
        return [
            f"{result.method}: {result.coverage:.0%} against a nominal {NOMINAL:.0%}; "
            f"reported SE is {result.se_to_sd_ratio:.2f}x the actual spread — "
            f"intervals are too narrow"
            + (f", plus a bias of {result.bias:+.3f}" if abs(result.bias) > 0.01 else "")
            + "."
        ]
    if high:
        return [
            f"{result.method}: {result.coverage:.0%} against a nominal {NOMINAL:.0%} — "
            "conservative; intervals are wider than they need to be."
        ]
    return [
        f"{result.method}: {result.coverage:.0%}, within Monte Carlo error of the "
        f"nominal {NOMINAL:.0%}."
    ]


__all__ = [
    "DEPLOYED_N",
    "NOMINAL",
    "PATIENT_GRID",
    "REFERENCE_ARM",
    "REFERENCE_COMPARATOR",
    "REFERENCE_FEATURES",
    "CoverageResult",
    "bootstrap_coverage",
    "JOINT_BOOTSTRAP_REPLICATES",
    "decision_rule_coverage",
    "joint_replicate_sweep",
    "joint_rule_coverage",
    "reference_truth",
    "sandwich_coverage",
    "top_two",
    "verdict",
]
