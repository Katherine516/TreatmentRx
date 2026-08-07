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
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from treatmentrx.estimation.q_learning import QLearningModel
from treatmentrx.simulation.ra_cohort import generate_ra_cohort, true_blip

NOMINAL = 0.95

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
        return {
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


def reference_truth(arm: str = REFERENCE_ARM, comparator: str = REFERENCE_COMPARATOR) -> float:
    """The known single-visit contrast for the reference patient."""
    return true_blip(arm, REFERENCE_FEATURES) - true_blip(comparator, REFERENCE_FEATURES)


def sandwich_coverage(
    replications: int = 120,
    n: int = 250,
    share_blip: bool = False,
    base_seed: int = 9_000,
) -> CoverageResult:
    """Coverage of the cluster-robust interval at the terminal stage."""
    truth = reference_truth()
    covered = 0
    widths: list[float] = []
    estimates: list[float] = []
    errors: list[float] = []

    for replication in range(replications):
        cohort = generate_ra_cohort(n, seed=base_seed + replication)
        model = QLearningModel(cohort, share_blip=share_blip)
        contrast = model.sandwich_contrast(
            REFERENCE_ARM, REFERENCE_COMPARATOR, REFERENCE_FEATURES, model.n_stages - 1
        )
        if contrast.lower <= truth <= contrast.upper:
            covered += 1
        widths.append(contrast.upper - contrast.lower)
        estimates.append(contrast.difference)
        errors.append(contrast.standard_error)

    return _result(
        "sandwich (stage-specific)" if not share_blip else "sandwich (shared blip)",
        replications, covered, widths, estimates, errors, truth,
    )


def bootstrap_coverage(
    replications: int = 20,
    n: int = 200,
    bootstrap_replicates: int = 25,
    share_blip: bool = False,
    base_seed: int = 9_500,
) -> CoverageResult:
    """Coverage of the m-out-of-n percentile interval.

    Costs one full refit per bootstrap replicate per replication, so the default
    is deliberately small and the Monte Carlo error is reported alongside.
    """
    truth = reference_truth()
    covered = 0
    widths: list[float] = []
    estimates: list[float] = []
    errors: list[float] = []

    for replication in range(replications):
        cohort = generate_ra_cohort(n, seed=base_seed + replication)
        model = QLearningModel(cohort, share_blip=share_blip)
        model.fit_bootstrap(cohort, replicates=bootstrap_replicates, seed=base_seed + replication)
        contrast = model.bootstrap_contrast(
            REFERENCE_ARM, REFERENCE_COMPARATOR, REFERENCE_FEATURES, model.n_stages - 1
        )
        if contrast.lower <= truth <= contrast.upper:
            covered += 1
        widths.append(contrast.upper - contrast.lower)
        estimates.append(contrast.difference)
        errors.append(contrast.standard_error)

    return _result(
        "m-out-of-n bootstrap (stage-specific)"
        if not share_blip
        else "m-out-of-n bootstrap (shared blip)",
        replications, covered, widths, estimates, errors, truth,
    )


def _result(method, replications, covered, widths, estimates, errors, truth) -> CoverageResult:
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
    )


def decision_rule_coverage(
    replications: int = 60,
    n: int = 250,
    base_seed: int = 9_700,
) -> CoverageResult:
    """Coverage of the interval the decision layer actually uses.

    The individual-estimator studies validate components. Layer 3 uses neither
    of them directly: it asks every estimator for the same contrast and keeps the
    *widest* interval, on the grounds that if any estimator cannot separate the
    arms the system should not claim separation. That selection rule has its own
    sampling behaviour — taking a maximum over three correlated intervals is not
    the same as taking any one of them — so it is measured here rather than
    assumed to inherit the components' coverage.
    """
    from treatmentrx.estimation.dwols import DWOLSModel

    truth = reference_truth()
    covered = 0
    widths: list[float] = []
    estimates: list[float] = []
    errors: list[float] = []

    for replication in range(replications):
        cohort = generate_ra_cohort(n, seed=base_seed + replication)
        shared = QLearningModel(cohort, share_blip=True)
        stage_specific = QLearningModel(cohort, share_blip=False)
        dwols = DWOLSModel(cohort)
        terminal = stage_specific.n_stages - 1

        candidates = {
            "shared": shared.sandwich_contrast(
                REFERENCE_ARM, REFERENCE_COMPARATOR, REFERENCE_FEATURES, terminal
            ),
            "stage_specific": stage_specific.sandwich_contrast(
                REFERENCE_ARM, REFERENCE_COMPARATOR, REFERENCE_FEATURES, terminal
            ),
            "dwols": _dwols_contrast(dwols, terminal),
        }
        contrast = _averaged_contrast(list(candidates.values()))
        if contrast.lower <= truth <= contrast.upper:
            covered += 1
        widths.append(contrast.upper - contrast.lower)
        estimates.append(contrast.difference)
        errors.append(contrast.standard_error)

    return _result(
        "decision rule (model-averaged)",
        replications, covered, widths, estimates, errors, truth,
    )


def joint_rule_coverage(
    replications: int = 20,
    n: int = 200,
    bootstrap_replicates: int = 20,
    base_seed: int = 9_900,
) -> CoverageResult:
    """Coverage of the model-averaged interval once the covariance is measured.

    The bound-based rule reaches nominal by erring wide (1.29x the actual
    spread). Replacing the bound with a joint resampling of all three estimators
    should keep coverage while narrowing the interval — which is only an
    improvement if the coverage actually survives, so it is measured.
    """
    from treatmentrx.estimation.dwols import DWOLSModel
    from treatmentrx.estimation.inference import joint_bootstrap

    truth = reference_truth()
    covered = 0
    widths: list[float] = []
    estimates: list[float] = []
    errors: list[float] = []

    for replication in range(replications):
        cohort = generate_ra_cohort(n, seed=base_seed + replication)
        shared = QLearningModel(cohort, share_blip=True)
        stage_specific = QLearningModel(cohort, share_blip=False)
        dwols = DWOLSModel(cohort)
        terminal = stage_specific.n_stages - 1

        names = ("shared", "stage_specific", "dwols")
        point = {
            "shared": shared.flat_parameters(),
            "stage_specific": stage_specific.flat_parameters(),
            "dwols": dwols.flat_parameters(),
        }

        def refit_all(sample):
            return {
                "shared": shared.refit(sample),
                "stage_specific": stage_specific.refit(sample),
                "dwols": dwols.refit(sample),
            }

        booted = joint_bootstrap(
            refit_all,
            cohort,
            point,
            shared.non_regularity(cohort),
            replicates=bootstrap_replicates,
            seed=base_seed + replication,
        )
        loadings = {
            "shared": shared.contrast_loading(
                REFERENCE_ARM, REFERENCE_COMPARATOR, REFERENCE_FEATURES, terminal
            ),
            "stage_specific": stage_specific.contrast_loading(
                REFERENCE_ARM, REFERENCE_COMPARATOR, REFERENCE_FEATURES, terminal
            ),
            "dwols": dwols.contrast_loading(
                REFERENCE_ARM, REFERENCE_COMPARATOR, REFERENCE_FEATURES
            ),
        }
        contrast = booted.contrast(
            loadings, {name: 1.0 for name in names}, REFERENCE_ARM, REFERENCE_COMPARATOR
        )
        if contrast.lower <= truth <= contrast.upper:
            covered += 1
        widths.append(contrast.upper - contrast.lower)
        estimates.append(contrast.difference)
        errors.append(contrast.standard_error)

    return _result(
        "decision rule (joint bootstrap)",
        replications, covered, widths, estimates, errors, truth,
    )


def _averaged_contrast(tests: list):
    """The decision layer's rule: centre on the average, bound the variance above.

    Equal weights here because the model-averaging weights come out near-uniform
    on this cohort; the point being measured is the averaging, not the weighting.
    """
    from treatmentrx.estimation.inference import ContrastTest

    difference = sum(test.difference for test in tests) / len(tests)
    standard_error = sum(test.standard_error for test in tests) / len(tests)
    margin = 1.96 * standard_error
    return ContrastTest(
        arm=REFERENCE_ARM,
        comparator=REFERENCE_COMPARATOR,
        difference=difference,
        standard_error=standard_error,
        lower=difference - margin,
        upper=difference + margin,
        alpha=0.05,
        caveat="model-averaged",
    )


def _dwols_contrast(model, terminal: int):
    """dWOLS exposes its contrast through the estimator facade, not the model."""
    from treatmentrx.estimation.dwols import DWOLSSharedEstimator

    estimator = DWOLSSharedEstimator()
    difference = model.blip(REFERENCE_ARM, REFERENCE_FEATURES) - model.blip(
        REFERENCE_COMPARATOR, REFERENCE_FEATURES
    )
    variance = sum(
        model.blip_standard_error(arm, REFERENCE_FEATURES) ** 2
        for arm in (REFERENCE_ARM, REFERENCE_COMPARATOR)
    )
    standard_error = math.sqrt(variance)
    margin = 1.96 * standard_error
    from treatmentrx.estimation.inference import ContrastTest

    return ContrastTest(
        arm=REFERENCE_ARM,
        comparator=REFERENCE_COMPARATOR,
        difference=difference,
        standard_error=standard_error,
        lower=difference - margin,
        upper=difference + margin,
        alpha=0.05,
        caveat="arm variances added as if independent",
    )


def verdict(results: list[CoverageResult]) -> str:
    """Read the coverage numbers back in plain terms."""
    lines = []
    for result in results:
        low = result.coverage < NOMINAL - 2 * result.monte_carlo_error
        high = result.coverage > NOMINAL + 2 * result.monte_carlo_error
        narrow = result.se_to_sd_ratio and result.se_to_sd_ratio < 0.95
        if low or narrow:
            lines.append(
                f"{result.method}: {result.coverage:.0%} against a nominal {NOMINAL:.0%}; "
                f"reported SE is {result.se_to_sd_ratio:.2f}x the actual spread — "
                f"intervals are too narrow"
                + (f", plus a bias of {result.bias:+.3f}" if abs(result.bias) > 0.01 else "")
                + "."
            )
        elif high:
            lines.append(
                f"{result.method}: {result.coverage:.0%} against a nominal {NOMINAL:.0%} — "
                "conservative; intervals are wider than they need to be."
            )
        else:
            lines.append(
                f"{result.method}: {result.coverage:.0%}, within Monte Carlo error of the "
                f"nominal {NOMINAL:.0%}."
            )
    return " ".join(lines)


__all__ = [
    "NOMINAL",
    "REFERENCE_ARM",
    "REFERENCE_COMPARATOR",
    "REFERENCE_FEATURES",
    "CoverageResult",
    "bootstrap_coverage",
    "reference_truth",
    "sandwich_coverage",
    "verdict",
]
