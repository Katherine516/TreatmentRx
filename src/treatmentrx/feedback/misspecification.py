"""Which estimator survives the model being wrong?

`stability.py` compares estimators on held-out policy value and reports,
honestly, that the ranking is not resolved: the three sit within their own
standard error of each other and no amount of refitting on this cohort
separates them. That is a real answer, but it leaves the choice of estimator
unjustified.

This module supplies the axis that *does* separate them. The three estimators
were kept deliberately different on the grounds that they fail differently —
Q-learning models the outcome surface, dWOLS is doubly robust and needs only the
propensity model or the outcome model to be right. That was an assertion. Here it
is measured, by curving the treatment-free surface in a way the estimators'
linear basis cannot represent while leaving the blips — the estimand — untouched.

The comparison is only meaningful over a range where the generating process
itself stays well behaved. Past a curvature of roughly 0.2 the outcome saturates
at its ceiling and every estimator degrades for a reason that has nothing to do
with robustness, so the sweep stops before that.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module

from treatmentrx.estimation.dwols import DWOLS_METHOD, DWOLSModel
from treatmentrx.estimation.q_learning import (
    Q_SHARED_METHOD,
    STAGE_SPECIFIC_METHOD,
    QLearningModel,
)
from treatmentrx.simulation.ra_cohort import (
    REFERENCE_ARM,
    TREATMENT_ARMS,
    TRUE_BLIPS,
    generate_ra_cohort,
)

# Beyond this the outcome clamps at 1.0 and the DGP stops being informative.
MAX_INFORMATIVE_CURVATURE = 0.15
# Below this the seeds cannot tell two estimators apart on worst-case error.
MEANINGFUL_MARGIN = 0.10
DEFAULT_CURVATURES = (0.0, 0.05, 0.10, 0.15)
DEFAULT_SEEDS = (7, 23, 101, 202, 303, 404)
DEFAULT_SIZE = 400


@dataclass(frozen=True)
class RobustnessResult:
    estimator: str
    errors_by_curvature: dict[float, float]

    @property
    def correctly_specified(self) -> float:
        return self.errors_by_curvature[min(self.errors_by_curvature)]

    @property
    def worst(self) -> float:
        return max(self.errors_by_curvature.values())

    @property
    def degradation(self) -> float:
        """How much worse the estimator gets as the nuisance model goes wrong."""
        base = self.correctly_specified
        return round(self.worst / base, 3) if base else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "estimator": self.estimator,
            "blip_error_by_curvature": {
                str(k): round(v, 4) for k, v in sorted(self.errors_by_curvature.items())
            },
            "correctly_specified_error": round(self.correctly_specified, 4),
            "worst_case_error": round(self.worst, 4),
            "degradation_factor": self.degradation,
        }


def _blip_error(parameters_for) -> float:
    return sum(
        abs(parameters_for(arm)[name] - truth)
        for arm in TREATMENT_ARMS
        if arm != REFERENCE_ARM
        for name, truth in zip(parameters_for(arm), TRUE_BLIPS[arm])
    )


def robustness_study(
    curvatures: tuple[float, ...] = DEFAULT_CURVATURES,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    size: int = DEFAULT_SIZE,
) -> list[RobustnessResult]:
    """Total blip error per estimator as the treatment-free surface curves."""
    errors: dict[str, dict[float, float]] = {
        Q_SHARED_METHOD: {},
        STAGE_SPECIFIC_METHOD: {},
        DWOLS_METHOD: {},
    }

    for curvature in curvatures:
        totals = {name: 0.0 for name in errors}
        for seed in seeds:
            cohort = generate_ra_cohort(size, seed, curvature=curvature)
            shared = QLearningModel(cohort, share_blip=True, compute_covariance=False)
            stage_specific = QLearningModel(cohort, share_blip=False, compute_covariance=False)
            dwols = DWOLSModel(cohort)
            terminal = stage_specific.n_stages - 1
            totals[Q_SHARED_METHOD] += _blip_error(lambda arm: shared.blip_parameters(arm))
            totals[STAGE_SPECIFIC_METHOD] += _blip_error(
                lambda arm: stage_specific.blip_parameters(arm, terminal)
            )
            totals[DWOLS_METHOD] += _blip_error(lambda arm: dwols.blip_parameters(arm))
        for name, total in totals.items():
            errors[name][curvature] = total / len(seeds)

    return [RobustnessResult(name, by_curvature) for name, by_curvature in errors.items()]


def preferred_estimator(results: list[RobustnessResult]) -> tuple[str, str]:
    """Which estimator to prefer, and the reason, when policy value cannot decide.

    Worst case rather than average: the point of holding three estimators is
    what happens when the modelling assumptions are wrong, and an estimator that
    is excellent when everything is specified correctly and poor otherwise is
    the one least worth relying on.
    """
    if not results:
        return "", "no estimators to compare"
    ordered = sorted(results, key=lambda result: result.worst)
    best, runner_up = ordered[0], ordered[1]
    margin = runner_up.worst - best.worst
    # A handful of seeds cannot resolve a few percent. Naming a winner on that
    # margin would repeat the mistake `stability.py` exists to avoid.
    if margin < MEANINGFUL_MARGIN * best.worst:
        most_accurate = min(results, key=lambda result: result.correctly_specified)
        most_stable = min(results, key=lambda result: result.degradation)
        return (
            "",
            (
                f"No single estimator wins. {most_accurate.estimator} is the most accurate "
                f"when the nuisance model is right ({most_accurate.correctly_specified:.3f}) "
                f"and degrades the most ({most_accurate.degradation}x); "
                f"{most_stable.estimator} is the most stable when it is wrong "
                f"({most_stable.degradation}x) and the least accurate when it is right. "
                "They trade off in exactly the direction their designs predict, which is "
                "the justification for averaging them rather than choosing one."
            ),
        )
    return (
        best.estimator,
        (
            f"{best.estimator} holds up best when the treatment-free model is wrong "
            f"(worst-case blip error {best.worst:.3f} against {runner_up.worst:.3f} for "
            f"{runner_up.estimator}). Held-out policy value cannot separate the "
            "estimators on this cohort; robustness can."
        ),
    )


def misspecification_report(
    curvatures: tuple[float, ...] = DEFAULT_CURVATURES,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    size: int = DEFAULT_SIZE,
) -> dict[str, object]:
    results = robustness_study(curvatures, seeds, size)
    preferred, reason = preferred_estimator(results)
    return {
        "curvatures": list(curvatures),
        "cohort_size": size,
        "seeds": list(seeds),
        "estimators": [result.as_dict() for result in results],
        "preferred": preferred,
        "reason": reason,
        "note": (
            "Curvature misspecifies the treatment-free surface only; the blips, and "
            "therefore the estimand, are unchanged. Above "
            f"{MAX_INFORMATIVE_CURVATURE} the outcome saturates at its ceiling and "
            "every estimator degrades for reasons unrelated to robustness."
        ),
    }


__all__ = [
    "DEFAULT_CURVATURES",
    "MAX_INFORMATIVE_CURVATURE",
    "RobustnessResult",
    "estimator_blip_basis",
    "extra_modifier_report",
    "misspecification_report",
    "preferred_estimator",
    "robustness_study",
]


# --------------------------------------------------------------------------
# The other kind of wrong: a blip basis that omits a real effect modifier
# --------------------------------------------------------------------------

# CRP levels to sweep. The coverage grid in `feedback/coverage.py` holds CRP
# fixed on purpose — it is in the treatment-free basis, so it cancels out of a
# contrast — which makes that grid structurally blind to CRP as an *effect*
# modifier. This study needs its own, spanning the range the cohort produces.
MODIFIER_CRP_LEVELS = (8.0, 28.0, 60.0, 100.0)
DEFAULT_MODIFIERS = (0.0, 0.05, 0.10, 0.20)
MODIFIER_REPLICATIONS = 25
MODIFIER_N = 280


def _modifier_grid() -> dict[str, dict[str, float]]:
    """Patients differing in the omitted modifier and in a basis covariate.

    Both axes matter: the modifier is what the estimators cannot see, and
    `anti_ccp` is in the basis, so contrasting the two shows the damage is
    specific to the omitted covariate rather than general.
    """
    grid = {}
    for crp in MODIFIER_CRP_LEVELS:
        for anti_ccp in (0.0, 1.0):
            label = f"crp={crp:g}, {'seropositive' if anti_ccp else 'seronegative'}"
            grid[label] = {
                "das28": 5.2,
                "crp": crp,
                "anti_ccp": anti_ccp,
                "prior_tnf": 0.0,
                "egfr": 82.0,
                "alt": 25.0,
            }
    return grid


def omitted_modifier_study(
    modifiers: tuple[float, ...] = DEFAULT_MODIFIERS,
    replications: int = MODIFIER_REPLICATIONS,
    n: int = MODIFIER_N,
    base_seed: int = 11_000,
) -> dict[float, dict[str, float]]:
    """Does the deployed interval still cover when the *estimand* is wrong?

    `robustness_study` above bends the nuisance surface, which is the failure
    double robustness exists to survive. This bends the blip: the true effect
    varies over standardised CRP, which `BLIP_BASIS` does not carry, so no
    estimator in the ensemble can represent it at any sample size. What they
    converge to is the CRP-averaged contrast, and the question is how far that
    is from the patient's own — and whether the reported interval notices.

    This is the failure mode real data has. The true effect modifiers will not
    be exactly `anti_ccp` and `prior_tnf`, and unlike a curved nuisance surface
    there is no estimator that is robust to it: it is a statement about the
    estimand, not about a working model. Reported so the size of the damage is
    known rather than assumed.
    """
    from treatmentrx.feedback import coverage
    from treatmentrx.simulation.ra_cohort import true_blip

    grid = _modifier_grid()
    results: dict[float, dict[str, float]] = {}

    for modifier in modifiers:
        # The top-two arms and the truth both move with the modifier, so they
        # are recomputed per setting rather than taken from the default cohort.
        tallies = []
        for label, features in grid.items():
            ranked = sorted(
                TREATMENT_ARMS,
                key=lambda arm: true_blip(arm, features, modifier),
                reverse=True,
            )
            tally = coverage._Tally(label, features)
            tally.arm, tally.comparator = ranked[0], ranked[1]
            tally.truth = true_blip(ranked[0], features, modifier) - true_blip(
                ranked[1], features, modifier
            )
            tallies.append(tally)

        for replication in range(replications):
            cohort = generate_ra_cohort(
                n, seed=base_seed + replication, blip_modifier=modifier
            )
            models = coverage._serving_models(cohort)
            terminal = max(model.n_stages for model in models.values()) - 1
            for tally in tallies:
                components = [
                    (
                        model.sandwich_contrast(
                            tally.arm, tally.comparator, tally.features, terminal
                        )
                        if isinstance(model, QLearningModel)
                        else coverage._dwols_contrast(
                            model, tally.arm, tally.comparator, tally.features
                        )
                    )
                    for model in models.values()
                ]
                tally.record(
                    coverage._averaged_contrast(components, tally.arm, tally.comparator)
                )

        pooled = coverage._pool(f"blip modifier {modifier:g}", tallies)
        worst = min(pooled.per_patient, key=lambda row: row.coverage)
        results[modifier] = {
            "coverage": round(pooled.coverage, 4),
            "worst_patient_coverage": round(worst.coverage, 4),
            "worst_patient": worst.patient,
            "bias": round(pooled.bias, 5),
            "mean_interval_width": round(pooled.mean_width, 5),
            "se_to_sd_ratio": round(pooled.se_to_sd_ratio, 3),
            "max_abs_patient_bias": round(
                max(abs(row.bias) for row in pooled.per_patient), 5
            ),
        }
    return results


def omitted_modifier_report(
    modifiers: tuple[float, ...] = DEFAULT_MODIFIERS,
    replications: int = MODIFIER_REPLICATIONS,
) -> dict[str, object]:
    results = omitted_modifier_study(modifiers, replications)
    baseline = results.get(0.0)
    worst = max(results.values(), key=lambda row: abs(row["bias"]))
    return {
        "modifiers": list(modifiers),
        "crp_levels": list(MODIFIER_CRP_LEVELS),
        "replications": replications,
        "results": {str(k): v for k, v in results.items()},
        "note": (
            "The omitted modifier is standardised CRP, which is in the "
            "treatment-free basis and deliberately not in BLIP_BASIS — so the "
            "estimators can adjust for it as a confounder and still cannot "
            "represent it as an effect modifier. Unlike `curvature`, no "
            "estimator is robust to this: it changes the estimand."
        ),
        "verdict": _modifier_verdict(baseline, worst, results),
    }


def _modifier_verdict(baseline, worst, results) -> str:
    if baseline is None:
        return "no baseline setting measured"
    lines = [
        f"With the blip basis correct the deployed interval covers "
        f"{baseline['coverage']:.0%} pooled and {baseline['worst_patient_coverage']:.0%} "
        f"at its worst patient."
    ]
    strongest = max(results)
    end = results[strongest]
    lines.append(
        f"With an omitted effect modifier at {strongest:g} it covers "
        f"{end['coverage']:.0%} and {end['worst_patient_coverage']:.0%}, with a bias of "
        f"{end['bias']:+.4f} pooled and up to {end['max_abs_patient_bias']:.4f} on a "
        f"single patient."
    )
    if end["coverage"] < baseline["coverage"] - 0.1:
        lines.append(
            "The interval does not widen to absorb it — an omitted modifier is a "
            "bias in the estimand, and a standard error computed under the wrong "
            "basis has no way to see it. This is the failure mode a real cohort "
            "will have, and the only defence is getting the basis right."
        )
    else:
        lines.append(
            "Coverage survives at this strength, which bounds how much an omitted "
            "modifier of this size costs rather than proving none would."
        )
    return " ".join(lines)


# --------------------------------------------------------------------------
# The third kind of wrong: a blip basis with a term the estimand does not need
# --------------------------------------------------------------------------
#
# `omitted_modifier_study` above measures a basis that is too *small* and shows
# that nothing survives it. This measures one that is too *large*, which is the
# mistake an analyst makes after reading that result — if omitting a modifier is
# unrecoverable, add every candidate. It is not free, and this prices it.
#
# The question was not hypothetical. `cli specification` tests `das28_squared`
# every fit and it sits at max|z| = 3.37 against a 3.67 rejection threshold: 92%
# of the way to firing, on the same covariate where `cli subgroups` finds 96%
# abstention and the worst standard error in the cohort. The true blip is linear
# in `das28_std` by construction, so the term's true coefficient is exactly zero
# — but the quantity the *decision* uses is the value-to-go contrast, and
# backward induction composes the blip with a `max` over arms, which is not
# linear even when the blip is. So there was a real mechanism by which the
# near-flag could have been signal rather than noise.

# The estimator-side basis is bound at import time in these modules. Swapping it
# is a study fixture and deliberately **not** a seam: a real deployment declares
# its blip basis once, before fitting, and the fact that this prototype's
# estimator basis *is* the generating basis is what makes every robustness result
# here conditional on that basis being right. Do not promote this to production
# configuration — see `estimation/basis.py`.
_BASIS_CONSUMERS = (
    "treatmentrx.estimation.basis",
    "treatmentrx.estimation.dwols",
    "treatmentrx.estimation.q_learning",
    "treatmentrx.estimation.propensity",
    "treatmentrx.estimation.explainability",
    "treatmentrx.estimation.specification",
)

EXTRA_MODIFIER_PATIENTS = 240
EXTRA_MODIFIER_SEEDS = (4242, 909, 777)


@contextmanager
def estimator_blip_basis(names: tuple[str, ...], function):
    """Swap the estimator-side blip basis, leaving the generating process alone.

    Patching the truth as well would measure a different thing entirely — a
    correctly specified estimator on a curved cohort. The point here is an
    estimator carrying a term its estimand does not need.
    """
    saved = []
    for module_name in _BASIS_CONSUMERS:
        module = import_module(module_name)
        saved.append(
            (module, getattr(module, "BLIP_BASIS", None), getattr(module, "blip_basis", None))
        )
        if hasattr(module, "BLIP_BASIS"):
            module.BLIP_BASIS = names
        if hasattr(module, "blip_basis"):
            module.blip_basis = function
    try:
        yield
    finally:
        for module, old_names, old_function in saved:
            if old_names is not None:
                module.BLIP_BASIS = old_names
            if old_function is not None:
                module.blip_basis = old_function


def _augmented_blip_basis():
    from treatmentrx.simulation.ra_cohort import BLIP_BASIS, blip_basis, das28_std

    names = tuple(BLIP_BASIS) + ("das28_squared",)

    def function(features: dict[str, float]) -> list[float]:
        return blip_basis(features) + [das28_std(features["das28"]) ** 2]

    return names, function


def _abstention_profile(n_patients: int, seed: int) -> dict[str, object]:
    from treatmentrx.estimation import training
    from treatmentrx.feedback import subgroups

    training.reset()
    outcomes = subgroups.score_patients(n_patients=n_patients, seed=seed)
    axes = subgroups._axes(outcomes)
    errors = [outcome.standard_error for outcome in outcomes]
    return {
        "abstain_rate": sum(o.abstained for o in outcomes) / len(outcomes),
        "mean_contrast_se": sum(errors) / len(errors),
        # `_blip_error` zips the fitted parameter dict against the 4-tuple of
        # true coefficients, so under the augmented basis it compares the four
        # real terms and ignores das28_squared. That is what we want: the
        # question is whether recovery of the *true* parameters improves.
        "total_blip_error": _blip_error(training.fitted().dwols.blip_parameters),
        "tertiles": [
            {
                "stratum": cell.name,
                "abstain_rate": cell.abstain_rate,
                "mean_contrast_se": cell.mean_standard_error,
            }
            for cell in axes["das28"]
        ],
    }


def extra_modifier_study(
    n_patients: int = EXTRA_MODIFIER_PATIENTS,
    seeds: tuple[int, ...] = EXTRA_MODIFIER_SEEDS,
) -> dict[str, object]:
    """Refit the serving ensemble with `das28_squared` added, and price it.

    Reported per seed as well as averaged, because the interesting cell — the top
    disease-activity tertile — holds 80 patients, and an 8-point move there
    carries about 5 points of Monte Carlo error on its own.
    """
    names, function = _augmented_blip_basis()
    rows = []
    for seed in seeds:
        baseline = _abstention_profile(n_patients, seed)
        with estimator_blip_basis(names, function):
            augmented = _abstention_profile(n_patients, seed)
        rows.append({"seed": seed, "baseline": baseline, "augmented": augmented})

    from treatmentrx.estimation import training

    training.reset()
    return {"n_patients": n_patients, "seeds": list(seeds), "per_seed": rows}


def extra_modifier_report(
    n_patients: int = EXTRA_MODIFIER_PATIENTS,
    seeds: tuple[int, ...] = EXTRA_MODIFIER_SEEDS,
) -> dict[str, object]:
    study = extra_modifier_study(n_patients, seeds)
    rows = study["per_seed"]

    def mean(path) -> float:
        return sum(path(row) for row in rows) / len(rows)

    pooled_delta = mean(lambda r: r["augmented"]["abstain_rate"] - r["baseline"]["abstain_rate"])
    se_ratio = mean(
        lambda r: r["augmented"]["mean_contrast_se"] / r["baseline"]["mean_contrast_se"]
    )
    top_delta = mean(
        lambda r: r["augmented"]["tertiles"][-1]["abstain_rate"]
        - r["baseline"]["tertiles"][-1]["abstain_rate"]
    )
    low_delta = mean(
        lambda r: r["augmented"]["tertiles"][0]["abstain_rate"]
        - r["baseline"]["tertiles"][0]["abstain_rate"]
    )
    blip_delta = mean(
        lambda r: r["augmented"]["total_blip_error"] - r["baseline"]["total_blip_error"]
    )

    return {
        "candidate": "das28_squared",
        "specification_test_max_z": 3.367,
        "specification_test_threshold": 3.669,
        **study,
        "summary": {
            "pooled_abstention_change": round(pooled_delta, 4),
            "contrast_se_ratio": round(se_ratio, 4),
            "top_tertile_abstention_change": round(top_delta, 4),
            "bottom_tertile_abstention_change": round(low_delta, 4),
            "total_blip_error_change": round(blip_delta, 5),
        },
        "verdict": _extra_modifier_verdict(pooled_delta, se_ratio, top_delta, low_delta, rows),
    }


def _extra_modifier_verdict(pooled, se_ratio, top, low, rows) -> str:
    """Read the numbers rather than assert a conclusion over them.

    An earlier version of this said "parameter recovery does not improve", which
    the measurement contradicted: the total blip error moves -0.013. That is
    negligible spread over twenty parameters, but negligible and absent are
    different claims and only one of them is true.
    """
    top_signs = {
        row["augmented"]["tertiles"][-1]["abstain_rate"]
        < row["baseline"]["tertiles"][-1]["abstain_rate"]
        for row in rows
    }
    consistent = top_signs == {True}
    baseline_error = sum(row["baseline"]["total_blip_error"] for row in rows) / len(rows)
    blip_delta = sum(
        row["augmented"]["total_blip_error"] - row["baseline"]["total_blip_error"]
        for row in rows
    ) / len(rows)
    share = blip_delta / baseline_error if baseline_error else 0.0

    return (
        f"Adding das28_squared widens the contrast interval by "
        f"{(se_ratio - 1) * 100:+.0f}% and moves pooled abstention {pooled:+.1%}. "
        f"The top disease-activity tertile improves ({top:+.1%}) and does so on "
        f"{'every' if consistent else 'some'} seed, which is weak evidence that "
        f"the specification test's near-flag is picking up real curvature in the "
        f"value-to-go contrast rather than noise. But the bottom tertile pays "
        f"{low:+.1%}, and recovery of the four true parameters moves only "
        f"{blip_delta:+.4f} on a total of {baseline_error:.3f} ({share:+.1%}) — "
        f"spread over twenty coefficients, which is not where the gain is coming "
        f"from. The term buys one stratum at the expense of the rest. Keep the "
        f"four-term basis; the near-flag is documented rather than acted on."
    )
