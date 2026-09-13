"""The agent abstains for many patients. For which clinical strata?

`cli power` prices abstention against cohort size and `cli audit` reports that it
is earned on average — the patients declined really do have closer arms than the
ones recommended. Both are pooled numbers, and a pooled number cannot see a
subgroup. An agent with one pooled rate but a much lower rate in one
kind of patient and a much higher rate in another is a different clinical object
from one that reproduces the pooled rate in every stratum.

**Two things drive abstention, and they mean opposite things.**

* *Signal.* The arms really are close for this patient, so the contrast is small.
  Declining is correct and more of it is not a defect.
* *Precision.* The contrast is not small, but the standard error is large, so the
  interval cannot exclude zero. Declining is a statement about the training data,
  not about the patient.

Pooled, these are indistinguishable. Stratified, they are not — and the
distinction is what decides whether a subgroup's higher abstention rate is the
agent working or the agent having less to work with. So every stratum reports the
true top-two gap (signal) beside the mean contrast standard error (precision),
plus a counterfactual: the abstention rate that stratum would have if its
standard errors were the pooled mean, holding each patient's own contrast fixed.
If equalizing precision collapses the spread between strata, the spread was
precision. If it does not, the spread was signal.

**What this is not.** It is not a fairness audit in the protected-attribute
sense, and it must not be reported as one. The synthetic generating process has
no age, gender, steroid or comorbidity effect — those nodes exist in the DAG as
`unmodelled_confounders` and are inert here — so there are no demographic strata
to slice. What it does slice is the three covariates the *treatment effect* is
actually allowed to vary over, `BLIP_BASIS` minus its intercept: seropositivity,
prior TNF exposure, and disease activity. Those are where differential
separability would be a clinical statement rather than an artefact, because they
are where the true effect genuinely differs. `ValidationLadder`'s `fairness_clean`
criterion is a separate and still-unmet thing; this does not satisfy it and does
not claim to.

Everything here re-scores patients against the deployed fit. It does not refit,
so it is cheap enough to be a CLI command with no excuse needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from treatmentrx.data import DataContractError, DataLayer
from treatmentrx.domain import RecommendationStatus
from treatmentrx.estimation.features import model_features
from treatmentrx.simulation.fhir_export import simulated_bundles
from treatmentrx.simulation.ra_cohort import TREATMENT_ARMS, oracle_action_value

DEFAULT_PATIENTS = 240
DEFAULT_SEED = 4242

# Enough patients in a cell before its rate is worth printing as a rate. Below
# this the tally is reported but the verdict abstains — a 2-of-3 abstention rate
# is not 67% of anything.
MIN_CELL = 20


@dataclass(frozen=True)
class PatientOutcome:
    """One patient, the decision made for them, and the truth behind it."""

    features: dict[str, float]
    stage_index: int
    abstained: bool
    difference: float
    standard_error: float
    half_width: float
    exact: bool
    true_gap: float

    @property
    def abs_z(self) -> float:
        if self.standard_error <= 0.0:
            return 0.0
        return abs(self.difference) / self.standard_error


@dataclass
class Cell:
    """One stratum of one axis."""

    name: str
    outcomes: list[PatientOutcome] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.outcomes)

    @property
    def abstain_rate(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(1 for outcome in self.outcomes if outcome.abstained) / self.n

    def _mean(self, attribute: str) -> float:
        if not self.outcomes:
            return 0.0
        return sum(getattr(outcome, attribute) for outcome in self.outcomes) / self.n

    @property
    def mean_true_gap(self) -> float:
        return self._mean("true_gap")

    @property
    def mean_standard_error(self) -> float:
        return self._mean("standard_error")

    @property
    def mean_abs_z(self) -> float:
        return self._mean("abs_z")

    def counterfactual_abstain_rate(self, standard_error: float) -> float:
        """Abstention if every patient here had `standard_error`, gaps unchanged.

        The separation rule is `ContrastTest.robustly_distinguishable`, which for
        a sandwich interval is `|difference| > half_width * SANDWICH_INFLATION`.
        Rather than assume the 1.96 behind the half-width, the implied multiplier
        is read off each patient's own interval and re-applied to the substituted
        standard error — so this stays exact if the interval method changes.
        """
        from treatmentrx.estimation.inference import SANDWICH_INFLATION

        if not self.outcomes:
            return 0.0
        abstained = 0
        for outcome in self.outcomes:
            if outcome.standard_error <= 0.0:
                abstained += int(outcome.abstained)
                continue
            multiplier = outcome.half_width / outcome.standard_error
            half_width = multiplier * standard_error
            if outcome.exact:
                separable = abs(outcome.difference) > half_width
            else:
                separable = abs(outcome.difference) > half_width * SANDWICH_INFLATION
            abstained += int(not separable)
        return abstained / self.n

    def precision_excess(self, pooled_standard_error: float) -> float:
        """Percentage points of this stratum's abstention that are precision.

        The difference between what it actually abstains and what it would
        abstain with average precision and its own contrasts. Positive means the
        stratum is being declined partly because the model knows less about it;
        negative means its precision is better than average and is *preventing*
        abstention its contrasts would otherwise earn.
        """
        return self.abstain_rate - self.counterfactual_abstain_rate(pooled_standard_error)

    def as_dict(self, pooled_standard_error: float) -> dict[str, object]:
        return {
            "stratum": self.name,
            "n": self.n,
            "abstain_rate": round(self.abstain_rate, 4),
            "mean_true_gap": round(self.mean_true_gap, 5),
            "mean_contrast_se": round(self.mean_standard_error, 5),
            "mean_abs_z": round(self.mean_abs_z, 3),
            "abstain_rate_at_pooled_se": round(
                self.counterfactual_abstain_rate(pooled_standard_error), 4
            ),
            "precision_excess": round(self.precision_excess(pooled_standard_error), 4),
            "underpowered": self.n < MIN_CELL,
        }


def _das28_tertiles(outcomes: list[PatientOutcome]) -> tuple[float, float]:
    values = sorted(outcome.features["das28"] for outcome in outcomes)
    if len(values) < 3:
        return (0.0, 0.0)
    return (values[len(values) // 3], values[2 * len(values) // 3])


def _axes(outcomes: list[PatientOutcome]) -> dict[str, list[Cell]]:
    """The three covariates the true treatment effect varies over.

    `BLIP_BASIS` is `(intercept, das28_std, anti_ccp, prior_tnf)`. Slicing on
    anything else — CRP, eGFR, ALT — would be slicing on prognosis rather than on
    effect modification, and a difference there says nothing about whether the
    arms are separable.
    """
    low, high = _das28_tertiles(outcomes)

    axes: dict[str, list[Cell]] = {
        "anti_ccp": [Cell("anti_ccp negative"), Cell("anti_ccp positive")],
        "prior_tnf": [Cell("TNF naive"), Cell("prior TNF exposure")],
        "das28": [
            Cell(f"das28 < {low:.2f}"),
            Cell(f"das28 {low:.2f}-{high:.2f}"),
            Cell(f"das28 >= {high:.2f}"),
        ],
    }
    for outcome in outcomes:
        axes["anti_ccp"][1 if outcome.features["anti_ccp"] else 0].outcomes.append(outcome)
        axes["prior_tnf"][1 if outcome.features["prior_tnf"] else 0].outcomes.append(outcome)
        das28 = outcome.features["das28"]
        index = 0 if das28 < low else (1 if das28 < high else 2)
        axes["das28"][index].outcomes.append(outcome)
    return axes


def score_patients(
    n_patients: int = DEFAULT_PATIENTS, seed: int = DEFAULT_SEED
) -> list[PatientOutcome]:
    """Run the deployed agent over a fresh patient sample and keep the evidence.

    Blocked and review cases are excluded rather than counted as abstentions:
    they are Layer 4 outcomes, and folding a contraindication into a statement
    about statistical separability would mix two different reasons for saying no.
    """
    from treatmentrx.decision import DecisionLayer
    from treatmentrx.estimation import EstimationLayer

    data_layer = DataLayer()
    decision_layer = DecisionLayer()
    estimation_layer = EstimationLayer()

    outcomes: list[PatientOutcome] = []
    for bundle in simulated_bundles(n_patients, seed=seed):
        try:
            state = data_layer.build_patient_state(bundle)
        except DataContractError:
            continue
        decision = decision_layer.decide(state, estimation_layer.estimate(state))
        contrast = decision.contrast
        if contrast is None:
            continue
        if decision.status not in (
            RecommendationStatus.EQUIPOISE,
            RecommendationStatus.RECOMMEND,
        ):
            continue

        features = model_features(state.stages)
        index = min(len(state.stages) - 1, training_horizon() - 1)
        ranked = sorted(
            TREATMENT_ARMS,
            key=lambda arm: oracle_action_value(features, arm, index),
            reverse=True,
        )
        true_gap = oracle_action_value(features, ranked[0], index) - oracle_action_value(
            features, ranked[1], index
        )
        outcomes.append(
            PatientOutcome(
                features=features,
                stage_index=index,
                abstained=decision.status is RecommendationStatus.EQUIPOISE,
                difference=contrast.difference,
                standard_error=contrast.standard_error,
                half_width=(contrast.upper - contrast.lower) / 2.0,
                exact=contrast.exact,
                true_gap=true_gap,
            )
        )
    return outcomes


def training_horizon() -> int:
    from treatmentrx.estimation import training

    return training.fitted().pooled.n_stages


def subgroup_report(
    n_patients: int = DEFAULT_PATIENTS, seed: int = DEFAULT_SEED
) -> dict[str, object]:
    outcomes = score_patients(n_patients, seed)
    if not outcomes:
        raise ValueError("no patients survived the data contract")

    pooled = Cell("all", outcomes)
    pooled_se = pooled.mean_standard_error
    axes = _axes(outcomes)

    report: dict[str, object] = {
        "patients_scored": len(outcomes),
        "pooled": pooled.as_dict(pooled_se),
        "axes": {
            name: {
                "strata": [cell.as_dict(pooled_se) for cell in cells],
                "verdict": _axis_verdict(cells, pooled_se),
            }
            for name, cells in axes.items()
        },
        "note": (
            "Strata are the non-intercept terms of BLIP_BASIS — the covariates "
            "the true treatment effect varies over. `mean_true_gap` is signal "
            "and `mean_contrast_se` is precision; `abstain_rate_at_pooled_se` "
            "holds each patient's own contrast and substitutes the pooled "
            "standard error, so the gap between it and `abstain_rate` is the "
            "part of this stratum's abstention that is precision rather than "
            "signal. This is a clinical-stratum analysis, not a fairness audit: "
            "the cohort has no demographic attributes to slice on."
        ),
    }
    return report


def _axis_verdict(cells: list[Cell], pooled_standard_error: float) -> dict[str, object]:
    """How far abstention spreads across this axis, and how much is precision.

    Deliberately not a labelled attribution. The first version of this divided
    the equalized spread by the actual one and bucketed the quotient into
    "signal" / "mixed" / "precision" at 0.75 and 0.25. Both halves were wrong:
    the cut points were chosen rather than measured, and the quotient is not a
    share — it came out at 1.76 on the `prior_tnf` axis, because equalizing
    precision can *widen* the spread when the better-powered stratum is also the
    one with the smaller contrasts. A number that exceeds 1 is not a share of
    anything, and a reader would have taken it for one.

    What is reported instead is per-stratum and additive: `precision_excess` is
    the percentage points of abstention that disappear when a stratum is given
    average precision, holding its own contrasts fixed. It has a direction, it
    has units, and it does not need a threshold to be read.
    """
    usable = [cell for cell in cells if cell.n >= MIN_CELL]
    if len(usable) < 2:
        return {
            "resolved": False,
            "reason": f"fewer than two strata reach {MIN_CELL} patients",
        }

    actual = [cell.abstain_rate for cell in usable]
    equalized = [cell.counterfactual_abstain_rate(pooled_standard_error) for cell in usable]
    worst = max(usable, key=lambda cell: cell.abstain_rate)
    most_penalised = max(usable, key=lambda cell: cell.precision_excess(pooled_standard_error))

    return {
        "resolved": True,
        "abstain_rate_spread": round(max(actual) - min(actual), 4),
        "abstain_rate_spread_at_pooled_se": round(max(equalized) - min(equalized), 4),
        "most_abstaining": {
            "stratum": worst.name,
            "abstain_rate": round(worst.abstain_rate, 4),
            "mean_true_gap": round(worst.mean_true_gap, 5),
        },
        "largest_precision_excess": {
            "stratum": most_penalised.name,
            "points": round(most_penalised.precision_excess(pooled_standard_error), 4),
        },
        "true_gap_range": [
            round(min(cell.mean_true_gap for cell in usable), 5),
            round(max(cell.mean_true_gap for cell in usable), 5),
        ],
        "contrast_se_range": [
            round(min(cell.mean_standard_error for cell in usable), 5),
            round(max(cell.mean_standard_error for cell in usable), 5),
        ],
    }


__all__ = [
    "DEFAULT_PATIENTS",
    "DEFAULT_SEED",
    "MIN_CELL",
    "Cell",
    "PatientOutcome",
    "score_patients",
    "subgroup_report",
]
