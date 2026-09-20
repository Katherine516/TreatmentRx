"""Mapping from a patient's stage history to the estimator covariate space.

One place decides what "das28", "prior_tnf" etc. mean for a live patient, so the
live path and the cohort the model was fit on cannot silently drift apart.
"""

from __future__ import annotations

from treatmentrx.domain import StageRecord

# Defaults are deliberately mid-range: a missing value must not look like an
# extreme one.
#
# What flags a covariate that fell back to one of these is `data/dag.py`'s
# identification check, not the data contract. The contract checks variable
# *families* and grades a missing one a warning, and the families are broader
# than the covariates: a HAQ-DI satisfies `disease_activity`, an ESR satisfies
# `inflammation`, a rheumatoid factor satisfies `serostatus`. Each of those is
# an ordinary RA record that leaves a default in here, and this comment used to
# say the contract caught them.
FEATURE_DEFAULTS = {
    "das28": 5.0,
    "crp": 15.0,
    "anti_ccp": 0.0,
    "prior_tnf": 0.0,
    "egfr": 90.0,
    "alt": 25.0,
}

_TNF_TOKENS = ("tnf", "adalimumab", "etanercept", "infliximab", "golimumab", "certolizumab")


def numeric_feature(stage: StageRecord, key: str, default: float) -> float:
    value = stage.features.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def prior_tnf_exposure(stages: list[StageRecord]) -> float:
    """1.0 if any *completed* stage used a TNF inhibitor.

    The current (open) stage is excluded: at the decision point the patient has
    not yet been exposed to whatever is being decided now.
    """
    for stage in stages[:-1]:
        haystack = f"{stage.treatment} {stage.response or ''}".lower()
        if any(token in haystack for token in _TNF_TOKENS):
            return 1.0
    if stages and stages[-1].features.get("prior_tnf"):
        return 1.0
    return 0.0


def model_features(stages: list[StageRecord]) -> dict[str, float]:
    """Covariates for the current decision point, in the cohort's feature space."""
    if not stages:
        raise ValueError("Cannot build model features from an empty stage history")
    latest = stages[-1]
    anti_ccp = latest.features.get("anti_ccp") or latest.features.get("anti_ccp_positive")
    return {
        "das28": numeric_feature(latest, "das28", FEATURE_DEFAULTS["das28"]),
        "crp": numeric_feature(latest, "crp", FEATURE_DEFAULTS["crp"]),
        "anti_ccp": 1.0 if anti_ccp else 0.0,
        "prior_tnf": prior_tnf_exposure(stages),
        "egfr": numeric_feature(latest, "egfr", FEATURE_DEFAULTS["egfr"]),
        "alt": numeric_feature(latest, "alt", FEATURE_DEFAULTS["alt"]),
    }


def top_tailoring_variables(
    blip_parameters: dict[str, float],
    features: dict[str, float],
    limit: int = 4,
) -> list[str]:
    """The covariates that actually move this arm's advantage, largest first.

    A tailoring variable is one the *treatment effect* varies over, so the honest
    ranking is by each term's contribution to the blip, `|psi_k * h_k(X)|` —
    exactly the decomposition `ModelExplainer` already reports.

    This replaces a Layer 1 heuristic that scored raw features by
    `sqrt(variance) + abs(latest)`. Unstandardised, that ranked by unit size: on
    the demo patient it returned `egfr=82, crp=28, das28=5.2, haq_di=1.4`, in
    descending order of magnitude and nothing else. Worse, it skipped booleans,
    so `anti_ccp` and `prior_tnf` — the two effect modifiers in the blip basis —
    could never appear. The card listed four "tailoring drivers" of which one was
    in the model, contributing -0.003, while the attribution three lines below
    credited `anti_ccp +0.138`. Two answers to the same question on one page.

    The intercept is excluded: it is the arm's average advantage, not something
    the effect is tailored on.
    """
    from treatmentrx.estimation.basis import BLIP_BASIS, blip_basis

    basis = dict(zip(BLIP_BASIS, blip_basis(features)))
    contributions = [
        (abs(coefficient * basis[name]), name, coefficient * basis[name])
        for name, coefficient in blip_parameters.items()
        if name in basis and name != "intercept"
    ]
    contributions.sort(reverse=True)
    return [f"{name}={value:+.3f}" for _, name, value in contributions[:limit]]


def stage_index(stages: list[StageRecord], n_fitted_stages: int) -> int:
    """Zero-based fitted-stage index for the patient's current decision point.

    Patients with a longer history than the fitted horizon are held at the last
    fitted stage rather than extrapolating to a stage the model never saw.
    """
    current = stages[-1].stage if stages else 1
    return max(0, min(current - 1, n_fitted_stages - 1))
