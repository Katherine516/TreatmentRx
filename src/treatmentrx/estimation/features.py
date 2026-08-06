"""Mapping from a patient's stage history to the estimator covariate space.

One place decides what "das28", "prior_tnf" etc. mean for a live patient, so the
live path and the cohort the model was fit on cannot silently drift apart.
"""

from __future__ import annotations

from treatmentrx.domain import StageRecord

# Defaults are deliberately mid-range: a missing value must not look like an
# extreme one. The data contract is what flags genuinely missing families.
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


def stage_index(stages: list[StageRecord], n_fitted_stages: int) -> int:
    """Zero-based fitted-stage index for the patient's current decision point.

    Patients with a longer history than the fitted horizon are held at the last
    fitted stage rather than extrapolating to a stage the model never saw.
    """
    current = stages[-1].stage if stages else 1
    return max(0, min(current - 1, n_fitted_stages - 1))
