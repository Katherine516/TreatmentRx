"""Shared design bases for the Layer 4 estimators.

Q-learning and dWOLS must assume the *same* covariate space, otherwise their
blip estimates are not comparable and model averaging across them is
meaningless. Both bases live here so there is exactly one definition.

`blip_basis` is re-exported from the simulation module: the tailoring variables
the treatment effect is allowed to vary over are a property of the causal model
(and, in the prototype, of the generating process the estimators are validated
against), not of any one estimator.
"""

from __future__ import annotations

from treatmentrx.simulation.ra_cohort import BLIP_BASIS, blip_basis, das28_std

__all__ = ["BLIP_BASIS", "TREATMENT_FREE_BASIS", "blip_basis", "treatment_free_basis"]

# f(X): the treatment-free / nuisance basis. Deliberately richer than the blip
# basis — it carries CRP and the ALT excess that drive prognosis but are not
# effect modifiers in the current causal model.
TREATMENT_FREE_BASIS = ("intercept", "das28_std", "crp_std", "anti_ccp", "prior_tnf", "alt_excess")


def treatment_free_basis(features: dict[str, float]) -> list[float]:
    return [
        1.0,
        das28_std(features["das28"]),
        (features["crp"] - 30.0) / 25.0,
        features["anti_ccp"],
        features["prior_tnf"],
        max(features["alt"] - 40.0, 0.0) / 25.0,
    ]
