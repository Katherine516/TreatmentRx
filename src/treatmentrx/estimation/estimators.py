"""Layer 4 estimator facades.

These are thin adapters: they map a patient's stage history into the estimator
covariate space, ask a model fitted in `training.py` for its Q-values, and wrap
the answer in the `RegimeEstimate` contract the rest of the pipeline consumes. All
of the statistics lives in `q_learning.py` (outcome-model Q-learning with
backward induction) and `dwols.py` (doubly-robust blip regression).

`policy_value` on every result is the estimator's **held-out IPW policy value**,
not a transform of its own scores — so Bayesian model averaging downstream
weights estimators by out-of-sample performance.
"""

from __future__ import annotations

from treatmentrx.estimation import training
from treatmentrx.estimation.features import (
    model_features,
    stage_index,
    top_tailoring_variables,
)
from treatmentrx.estimation.q_learning import (
    Q_POOLED_METHOD,
    Q_SHARED_METHOD,
    STAGE_SPECIFIC_METHOD,
    QLearningModel,
)
from treatmentrx.simulation.ra_cohort import TREATMENT_ARMS
from treatmentrx.contracts import RegimeEstimate
from treatmentrx.domain import RegimeType, StageRecord
from treatmentrx.estimation.inference import DEFAULT_ALPHA, ContrastTest

DEFAULT_TREATMENT_MENU = TREATMENT_ARMS

# Two-sided 95% normal quantile for the reported confidence bands.
Z_95 = 1.96


class _QLearningEstimator:
    """Shared plumbing for the two Q-learning facades."""

    method_name = "Q-learning"
    regime_type = RegimeType.SPTR

    def model(self) -> QLearningModel:
        raise NotImplementedError

    def fit_predict(
        self,
        stages: list[StageRecord],
        treatment_menu: tuple[str, ...] = DEFAULT_TREATMENT_MENU,
    ) -> RegimeEstimate:
        model = self.model()
        features = model_features(stages)
        index = stage_index(stages, model.n_stages)
        q_values = model.q_values(features, index, treatment_menu)
        # Ranked by the model, not by the dict above. `q_values` is clamped to
        # [Q_FLOOR, Q_CEILING] and rounded to 3dp for display, and both of those
        # are many-to-one: a patient whose response saturates the ceiling has two
        # arms collapse to 0.99 and the argmax then falls through to dict order.
        # `recommend` reads the unclamped value-to-go.
        recommended = model.recommend(features, index, treatment_menu)
        best = q_values[recommended]
        # A real interval: the standard error of the recommended arm's blip,
        # from the cluster-robust covariance of the fit.
        half_width = Z_95 * model.blip_standard_error(recommended, features, index)
        return RegimeEstimate(
            estimator=self.method_name,
            regime_type=self.regime_type,
            recommended_arm=recommended,
            q_values=q_values,
            policy_value=training.policy_value_for(self.method_name),
            confidence_band=(
                round(max(best - half_width, 0.0), 3),
                round(min(best + half_width, 1.0), 3),
            ),
            coefficients=model.coefficient_summary(recommended, index),
            top_tailoring_variables=top_tailoring_variables(
                model.blip_parameters(recommended, index), features
            ),
        )

    def contrast(
        self,
        stages: list[StageRecord],
        arm: str,
        comparator: str,
        alpha: float = DEFAULT_ALPHA,
    ) -> ContrastTest:
        model = self.model()
        return model.contrast(
            arm, comparator, model_features(stages), stage_index(stages, model.n_stages), alpha
        )


class QSharedEstimator(_QLearningEstimator):
    """Q-Shared: blip parameters shared across stages, ridge-penalized."""

    method_name = Q_SHARED_METHOD
    regime_type = RegimeType.SPTR

    def model(self) -> QLearningModel:
        return training.fitted().q_shared


class StageSpecificQEstimator(_QLearningEstimator):
    """Stage-specific Q-learning: an independent blip per stage."""

    method_name = STAGE_SPECIFIC_METHOD
    regime_type = RegimeType.DTR

    def model(self) -> QLearningModel:
        return training.fitted().stage_specific


class PooledQEstimator(_QLearningEstimator):
    """Partially pooled Q-learning — the serving Q-learning model.

    A shared blip level plus penalized per-stage deviations, which is the same
    axis `QSharedEstimator` and `StageSpecificQEstimator` sit at the ends of.
    Both of those remain available as comparators; this is the one averaged into
    a recommendation, because it is the only one that borrows strength across
    stages without inheriting the shared fit's stage bias.
    """

    method_name = Q_POOLED_METHOD
    regime_type = RegimeType.HYBRID

    def model(self) -> QLearningModel:
        return training.fitted().pooled


# The doubly-robust dWOLS-Shared estimator lives in `dwols.py`; re-exported here
# so existing imports (pipeline, package __init__) keep working.
from treatmentrx.estimation.dwols import DWOLS_METHOD, DWOLSSharedEstimator  # noqa: E402


# `PolicyValueSelector` is gone, and it was worse than unused.
#
# It chose "the most interpretable estimator among those it cannot separate" —
# the job `training.best_score()` does (invariant 30) with the same held-out
# interval logic. Zero constructions anywhere, so its copy of the tie-break had
# drifted invisibly, and the two no longer described the same preference:
#
#     PolicyValueSelector            training.INTERPRETABILITY_ORDER
#     0  Q-Shared + Penalized        0  dWOLS-Shared
#     1  dWOLS-Shared                1  Q-Pooled
#     2  Bayesian Hierarchical Q     2  Stage-Specific Q-learning
#     3  Stage-Specific Q-learning   3  Q-Shared + Penalized
#     4  Survival Forest DTR
#
# Every shared name disagrees on position, and the reversal is the one that
# matters: it ranked `Q-Shared + Penalized` **first** where the live order ranks
# it last. That is the shared-blip fit invariant 18 keeps out of the ensemble and
# `cli coverage` measures at 28% pooled, 0% at the worst patient. It also named
# two estimators this package has never contained and omitted `Q-Pooled`, an
# actual `SERVING_ENSEMBLE` member, which `.get(name, 99)` would have ranked
# behind everything. Had anything called it, it would have selected the estimator
# with the worst measured coverage in the repo.


