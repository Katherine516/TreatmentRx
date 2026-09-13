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
        recommended = max(q_values, key=q_values.get)
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


class PolicyValueSelector:
    """Pick the most interpretable estimator among those it cannot separate.

    Used when a single named method is wanted instead of a model average.

    The tolerance is no longer a hand-set 0.02. It comes from the held-out
    bootstrap intervals in `training`: two estimators are contenders unless the
    leader's interval clears the other's. A fixed tolerance answers "is the gap
    smaller than a number I chose"; this answers "can the holdout tell these
    apart", which is the question. Where an interval is unavailable the estimator
    is treated as a contender, because an unmeasured difference is not a
    demonstrated one.
    """

    interpretability_order = {
        Q_SHARED_METHOD: 0,
        DWOLS_METHOD: 1,
        "Bayesian Hierarchical Q": 2,
        STAGE_SPECIFIC_METHOD: 3,
        "Survival Forest DTR": 4,
    }

    def choose(self, results: list[RegimeEstimate]) -> RegimeEstimate:
        if not results:
            raise ValueError("No method results to select from")
        ranked = sorted(results, key=lambda result: result.policy_value, reverse=True)
        leader = training.score_for(ranked[0].estimator)
        contenders = [
            result
            for result in ranked
            if result is ranked[0]
            or leader is None
            or not leader.beats(training.score_for(result.estimator) or leader)
        ]
        return sorted(
            contenders, key=lambda result: self.interpretability_order.get(result.estimator, 99)
        )[0]
