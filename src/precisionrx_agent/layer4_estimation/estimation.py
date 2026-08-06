"""Layer 4 estimator facades.

These are thin adapters: they map a patient's stage history into the estimator
covariate space, ask a model fitted in `training.py` for its Q-values, and wrap
the answer in the `MethodResult` contract the rest of the pipeline consumes. All
of the statistics lives in `q_learning.py` (outcome-model Q-learning with
backward induction) and `dwols.py` (doubly-robust blip regression).

`policy_value` on every result is the estimator's **held-out IPW policy value**,
not a transform of its own scores — so Bayesian model averaging downstream
weights estimators by out-of-sample performance.
"""

from __future__ import annotations

from precisionrx_agent.layer4_estimation import training
from precisionrx_agent.layer4_estimation.features import model_features, stage_index
from precisionrx_agent.layer4_estimation.q_learning import (
    Q_SHARED_METHOD,
    STAGE_SPECIFIC_METHOD,
    QLearningModel,
)
from precisionrx_agent.simulation.ra_cohort import TREATMENT_ARMS
from precisionrx_agent.shared.models import MethodResult, RegimeAssignment, RegimeType, StageRecord

DEFAULT_TREATMENT_MENU = TREATMENT_ARMS


class _QLearningEstimator:
    """Shared plumbing for the two Q-learning facades."""

    method_name = "Q-learning"
    regime_type = RegimeType.SPTR

    def model(self) -> QLearningModel:
        raise NotImplementedError

    def fit_predict(
        self,
        stages: list[StageRecord],
        assignment: RegimeAssignment,
        tailoring_variables: list[str],
        treatment_menu: tuple[str, ...] = DEFAULT_TREATMENT_MENU,
    ) -> MethodResult:
        model = self.model()
        features = model_features(stages)
        index = stage_index(stages, model.n_stages)
        q_values = model.q_values(features, index, treatment_menu)
        recommended = max(q_values, key=q_values.get)
        best = q_values[recommended]
        half_width = self._half_width(model, stages)
        return MethodResult(
            method_name=self.method_name,
            regime_type=self.regime_type,
            recommended_action=recommended,
            q_values=q_values,
            policy_value=training.policy_value_for(self.method_name),
            confidence_band=(
                round(max(best - half_width, 0.0), 3),
                round(min(best + half_width, 1.0), 3),
            ),
            coefficients=model.coefficient_summary(recommended, index),
            top_tailoring_variables=_format_tailoring_vars(stages[-1], tailoring_variables),
        )

    def _half_width(self, model: QLearningModel, stages: list[StageRecord]) -> float:
        """Width shrinks with training support and widens with a short history.

        A crude interval, and labelled as such: a proper sandwich/bootstrap
        interval for the blip parameters is the next thing this deserves.
        """
        support = model.n_train * model.n_stages
        base = 1.6 / max(support, 1) ** 0.5
        history_penalty = 0.02 * max(3 - len(stages), 0)
        return round(min(max(base + history_penalty, 0.04), 0.25), 3)


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


def _format_tailoring_vars(stage: StageRecord, variables: list[str]) -> list[str]:
    formatted = []
    for variable in variables:
        value = stage.features.get(variable)
        if value is not None:
            formatted.append(f"{variable}={value}")
    return formatted[:5]


# The doubly-robust dWOLS-Shared estimator lives in `dwols.py`; re-exported here
# so existing imports (pipeline, package __init__) keep working.
from precisionrx_agent.layer4_estimation.dwols import DWOLS_METHOD, DWOLSSharedEstimator  # noqa: E402


class PolicyValueSelector:
    """Pick the most interpretable estimator among those within `tolerance`.

    Used when a single named method is wanted instead of a model average.
    """

    interpretability_order = {
        Q_SHARED_METHOD: 0,
        DWOLS_METHOD: 1,
        "Bayesian Hierarchical Q": 2,
        STAGE_SPECIFIC_METHOD: 3,
        "Survival Forest DTR": 4,
    }

    def choose(self, results: list[MethodResult], tolerance: float = 0.02) -> MethodResult:
        if not results:
            raise ValueError("No method results to select from")
        best_value = max(result.policy_value for result in results)
        contenders = [result for result in results if best_value - result.policy_value <= tolerance]
        return sorted(contenders, key=lambda result: self.interpretability_order.get(result.method_name, 99))[0]
