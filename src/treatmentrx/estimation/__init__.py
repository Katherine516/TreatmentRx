"""Layer 2 — `PatientState` to a set of `RegimeEstimate`s.

Three estimators score the same arm menu on the same scale, fit once per process
on the same training split (`training.py`). They are kept deliberately different
in their failure modes: Q-learning is an outcome-model method, dWOLS is
doubly-robust, and the stage-specific fit relaxes the shared-blip assumption.
Averaging them is only informative because they can fail differently.

This layer hands the estimators the patient's `StageRecord`s untouched, so the
timing, belief, switching and competing-risk annotations from Layer 1 reach the
covariate mapping intact.
"""

from __future__ import annotations

from dataclasses import replace

from treatmentrx.contracts import LayerDiagnostic, PatientState, RegimeEstimate
from treatmentrx.domain import RegimeAssignment, RegimeType
from treatmentrx.estimation.dwols import DWOLSSharedEstimator
from treatmentrx.estimation.estimators import QSharedEstimator, StageSpecificQEstimator
from treatmentrx.estimation.regime import AdaptiveRegimeSelector


class EstimationLayer:
    def __init__(self) -> None:
        self.regime_selector = AdaptiveRegimeSelector()
        self.estimators = (
            QSharedEstimator(),
            DWOLSSharedEstimator(),
            StageSpecificQEstimator(),
        )

    def estimate(self, state: PatientState) -> list[RegimeEstimate]:
        stages = state.stages
        assignment = self.regime_selector.select(stages) if stages else self._fallback_assignment()
        menu = tuple(state.feasible_arms)
        return [
            self._with_diagnostic(
                estimator.fit_predict(stages, assignment, state.tailoring_variables, menu)
            )
            for estimator in self.estimators
        ]

    def _with_diagnostic(self, estimate: RegimeEstimate) -> RegimeEstimate:
        return replace(
            estimate,
            diagnostics=[
                LayerDiagnostic(
                    name=f"estimator:{estimate.estimator}",
                    passed=True,
                    severity="info",
                    message=(
                        f"{estimate.estimator} recommends {estimate.recommended_arm}; "
                        f"held-out policy value {estimate.policy_value:.3f}"
                    ),
                )
            ],
        )

    def _fallback_assignment(self) -> RegimeAssignment:
        return RegimeAssignment(
            regime_type=RegimeType.SPTR,
            reason="fallback when no stages are available",
            shared_bic=0.0,
            stage_specific_bic=0.0,
        )


__all__ = ["EstimationLayer"]
