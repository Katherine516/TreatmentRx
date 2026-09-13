"""Layer 2 — `PatientState` to a set of `RegimeEstimate`s.

Two estimators score the same arm menu on the same scale, fit once per process on
the same training split (`training.py`). They are kept deliberately different in
their failure modes: stage-specific Q-learning is an outcome-model method, dWOLS
is doubly-robust. Averaging them is only informative because they can fail
differently — and only *valid* because they estimate the same quantity.

That second condition is why the shared-blip fit is not here. It is still fitted
and still scored; `training.SERVING_ENSEMBLE` carries the membership and the
measurements behind it. Averaging a stage-pooled ψ with a stage-resolved one
produced an ensemble centred between two parameters, which no weighting can
repair.

This layer hands the estimators the patient's `StageRecord`s untouched, so the
timing, belief, switching and competing-risk annotations from Layer 1 reach the
covariate mapping intact.

**There is no regime-selection step, deliberately.** An `AdaptiveRegimeSelector`
used to run here and route the patient to SPTR, DTR or a hybrid. Its result was
passed to every estimator and read by none of them — `RegimeEstimate.regime_type`
comes from the estimator's own class — and the "BIC" it selected on was
degenerate: the stage-specific term was `sum((outcome - stage.outcome) ** 2)`
over the same sequence twice, which is identically zero, so one side of the
comparison was a constant. The shared-versus-stage-specific question it purported
to answer is answered better by fitting both and averaging them, which is what
`cli misspecification` justifies: they trade accuracy against robustness in
opposite directions, so choosing one is the thing to avoid.
"""

from __future__ import annotations

from dataclasses import replace

from treatmentrx.contracts import LayerDiagnostic, PatientState, RegimeEstimate
from treatmentrx.estimation.dwols import DWOLSSharedEstimator
from treatmentrx.estimation.estimators import PooledQEstimator
from treatmentrx.scientific import ScientificMode


class EstimationLayer:
    def __init__(self) -> None:
        self.estimators = (
            PooledQEstimator(),
            DWOLSSharedEstimator(),
        )

    def estimate(self, state: PatientState) -> list[RegimeEstimate]:
        if state.operating_mode is not ScientificMode.DTR_RESEARCH:
            raise ValueError(
                "EstimationLayer is authorized only for dtr_research patient states"
            )
        if state.estimand_contract is None:
            raise ValueError("PatientState is missing its estimand contract")
        menu = tuple(state.feasible_arms)
        return [
            self._with_diagnostic(
                replace(
                    estimator.fit_predict(state.stages, menu),
                    estimand_fingerprint=state.estimand_contract.fingerprint,
                )
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


__all__ = ["EstimationLayer"]
