"""Scientific operating modes and versioned estimand contracts.

The same numerical object can mean very different things under randomization,
sequential observational identification, and pure outcome prediction.  These
types make that distinction executable: a workflow declares its mode and the
causal target before any model is allowed to score data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum


class ScientificMode(str, Enum):
    DTR_RESEARCH = "dtr_research"
    RANDOMIZED_TRIAL = "randomized_trial"
    BIOMARKER_RESEARCH = "biomarker_research"


class OutcomeDirection(str, Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


class EstimandContractError(ValueError):
    """The requested analysis does not have a coherent statistical target."""


@dataclass(frozen=True)
class EstimandContract:
    """The minimum information required to interpret an effect or policy value."""

    estimand_id: str
    version: str
    mode: ScientificMode
    target_population: str
    treatment_strategies: tuple[str, ...]
    reference_action: str
    outcome: str
    outcome_direction: OutcomeDirection
    horizon_days: int
    summary_measure: str
    contrast_scale: str
    intercurrent_event_strategy: tuple[tuple[str, str], ...]
    deterministic_regime: bool = True
    value_scope: str = "remaining_value_to_go"

    def __post_init__(self) -> None:
        required = {
            "estimand_id": self.estimand_id,
            "version": self.version,
            "target_population": self.target_population,
            "reference_action": self.reference_action,
            "outcome": self.outcome,
            "summary_measure": self.summary_measure,
            "contrast_scale": self.contrast_scale,
            "value_scope": self.value_scope,
        }
        empty = sorted(name for name, value in required.items() if not value.strip())
        if empty:
            raise EstimandContractError(
                "estimand fields may not be blank: " + ", ".join(empty)
            )
        if self.horizon_days <= 0:
            raise EstimandContractError("horizon_days must be positive")
        if len(self.treatment_strategies) < 2:
            raise EstimandContractError(
                "an effect estimand requires at least two treatment strategies"
            )
        if len(set(self.treatment_strategies)) != len(self.treatment_strategies):
            raise EstimandContractError("treatment strategies must be unique")
        if self.reference_action not in self.treatment_strategies:
            raise EstimandContractError(
                "reference_action must be one of treatment_strategies"
            )
        events = [event for event, _ in self.intercurrent_event_strategy]
        if len(events) != len(set(events)):
            raise EstimandContractError(
                "each intercurrent event may have only one declared strategy"
            )

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(include_fingerprint=False), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def as_dict(self, include_fingerprint: bool = True) -> dict[str, object]:
        payload = asdict(self)
        payload["mode"] = self.mode.value
        payload["outcome_direction"] = self.outcome_direction.value
        payload["treatment_strategies"] = list(self.treatment_strategies)
        payload["intercurrent_event_strategy"] = dict(
            self.intercurrent_event_strategy
        )
        if include_fingerprint:
            payload["fingerprint"] = self.fingerprint
        return payload

    def assert_compatible(self, other: "EstimandContract") -> None:
        """Block averaging of methods that estimate different quantities."""
        if self.fingerprint != other.fingerprint:
            raise EstimandContractError(
                f"estimands {self.estimand_id!r} and {other.estimand_id!r} "
                "are not compatible for model averaging"
            )


@dataclass(frozen=True)
class EvaluationPartitionContract:
    """Patient-level partitions with one-use roles and a locked final test.

    **This type used to be decoration and that was worse than not having it.**
    It was exported, unit-tested, and never constructed anywhere in the
    pipeline, which meant the package's public surface advertised a locked final
    test that did not exist. `training.evaluation_partition()` now builds the
    partition this build actually has — a training split and an evaluation split
    and nothing else — and `has_final_test` reports the absence rather than
    letting `final_test_locked=True` imply a discipline nobody is keeping.

    The distinction matters here more than usual. The evaluation split does four
    jobs at once: it sets the model-averaging weights, it selects the serving
    estimator through `best_score()`, it supplies the calibration the validation
    ladder gates on, and it is the headline policy value. Every constant that was
    tuned against a Monte Carlo study was at least tuned on fresh seeds — see
    `feedback/coverage.py` — but the numbers on the model card come from a split
    that has been looked at repeatedly, and there is no untouched partition left
    to check them against.
    """

    training: frozenset[str]
    tuning: frozenset[str]
    calibration: frozenset[str]
    final_test: frozenset[str]
    final_test_locked: bool = True

    def __post_init__(self) -> None:
        named = {
            "training": self.training,
            "tuning": self.tuning,
            "calibration": self.calibration,
            "final_test": self.final_test,
        }
        for left_name, left in named.items():
            for right_name, right in named.items():
                if left_name >= right_name:
                    continue
                overlap = left & right
                if overlap:
                    raise EstimandContractError(
                        f"{left_name} and {right_name} overlap for "
                        f"{len(overlap)} patient(s)"
                    )
        if not self.final_test_locked:
            raise EstimandContractError(
                "the final test partition must be locked before model development"
            )

    @property
    def has_final_test(self) -> bool:
        """Is there a partition held back from model development at all?

        `final_test_locked` says the set may not be touched; this says whether
        there is a set. An empty partition satisfies "locked" vacuously, which is
        the reading that let this type describe a discipline the codebase was not
        following.
        """
        return bool(self.final_test)

    @property
    def roles_in_use(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, members in (
                ("training", self.training),
                ("tuning", self.tuning),
                ("calibration", self.calibration),
                ("final_test", self.final_test),
            )
            if members
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "sizes": {
                "training": len(self.training),
                "tuning": len(self.tuning),
                "calibration": len(self.calibration),
                "final_test": len(self.final_test),
            },
            "roles_in_use": list(self.roles_in_use),
            "has_final_test": self.has_final_test,
            "final_test_locked": self.final_test_locked,
        }


def ra_dtr_estimand(treatment_arms: tuple[str, ...]) -> EstimandContract:
    return EstimandContract(
        estimand_id="ra_sequential_response_value",
        version="ra-estimand-v1",
        mode=ScientificMode.DTR_RESEARCH,
        target_population=(
            "research-eligible rheumatoid-arthritis trajectories satisfying "
            "the versioned RA data contract"
        ),
        treatment_strategies=treatment_arms,
        reference_action="continue-current",
        outcome="stage response utility (synthetic-cohort default)",
        outcome_direction=OutcomeDirection.HIGHER_IS_BETTER,
        horizon_days=90,
        summary_measure="mean remaining value-to-go",
        contrast_scale="utility difference",
        intercurrent_event_strategy=(
            ("treatment_switch", "treatment-policy strategy"),
            ("rescue_therapy", "treatment-policy strategy"),
            ("serious_toxicity", "composite utility penalty"),
            ("death", "composite worst outcome"),
            ("dropout", "censoring-weight strategy"),
        ),
    )


__all__ = [
    "EstimandContract",
    "EstimandContractError",
    "EvaluationPartitionContract",
    "OutcomeDirection",
    "ScientificMode",
    "ra_dtr_estimand",
]
