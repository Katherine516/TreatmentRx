from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

from precisionrx_agent.demo_data import sample_ra_bundle
from precisionrx_agent.layer4_estimation import training
from precisionrx_agent.pipeline import PrecisionRxAgent
from precisionrx_agent.simulation.ra_cohort import behaviour_policy, myopic_optimal_policy, rollout_value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PrecisionRx agent prototype")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("demo", help="Run a synthetic RA patient through the agent")
    subparsers.add_parser("evaluate", help="Score every estimator on the held-out cohort")
    args = parser.parse_args(argv)

    if args.command == "evaluate":
        print(json.dumps(_evaluation_report(), indent=2, sort_keys=True))
        return

    if args.command in {None, "demo"}:
        recommendation = PrecisionRxAgent().recommend_from_fhir(sample_ra_bundle())
        print(json.dumps(_to_json(recommendation), indent=2, sort_keys=True))


def _evaluation_report() -> dict[str, Any]:
    fitted = training.fitted()
    return {
        "cohort": {
            "size": training.COHORT_SIZE,
            "seed": training.COHORT_SEED,
            "train_trajectories": len(fitted.train),
            "holdout_trajectories": len(fitted.holdout),
            "stages_per_trajectory": fitted.q_shared.n_stages,
        },
        "reference_policies": {
            "clinician_behaviour": rollout_value(behaviour_policy),
            "myopic_oracle": rollout_value(myopic_optimal_policy),
        },
        "estimators": training.scorecard(),
        "note": (
            "ipw_policy_value is the observational (Hajek) estimate on held-out stages; "
            "oracle_rollout_value is the total-trajectory reward under the generating "
            "process and exists only in simulation. Differences smaller than the IPW "
            "standard error should not be read as a ranking."
        ),
    }


def _to_json(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _to_json(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _to_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_json(item) for item in value]
    if isinstance(value, tuple):
        return [_to_json(item) for item in value]
    return value


if __name__ == "__main__":
    main()
