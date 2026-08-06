from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

from treatmentrx import TreatmentRxOrchestrator
from treatmentrx.demo_data import sample_ra_bundle
from treatmentrx.estimation import training
from treatmentrx.simulation.ra_cohort import behaviour_policy, myopic_optimal_policy, rollout_value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="TreatmentRx research agent")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("demo", help="Run a synthetic RA patient through all six layers")
    subparsers.add_parser("evaluate", help="Score every estimator on the held-out cohort")
    subparsers.add_parser("stability", help="Re-fit across seeds and folds to test the ranking")
    args = parser.parse_args(argv)

    if args.command == "evaluate":
        print(json.dumps(_evaluation_report(), indent=2, sort_keys=True))
        return

    if args.command == "stability":
        from treatmentrx.feedback.stability import stability_report

        print(json.dumps(stability_report(), indent=2, sort_keys=True))
        return

    recommendation = TreatmentRxOrchestrator().run(sample_ra_bundle())
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
            "process and exists only in simulation. Run `stability` before reading the "
            "ordering of these estimators as a ranking."
        ),
    }


def _to_json(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _to_json(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json(item) for item in value]
    return value


if __name__ == "__main__":
    main()
