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
    bootstrap = subparsers.add_parser(
        "inference", help="Compare sandwich and m-out-of-n bootstrap intervals"
    )
    bootstrap.add_argument("--replicates", type=int, default=200)
    args = parser.parse_args(argv)

    if args.command == "inference":
        print(json.dumps(_inference_report(args.replicates), indent=2, sort_keys=True))
        return

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
        "estimators": training.scorecard(include_oracle=True),
        "note": (
            "ipw_policy_value is the observational (Hajek) estimate on held-out stages; "
            "oracle_rollout_value is the total-trajectory reward under the generating "
            "process and exists only in simulation. Run `stability` before reading the "
            "ordering of these estimators as a ranking."
        ),
    }


def _inference_report(replicates: int) -> dict[str, Any]:
    """How far the sandwich understates the interval at non-terminal stages.

    The sandwich treats the pseudo-outcomes as fixed; the bootstrap re-runs the
    whole procedure, so the ratio between them is the size of the understatement.
    """
    from treatmentrx.data import DataLayer
    from treatmentrx.estimation.features import model_features

    setup = training.enable_bootstrap_inference(replicates=replicates)
    fit = training.fitted()
    model = fit.q_shared
    features = model_features(DataLayer().build_patient_state(sample_ra_bundle()).stages)

    stages = []
    for index in range(model.n_stages):
        ordered = sorted(model.arms, key=lambda arm: model.raw_q(features, arm, index), reverse=True)
        sandwich = model.sandwich_contrast(ordered[0], ordered[1], features, index)
        booted = model.bootstrap_contrast(ordered[0], ordered[1], features, index)
        stages.append(
            {
                "stage_index": index,
                "terminal": index == model.n_stages - 1,
                "arm": ordered[0],
                "comparator": ordered[1],
                "sandwich": sandwich.as_dict(),
                "bootstrap": booted.as_dict(),
                "se_ratio_bootstrap_over_sandwich": (
                    round(booted.standard_error / sandwich.standard_error, 3)
                    if sandwich.standard_error
                    else None
                ),
            }
        )

    return {
        "setup": setup,
        "patient": "demo (seropositive, prior TNF failure)",
        "stages": stages,
        "note": (
            "A ratio above 1 is the sandwich understating the interval. It is expected "
            "at non-terminal stages, where the regression target is a pseudo-outcome "
            "built from the fitted downstream model and the sandwich treats it as fixed "
            "data. At the terminal stage the two should agree."
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
