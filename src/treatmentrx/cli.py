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


def build_parser() -> argparse.ArgumentParser:
    """The subcommand surface, built separately so it can be inspected.

    Split out of `main` for `tests/test_docs.py`, which checks that every command
    `CLAUDE.md` documents exists and that every command that exists is
    documented. Built inline, the parser could only be reached by running the
    command, so that test could not close — it skipped, which is a test that
    cannot fail.
    """
    parser = argparse.ArgumentParser(description="TreatmentRx research agent")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("demo", help="Run a synthetic RA patient through all six layers")
    subparsers.add_parser("evaluate", help="Score every estimator on the held-out cohort")
    subparsers.add_parser("stability", help="Re-fit across seeds and folds to test the ranking")
    bootstrap = subparsers.add_parser(
        "inference", help="Compare sandwich and m-out-of-n bootstrap intervals"
    )
    bootstrap.add_argument("--replicates", type=int, default=200)
    subparsers.add_parser("audit", help="Layer-by-layer evaluation of the whole agent")
    subparsers.add_parser(
        "specification", help="Is the blip basis missing an effect modifier?"
    )
    service = subparsers.add_parser("serve", help="Run the HTTP service (stdlib only)")
    service.add_argument("--host", default=None)
    service.add_argument("--port", type=int, default=None)
    transfer = subparsers.add_parser(
        "transfer", help="Does any of it survive data it was not fit on?"
    )
    transfer.add_argument("--patients", type=int, default=None)
    transfer.add_argument("--replications", type=int, default=None)
    subgroups = subparsers.add_parser(
        "subgroups", help="Who does the agent abstain on, and is it earned?"
    )
    subgroups.add_argument("--patients", type=int, default=None)
    subgroups.add_argument("--seed", type=int, default=None)
    power = subparsers.add_parser(
        "power", help="How much training data before the agent stops abstaining?"
    )
    power.add_argument("--patients", type=int, default=None)
    power.add_argument(
        "--target", type=float, default=None, help="abstention rate to solve for"
    )
    misspec = subparsers.add_parser(
        "misspecification", help="Which estimator survives the model being wrong"
    )
    misspec.add_argument(
        "--omitted-modifier",
        action="store_true",
        help="also bend the blip basis, not just the nuisance surface",
    )
    misspec.add_argument(
        "--extra-modifier",
        action="store_true",
        help="price a blip basis that carries a term the estimand does not need",
    )
    cover = subparsers.add_parser("coverage", help="Do the confidence intervals actually cover?")
    cover.add_argument("--replications", type=int, default=100)
    cover.add_argument("--bootstrap", action="store_true", help="also run the (slow) bootstrap arm")
    cover.add_argument(
        "--decision-rule", action="store_true", help="also measure the rule Layer 3 deploys"
    )
    cover.add_argument(
        "--multiplicity",
        action="store_true",
        help="what the family-wise correction buys, and what it costs in decisiveness",
    )
    cover.add_argument(
        "--candidate-set",
        action="store_true",
        help="does the set of indistinguishable arms contain the optimal one?",
    )
    cover.add_argument(
        "--stages",
        action="store_true",
        help="coverage at every stage index, not only the terminal block",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "audit":
        from treatmentrx.feedback.audit import full_audit

        print(json.dumps(full_audit(), indent=2, default=str))
        return

    if args.command == "serve":
        from treatmentrx.service import DEFAULT_HOST, DEFAULT_PORT, serve

        serve(host=args.host or DEFAULT_HOST, port=args.port or DEFAULT_PORT)
        return

    if args.command == "specification":
        from treatmentrx.estimation.specification import specification_report

        print(json.dumps(specification_report(), indent=2))
        return

    if args.command == "transfer":
        from treatmentrx.feedback.transfer import (
            COVERAGE_REPLICATIONS,
            DEFAULT_EVAL_SIZE,
            transfer_report,
        )

        print(
            json.dumps(
                transfer_report(
                    n=args.patients or DEFAULT_EVAL_SIZE,
                    replications=args.replications or COVERAGE_REPLICATIONS,
                ),
                indent=2,
                default=str,
            )
        )
        return

    if args.command == "subgroups":
        from treatmentrx.feedback.subgroups import (
            DEFAULT_PATIENTS,
            DEFAULT_SEED,
            subgroup_report,
        )

        print(
            json.dumps(
                subgroup_report(
                    n_patients=args.patients or DEFAULT_PATIENTS,
                    seed=args.seed if args.seed is not None else DEFAULT_SEED,
                ),
                indent=2,
            )
        )
        return

    if args.command == "power":
        from treatmentrx.feedback.power import (
            DEFAULT_PATIENTS,
            TARGET_ABSTENTION,
            power_report,
        )

        print(
            json.dumps(
                power_report(
                    n_patients=args.patients or DEFAULT_PATIENTS,
                    target=args.target if args.target is not None else TARGET_ABSTENTION,
                ),
                indent=2,
            )
        )
        return

    if args.command == "misspecification":
        from treatmentrx.feedback.misspecification import (
            extra_modifier_report,
            misspecification_report,
            omitted_modifier_report,
        )

        report = misspecification_report()
        if args.omitted_modifier:
            report["omitted_effect_modifier"] = omitted_modifier_report()
        if args.extra_modifier:
            report["superfluous_effect_modifier"] = extra_modifier_report()
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    if args.command == "coverage":
        print(
            json.dumps(
                _coverage_report(
                    args.replications,
                    args.bootstrap,
                    args.decision_rule,
                    args.candidate_set,
                    args.multiplicity,
                    args.stages,
                ),
                indent=2,
            )
        )
        return

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
        "propensity": training.propensity_comparison(),
        "selection": {
            "ranking_resolved": training.ranking_is_resolved(),
            "selected_estimator": training.best_score().estimator,
            "basis": (
                "held-out interval clears the runner-up's"
                if training.ranking_is_resolved()
                else "values overlap; broken by INTERPRETABILITY_ORDER, not by measurement"
            ),
        },
        "note": (
            "ipw_policy_value is the observational (Hajek) estimate on held-out stages, "
            "with a percentile interval from resampling held-out trajectories; "
            "oracle_rollout_value is the total-trajectory reward under the generating "
            "process and exists only in simulation. The intervals here cover evaluation "
            "noise on a fixed fit — run `stability` for the fit's own variability across "
            "seeds before reading the ordering of these estimators as a ranking."
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


def _coverage_report(
    replications: int,
    include_bootstrap: bool,
    include_decision_rule: bool = False,
    include_candidate_set: bool = False,
    include_multiplicity: bool = False,
    include_stages: bool = False,
) -> dict[str, Any]:
    """Empirical coverage of the nominal 95% intervals.

    The one number that validates everything else the system says about
    uncertainty. A standard error can shrink correctly with sqrt(n), be reported
    on the right scale, and still systematically miss.
    """
    from treatmentrx.feedback import coverage

    results = [
        coverage.sandwich_coverage(replications=replications, share_blip=False),
        coverage.sandwich_coverage(replications=replications, share_blip=True),
        # The rule Layer 3 actually deploys. It used to be opt-in, which meant
        # the default run validated two components and skipped the composite.
        coverage.decision_rule_coverage(replications=min(replications, 40)),
    ]
    if include_bootstrap:
        results.append(coverage.bootstrap_coverage())
    if include_decision_rule:
        results.append(coverage.joint_rule_coverage())
    candidate_set = (
        coverage.candidate_set_coverage(replications=min(replications, 40))
        if include_candidate_set
        else None
    )
    multiplicity = coverage.multiplicity_sweep() if include_multiplicity else None
    stages = (
        coverage.decision_rule_stage_sweep(replications=min(replications, 40))
        if include_stages
        else None
    )
    return {
        "cohort_size": coverage.DEPLOYED_N,
        **({"candidate_set": candidate_set} if candidate_set else {}),
        **({"multiplicity": multiplicity} if multiplicity else {}),
        **({"stage_sweep": stages} if stages else {}),
        "patient_grid": {
            name: _grid_entry(features) for name, features in coverage.PATIENT_GRID.items()
        },
        "results": [result.as_dict() for result in results],
        "verdict": coverage.verdict(results),
        "note": (
            "Each row pools a sweep over the patient grid off one fit per "
            "replication; `per_patient` carries the individual rows. Coverage is "
            "measured at the size the deployed models are fit on, and each patient "
            "is contrasted on their own true top-two arms, which is the quantity "
            "Layer 3 reports. `blip_truth` is the single-visit contrast the "
            "coverage count is scored against; `value_to_go_range` is the span of "
            "the true contrast across the decision points a shared-blip fit pools "
            "over. An estimate inside that span is answering a different question, "
            "not getting this one wrong. Every row here is measured at the "
            "terminal block, the one stage where both serving estimators target "
            "the same quantity; `--stages` sweeps the stage index the way the "
            "patient grid sweeps the covariates."
        ),
    }


def _grid_entry(features: dict[str, float]) -> dict[str, Any]:
    from treatmentrx.feedback import coverage
    from treatmentrx.simulation.ra_cohort import true_blip

    arm, comparator = coverage.top_two(features)
    low, high = coverage.estimand_range(features, arm, comparator)
    return {
        "features": features,
        "contrast": f"{arm} vs {comparator}",
        "blip_truth": round(true_blip(arm, features) - true_blip(comparator, features), 4),
        "value_to_go_range": [round(low, 4), round(high, 4)],
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
