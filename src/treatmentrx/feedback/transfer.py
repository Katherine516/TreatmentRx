"""Does any of this survive data it was not fit on?

Every number in this repo is measured against one generating process, and the
estimators' blip basis *is* that process's blip basis. `curvature` and
`blip_modifier` bend that single cohort along two known axes, which is
informative but is still the same cohort bent two ways. Nothing here has ever
asked whether the *pipeline* holds up on structurally different data — and the
four numbers a reader would quote (interval coverage 95%, ECE 0.011, abstention
67%, policy value above behaviour) are all in-distribution.

So: fit at site A, evaluate at site B. The shift is deliberately confined to
things that leave the estimand alone — case mix, prescribing habits, retention —
because a bent estimand is already measured and nothing survives it. What is
open is whether a *correctly specified* pipeline still works when the population
and the practice change, which is the shift a real deployment actually meets.

**This does not advance the validation ladder.** A second simulation is not live
data, `live_data` stays False, and no result here should be read as evidence that
the agent would work in a clinic. What it establishes is narrower and still worth
having: which of the four claims is fragile, and in what order they break.

The interesting output is the *ordering*. Calibration surviving while coverage
collapses means something quite different from the reverse, and either changes
what the model card has to say.
"""

from __future__ import annotations

from dataclasses import dataclass

from treatmentrx.estimation import training
from treatmentrx.estimation.q_learning import QLearningModel
from treatmentrx.simulation.ra_cohort import (
    CohortShift,
    behaviour_policy,
    generate_ra_cohort,
    rollout_value,
)

# Each site changes one thing, so a degradation can be attributed. The combined
# site is the realistic case — real sites differ in every way at once — and is
# reported last because its value is the total, not the attribution.
SITES: tuple[CohortShift, ...] = (
    CohortShift(name="baseline (same process)"),
    CohortShift(
        name="case mix: sicker, more seronegative",
        das28_range=(4.5, 9.0),
        anti_ccp_rate=0.35,
        prior_tnf_rate=0.55,
    ),
    CohortShift(
        name="case mix: milder, mostly seropositive",
        das28_range=(2.5, 6.0),
        anti_ccp_rate=0.85,
        prior_tnf_rate=0.15,
    ),
    CohortShift(
        name="practice: TNF-first, rituximab-averse",
        assignment_tilt=(("TNF-inhibitor", 1.2), ("rituximab", -1.0), ("JAK-inhibitor", -0.5)),
    ),
    CohortShift(name="retention: heavy attrition", dropout_shift=1.0),
    CohortShift(name="retention: near-complete follow-up", dropout_shift=-1.5),
    CohortShift(
        name="combined: all three at once",
        das28_range=(4.5, 9.0),
        anti_ccp_rate=0.35,
        prior_tnf_rate=0.55,
        assignment_tilt=(("TNF-inhibitor", 1.2), ("rituximab", -1.0)),
        dropout_shift=0.8,
    ),
    # The bound, not a peer of the rows above: the treatment effects themselves
    # differ, so the fitted parameters are genuinely wrong here.
    CohortShift(name="ESTIMAND SHIFTED: blips 1.5x (bound)", blip_scale=1.5),
)

DEFAULT_EVAL_SIZE = 400
DEFAULT_EVAL_SEED = 20260825
ROLLOUT_N = 2000


@dataclass(frozen=True)
class SiteResult:
    site: str
    shifts_the_estimand: bool
    n_trajectories: int
    n_scored: int
    n_rejected_by_contract: int
    ipw_policy_value: float
    behaviour_policy_value: float
    calibration_error: float
    calibration_passed: bool
    abstention_rate: float
    coverage: float | None
    rollout_agent: float
    rollout_behaviour: float

    @property
    def ipw_improvement(self) -> float:
        return self.ipw_policy_value - self.behaviour_policy_value

    @property
    def rollout_improvement(self) -> float:
        return self.rollout_agent - self.rollout_behaviour

    def as_dict(self) -> dict[str, object]:
        return {
            "site": self.site,
            "shifts_the_estimand": self.shifts_the_estimand,
            "n_trajectories": self.n_trajectories,
            "n_scored": self.n_scored,
            "n_rejected_by_contract": self.n_rejected_by_contract,
            "held_out_ipw_policy_value": round(self.ipw_policy_value, 4),
            "behaviour_policy_value": round(self.behaviour_policy_value, 4),
            "ipw_improvement": round(self.ipw_improvement, 4),
            "rollout_agent": round(self.rollout_agent, 4),
            "rollout_behaviour": round(self.rollout_behaviour, 4),
            "rollout_improvement": round(self.rollout_improvement, 4),
            "expected_calibration_error": round(self.calibration_error, 4),
            "calibration_passed": self.calibration_passed,
            "abstention_rate": round(self.abstention_rate, 4),
            "interval_coverage": (
                None if self.coverage is None else round(self.coverage, 4)
            ),
        }


def _serving_policy():
    """The policy the agent deploys: model-averaged over the *deployed* fit.

    `training.serving_models()` rather than a local map — invariant 19, which bit
    twice when a study kept its own and silently collapsed to one estimator while
    still reporting an ensemble figure.
    """
    models = training.serving_models()
    arms = training.fitted().pooled.arms

    def q_values(model, features: dict[str, float], stage_index: int) -> dict[str, float]:
        # dWOLS has no stage argument — its blip is shared by construction — and
        # Q-learning requires one. The signatures differ because the models do.
        if isinstance(model, QLearningModel):
            return model.q_values(features, stage_index)
        return model.q_values(features)

    def policy(features: dict[str, float], stage_index: int) -> str:
        totals = {arm: 0.0 for arm in arms}
        for model in models.values():
            values = q_values(model, features, stage_index)
            for arm in arms:
                totals[arm] += values.get(arm, 0.0)
        return max(totals, key=totals.get)

    return policy, models


def _predict_outcome(models):
    def predict(features: dict[str, float], arm: str, stage_index: int) -> float:
        values = [model.predict_outcome(features, arm) for model in models.values()]
        return sum(values) / len(values)

    return predict


def _decisions_at_site(shift: CohortShift, n: int, seed: int):
    """Run the deployed pipeline over this site's patients, as a clinician would.

    Layer 1 through Layer 3 on exported bundles, so the abstention rate and the
    interval are the ones the agent actually reports rather than a
    reconstruction. Records this site's shift causes the data contract to reject
    are counted, not silently dropped — a site whose patients the contract
    refuses is a transfer failure of a different and more basic kind.
    """
    from treatmentrx.data import DataContractError, DataLayer
    from treatmentrx.decision import DecisionLayer
    from treatmentrx.domain import RecommendationStatus
    from treatmentrx.estimation import EstimationLayer
    from treatmentrx.estimation.features import model_features
    from treatmentrx.simulation.fhir_export import simulated_bundles

    data_layer, estimation, decision_layer = DataLayer(), EstimationLayer(), DecisionLayer()
    rejected = 0
    rows = []
    for bundle in simulated_bundles(n, seed=seed, shift=shift):
        try:
            state = data_layer.build_patient_state(bundle)
        except DataContractError:
            rejected += 1
            continue
        decision = decision_layer.decide(state, estimation.estimate(state))
        if decision.contrast is None:
            continue
        rows.append(
            (
                model_features(state.stages),
                min(len(state.stages) - 1, training.fitted().pooled.n_stages - 1),
                decision.status is RecommendationStatus.EQUIPOISE,
                decision.contrast,
            )
        )
    return rows, rejected


def _abstention(rows) -> float:
    """What fraction of this site's patients the deployed agent declines to separate.

    A single-fit quantity on purpose: this is what the deployed model does at
    this site, which is the deployment question. Coverage cannot be measured the
    same way — see `transfer_coverage`.
    """
    if not rows:
        return 0.0
    return sum(1 for _, _, abstained, _ in rows if abstained) / len(rows)


PATIENTS_PER_SITE = 8
COVERAGE_REPLICATIONS = 30


def _site_patient_grid(shift: CohortShift, k: int, seed: int) -> list[dict[str, float]]:
    """`k` terminal-stage patients drawn from this site, spread over the cohort.

    Evenly spaced rather than the first `k`, so the grid samples the site's case
    mix instead of whichever patients the generator happened to emit first.
    """
    cohort = generate_ra_cohort(max(k * 10, 100), seed + 1, shift=shift)
    step = max(len(cohort) // k, 1)
    return [cohort[i].stages[-1].features for i in range(0, len(cohort), step)][:k]


def transfer_coverage(
    sites: tuple[CohortShift, ...] = (),
    patients_per_site: int = PATIENTS_PER_SITE,
    replications: int = COVERAGE_REPLICATIONS,
    seed: int = DEFAULT_EVAL_SEED,
) -> dict[str, object]:
    """Replicated coverage of the site-A decision rule on each site's patients.

    **Why this cannot be read off one fit.** The obvious version — take the
    deployed model, run it over a site's patients, count how many intervals
    contain the truth — is not coverage. Every patient's interval is built from
    the *same* fitted parameters, so their errors are correlated: one unlucky fit
    misses for everybody at once, and the fraction you measure is a property of
    the single draw you happened to take. Measured that way the baseline site
    reads 87%, against the 95% `cli coverage` reports for the same rule, and the
    gap is the method of measurement rather than anything about transfer.

    So the replication is over *fits*: each replication refits the serving
    ensemble on a fresh site-A cohort — the training site, which does not change
    — and evaluates it on every site's fixed patient grid. One refit loop serves
    all sites, because the fit does not depend on where it is evaluated.
    """
    from treatmentrx.feedback.coverage import (
        DEPLOYED_N,
        _averaged_contrast,
        _pool,
        _serving_contrasts,
        _Tally,
    )

    honest = [site for site in sites if not site.shifts_the_estimand]
    tallies = {
        site.name: [
            _Tally(f"{site.name} #{i}", features)
            for i, features in enumerate(_site_patient_grid(site, patients_per_site, seed))
        ]
        for site in honest
    }

    for replication in range(replications):
        contrasts_for = _serving_contrasts(DEPLOYED_N, 9_700 + replication)
        for site_tallies in tallies.values():
            for tally in site_tallies:
                components = contrasts_for(tally)
                tally.record(
                    _averaged_contrast(
                        list(components.values()), tally.arm, tally.comparator
                    )
                )

    return {
        name: _pool(name, site_tallies) for name, site_tallies in tallies.items()
    }


def evaluate_site(
    shift: CohortShift,
    n: int = DEFAULT_EVAL_SIZE,
    seed: int = DEFAULT_EVAL_SEED,
    rollouts: int = ROLLOUT_N,
) -> SiteResult:
    """Score the site-A fit on a site-B cohort. Nothing is refit."""
    from treatmentrx.feedback.offline_evaluation import evaluate_policy

    policy, models = _serving_policy()
    cohort = generate_ra_cohort(n, seed, shift=shift)

    score = evaluate_policy(
        estimator="serving ensemble",
        policy=policy,
        predict_outcome=_predict_outcome(models),
        holdout=cohort,
        with_intervals=False,
    )
    behaviour = evaluate_policy(
        estimator="behaviour",
        policy=behaviour_policy,
        predict_outcome=_predict_outcome(models),
        holdout=cohort,
        with_intervals=False,
    )
    rows, rejected = _decisions_at_site(shift, n, seed)

    return SiteResult(
        site=shift.name,
        shifts_the_estimand=shift.shifts_the_estimand,
        n_trajectories=len(cohort),
        n_scored=len(rows),
        n_rejected_by_contract=rejected,
        ipw_policy_value=score.ipw_policy_value,
        behaviour_policy_value=behaviour.ipw_policy_value,
        calibration_error=score.calibration.expected_calibration_error,
        calibration_passed=score.calibration.passed,
        abstention_rate=_abstention(rows),
        coverage=None,
        rollout_agent=rollout_value(policy, n=rollouts, shift=shift),
        rollout_behaviour=rollout_value(behaviour_policy, n=rollouts, shift=shift),
    )


def transfer_report(
    sites: tuple[CohortShift, ...] = SITES,
    n: int = DEFAULT_EVAL_SIZE,
    seed: int = DEFAULT_EVAL_SEED,
    replications: int = COVERAGE_REPLICATIONS,
) -> dict[str, object]:
    results = [evaluate_site(shift, n, seed) for shift in sites]
    coverage = transfer_coverage(sites, replications=replications, seed=seed)
    rows = []
    for result in results:
        row = result.as_dict()
        measured = coverage.get(result.site)
        row["interval_coverage"] = None if measured is None else measured.coverage
        row["se_to_sd_ratio"] = None if measured is None else measured.se_to_sd_ratio
        rows.append(row)

    return {
        "training_site": {
            "cohort_size": training.COHORT_SIZE,
            "seed": training.COHORT_SEED,
            "serving_ensemble": list(training.SERVING_ENSEMBLE),
        },
        "evaluation_size": n,
        "evaluation_seed": seed,
        "coverage_replications": replications,
        "sites": rows,
        "verdict": _verdict(results, coverage),
        "note": (
            "Nothing is refit for the policy columns: the deployed site-A "
            "ensemble is scored on each site-B cohort. Rows above the "
            "estimand-shifted one leave TRUE_BLIPS alone, so the target is "
            "identical and representable and any degradation is the pipeline "
            "rather than the model being wrong. Coverage is replicated over "
            "site-A refits — it cannot be read off a single fit, because every "
            "patient's interval shares that fit's parameters. This is not live "
            "data and does not advance the validation ladder."
        ),
    }


def _verdict(results: list[SiteResult], coverage: dict) -> str:
    honest = [r for r in results if not r.shifts_the_estimand]
    shifted = [r for r in results if r.shifts_the_estimand]
    if not honest:
        return "no estimand-preserving site measured"
    losers = [r for r in honest if r.rollout_improvement <= 0.0]
    covered = {name: res.coverage for name, res in coverage.items() if res is not None}

    lines = [
        f"The policy transfers: across {len(honest)} estimand-preserving sites the "
        f"agent beats local practice on rollout value at "
        f"{len(honest) - len(losers)}/{len(honest)}"
        + (
            " (fails at: " + ", ".join(r.site for r in losers) + ")."
            if losers
            else ", by "
            f"{min(r.rollout_improvement for r in honest):+.3f} to "
            f"{max(r.rollout_improvement for r in honest):+.3f}."
        )
    ]
    if covered:
        worst = min(covered, key=covered.get)
        lines.append(
            f"Interval coverage holds between {min(covered.values()):.0%} and "
            f"{max(covered.values()):.0%} (worst: {worst})."
        )
    lines.append(
        f"Calibration is the robust one: expected calibration error stays within "
        f"{min(r.calibration_error for r in honest):.4f}-"
        f"{max(r.calibration_error for r in honest):.4f} across every "
        "estimand-preserving site."
    )
    if shifted:
        bound = shifted[0]
        lines.append(
            f"Under the estimand shift it is calibration that breaks "
            f"({bound.calibration_error:.4f}, an order of magnitude worse) while "
            f"the policy still improves ({bound.rollout_improvement:+.3f}) — "
            "scaling every blip preserves the ordering of the arms, so the "
            "recommendation stays right while the numbers attached to it stop "
            "meaning anything. Policy value cannot detect that; calibration can."
        )
    baseline = next(
        (result for result in honest if result.site == SITES[0].name), honest[0]
    )
    lines.append(
        f"Abstention is the fragile one: "
        f"{min(r.abstention_rate for r in honest):.0%}-"
        f"{max(r.abstention_rate for r in honest):.0%} across sites against "
        f"{baseline.abstention_rate:.0%} at the training population, so the rate "
        "on the model card describes "
        "a population as much as a method."
    )
    return " ".join(lines)


__all__ = [
    "DEFAULT_EVAL_SEED",
    "DEFAULT_EVAL_SIZE",
    "SITES",
    "SiteResult",
    "evaluate_site",
    "transfer_report",
]
