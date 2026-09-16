"""A layer-by-layer evaluation of the whole agent.

The scorecard in `training.py` answers one question — is the learned policy
better than the clinician policy that produced the data. That is not an
evaluation of the *agent*, which is six layers of which estimation is one. This
module measures each layer against something it could fail at:

    Layer 1  ingestion fidelity     did the recovered record match what was generated?
    Layer 2  parameter recovery     are the blips right, and are the intervals honest?
    Layer 3  decision quality       regret against the oracle, and is abstention earned?
    Layer 4  safety                 does the gate catch contraindications, and only those?
    Layer 5  explanation            do the explanations decompose the model faithfully?
    Layer 6  governance             are the estimands separated and the gates closed?

Everything here is measured against the generating process, so it is a
simulation result and not clinical evidence. It is a statement about whether the
machinery does what the docstrings claim.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

from treatmentrx.arms import TREATMENT_ARMS, normalize_arm
from treatmentrx.data import DataContractError, DataLayer
from treatmentrx.demo_data import sample_ra_bundle
from treatmentrx.domain import RecommendationStatus
from treatmentrx.estimation import training
from treatmentrx.estimation.basis import BLIP_BASIS, blip_basis
from treatmentrx.estimation.features import model_features, stage_index
from treatmentrx.orchestrator import TreatmentRxOrchestrator
from treatmentrx.simulation.fhir_export import simulated_bundles, trajectory_to_bundle
from treatmentrx.simulation.ra_cohort import (
    TRUE_BLIPS,
    REFERENCE_ARM,
    generate_ra_cohort,
    optimal_arm,
    oracle_action_value,
    oracle_arm,
    oracle_value,
    true_blip,
)

AUDIT_PATIENTS = 120
AUDIT_SEED = 4242


@dataclass
class Section:
    layer: str
    metrics: dict[str, object] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {"layer": self.layer, "metrics": self.metrics, "notes": self.notes}


# ---------------------------------------------------------------- Layer 1

def audit_ingestion(n: int = 40, seed: int = AUDIT_SEED) -> Section:
    """Does Layer 1 recover what the generator wrote into the record?

    **`switch_detection_recall` used to be the headline here and it could not
    fail.** It asked whether any stage was flagged for a patient whose arm
    changed — and `SwitchingCapture`'s third condition *is* "the arm changed", so
    the metric tested the same predicate it used as truth. It read 1.0 because it
    is 1.0 by construction, which is invariant 25's defect wearing an accuracy
    figure. It also used `any()`, so flagging the wrong stage counted as a hit,
    and its denominator excluded every patient who never switched — the only
    patients where a false positive could appear.

    `switched` is the union of four conditions: a recorded discontinuation
    reason, a dispensed name that differs from the order, an arm change, and a
    loss-of-response note. Only the third has ground truth in the simulator. So
    this now reports what is actually knowable — how much of the flag rests on
    the condition that can be checked, and how much on the three that cannot —
    rather than one number that cannot move.

    Measured over 120 trajectories: 277 of 443 stages flagged, **246 of them
    (89%) by the arm-change condition**, 31 by the others. And two conditions are
    dead on this fixture, reported as zeros the way `SwitchingAwareOPE` reports
    its own: no stage has a dispensed name differing from the order, so
    `realized` always echoes `assigned` and the ITT / per-protocol / as-treated
    seam has no input (`FHIRAdapter._supply_records` says the same); and
    `adherence` takes exactly one distinct value, so the days-covered path never
    runs.
    """
    trajectories = generate_ra_cohort(n, seed=seed)
    layer = DataLayer()
    stage_matches = interval_matches = interval_total = 0
    ingested = 0
    flagged = definitional = stages_seen = 0
    realized_differs = 0
    adherence_values: set[float] = set()
    arm_change_stages = arm_change_flagged = 0

    for trajectory in trajectories:
        try:
            state = layer.build_patient_state(trajectory_to_bundle(trajectory))
        except DataContractError:
            continue
        ingested += 1
        # +1 for the open decision point the exporter appends.
        if len(state.stages) == trajectory.n_observed + 1:
            stage_matches += 1
        for generated, recovered in zip(trajectory.stages[1:], state.stages[1:]):
            if generated.interval_days is None or recovered.timing is None:
                continue
            interval_total += 1
            if recovered.timing.time_since_last_treatment == generated.interval_days:
                interval_matches += 1

        for index, stage in enumerate(state.stages):
            switching = stage.switching
            if switching is None:
                continue
            stages_seen += 1
            adherence_values.add(switching.adherence)
            if normalize_arm(switching.realized) != normalize_arm(switching.assigned):
                realized_differs += 1
            previous = state.stages[index - 1] if index else None
            arm_changed = previous is not None and normalize_arm(
                previous.treatment
            ) != normalize_arm(stage.treatment)
            if arm_changed:
                arm_change_stages += 1
                arm_change_flagged += bool(switching.switched)
            if switching.switched:
                flagged += 1
                definitional += bool(arm_changed)

    section = Section("1 data")
    section.metrics = {
        "patients_ingested": f"{ingested}/{n}",
        "stage_count_exact": _rate(stage_matches, ingested),
        "visit_interval_exact": _rate(interval_matches, interval_total),
        "stages_flagged_switched": f"{flagged}/{stages_seen}",
        # Per *stage*, not per patient, and it is 1.0 by construction: the
        # detector's third condition is this predicate. Kept so a wiring
        # regression is visible, labelled so nobody reads it as accuracy.
        "arm_change_always_flagged": _rate(arm_change_flagged, arm_change_stages),
        # The part of the flag that rests on conditions the simulator cannot
        # check — a free-text discontinuation reason or a loss-of-response note.
        "switches_beyond_arm_change": _rate(flagged - definitional, flagged),
        # Dead seams, stated rather than left to be inferred from a silent field.
        "stages_where_realized_differs": realized_differs,
        "distinct_adherence_values": len(adherence_values),
    }
    section.notes.append(
        "Timing and stage structure are reconstructed from dates and drug names, "
        "not read back from the generator."
    )
    section.notes.append(
        f"`arm_change_always_flagged` is 1.0 by construction and is a wiring "
        f"check, not an accuracy figure: `SwitchingCapture`'s third condition is "
        f"the same predicate this uses as truth. It replaced a "
        f"`switch_detection_recall` that read 1.0 for the same reason without "
        f"saying so. Of {flagged} flagged stages, "
        f"{flagged - definitional} rest on conditions the simulator carries no "
        f"ground truth for."
    )
    if realized_differs == 0:
        section.notes.append(
            "No stage has a dispensed name differing from the order, so "
            "`SwitchingRecord.realized` only echoes `assigned` and the "
            "ITT / per-protocol / as-treated distinction has no input from this "
            "field on this fixture. A bundle carrying MedicationDispense "
            "resources would supply one."
        )
    if len(adherence_values) <= 1:
        section.notes.append(
            f"`adherence` takes {len(adherence_values)} distinct value(s) across "
            f"{stages_seen} stages, so the days-covered path never runs and the "
            "default is the only thing being read."
        )
    return section


# ---------------------------------------------------------------- Layer 2

def audit_estimation() -> Section:
    """Parameter recovery at the terminal stage, plus the held-out scorecard."""
    fit = training.fitted()
    terminal = fit.stage_specific.n_stages - 1
    errors = {}
    for arm in TREATMENT_ARMS:
        if arm == REFERENCE_ARM:
            continue
        estimated = fit.stage_specific.blip_parameters(arm, terminal)
        errors[arm] = round(
            max(abs(estimated[name] - truth) for name, truth in zip(estimated, TRUE_BLIPS[arm])), 4
        )

    section = Section("2 estimation")
    section.metrics = {
        "worst_blip_parameter_error": max(errors.values()),
        "per_arm_worst_error": errors,
        "held_out_ipw_policy_value": {
            name: [score.ipw_policy_value, list(score.value_interval or ())]
            for name, score in fit.scores.items()
        },
        "behaviour_policy_value": next(iter(fit.scores.values())).behaviour_value,
        "improvement_over_behaviour": {
            name: list(score.improvement_interval or ()) for name, score in fit.scores.items()
        },
        "blip_basis_flagged": list(training.basis_specification().get("flagged", [])),
        "blip_basis_max_z": {
            name: entry["max_abs_z"]
            for name, entry in training.basis_specification()["candidates"].items()
        },
        "estimator_ranking_resolved": training.ranking_is_resolved(),
        "selected_estimator": training.best_score().estimator,
        "expected_calibration_error": training.holdout_calibration().expected_calibration_error,
        "calibration_passed": training.holdout_calibration().passed,
    }
    section.notes.append(
        "Recovery is asserted at the terminal stage, the only block whose estimand "
        "is the single-visit blip; earlier stages legitimately absorb delayed effects."
    )
    section.notes.append(
        "Every estimator's gain over the behaviour policy excludes zero, so the "
        "improvement is real. Their values overlap each other, so the *ordering* is "
        "not — `selected_estimator` is then an interpretability tie-break, not a "
        "measurement, and `estimator_ranking_resolved` says which of the two happened."
        if not training.ranking_is_resolved()
        else "The leading estimator's held-out interval clears the runner-up's."
    )
    return section


# ---------------------------------------------------------------- Layer 3

def _regret_block(n: int, hits: int, regrets: list) -> dict:
    """A regret summary that carries its own denominator.

    `n` is not decoration. The previous version reported `oracle_arm_rate` and
    `mean_regret_vs_oracle` as bare numbers, and when `recommended_arm` stopped
    being populated for undecided patients the denominator fell from 120 to 43
    without a word — the metrics went to a perfect 1.0 and 0.0 and read as an
    improvement. Any rate printed here shows what it was computed over.
    """
    return {
        "patients": n,
        "oracle_arm_rate": _rate(hits, n),
        "mean_regret": round(sum(regrets) / len(regrets), 4) if regrets else None,
        "max_regret": round(max(regrets), 4) if regrets else None,
    }


def _abstention_price(argmax: list, worst_in_set: list, worst_overall: list) -> dict:
    """What declining costs, as a function of what the clinician does next.

    Abstention is the agent's defining behaviour and its cost has been asserted
    rather than measured. It is not one number: handing the choice back is cheap
    if the clinician takes the model's own ordering anyway, and expensive if they
    pick the worst thing on the menu. The candidate set is what sits between
    those two, so this is also the retrospective case for it.
    """
    if not argmax:
        return {"declined_patients": 0}

    def summary(values):
        return {
            "mean": round(sum(values) / len(values), 4),
            "max": round(max(values), 4),
        }

    return {
        "declined_patients": len(argmax),
        "clinician_takes_the_models_top_arm": summary(argmax),
        "clinician_takes_the_worst_arm_in_the_candidate_set": summary(worst_in_set),
        "clinician_takes_the_worst_arm_on_the_menu": summary(worst_overall),
    }


def audit_decision(n: int = AUDIT_PATIENTS, seed: int = AUDIT_SEED) -> Section:
    """Regret against the oracle, and whether abstention is earned.

    **Which oracle.** Regret used to be measured against `optimal_arm` — the
    per-visit blip argmax — which is not the optimal policy for a sequential
    problem with a delayed toxicity cost and differential dropout. Measured over
    4000 rollouts it scores 2.134 against 2.149 for the agent's own policy, so
    the agent was being charged regret against a rule it beats, and the companion
    "optimal arm rate" rewarded whichever estimator was most myopic. Both are now
    measured against `oracle_arm`, the backward-induction optimum under the
    generating process (2.155). The myopic agreement rate is still reported,
    under a name that says what it is.

    The equipoise check is the one that matters clinically: when the agent
    declines to separate two arms, the arms should actually be close. An agent
    that abstains at random is useless however well calibrated its Q-values are.
    """
    orchestrator = TreatmentRxOrchestrator()
    horizon = training.fitted().pooled.n_stages
    statuses: dict[str, int] = {}
    gaps: dict[str, list[float]] = {}
    regrets: list[float] = []
    myopic_regrets: list[float] = []
    ranked_regrets: list[float] = []
    declined_argmax: list[float] = []
    declined_worst_in_set: list[float] = []
    declined_worst_overall: list[float] = []
    oracle_hits = 0
    ranked_oracle_hits = 0
    myopic_hits = 0
    scored = 0
    ranked_scored = 0

    for bundle in simulated_bundles(n, seed=seed):
        try:
            recommendation = orchestrator.run(bundle)
        except DataContractError:
            continue
        status = recommendation.status.value
        statuses[status] = statuses.get(status, 0) + 1

        state = DataLayer().build_patient_state(bundle)
        features = model_features(state.stages)
        index = stage_index(state.stages, horizon)
        ranked = sorted(
            TREATMENT_ARMS,
            key=lambda arm: oracle_action_value(features, arm, index),
            reverse=True,
        )
        true_gap = oracle_action_value(features, ranked[0], index) - oracle_action_value(
            features, ranked[1], index
        )
        gaps.setdefault(status, []).append(true_gap)

        # Two questions, two denominators, and conflating them is how this
        # metric came to read as perfection.
        #
        # `recommended_arm` is None unless the agent committed, so scoring only
        # those rows asks "when it commits, is it right?" — and the answer is
        # trivially yes, because it commits only when the gap is large. Measured
        # that way the section reported oracle-arm rate 1.0 and max regret 0.0
        # over 43 of 120 patients, while the same audit had read 0.9083 and
        # 0.0406 when the denominator was all of them. Nothing improved; the
        # question changed underneath the name.
        #
        # `top_scored_arm` is the ranking regardless of commitment, so it keeps
        # the full denominator and answers "how good is the ordering?".
        top = recommendation.top_scored_arm
        if top is not None:
            ranked_regrets.append(
                oracle_value(features, index) - oracle_action_value(features, top, index)
            )
            ranked_scored += 1
            if top == oracle_arm(features, index):
                ranked_oracle_hits += 1
            if top == optimal_arm(features):
                myopic_hits += 1
            myopic_regrets.append(
                max(true_blip(arm, features) for arm in TREATMENT_ARMS)
                - true_blip(top, features)
            )

        # What abstention actually costs depends on what the clinician does with
        # it, so all three are priced rather than one asserted.
        if recommendation.status is RecommendationStatus.EQUIPOISE:
            best = oracle_value(features, index)
            candidates = recommendation.audit_event.get("candidate_arms") or [top]
            declined_argmax.append(best - oracle_action_value(features, top, index))
            declined_worst_in_set.append(
                best - min(oracle_action_value(features, arm, index) for arm in candidates)
            )
            declined_worst_overall.append(
                best - min(oracle_action_value(features, arm, index) for arm in TREATMENT_ARMS)
            )

        if recommendation.recommended_arm is None:
            continue
        scored += 1
        chosen = recommendation.recommended_arm
        regrets.append(
            oracle_value(features, index) - oracle_action_value(features, chosen, index)
        )
        if chosen == oracle_arm(features, index):
            oracle_hits += 1

    section = Section("3 decision")
    section.metrics = {
        "status_distribution": statuses,
        # Named for the denominator, so neither can be read as the other.
        "when_it_commits": _regret_block(scored, oracle_hits, regrets),
        "if_forced_to_commit": _regret_block(
            ranked_scored, ranked_oracle_hits, ranked_regrets
        ),
        "myopic_oracle_agreement_rate": _rate(myopic_hits, ranked_scored),
        "mean_single_visit_blip_regret": (
            round(sum(myopic_regrets) / len(myopic_regrets), 4) if myopic_regrets else None
        ),
        "abstention_price": _abstention_price(
            declined_argmax, declined_worst_in_set, declined_worst_overall
        ),
        "mean_true_gap_by_status": {
            status: round(sum(values) / len(values), 4) for status, values in sorted(gaps.items())
        },
    }
    recommend_gap = section.metrics["mean_true_gap_by_status"].get("recommend")
    equipoise_gap = section.metrics["mean_true_gap_by_status"].get("equipoise")
    if recommend_gap is not None and equipoise_gap is not None:
        section.metrics["abstention_is_earned"] = equipoise_gap < recommend_gap
        section.notes.append(
            f"Patients the agent declined to separate had a true top-two gap of "
            f"{equipoise_gap:.4f} against {recommend_gap:.4f} for those it recommended."
        )
    forced = section.metrics["if_forced_to_commit"]
    committed = section.metrics["when_it_commits"]
    section.notes.append(
        f"Two denominators. Over the {committed['patients']} patients it committed "
        f"to, the agent picks the oracle arm {committed['oracle_arm_rate']:.0%} of "
        f"the time — trivially high, because it commits only when the gap is large. "
        f"Over all {forced['patients']}, its top-scored arm is the oracle arm "
        f"{forced['oracle_arm_rate']:.0%} of the time at mean regret "
        f"{forced['mean_regret']}. The second is the ranking; the first is the "
        "ranking after selection, and reporting only the first once made this "
        "section read as a flawless decision layer."
    )
    section.notes.append(
        "Regret is against the backward-induction optimum under the generating "
        "process, which charges for delayed toxicity and lost visits. The myopic "
        "agreement rate is reported for continuity and is not an accuracy score: "
        "the myopic rule is itself worse than the agent's policy."
    )
    return section


# ---------------------------------------------------------------- Layer 4

# Organ-function values swept against each arm. `safe` values sit inside the
# thresholds in `safety/rules.py`; `unsafe` ones sit clearly outside, so a case is
# never scored on a boundary where either answer is defensible.
_ALT_LEVELS = ((25.0, False), (100.0, False), (400.0, True))
_EGFR_LEVELS = ((90.0, False), (45.0, False), (12.0, True), (0.0, True))
_PREGNANCY_LEVELS = ((False, False), (True, True))

# Which arms each condition contraindicates. Derived from the same clinical
# facts `safety/feasible_set.py` encodes, written out independently here so the
# audit is a check rather than a restatement of the implementation.
_RENAL_HEPATIC_ARMS = frozenset({"JAK-inhibitor", "methotrexate-optimization"})
_TERATOGENIC_ARMS = frozenset({"JAK-inhibitor", "methotrexate-optimization"})

# Allergy tokens, with the arms each must remove *at the arm level*.
#
# A class-level allergy removes the whole arm. A drug-level allergy removes only
# that molecule's composites, and the arm survives if another molecule in it
# does — an adalimumab allergy is not an etanercept contraindication, and
# treating it as one would cost the patient a viable option. So the arm-level
# expectation for a drug-level allergy is *empty*, and the real assertion is the
# composite one below: the allergen must never appear in a feasible action.
# Getting this distinction wrong in the audit is how a correct filter gets
# "fixed" into an over-removing one.
_ALLERGY_CASES = (
    ("TNF-inhibitor", frozenset({"TNF-inhibitor"})),
    ("adalimumab", frozenset()),
    ("etanercept", frozenset()),
    ("tocilizumab", frozenset({"IL-6 inhibitor"})),
    ("IL-6 inhibitor", frozenset({"IL-6 inhibitor"})),
    ("upadacitinib", frozenset({"JAK-inhibitor"})),
    ("rituximab", frozenset({"rituximab"})),
    ("methotrexate", frozenset({"methotrexate-optimization"})),
)

# Values that must never remove anything: a blank record, and a measured zero
# that is not clinically extreme for the field it is in.
_INERT_RECORDS = ("", "   ")


def _with_observation(code: str, value):
    bundle = copy.deepcopy(sample_ra_bundle())
    key = "valueBoolean" if isinstance(value, bool) else "valueQuantity"
    payload = value if isinstance(value, bool) else {"value": value}
    bundle["entry"].append(
        {
            "resource": {
                "resourceType": "Observation",
                "code": {"text": code},
                key: payload,
                "effectiveDay": 365,
            }
        }
    )
    return bundle


def _with_allergy(text: str):
    bundle = copy.deepcopy(sample_ra_bundle())
    bundle["entry"].append(
        {"resource": {"resourceType": "AllergyIntolerance", "code": {"text": text}}}
    )
    return bundle


def _contraindication_routing(n: int = 60, seed: int = 606) -> dict[str, int]:
    """How a contraindication is routed, over patients rather than one fixture.

    The sweep above runs the demo patient, for whom the model has a clear
    recommendation, so it cannot reach the branch where a contraindication lands
    on an arm that was never recommended. That branch is the whole of Phase 2 and
    it needs to be visible somewhere a reader looks.

    Four outcomes, and only the third is new:

    * leader feasible          -> the decision layer's status stands
    * recommended, leader out  -> BLOCKED (invariant 2; the only status that stops)
    * undecided, survivors     -> REVIEW with the remaining candidates
    * undecided, no survivors  -> BLOCKED
    """
    from treatmentrx.data import DataContractError, DataLayer
    from treatmentrx.decision import DecisionLayer
    from treatmentrx.estimation import EstimationLayer
    from treatmentrx.safety import SafetyLayer
    from treatmentrx.simulation.fhir_export import simulated_bundles

    data, estimation, decision, safety = (
        DataLayer(), EstimationLayer(), DecisionLayer(), SafetyLayer()
    )
    counts = {
        "leader_feasible": 0,
        "recommended_arm_infeasible_blocked": 0,
        "undecided_contraindication_reviewed": 0,
        "undecided_contraindication_blocked": 0,
    }
    for bundle in simulated_bundles(n, seed=seed):
        try:
            state = data.build_patient_state(_with_pregnancy(bundle))
        except DataContractError:
            continue
        made = decision.decide(state, estimation.estimate(state))
        safe = safety.apply(made, state)
        feasible = set(safe.feasible_arms)
        if made.recommended_arm in feasible:
            counts["leader_feasible"] += 1
        elif made.status is not RecommendationStatus.EQUIPOISE:
            counts["recommended_arm_infeasible_blocked"] += 1
        elif any(arm in feasible for arm in made.candidate_arms):
            counts["undecided_contraindication_reviewed"] += 1
        else:
            counts["undecided_contraindication_blocked"] += 1
    return counts


def _with_pregnancy(bundle: dict) -> dict:
    """Attach the contraindication inside the patient's final stage.

    A fixed day does not work on the simulated cohort — visits are irregular, so
    it lands inside only some final stages, and anything after the decision point
    is dropped by the temporal firewall. Getting this wrong is silent: the
    contraindication simply never reaches the model.
    """
    from treatmentrx.data import DataLayer

    day = DataLayer().build_patient_state(bundle).stages[-1].start_day
    bundle = copy.deepcopy(bundle)
    bundle["entry"].append(
        {
            "resource": {
                "resourceType": "Observation",
                "code": {"text": "pregnant"},
                "valueBoolean": True,
                "effectiveDay": day,
            }
        }
    )
    return bundle


def audit_safety() -> Section:
    """Does the gate remove what it should, and leave the rest alone?

    Reported as recall *and* precision rather than one accuracy number: the two
    failure modes are not interchangeable. Missing a contraindication can harm a
    patient; removing a safe arm costs them an option and, because the layer
    refuses to substitute, can block the recommendation outright.

    The sweep replaced three hand-picked scenarios. `contraindication_recall: 1.0`
    over six expected removals reads like a validated recall estimate and was
    nothing of the kind — it could not distinguish a filter that works from one
    that removes the two arms in question unconditionally. Every case here is a
    labelled (record, arm) pair, and the safe levels are what give the count a
    denominator: a filter that removes everything now scores precision 0.
    """
    orchestrator = TreatmentRxOrchestrator()
    # (label, bundle, arms that must be removed, token that must not survive)
    cases: list[tuple[str, dict, frozenset, str | None]] = []

    for level, unsafe in _ALT_LEVELS:
        cases.append((f"ALT={level:g}", _with_observation("ALT", level),
                      _RENAL_HEPATIC_ARMS if unsafe else frozenset(), None))
    for level, unsafe in _EGFR_LEVELS:
        cases.append((f"eGFR={level:g}", _with_observation("eGFR", level),
                      _RENAL_HEPATIC_ARMS if unsafe else frozenset(), None))
    for level, unsafe in _PREGNANCY_LEVELS:
        cases.append((f"pregnant={level}", _with_observation("pregnant", level),
                      _TERATOGENIC_ARMS if unsafe else frozenset(), None))
    for token, arms in _ALLERGY_CASES:
        cases.append((f"allergy={token!r}", _with_allergy(token), arms, token))
    for blank in _INERT_RECORDS:
        cases.append((f"allergy={blank!r} (blank record)", _with_allergy(blank), frozenset(), None))
    cases.append(("baseline (healthy)", sample_ra_bundle(), frozenset(), None))

    caught = expected = removed_total = spurious = 0
    misses: dict[str, list[str]] = {}
    over_removals: dict[str, list[str]] = {}
    surviving_allergens: dict[str, list[str]] = {}

    for label, bundle, should_remove, token in cases:
        try:
            recommendation = orchestrator.run(bundle)
        except DataContractError:
            over_removals[label] = ["record rejected by the data contract"]
            continue
        removed = set(recommendation.audit_event["removed_arms"])
        caught += len(removed & should_remove)
        expected += len(should_remove)
        removed_total += len(removed)
        spurious += len(removed - should_remove)
        if should_remove - removed:
            misses[label] = sorted(should_remove - removed)
        if removed - should_remove:
            over_removals[label] = sorted(removed - should_remove)
        # The arm-level check cannot see this: an arm survives on one molecule
        # while another of its composites is the one the patient reacts to.
        if token:
            leaked = [
                action
                for action in _feasible_actions(bundle)
                if token.lower() in action.lower()
            ]
            if leaked:
                surviving_allergens[label] = leaked

    section = Section("4 safety")
    section.metrics = {
        "cases": len(cases),
        "labelled_removals_expected": expected,
        "contraindication_recall": _rate(caught, expected),
        "removal_precision": _rate(caught, removed_total),
        "allergen_composites_surviving": surviving_allergens,
        "missed": misses,
        "over_removed": over_removals,
        "spurious_removals": spurious,
"contraindication_routing": _contraindication_routing(),
        "healthy_patient_removals": len(
            orchestrator.run(sample_ra_bundle()).audit_event["removed_arms"]
        ),
    }
    section.notes.append(
        f"{len(cases)} labelled cases spanning organ function, pregnancy, drug- and "
        f"class-level allergies, and records that must remove nothing. Recall and "
        f"precision are reported separately because a filter that removes every arm "
        f"scores perfect recall."
    )
    section.notes.append(
        "A drug-level allergy is expected to remove the molecule's composites and "
        "leave the arm, which survives on another molecule; the arm-level count "
        "would call that a miss, so the allergen is also checked against the "
        "surviving composite set, where it must not appear at all."
    )
    return section


def _feasible_actions(bundle) -> list[str]:
    """Composite-level survivors, which the recommendation does not carry."""
    from treatmentrx.decision import DecisionLayer
    from treatmentrx.estimation import EstimationLayer
    from treatmentrx.safety import SafetyLayer

    state = DataLayer().build_patient_state(bundle)
    decision = DecisionLayer().decide(state, EstimationLayer().estimate(state))
    return SafetyLayer().apply(decision, state).feasible_actions


# ---------------------------------------------------------------- Layer 5

def audit_explanation(n: int = 30, seed: int = AUDIT_SEED) -> Section:
    """Do the explanations decompose the model, and does memory stay out?

    Every metric here used to sit at its ceiling — 1.0, 0.0, 0 — and two of the
    three could not have done anything else. That is invariant 49's defect, in
    the layer whose output a clinician reads.

    *The attribution check tested its own arithmetic.* `_attributions` sets
    `total = round(sum(contributions.values()), 4)` from contributions that are
    themselves rounded to 4dp, so `sum(parts) == total` is an identity: measured,
    the residual is **5.6e-17**, which is float noise and not evidence. It is
    kept — a rounding change would break it and that is worth catching — but it
    is labelled a wiring check, and the question it was standing in for is now
    asked separately: does the decomposition match the *fitted model* it claims
    to come from? That is a real round-trip, and it discriminates: against the
    serving member it names, the residual is **1.6e-4** (the 4dp rounding);
    against the other serving member it is **0.041**.

    *The PHI check was measured where the channel it guards does not run.* It
    scans the card for the patient hash, and the card carries patient free text
    in exactly two places, both fed from episodic memory: `preference` renders
    under `Recorded patient preferences:`, and `outcome_summary` is interpolated
    into `Continuity:`. Memory is empty for the fresh simulated patients this
    loop scores, so **0 of 30** cards could hold what the guard looked for. That
    zero is measured and reported rather than left implicit — invariant 49
    reports its dead seams the same way — and the guard is additionally scored on
    a card built to carry *both* routes, so a route that stops rendering shows up
    as a falling denominator rather than as a quietly narrower scan.

    *What does not reach the card is `history_summary`*, which is the obvious
    suspect and worth naming as a negative. Layer 1 builds it from the stage
    list, it reaches Layer 5 through `provenance`, and the only thing that reads
    it is the retrieval query. The card renders `care_goal` and
    `top_tailoring_vars` from the patient context and nothing else. It *is*
    returned in the served provenance block, which is a different question from
    this one: that block is addressed to the caller who supplied the record.
    """
    orchestrator = TreatmentRxOrchestrator()
    faithful = checked = 0
    worst_residual = 0.0
    matched_source = source_checked = 0
    worst_source_residual = 0.0
    sources: set[str] = set()
    leaked = scanned = carrying = 0

    for bundle in simulated_bundles(n, seed=seed):
        try:
            recommendation = orchestrator.run(bundle)
        except DataContractError:
            continue
        if recommendation.explanation is None or not recommendation.explanation.attributions:
            continue
        attribution = recommendation.explanation.attributions[0]
        if not attribution.contributions:
            continue
        checked += 1
        residual = abs(sum(attribution.contributions.values()) - attribution.total_advantage)
        worst_residual = max(worst_residual, residual)
        if residual < 1e-6:
            faithful += 1

        source, model_residual = _attribution_against_its_model(bundle, recommendation)
        if source is not None:
            sources.add(source)
            source_checked += 1
            worst_source_residual = max(worst_source_residual, model_residual)
            # 4dp rounding on every published coefficient sets the floor.
            matched_source += model_residual < 1e-3

        scanned += 1
        # Measured rather than asserted in prose: this is the denominator the
        # guard could actually have failed on, and for a patient with no
        # recorded history it is zero.
        if _episodic_sections(recommendation.clinician_card):
            carrying += 1
        if _identifiers_in_narrative(recommendation):
            leaked += 1

    # Memory must not be able to move a statistical quantity, by assertion.
    first = orchestrator.run(sample_ra_bundle())
    orchestrator.submit_override(first, "JAK-inhibitor", "local protocol prefers JAK")
    second = orchestrator.run(sample_ra_bundle())

    # The PHI scan, on a card that actually carries the narrative channel. The
    # preference is free text a clinician typed, which is the only place on this
    # card an identifier could realistically arrive.
    with_memory, memory_leaked = _narrative_with_memory()

    section = Section("5 agent")
    section.metrics = {
        "attribution_parts_sum_to_total": {
            "rate": _rate(faithful, checked),
            # Significant digits, not decimal places. The residual is float
            # noise at 5.6e-17 and `round(x, 12)` renders that as an exact 0.0,
            # which asserts an identity the arithmetic does not have and hides
            # the one magnitude that shows this is a wiring check.
            "worst_residual": float(f"{worst_residual:.3g}"),
            "note": "wiring check, not an accuracy figure: both sides are rounded to 4dp",
        },
        "attribution_matches_its_source_model": {
            "patients": source_checked,
            "rate": _rate(matched_source, source_checked),
            "worst_residual": round(worst_source_residual, 6),
            "sources": sorted(sources),
        },
        "phi_leaks_into_narrative": {
            # `leaks` is counted over every card scanned, the fixture included,
            # so the denominator has to say so — reporting the cohort loop alone
            # beside a count taken over one more card is invariant 36 inside the
            # fix for it.
            "cards_scanned": scanned + with_memory,
            "cards_carrying_episodic_text": carrying + with_memory,
            "leaks": leaked + memory_leaked,
            "note": (
                f"{carrying} of {scanned} cohort cards carry an episodic section: "
                "the channel is empty for a patient with no recorded history, so "
                f"the guard is also scored on {with_memory} card built to carry it."
            ),
        },
        "memory_changed_recommendation": second.recommended_arm != first.recommended_arm,
        "memory_changed_q_values": second.q_values != first.q_values,
        "memory_changed_narrative": second.clinician_card != first.clinician_card,
    }
    section.notes.append(
        "Memory is expected to change the narrative and nothing else; that asymmetry "
        "is the whole boundary."
    )
    section.notes.append(
        "`attribution_matches_its_source_model` is the faithfulness figure. psi cannot "
        "be averaged across members that parameterise it differently, so the card "
        "decomposes one member's blip and names it; `sources` is which. On the "
        "deployed fit that choice turns on a BMA weight margin of 0.003 while the two "
        "members' blips differ by up to 0.041 for the same patient."
    )
    return section


def _attribution_against_its_model(bundle, recommendation) -> tuple[str | None, float]:
    """Recompute psi . h(X) from the fitted model the estimate names.

    Independent of the coefficients the estimate carries, which is the point: the
    identity check reads those back to themselves.

    `Q-Pooled`'s psi is a value-to-go stage parameter and the attribution is
    published per remaining visit, so the horizon division is applied here too —
    it is the same division `coefficient_summary` now makes, and recomputing
    without it would measure that rescaling rather than the model. dWOLS's blip
    is single-visit and has no horizon to divide by.
    """
    source = (recommendation.audit_event or {}).get("attribution_source")
    if not source:
        return None, 0.0
    attribution = recommendation.explanation.attributions[0]
    fit = training.fitted()
    state = DataLayer().build_patient_state(bundle)
    features = model_features(state.stages)
    basis = dict(zip(BLIP_BASIS, blip_basis(features)))
    horizon = 1
    if source == training.Q_POOLED:
        index = stage_index(state.stages, fit.pooled.n_stages)
        psi = fit.pooled.blip_parameters(attribution.action, index)
        horizon = fit.pooled.remaining_stages(index)
    elif source == training.DWOLS_SHARED:
        psi = fit.dwols.blip_parameters(attribution.action)
    else:
        return None, 0.0
    value = sum(psi[name] * basis[name] for name in BLIP_BASIS if name in psi) / horizon
    return source, abs(value - attribution.total_advantage)


def _identifiers_in_narrative(recommendation) -> bool:
    rendered = f"{recommendation.clinician_card}{recommendation.patient_summary}"
    return "sim-" in rendered or recommendation.patient_hash in rendered


#: The card sections fed from episodic memory. Everything else on the card is a
#: number, an arm name, or fixed prose, so these are the only place a string a
#: human typed about this patient can arrive.
_EPISODIC_SECTIONS = ("Continuity:", "Recorded patient preferences:")

#: One benign string per free-text route, so a route that stops rendering is
#: visible rather than masked by the other. Benign on purpose: the question is
#: whether the *renderer* introduces an identifier, not whether a string handed
#: to it round-trips.
_FIXTURE_PREFERENCE = "prefers an oral route"
_FIXTURE_OUTCOME = "partial response by week 12, no infusion reactions"


def _episodic_sections(card: str) -> list[str]:
    """The card's episodic sections, which is the denominator the guard needs."""
    return [
        section
        for section in card.split("\n\n")
        if section.startswith(_EPISODIC_SECTIONS)
    ]


def _narrative_with_memory() -> tuple[int, int]:
    """Score the PHI guard on a card that carries the sections it guards.

    Returns (cards carrying episodic text, leaks among them).

    Its own orchestrator, deliberately. The caller has already recorded an
    override against this same patient hash, so sharing one would measure that
    memory rather than this item — and clearing up afterwards would delete the
    override the caller's own metrics were computed from.

    The item fills **both** free-text fields that reach the card — `preference`
    renders under `Recorded patient preferences:` and `outcome_summary` is
    interpolated into `Continuity:` — because those are the two routes and a
    fixture exercising one leaves the other unscanned. `override_reason` is
    stored and never rendered, so it is not a route.

    If either stops rendering this returns 0 and the test asserting the guard is
    scored where its channel runs fails, which is the point: the alternative is a
    fixture that silently narrows to half the surface it claims to cover.
    """
    from treatmentrx.agent.memory import EpisodicItem

    orchestrator = TreatmentRxOrchestrator()
    bundle = sample_ra_bundle()
    recommendation = orchestrator.run(bundle)
    orchestrator.agent.memory.record(
        recommendation.patient_hash,
        EpisodicItem(
            stage=1,
            recommended_arm="TNF-inhibitor",
            clinician_action="TNF-inhibitor",
            override_reason=None,
            outcome_summary=_FIXTURE_OUTCOME,
            preference=_FIXTURE_PREFERENCE,
        ),
    )
    rendered = orchestrator.run(bundle)
    card = rendered.clinician_card
    if not _episodic_sections(card):
        return 0, 0
    if _FIXTURE_PREFERENCE not in card or _FIXTURE_OUTCOME not in card:
        return 0, 0
    return 1, int(_identifiers_in_narrative(rendered))


# ---------------------------------------------------------------- Layer 6

def audit_governance(n: int = 60, seed: int = AUDIT_SEED) -> Section:
    """Are the estimands model-level, do the three tracks stay apart, and are the
    gates shut?

    **The section measured two patients and never looked at what the layer is
    for.** `estimands_are_model_level` compared one patient against one other, so
    a patient-dependent estimand had to disagree on exactly that pair to be
    caught — invariant 36's denominator problem at n=2. It sweeps now, and the
    count is reported beside the verdict.

    What was missing entirely is the separation this layer's own docstring opens
    with. Layer 6 keeps three tracks and the rule is that **an abstention must
    not reach Track B**: there is no policy action to evaluate, and the
    diagnostic top-scored arm must never be smuggled in as though it were a
    recommendation. That is invariant 2's shape one layer down — a name appearing
    where nothing was recommended — and the agent abstains on most patients, so
    the denominator is large and the property can genuinely fail. Nothing checked
    it. Measured over 120 patients: 120 observational rows, 33 on the OPE track
    against 33 recommendations, and 87 abstentions all carrying
    `clinician-usual-care`, none of them the top-scored arm.

    The rung is reported with its **blockers** now. `validation_rung: silent`
    beside `retraining_allowed: false` told a reader the gate was shut and
    nothing about why, while `ValidationStatus.blockers` sat populated and
    unread — the defect invariant 41 fixed in `PolicyScore.notes`, in the section
    whose subject is the gate.

    Three metrics here are regression tripwires rather than measurements, and are
    grouped and labelled as such. Each guards a specific defect that was fixed
    and could silently return; none of them has a denominator, because a
    structural assertion does not have one.
    """
    orchestrator = TreatmentRxOrchestrator()
    estimand_sets: dict[tuple, int] = {}
    values: dict[str, float] = {}
    scored = coincident = 0
    top_scored: dict[str, str] = {}
    published: dict[str, str | None] = {}
    statuses: dict[str, int] = {}
    # The last patient of the sweep, not a fresh run. Scoring one more here would
    # append a row to every track and report it against a denominator of `n` —
    # which is the defect this section was rewritten to remove.
    latest = None

    for bundle in simulated_bundles(n, seed=seed):
        try:
            recommendation = orchestrator.run(bundle)
        except DataContractError:
            continue
        scored += 1
        latest = recommendation
        statuses[recommendation.status.value] = statuses.get(recommendation.status.value, 0) + 1
        top_scored[recommendation.patient_hash] = recommendation.top_scored_arm
        published[recommendation.patient_hash] = recommendation.recommended_arm
        values = {result.estimand: result.policy_value for result in recommendation.estimands}
        key = tuple(sorted(values.items()))
        estimand_sets[key] = estimand_sets.get(key, 0) + 1
        if len(set(values.values())) != len(values):
            coincident += 1

    feedback = orchestrator.feedback
    recommended = statuses.get(RecommendationStatus.RECOMMEND.value, 0)
    abstained = scored - recommended

    # The rule: no abstention on Track B, and no abstention carrying the
    # diagnostic top-scored arm as though it were a policy action.
    abstentions_on_ope = sum(
        1 for row in feedback.ope_track
        if published.get(row["patient_hash"]) != row["policy_arm"]
    )
    leaked_top_arm = sum(
        1 for row in feedback.full_system_track
        if row["agent_status"] != RecommendationStatus.RECOMMEND.value
        and row["policy_action"] == top_scored.get(row["patient_hash"])
    )

    validation = latest.validation if latest else None

    section = Section("6 feedback")
    section.metrics = {
        "estimands": values,
        "estimands_are_model_level": {
            "patients": scored,
            "distinct_value_sets": len(estimand_sets),
            "model_level": len(estimand_sets) == 1,
        },
        "track_separation": {
            "patients": scored,
            "recommendations": recommended,
            "abstentions": abstained,
            "observational_rows": len(feedback.observational_track),
            "ope_rows": len(feedback.ope_track),
            "full_system_rows": len(feedback.full_system_track),
            "ope_rows_not_the_published_arm": abstentions_on_ope,
            "abstentions_carrying_the_top_scored_arm": leaked_top_arm,
        },
        "validation_rung": validation.rung.value if validation else None,
        "validation_gate_passed": validation.gate_passed if validation else None,
        "validation_blockers": list(validation.blockers) if validation else [],
        "retraining_allowed": latest.provenance["feedback"]["retraining_allowed"],
        "regression_tripwires": {
            "estimands_are_distinct": coincident == 0,
            "ope_is_patient_level": (
                latest.audit_event["ope"]["observed_mean_outcome"]
                != latest.audit_event["ope"]["model_policy_value"]
            ),
            # The patient-level block used to carry an `iptw_policy_value` built
            # from hand-set constants. It is gone; what is left is descriptive.
            "ope_is_descriptive_only": "observed_mean_outcome" in latest.audit_event["ope"]
            and "iptw_policy_value" not in latest.audit_event["ope"],
            "note": (
                "structural assertions, not measurements: each guards a defect "
                "that was fixed and could silently return, and none has a "
                "denominator because a structural assertion does not have one"
            ),
        },
    }
    section.notes.append(
        "Estimands describe the policy and must not vary by patient; the OPE summarises "
        "the patient and must not be averaged into them."
    )
    section.notes.append(
        "Track B is the off-policy evaluation cohort and takes recommendations only. "
        "An abstention is a referral to clinician-led usual care, not the top-scored "
        "arm — `abstentions_carrying_the_top_scored_arm` is the count that would show "
        "the diagnostic argmax being promoted into a policy action, and it is the "
        "abstentions that form its denominator."
    )
    return section


# ---------------------------------------------------------------- runtime

def audit_runtime(repeats: int = 30) -> Section:
    orchestrator = TreatmentRxOrchestrator()
    training.fitted()  # exclude the fit from the warm measurement
    start = time.time()
    for _ in range(repeats):
        orchestrator.run(sample_ra_bundle())
    warm = (time.time() - start) / repeats

    first = orchestrator.run(sample_ra_bundle()).audit_event
    second = orchestrator.run(sample_ra_bundle()).audit_event
    first.pop("timestamp"), second.pop("timestamp")

    section = Section("cross-cutting")
    section.metrics = {
        "warm_inference_ms": round(warm * 1000, 2),
        "deterministic_across_runs": first == second,
    }
    return section


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def full_audit() -> dict[str, object]:
    """Every layer, in order."""
    sections = [
        audit_ingestion(),
        audit_estimation(),
        audit_decision(),
        audit_safety(),
        audit_explanation(),
        audit_governance(),
        audit_runtime(),
    ]
    return {
        "sections": [section.as_dict() for section in sections],
        "note": (
            "Measured against the generating process. This says whether the machinery "
            "does what it claims, not whether the recommendations are clinically correct."
        ),
    }


__all__ = [
    "Section",
    "audit_decision",
    "audit_estimation",
    "audit_explanation",
    "audit_governance",
    "audit_ingestion",
    "audit_runtime",
    "audit_safety",
    "full_audit",
]
