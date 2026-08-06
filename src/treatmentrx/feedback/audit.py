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

from treatmentrx.arms import TREATMENT_ARMS
from treatmentrx.data import DataContractError, DataLayer
from treatmentrx.demo_data import sample_ra_bundle
from treatmentrx.domain import RecommendationStatus
from treatmentrx.estimation import training
from treatmentrx.estimation.features import model_features
from treatmentrx.orchestrator import TreatmentRxOrchestrator
from treatmentrx.simulation.fhir_export import simulated_bundles, trajectory_to_bundle
from treatmentrx.simulation.ra_cohort import (
    TRUE_BLIPS,
    REFERENCE_ARM,
    generate_ra_cohort,
    optimal_arm,
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
    """Does Layer 1 recover what the generator wrote into the record?"""
    trajectories = generate_ra_cohort(n, seed=seed)
    layer = DataLayer()
    stage_matches = interval_matches = interval_total = 0
    switch_recall = switch_total = 0
    ingested = 0

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
        arms = [stage.arm for stage in trajectory.stages]
        if len(set(arms)) > 1:
            switch_total += 1
            if any(stage.switching and stage.switching.switched for stage in state.stages):
                switch_recall += 1

    section = Section("1 data")
    section.metrics = {
        "patients_ingested": f"{ingested}/{n}",
        "stage_count_exact": _rate(stage_matches, ingested),
        "visit_interval_exact": _rate(interval_matches, interval_total),
        "switch_detection_recall": _rate(switch_recall, switch_total),
    }
    section.notes.append(
        "Timing and stage structure are reconstructed from dates and drug names, "
        "not read back from the generator."
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
            name: score.ipw_policy_value for name, score in fit.scores.items()
        },
        "behaviour_policy_value": next(iter(fit.scores.values())).behaviour_value,
        "expected_calibration_error": training.holdout_calibration().expected_calibration_error,
        "calibration_passed": training.holdout_calibration().passed,
    }
    section.notes.append(
        "Recovery is asserted at the terminal stage, the only block whose estimand "
        "is the single-visit blip; earlier stages legitimately absorb delayed effects."
    )
    return section


# ---------------------------------------------------------------- Layer 3

def audit_decision(n: int = AUDIT_PATIENTS, seed: int = AUDIT_SEED) -> Section:
    """Regret against the oracle, and whether abstention is earned.

    The equipoise check is the one that matters clinically: when the agent
    declines to separate two arms, the arms should actually be close. An agent
    that abstains at random is useless however well calibrated its Q-values are.
    """
    orchestrator = TreatmentRxOrchestrator()
    statuses: dict[str, int] = {}
    gaps: dict[str, list[float]] = {}
    regrets: list[float] = []
    optimal_hits = 0
    scored = 0

    for bundle in simulated_bundles(n, seed=seed):
        try:
            recommendation = orchestrator.run(bundle)
        except DataContractError:
            continue
        status = recommendation.status.value
        statuses[status] = statuses.get(status, 0) + 1

        state = DataLayer().build_patient_state(bundle)
        features = model_features(state.stages)
        ranked = sorted(TREATMENT_ARMS, key=lambda arm: true_blip(arm, features), reverse=True)
        true_gap = true_blip(ranked[0], features) - true_blip(ranked[1], features)
        gaps.setdefault(status, []).append(true_gap)

        if recommendation.recommended_arm is None:
            continue
        scored += 1
        chosen = true_blip(recommendation.recommended_arm, features)
        regrets.append(true_blip(ranked[0], features) - chosen)
        if recommendation.recommended_arm == optimal_arm(features):
            optimal_hits += 1

    section = Section("3 decision")
    section.metrics = {
        "status_distribution": statuses,
        "optimal_arm_rate": _rate(optimal_hits, scored),
        "mean_regret_vs_oracle": round(sum(regrets) / len(regrets), 4) if regrets else None,
        "max_regret": round(max(regrets), 4) if regrets else None,
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
    return section


# ---------------------------------------------------------------- Layer 4

_CONTRAINDICATIONS = [
    ("pregnant", True, {"JAK-inhibitor", "methotrexate-optimization"}),
    ("ALT", 400, {"JAK-inhibitor", "methotrexate-optimization"}),
    ("eGFR", 12, {"JAK-inhibitor", "methotrexate-optimization"}),
]


def audit_safety() -> Section:
    """Does the gate remove what it should, and leave the rest alone?

    Reported as recall and false-removal rate rather than one accuracy number:
    the two failure modes are not interchangeable. Missing a contraindication
    can harm a patient; removing a safe arm costs them an option and, because
    the layer refuses to substitute, can block the recommendation outright.
    """
    orchestrator = TreatmentRxOrchestrator()
    caught = expected = spurious = 0
    detail = {}

    for code, value, should_remove in _CONTRAINDICATIONS:
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
        removed = set(orchestrator.run(bundle).audit_event["removed_arms"])
        caught += len(removed & should_remove)
        expected += len(should_remove)
        spurious += len(removed - should_remove)
        detail[f"{code}={value}"] = {
            "removed": sorted(removed),
            "missed": sorted(should_remove - removed),
            "unexpected": sorted(removed - should_remove),
        }

    baseline_removed = orchestrator.run(sample_ra_bundle()).audit_event["removed_arms"]
    allergy_bundle = copy.deepcopy(sample_ra_bundle())
    allergy_bundle["entry"].append(
        {"resource": {"resourceType": "AllergyIntolerance", "code": {"text": "TNF-inhibitor"}}}
    )
    class_allergy = set(orchestrator.run(allergy_bundle).audit_event["removed_arms"])

    section = Section("4 safety")
    section.metrics = {
        "contraindication_recall": _rate(caught, expected),
        "spurious_removals": spurious,
        "healthy_patient_removals": len(baseline_removed),
        "class_level_allergy_removed": "TNF-inhibitor" in class_allergy,
        "detail": detail,
    }
    return section


# ---------------------------------------------------------------- Layer 5

def audit_explanation(n: int = 30, seed: int = AUDIT_SEED) -> Section:
    """Do the explanations decompose the model, and does memory stay out?"""
    orchestrator = TreatmentRxOrchestrator()
    faithful = checked = 0
    worst_residual = 0.0
    leaked = 0

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
        rendered = f"{recommendation.clinician_card}{recommendation.patient_summary}"
        if "sim-" in rendered or recommendation.patient_hash in rendered:
            leaked += 1

    # Memory must not be able to move a statistical quantity, by assertion.
    first = orchestrator.run(sample_ra_bundle())
    orchestrator.submit_override(first, "JAK-inhibitor", "local protocol prefers JAK")
    second = orchestrator.run(sample_ra_bundle())

    section = Section("5 agent")
    section.metrics = {
        "attribution_sums_to_advantage": _rate(faithful, checked),
        "worst_attribution_residual": round(worst_residual, 9),
        "phi_leaks_into_narrative": leaked,
        "memory_changed_recommendation": second.recommended_arm != first.recommended_arm,
        "memory_changed_q_values": second.q_values != first.q_values,
        "memory_changed_narrative": second.clinician_card != first.clinician_card,
    }
    section.notes.append(
        "Memory is expected to change the narrative and nothing else; that asymmetry "
        "is the whole boundary."
    )
    return section


# ---------------------------------------------------------------- Layer 6

def audit_governance() -> Section:
    """Are the estimands separated, and are the deployment gates still closed?"""
    orchestrator = TreatmentRxOrchestrator()
    first = orchestrator.run(sample_ra_bundle())
    other = orchestrator.run(simulated_bundles(1, seed=77)[0])

    values = {result.estimand: result.policy_value for result in first.estimands}
    other_values = {result.estimand: result.policy_value for result in other.estimands}

    section = Section("6 feedback")
    section.metrics = {
        "estimands": values,
        "estimands_are_model_level": values == other_values,
        "estimands_are_distinct": len(set(values.values())) == len(values),
        "validation_rung": first.validation.rung.value if first.validation else None,
        "retraining_allowed": first.provenance["feedback"]["retraining_allowed"],
        "ope_is_patient_level": (
            first.audit_event["ope"]["naive_policy_value"]
            != first.audit_event["ope"]["model_policy_value"]
        ),
    }
    section.notes.append(
        "Estimands describe the policy and must not vary by patient; the OPE summarises "
        "the patient and must not be averaged into them."
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
