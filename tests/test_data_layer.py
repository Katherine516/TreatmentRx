"""Layer 1 — ingestion, stage construction, and the guards that must raise."""

import copy
import dataclasses
import unittest

from treatmentrx.contracts import PatientState
from treatmentrx.arms import normalize_arm
from treatmentrx.data import DataLayer
from treatmentrx.data.contract import RADataContract
from treatmentrx.data.dag import CausalDAGRegistry
from treatmentrx.data.encoders import HandcraftedFeatureEncoder
from treatmentrx.data.fhir import FHIRAdapter
from treatmentrx.data.leakage import LeakageError, LeakageTestSuite, TemporalFirewall
from treatmentrx.data.stages import StageHistoryBuilder
from treatmentrx.arms import TREATMENT_ARMS
from treatmentrx.demo_data import sample_ra_bundle
from treatmentrx.domain import (
    CareGoal,
    Observation,
    PatientRecord,
    RecommendationStatus,
    StageRecord,
)
from treatmentrx.orchestrator import TreatmentRxOrchestrator
from treatmentrx.simulation.fhir_export import simulated_bundles
from treatmentrx.simulation.ra_cohort import generate_ra_cohort


def _patient():
    return FHIRAdapter().parse_bundle(sample_ra_bundle())


def _stages():
    patient = _patient()
    return patient, StageHistoryBuilder().build(patient)


class IngestionTests(unittest.TestCase):
    def test_fhir_adapter_parses_patient_bundle(self):
        patient = _patient()
        self.assertEqual(patient.patient_id, "patient-demo-001")
        self.assertEqual(patient.disease, "Rheumatoid Arthritis")
        self.assertEqual(len(patient.medications), 3)
        self.assertTrue(any(observation.code == "DAS28" for observation in patient.observations))

    def test_patient_id_never_leaves_the_data_layer(self):
        state = DataLayer().build_patient_state(sample_ra_bundle())
        self.assertNotIn("patient-demo-001", state.patient_hash)
        self.assertEqual(len(state.patient_hash), 16)

    def test_data_contract_maps_medications_to_the_shared_vocabulary(self):
        contract = RADataContract()
        report = contract.validate(_patient())
        self.assertTrue(report.passed)
        self.assertEqual(contract.normalize_treatment_arm("adalimumab TNF inhibitor"), "TNF-inhibitor")
        self.assertEqual(contract.normalize_treatment_arm("tocilizumab"), "IL-6 inhibitor")
        self.assertEqual(contract.normalize_treatment_arm("upadacitinib"), "JAK-inhibitor")
        self.assertEqual(contract.normalize_treatment_arm("rituximab"), "rituximab")

    def test_dag_returns_adjustment_set_and_causal_path(self):
        patient, stages = _stages()
        result = CausalDAGRegistry().validate(patient, stages)
        self.assertTrue(result.identified)
        self.assertIn("baseline_disease_activity", result.adjustment_set)
        self.assertIn("colliders", result.causal_path_text)

    def test_the_encoder_produces_a_stable_state_vector(self):
        _, stages = _stages()
        encoded = HandcraftedFeatureEncoder().encode(stages)
        self.assertEqual(len(encoded.vector), 32)
        self.assertEqual(encoded.vector, HandcraftedFeatureEncoder().encode(stages).vector)

    def test_everything_the_state_carries_from_the_encoder_is_read(self):
        """The encoder used to be wrapped by a GRU-shaped one whose 256-entry
        vector nothing consumed — its tail held seven distinct values and its
        `feature_map` was a copy of this one's. Both fields here reach
        `PatientState` and are read, which is what makes keeping them honest.
        """
        _, stages = _stages()
        encoded = HandcraftedFeatureEncoder().encode(stages)
        state = DataLayer().build_patient_state(sample_ra_bundle())
        self.assertEqual(state.features, state.encoded_state.vector)
        self.assertEqual(state.feature_names, sorted(state.encoded_state.feature_map))
        self.assertEqual(len(encoded.feature_map), len(state.feature_names))


class ClinicalRealismTests(unittest.TestCase):
    """The v5.1 annotations must survive all the way into the PatientState."""

    def test_every_annotation_reaches_the_contract(self):
        state = DataLayer().build_patient_state(sample_ra_bundle())
        self.assertIsInstance(state, PatientState)
        latest = state.latest
        self.assertIsNotNone(latest.timing, "timing annotation was dropped")
        self.assertIsNotNone(latest.belief, "belief annotation was dropped")
        self.assertIsNotNone(latest.event, "competing-risk annotation was dropped")
        self.assertIsNotNone(latest.switching, "switching annotation was dropped")

    def test_timing_captures_irregular_intervals(self):
        state = DataLayer().build_patient_state(sample_ra_bundle())
        self.assertIsNone(state.stages[0].timing.time_since_last_treatment)
        self.assertIsNotNone(state.stages[1].timing.time_since_last_treatment)
        self.assertGreater(state.latest.timing.inverse_intensity_weight, 0)

    def test_switching_flags_loss_of_response(self):
        state = DataLayer().build_patient_state(sample_ra_bundle())
        self.assertTrue(any(s.switching and s.switching.switched for s in state.stages))

    def test_belief_is_bounded(self):
        belief = DataLayer().build_patient_state(sample_ra_bundle()).latest.belief
        self.assertTrue(0.0 <= belief.activity <= 1.0)
        self.assertTrue(0.0 <= belief.uncertainty <= 0.5)

    def test_care_goal_is_inferred_from_the_trajectory(self):
        state = DataLayer().build_patient_state(sample_ra_bundle())
        self.assertEqual(state.care_goal, CareGoal.INDUCTION)
        self.assertTrue(all(stage.care_goal == state.care_goal for stage in state.stages))

    def test_toxicity_overrides_the_inferred_goal(self):
        """A failing liver is not an induction conversation."""
        patient, stages = _stages()
        toxic = dict(stages[-1].features)
        toxic["alt"] = 180.0
        stages[-1] = StageRecord(**(stages[-1].__dict__ | {"features": toxic}))
        self.assertEqual(DataLayer().infer_care_goal(stages), CareGoal.TOXICITY_CONTROL)


def _future_observation_bundle():
    """The demo record plus an observation dated long after the decision.

    Ingests cleanly as it stands — `_features_until` never admits it — and is
    the record that exposes the firewall once that filter is broken.
    """
    import copy

    bundle = copy.deepcopy(sample_ra_bundle())
    bundle["entry"].append(
        {
            "resource": {
                "resourceType": "Observation",
                "code": {"text": "future marker"},
                "valueQuantity": {"value": 1.0},
                "effectiveDay": 10_000,
            }
        }
    )
    return bundle


class LeakageTests(unittest.TestCase):
    def test_firewall_catches_a_future_feature(self):
        patient, stages = _stages()
        tampered_features = dict(stages[0].features)
        tampered_features["future_marker"] = 1.0
        tampered = StageRecord(**(stages[0].__dict__ | {"features": tampered_features}))
        leaky = PatientRecord(
            patient_id=patient.patient_id,
            disease=patient.disease,
            demographics=patient.demographics,
            conditions=patient.conditions,
            allergies=patient.allergies,
            medications=patient.medications,
            observations=patient.observations
            + [Observation(code="future marker", value=1.0, days_from_baseline=10_000)],
            encounters=patient.encounters,
            outcomes=patient.outcomes,
        )
        self.assertTrue(TemporalFirewall().check(leaky, [tampered]))
        with self.assertRaises(LeakageError):
            TemporalFirewall().assert_clean(leaky, [tampered])

    def test_suite_passes_on_the_clean_demo(self):
        patient, stages = _stages()
        self.assertTrue(LeakageTestSuite().run(patient, stages).passed)

    def test_every_assertion_fires_on_its_own(self):
        """Invariant 25's rule, on the module whose whole claim is that it stops
        things. Each check guards a different upstream guarantee, so each is
        broken separately — and breaking one must not be reported as another.
        """
        from dataclasses import replace

        patient, stages = _stages()
        suite = LeakageTestSuite()
        self.assertTrue(suite.run(patient, stages).passed)

        # Firewall: a stage whose decision day precedes every observation.
        early = [replace(stages[0], start_day=-1)] + list(stages[1:])
        report = suite.run(patient, early)
        self.assertFalse(report.temporal_firewall_passed)
        self.assertTrue(report.immortal_time_passed)
        self.assertTrue(report.timestamp_monotonic_passed)

        # Immortal time: medication starts out of order.
        unsorted_meds = replace(patient, medications=list(reversed(patient.medications)))
        report = suite.run(unsorted_meds, stages)
        self.assertFalse(report.immortal_time_passed)
        self.assertTrue(report.temporal_firewall_passed)

        # Monotonic: stages out of order.
        report = suite.run(patient, list(reversed(stages)))
        self.assertFalse(report.timestamp_monotonic_passed)
        self.assertTrue(report.temporal_firewall_passed)

    def test_any_violation_raises_out_of_the_pipeline(self):
        """It used to be only the firewall's.

        `build_patient_state` read `temporal_firewall_passed` while the suite ran
        four checks, so the rest were computed, appended to `violations`, and
        reduced to a warning-severity diagnostic nothing reads — three of the
        four booleans were written and never read anywhere in the package.

        Broken at the guarantee rather than at the check, because these are
        assertions on properties enforced upstream: unfilter `_features_until`
        and the firewall is what catches it.
        """
        from treatmentrx.data.stages import StageHistoryBuilder

        layer = DataLayer()
        original = StageHistoryBuilder._features_until

        def unfiltered(self, observations, day):
            return {
                self._feature_name(observation.code): observation.value
                for observation in observations
            }

        try:
            StageHistoryBuilder._features_until = unfiltered
            with self.assertRaises(LeakageError) as raised:
                layer.build_patient_state(_future_observation_bundle())
        finally:
            StageHistoryBuilder._features_until = original

        self.assertIn("only available after decision day", str(raised.exception))
        # And the guarantee restored, the same record is served.
        self.assertTrue(layer.build_patient_state(_future_observation_bundle()).stages)

    def test_a_non_firewall_violation_also_raises(self):
        """The discriminating case, and the one the old code let through.

        `build_patient_state` read `temporal_firewall_passed`, so a record that
        tripped only `_timestamp_monotonic` was served with the violation filed
        in a diagnostic. Stage order is guarded by the adapter's medication sort,
        so the guarantee is broken there — the contract checks *medications* for
        chronology and would not see this.
        """
        from treatmentrx.data.stages import StageHistoryBuilder

        layer = DataLayer()
        original = StageHistoryBuilder.build

        def reversed_stages(self, patient):
            return list(reversed(original(self, patient)))

        try:
            StageHistoryBuilder.build = reversed_stages
            with self.assertRaises(LeakageError) as raised:
                layer.build_patient_state(sample_ra_bundle())
        finally:
            StageHistoryBuilder.build = original

        message = str(raised.exception)
        self.assertIn("precedes prior", message)
        self.assertNotIn("only available after decision day", message)

    def test_a_feature_named_outcome_is_history_not_leakage(self):
        """The check that was removed rather than repaired.

        `_outcome_in_features` flagged any feature whose *name* contained
        "outcome". `_features_until` has already excluded everything after the
        decision, so the only thing it could catch is a legitimately recorded
        past outcome. It was also the one check in the module that could fire,
        and firing it changed nothing — the record was served anyway.
        """
        import copy

        bundle = copy.deepcopy(sample_ra_bundle())
        bundle["entry"].append(
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"text": "outcome"},
                    "valueQuantity": {"value": 0.9},
                    "effectiveDay": 0,
                }
            }
        )
        state = DataLayer().build_patient_state(bundle)
        self.assertIn("outcome", state.stages[-1].features)
        self.assertTrue(state.diagnostics)

    def test_the_report_names_which_guarantee_broke(self):
        """Three booleans, because the repairs are different. A single `passed`
        would say a record leaked without saying how."""
        patient, stages = _stages()
        report = LeakageTestSuite().run(patient, stages)
        self.assertEqual(
            sorted(f.name for f in dataclasses.fields(report)),
            [
                "immortal_time_passed",
                "temporal_firewall_passed",
                "timestamp_monotonic_passed",
                "violations",
            ],
        )

    def test_a_leaking_record_cannot_produce_a_patient_state(self):
        """The firewall raises when asserted directly, which is the seam a
        cohort builder would use outside the pipeline."""
        bundle = sample_ra_bundle()
        bundle["entry"].append(
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"text": "future marker"},
                    "valueQuantity": {"value": 1.0},
                    "effectiveDay": 10_000,
                }
            }
        )
        layer = DataLayer()
        patient = FHIRAdapter().parse_bundle(bundle)
        stages = layer.stage_builder.build(patient)
        leaked = dict(stages[0].features)
        leaked["future_marker"] = 1.0
        stages[0] = StageRecord(**(stages[0].__dict__ | {"features": leaked}))
        with self.assertRaises(LeakageError):
            TemporalFirewall().assert_clean(patient, stages)


class SimulatedIngestionTests(unittest.TestCase):
    """Layer 1 against patients whose truth is known by construction.

    Until the cohort could be exported as bundles, every ingestion module was
    tested on one hand-written demo patient. These push simulated trajectories —
    whose stage count, visit intervals, arm sequence and terminal event were
    *generated* — through the real DataLayer and check what comes back.
    """

    @classmethod
    def setUpClass(cls):
        cls.trajectories = generate_ra_cohort(20, seed=991)
        cls.bundles = simulated_bundles(20, seed=991)
        layer = DataLayer()
        cls.states = [layer.build_patient_state(bundle) for bundle in cls.bundles]

    def test_every_simulated_patient_ingests(self):
        self.assertEqual(len(self.states), 20)
        self.assertTrue(all(state.stages for state in self.states))

    def test_stage_count_matches_what_was_generated(self):
        """Plus one: the open decision point the agent is being asked about."""
        for trajectory, state in zip(self.trajectories, self.states):
            self.assertEqual(
                len(state.stages),
                trajectory.n_observed + 1,
                msg=f"patient {trajectory.patient_index}",
            )

    def test_recovered_intervals_match_the_generated_ones(self):
        """The timing model has to reconstruct irregular spacing from dates."""
        checked = 0
        for trajectory, state in zip(self.trajectories, self.states):
            for generated, recovered in zip(trajectory.stages[1:], state.stages[1:]):
                if generated.interval_days is None or recovered.timing is None:
                    continue
                self.assertEqual(
                    recovered.timing.time_since_last_treatment,
                    generated.interval_days,
                    msg=f"patient {trajectory.patient_index} stage {generated.stage}",
                )
                checked += 1
        self.assertGreater(checked, 10, "no intervals were actually compared")

    def test_switching_is_detected_where_the_arm_changed(self):
        for trajectory, state in zip(self.trajectories, self.states):
            arms = [stage.arm for stage in trajectory.stages]
            if len(set(arms)) <= 1:
                continue
            self.assertTrue(
                any(stage.switching is not None for stage in state.stages),
                msg=f"patient {trajectory.patient_index} changed arm but no switching was captured",
            )

    def test_dropouts_carry_a_discontinuation_reason(self):
        censored = [
            (t, b) for t, b in zip(self.trajectories, self.bundles) if t.censored
        ]
        self.assertTrue(censored, "the seed produced no dropouts to check")
        for trajectory, bundle in censored:
            reasons = [
                entry["resource"].get("discontinuationReason")
                for entry in bundle["entry"]
                if entry["resource"].get("resourceType") == "MedicationRequest"
            ]
            self.assertTrue(
                any(reasons), msg=f"patient {trajectory.patient_index} left with no reason recorded"
            )

    def test_the_whole_pipeline_runs_on_every_simulated_patient(self):
        """One demo patient exercises one path; twenty exercise the branches."""
        recommendations = [TreatmentRxOrchestrator().run(bundle) for bundle in self.bundles]
        self.assertEqual(len(recommendations), 20)
        statuses = {recommendation.status for recommendation in recommendations}
        self.assertGreater(len(statuses), 1, "every patient took the same path")
        for recommendation in recommendations:
            if recommendation.status is RecommendationStatus.RECOMMEND:
                self.assertIn(recommendation.recommended_arm, TREATMENT_ARMS)
            else:
                self.assertIsNone(recommendation.recommended_arm)

    def test_no_simulated_patient_trips_the_leakage_guard(self):
        """The exporter must not leak the future into the record it writes."""
        for bundle in self.bundles:
            DataLayer().build_patient_state(bundle)  # raises LeakageError if it did



class BackdoorCriterionTests(unittest.TestCase):
    """The DAG licenses calling any of this causal, so check it against graphs
    with known answers rather than against the one it ships with.

    M-bias and descendant-of-collider are the two cases implementations get
    wrong, and getting them right is what `path_is_blocked` is claiming.
    """

    @staticmethod
    def _graph(edges):
        from treatmentrx.data.dag import CausalDAG

        return CausalDAG(
            "test", "v", tuple({n for e in edges for n in e}), tuple(edges), (), (), ()
        )

    @staticmethod
    def _brute_force(graph, treatment="T", outcome="Y"):
        """Smallest satisfying subset, found by exhaustion — the reference."""
        import itertools

        from treatmentrx.data.dag import _descendants, satisfies_backdoor

        descendants = _descendants(graph.edges, treatment)
        candidates = sorted(
            node
            for node in graph.nodes
            if node not in descendants and node not in {treatment, outcome}
        )
        for size in range(len(candidates) + 1):
            for combination in itertools.combinations(candidates, size):
                if satisfies_backdoor(graph, set(combination), treatment, outcome)[0]:
                    return set(combination)
        return None

    def test_a_plain_confounder_must_be_adjusted(self):
        from treatmentrx.data.dag import minimal_backdoor_set, satisfies_backdoor

        graph = self._graph([("Z", "T"), ("Z", "Y")])
        self.assertFalse(satisfies_backdoor(graph, set(), "T", "Y")[0])
        self.assertEqual(minimal_backdoor_set(graph, "T", "Y"), {"Z"})

    def test_a_mediator_is_a_descendant_and_is_refused(self):
        """Adjusting a descendant of treatment is the first backdoor condition."""
        from treatmentrx.data.dag import minimal_backdoor_set, satisfies_backdoor

        graph = self._graph([("T", "M"), ("M", "Y")])
        self.assertTrue(satisfies_backdoor(graph, set(), "T", "Y")[0])
        self.assertFalse(satisfies_backdoor(graph, {"M"}, "T", "Y")[0])
        self.assertEqual(minimal_backdoor_set(graph, "T", "Y"), set())

    def test_m_bias_conditioning_opens_a_path_that_was_closed(self):
        """The case that separates a real implementation from a plausible one.

        `T <- U1 -> M <- U2 -> Y` is blocked at the collider M with no
        adjustment at all. Conditioning on M *opens* it — so the empty set is
        valid and `{M}` is not, which is the opposite of the "adjust for
        everything you measured" instinct.
        """
        from treatmentrx.data.dag import minimal_backdoor_set, satisfies_backdoor

        graph = self._graph([("U1", "T"), ("U1", "M"), ("U2", "M"), ("U2", "Y")])
        self.assertTrue(satisfies_backdoor(graph, set(), "T", "Y")[0])
        self.assertFalse(satisfies_backdoor(graph, {"M"}, "T", "Y")[0])
        self.assertEqual(minimal_backdoor_set(graph, "T", "Y"), set())

    def test_a_descendant_of_a_collider_opens_it_too(self):
        """Conditioning on a collider's child is conditioning on the collider."""
        from treatmentrx.data.dag import satisfies_backdoor

        graph = self._graph(
            [("U1", "T"), ("U1", "M"), ("U2", "M"), ("U2", "Y"), ("M", "D")]
        )
        self.assertTrue(satisfies_backdoor(graph, set(), "T", "Y")[0])
        self.assertFalse(satisfies_backdoor(graph, {"D"}, "T", "Y")[0])

    def test_the_result_is_irreducible(self):
        """No proper subset may also satisfy the criterion.

        The previous implementation walked candidates in sorted order and added
        one whenever the set so far did not yet block, without testing whether
        that node helped — so on this graph, where `{Z}` blocks both paths, it
        returned `{W, Z}`.
        """
        from treatmentrx.data.dag import minimal_backdoor_set, satisfies_backdoor

        graph = self._graph([("Z", "T"), ("Z", "Y"), ("W", "Z"), ("W", "Y")])
        required = minimal_backdoor_set(graph, "T", "Y")
        self.assertEqual(required, {"Z"})
        for node in sorted(required):
            with self.subTest(drop=node):
                self.assertFalse(
                    satisfies_backdoor(graph, required - {node}, "T", "Y")[0],
                    f"{node} is redundant, so the set is not irreducible",
                )

    def test_the_deployed_graph_matches_brute_force(self):
        """The shipped adjustment set is the reference answer, not a coincidence."""
        from treatmentrx.data.dag import CausalDAGRegistry, minimal_backdoor_set

        graph = CausalDAGRegistry()._ra_v1()
        self.assertEqual(
            minimal_backdoor_set(graph, "treatment", "ra_response"),
            self._brute_force(graph, "treatment", "ra_response"),
        )

    def test_the_split_by_basis_covers_the_whole_required_set(self):
        """Every required adjuster lands in exactly one of the two lists.

        `unmodelled_confounders` is what the model card reports as unadjusted, so
        a spurious entry there claims a failure that never happened.
        """
        from treatmentrx.data.dag import CausalDAGRegistry, minimal_backdoor_set

        graph = CausalDAGRegistry()._ra_v1()
        required = minimal_backdoor_set(graph, "treatment", "ra_response")
        self.assertEqual(
            set(graph.adjustment_set) | set(graph.unmodelled_confounders), required
        )
        self.assertFalse(
            set(graph.adjustment_set) & set(graph.unmodelled_confounders)
        )

    def test_the_colliders_are_kept_out_of_the_adjustment_set(self):
        from treatmentrx.data.dag import CausalDAGRegistry

        graph = CausalDAGRegistry()._ra_v1()
        for collider in graph.colliders:
            with self.subTest(collider=collider):
                self.assertNotIn(collider, graph.adjustment_set)
                self.assertNotIn(collider, graph.unmodelled_confounders)

if __name__ == "__main__":
    unittest.main()


class RealizedTreatmentTests(unittest.TestCase):
    """`SwitchingRecord.realized` used to be a copy of `assigned`.

    That made the ITT / per-protocol / as-treated split a distinction with no
    input: every estimand was computed from the same assignment sequence. It now
    comes from `MedicationDispense` / `MedicationAdministration` when the bundle
    carries one.
    """

    def _bundle(self, dispensed=None, days_supply=None, stop_day=180):
        entries = [
            {"resource": {"resourceType": "Patient", "id": "p1"}},
            {"resource": {"resourceType": "Condition", "code": {"text": "Rheumatoid Arthritis"}}},
            {"resource": {"resourceType": "Encounter", "day": 0}},
            {"resource": {"resourceType": "Encounter", "day": stop_day}},
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"text": "DAS28"},
                    "valueQuantity": {"value": 5.2, "unit": "score"},
                    "effectiveDay": 0,
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "medicationCodeableConcept": {"text": "adalimumab"},
                    "authoredOnDay": 0,
                    "stopDay": stop_day,
                    "response": "partial response",
                }
            },
        ]
        if dispensed is not None:
            entries.append(
                {
                    "resource": {
                        "resourceType": "MedicationDispense",
                        "medicationCodeableConcept": {"text": dispensed},
                        "whenHandedOverDay": 5,
                        "daysSupply": days_supply,
                    }
                }
            )
        entries.append(
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "medicationCodeableConcept": {"text": "current decision point"},
                    "authoredOnDay": stop_day,
                }
            }
        )
        return {"resourceType": "Bundle", "entry": entries}

    def _switching(self, **kwargs):
        return DataLayer().build_patient_state(self._bundle(**kwargs)).stages[0].switching

    def test_no_dispense_record_leaves_realized_equal_to_assigned(self):
        """An absent supply chain is a missing measurement, not evidence that
        nothing was supplied — the two must not read the same."""
        switching = self._switching()
        self.assertEqual(switching.realized, switching.assigned)
        self.assertEqual(switching.adherence, 1.0)

    def test_a_cross_arm_substitution_is_a_switch(self):
        switching = self._switching(dispensed="tocilizumab", days_supply=180)
        self.assertNotEqual(
            normalize_arm(switching.realized), normalize_arm(switching.assigned)
        )
        self.assertTrue(switching.switched)
        self.assertIn("dispensed", switching.discontinuation_reason)

    def test_a_within_class_substitution_is_not(self):
        """Etanercept against an adalimumab order is the same arm, and the arm
        is what the model reasons about."""
        switching = self._switching(dispensed="etanercept", days_supply=180)
        self.assertEqual(switching.realized, "etanercept")
        self.assertEqual(
            normalize_arm(switching.realized), normalize_arm(switching.assigned)
        )
        self.assertFalse(switching.switched)

    def test_adherence_comes_from_days_covered_when_it_can(self):
        full = self._switching(dispensed="adalimumab", days_supply=180)
        half = self._switching(dispensed="adalimumab", days_supply=90)
        self.assertEqual(full.adherence, 1.0)
        self.assertEqual(half.adherence, 0.5)

    def test_oversupply_does_not_exceed_full_adherence(self):
        switching = self._switching(dispensed="adalimumab", days_supply=400)
        self.assertEqual(switching.adherence, 1.0)

    def test_a_dispense_without_days_supply_does_not_invent_adherence(self):
        switching = self._switching(dispensed="adalimumab", days_supply=None)
        self.assertEqual(switching.realized, "adalimumab")
        self.assertEqual(switching.adherence, 1.0)


class ConcomitantMedicationTests(unittest.TestCase):
    """A steroid bridge is not a change of treatment line.

    Every `MedicationRequest` used to become a stage, so a prednisone taper
    recorded mid-line produced a phantom decision point: the agent read it as a
    switch to an arm mapping to `manual-review`, renumbered every later stage,
    and truncated the real DMARD line to end at the steroid's start day. Steroid
    bridging is standard RA practice, so this would have hit real data at once.
    """

    def _bundle(self, *extra):
        bundle = copy.deepcopy(sample_ra_bundle())
        for resource in extra:
            bundle["entry"].insert(-1, {"resource": resource})
        return bundle

    def _steroid(self, text="prednisone 20mg taper", start=120, stop=160):
        return {
            "resourceType": "MedicationRequest",
            "medicationCodeableConcept": {"text": text},
            "authoredOnDay": start,
            "stopDay": stop,
        }

    def test_a_steroid_bridge_does_not_become_a_decision_point(self):
        clean = DataLayer().build_patient_state(sample_ra_bundle())
        bridged = DataLayer().build_patient_state(self._bundle(self._steroid()))
        self.assertEqual(len(bridged.stages), len(clean.stages))
        self.assertEqual(
            [(s.treatment, s.start_day, s.end_day) for s in bridged.stages],
            [(s.treatment, s.start_day, s.end_day) for s in clean.stages],
            "the concomitant record changed the line sequence",
        )

    def test_it_flags_rescue_on_the_stage_it_overlaps(self):
        state = DataLayer().build_patient_state(self._bundle(self._steroid(start=120)))
        rescued = [s.stage for s in state.stages if s.switching.rescue_therapy]
        self.assertEqual(rescued, [1], "day 120 falls inside stage 1 (0-240)")

    def test_rescue_is_not_flagged_on_a_stage_it_misses(self):
        state = DataLayer().build_patient_state(self._bundle(self._steroid(start=300, stop=330)))
        rescued = [s.stage for s in state.stages if s.switching.rescue_therapy]
        self.assertEqual(rescued, [2], "day 300 falls inside stage 2 (240-365)")

    def test_a_clean_record_flags_no_rescue(self):
        """The old check searched the arm name for 'steroid', which no canonical
        arm contains, so the flag was permanently False either way."""
        state = DataLayer().build_patient_state(sample_ra_bundle())
        self.assertFalse(any(s.switching.rescue_therapy for s in state.stages))

    def test_an_unrecognised_dmard_still_becomes_a_stage(self):
        """Conservative on purpose: only positively-recognised concomitants are
        dropped. A biologic newer than `ARM_SYNONYMS` must still surface for
        manual review rather than vanish from the sequence.
        """
        from treatmentrx.arms import MANUAL_REVIEW, is_concomitant, normalize_arm

        name = "some-new-biologic-2029"
        self.assertEqual(normalize_arm(name), MANUAL_REVIEW)
        self.assertFalse(is_concomitant(name))
        clean = DataLayer().build_patient_state(sample_ra_bundle())
        state = DataLayer().build_patient_state(
            self._bundle(
                {
                    "resourceType": "MedicationRequest",
                    "medicationCodeableConcept": {"text": name},
                    "authoredOnDay": 300,
                }
            )
        )
        self.assertEqual(len(state.stages), len(clean.stages) + 1)

    def test_a_record_of_only_concomitants_is_rejected_clearly(self):
        from treatmentrx.data.stages import StageHistoryBuilder
        from treatmentrx.domain import PatientRecord, TreatmentEvent

        patient = PatientRecord(
            patient_id="p",
            disease="Rheumatoid Arthritis",
            demographics={},
            conditions=[],
            allergies=[],
            medications=[TreatmentEvent(name="prednisone 10mg", start_day=0)],
            observations=[],
            encounters=[0, 90],
        )
        with self.assertRaises(ValueError) as raised:
            StageHistoryBuilder().build(patient)
        self.assertIn("concomitant", str(raised.exception))


class IdentificationReadsWhatTheModelReadsTests(unittest.TestCase):
    """The certificate and the covariate space were two vocabularies.

    `data/dag.py` opens by saying the old check asked whether the *bundle
    mentioned* an adjuster, and that what matters is whether the estimator
    conditions on the variable. `_has_adjuster` then hand-listed observation
    codes: `baseline_disease_activity` was satisfied by `cdai`, `sdai` or
    `haq_di`, none of which produces a DAS28. A patient with a HAQ-DI and no
    DAS28 was certified **identified** while `estimation.features` handed the
    estimators `FEATURE_DEFAULTS["das28"]` — the exact case that docstring says
    the check exists to catch.

    These assert the agreement against `model_features` rather than against
    `OBSERVED_AS`, because a check scored on the map it uses as truth reads 1.0
    by construction.
    """

    @staticmethod
    def _state(withheld=(), added=()):
        import copy

        from treatmentrx.data import DataLayer

        bundle = copy.deepcopy(sample_ra_bundle())
        drop = {code.lower() for code in withheld}
        bundle["entry"] = [
            entry
            for entry in bundle["entry"]
            if not (
                entry["resource"].get("resourceType") == "Observation"
                and (entry["resource"].get("code", {}).get("text") or "").lower() in drop
            )
        ]
        for code, value in added:
            key = "valueBoolean" if isinstance(value, bool) else "valueString"
            bundle["entry"].append(
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {"text": code},
                        key: value,
                        "effectiveDay": 365,
                    }
                }
            )
        return DataLayer().fhir.parse_bundle(bundle), bundle

    def _validate(self, withheld=(), added=()):
        from treatmentrx.data import DataLayer
        from treatmentrx.data.dag import CausalDAGRegistry

        patient, _ = self._state(withheld, added)
        stages = DataLayer().stage_builder.build(patient)
        return CausalDAGRegistry().validate(patient, stages), stages

    def test_a_family_member_is_not_the_covariate(self):
        """A HAQ-DI satisfies the contract's `disease_activity` family and
        produces no DAS28. The contract is right to grade that a warning; the
        certificate must not read it as the adjuster being present."""
        from treatmentrx.estimation.features import FEATURE_DEFAULTS, model_features

        result, stages = self._validate(withheld=("DAS28",))
        self.assertTrue(
            any("haq" in key for key in stages[-1].features),
            "the point of this record is that a family member survives",
        )
        self.assertEqual(model_features(stages)["das28"], FEATURE_DEFAULTS["das28"])
        self.assertFalse(result.identified)
        self.assertIn("baseline_disease_activity", result.blocked_reason)

    def test_a_non_numeric_value_is_not_a_measurement(self):
        """`numeric_feature` falls back for a DAS28 recorded as "high", so the
        record claims to carry the covariate and the model cannot read it."""
        from treatmentrx.estimation.features import FEATURE_DEFAULTS, model_features

        result, stages = self._validate(
            withheld=("DAS28",), added=(("DAS28", "high"),)
        )
        self.assertEqual(model_features(stages)["das28"], FEATURE_DEFAULTS["das28"])
        self.assertFalse(result.identified)

    def test_a_recorded_negative_serostatus_is_an_observation(self):
        """The half a value-against-default check would get wrong.

        `FEATURE_DEFAULTS["anti_ccp"]` is 0.0 and a recorded negative is also
        0.0, so "never tested" and "tested negative" are one number. Only the
        second is a measurement, and refusing it would abstain on every
        seronegative patient for having been tested.
        """
        result, _ = self._validate(
            withheld=("anti_CCP",), added=(("anti_CCP", False),)
        )
        self.assertTrue(result.identified, result.blocked_reason)

    def test_a_missing_serostatus_is_not(self):
        result, _ = self._validate(withheld=("anti_CCP",))
        self.assertFalse(result.identified)
        self.assertIn("anti_ccp", result.blocked_reason)

    def test_the_complete_record_is_identified(self):
        """Precision. A certificate that refused everything would satisfy every
        assertion above."""
        result, _ = self._validate()
        self.assertTrue(result.identified, result.blocked_reason)

    def test_every_adjuster_the_model_carries_names_where_it_is_read_from(self):
        """A new covariate in the basis becomes an adjuster automatically
        (`_split_by_what_the_model_carries`). It must not become one whose
        presence nothing can check."""
        from treatmentrx.data.dag import OBSERVED_AS, CausalDAGRegistry

        dag = CausalDAGRegistry()._ra_v1()
        # `prior_biologic_exposure` is read from the medication history rather
        # than an observation, and "no prior biologic" is a value of it.
        self.assertEqual(
            set(dag.adjustment_set),
            set(OBSERVED_AS) | {"prior_biologic_exposure"},
        )

    def test_the_adjuster_keys_are_the_ones_the_features_are_built_from(self):
        """`OBSERVED_AS` and `model_features` read the same record keys. Asserted
        by withholding each one and watching the covariate fall to its default,
        rather than by comparing the two lists to each other."""
        from treatmentrx.data.dag import OBSERVED_AS
        from treatmentrx.estimation.features import FEATURE_DEFAULTS, model_features

        covariate_for = {
            "baseline_disease_activity": "das28",
            "crp": "crp",
            "anti_ccp": "anti_ccp",
        }
        for node, (keys, _numeric) in OBSERVED_AS.items():
            with self.subTest(node=node):
                result, stages = self._validate(withheld=tuple(keys))
                covariate = covariate_for[node]
                self.assertEqual(
                    model_features(stages)[covariate], FEATURE_DEFAULTS[covariate]
                )
                self.assertFalse(result.identified)
                self.assertIn(node, result.blocked_reason)


class RetrospectiveBundleTests(unittest.TestCase):
    """`through_stage`: re-ask a decision the record already answered.

    Nothing could do this. `trajectory_to_bundle` always ends in a synthetic
    open decision point extrapolated past the last visit, so `stages[-1]`
    carries the sentinel `'current decision point'` and the record holds no
    clinician answer to compare an agent's against. Measuring concordance — the
    quantity the SHADOW gate asks for — needs the state as it stood *before* a
    decision, with the arm taken then withheld.
    """

    @classmethod
    def setUpClass(cls):
        from treatmentrx.simulation.ra_cohort import generate_ra_cohort

        cls.cohort = [
            t for t in generate_ra_cohort(20, seed=991) if len(t.stages) >= 3
        ]
        assert cls.cohort, "no trajectory long enough to truncate"

    def test_the_default_bundle_is_unchanged(self):
        """The parameter is additive: omitted, every byte is what it was."""
        import hashlib
        import json

        from treatmentrx.simulation.fhir_export import (
            simulated_bundles,
            trajectory_to_bundle,
        )
        from treatmentrx.simulation.ra_cohort import generate_ra_cohort

        batch = simulated_bundles(20, seed=991)
        rebuilt = [trajectory_to_bundle(t) for t in generate_ra_cohort(20, seed=991)]
        digest = lambda o: hashlib.sha256(
            json.dumps(o, sort_keys=True).encode()
        ).hexdigest()
        self.assertEqual(digest(batch), digest(rebuilt))

    def test_it_withholds_the_arm_it_is_asking_about(self):
        """The whole point: the answer must not be in the question."""
        from treatmentrx.simulation.fhir_export import trajectory_to_bundle

        trajectory = self.cohort[0]
        for index in range(1, len(trajectory.stages)):
            bundle = trajectory_to_bundle(trajectory, through_stage=index)
            prescribed = [
                entry["resource"]["medicationCodeableConcept"]["text"]
                for entry in bundle["entry"]
                if entry["resource"]["resourceType"] == "MedicationRequest"
            ]
            with self.subTest(through_stage=index):
                # The history is exactly the stages before it, in order.
                self.assertEqual(
                    prescribed[:-1],
                    [s.arm for s in trajectory.stages[:index]],
                )
                self.assertEqual(prescribed[-1], "current decision point")

    def test_the_open_point_carries_the_covariates_of_that_decision(self):
        """Not an extrapolation past the last visit — the state the clinician
        actually saw, which is what makes the comparison fair."""
        from treatmentrx.simulation.fhir_export import trajectory_to_bundle

        trajectory = self.cohort[0]
        index = 1
        bundle = trajectory_to_bundle(trajectory, through_stage=index)
        days = [
            entry["resource"]["day"]
            for entry in bundle["entry"]
            if entry["resource"]["resourceType"] == "Encounter"
        ]
        self.assertEqual(days[-1], trajectory.stages[index].day)

    def test_a_truncated_bundle_still_ingests(self):
        """It has to survive Layer 1, or it cannot be scored."""
        from treatmentrx.data import DataLayer
        from treatmentrx.simulation.fhir_export import trajectory_to_bundle

        layer = DataLayer()
        for trajectory in self.cohort[:5]:
            for index in range(1, len(trajectory.stages)):
                with self.subTest(patient=trajectory.patient_index, stage=index):
                    state = layer.build_patient_state(
                        trajectory_to_bundle(trajectory, through_stage=index)
                    )
                    self.assertEqual(len(state.stages), index + 1)
                    self.assertEqual(
                        state.stages[-1].treatment, "current decision point"
                    )

    def test_it_refuses_a_truncation_that_leaves_nothing_to_ask(self):
        """Stage 0 has no history before it and the last stage has no decision
        after it; both would produce a bundle that looks fine and means
        nothing."""
        from treatmentrx.simulation.fhir_export import trajectory_to_bundle

        trajectory = self.cohort[0]
        for index in (0, len(trajectory.stages), len(trajectory.stages) + 1):
            with self.subTest(through_stage=index):
                with self.assertRaises(ValueError):
                    trajectory_to_bundle(trajectory, through_stage=index)
