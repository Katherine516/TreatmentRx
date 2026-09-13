"""Layer 1 — ingestion, stage construction, and the guards that must raise."""

import copy
import unittest

from treatmentrx.contracts import PatientState
from treatmentrx.arms import normalize_arm
from treatmentrx.data import DataLayer
from treatmentrx.data.contract import RADataContract
from treatmentrx.data.dag import CausalDAGRegistry
from treatmentrx.data.encoders import GRUBaselineEncoder, HandcraftedFeatureEncoder
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
        result = CausalDAGRegistry().validate(_patient(), treatment="IL-6 inhibitor")
        self.assertTrue(result.identified)
        self.assertIn("baseline_disease_activity", result.adjustment_set)
        self.assertIn("colliders", result.causal_path_text)

    def test_encoders_produce_stable_state_vectors(self):
        _, stages = _stages()
        self.assertEqual(len(HandcraftedFeatureEncoder().encode(stages).vector), 32)
        self.assertEqual(len(GRUBaselineEncoder().encode(stages).vector), 256)

    def test_only_the_handcrafted_prefix_of_the_encoding_is_read(self):
        """Documented rather than assumed: the recurrent tail holds the shape of
        the planned `z_t` interface, and no consumer reads it yet. The OOD score
        slices `vector[:32]`."""
        _, stages = _stages()
        vector = GRUBaselineEncoder().encode(stages).vector
        handcrafted = HandcraftedFeatureEncoder().encode(stages, dimension=32).vector
        self.assertEqual(vector[:32], handcrafted)


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

    def test_a_leaking_record_cannot_produce_a_patient_state(self):
        """The guard raises out of the pipeline; it is never a soft diagnostic."""
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
