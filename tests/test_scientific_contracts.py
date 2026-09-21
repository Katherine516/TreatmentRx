"""Scientific targets are executable contracts, not prose on a model card."""

import unittest
from dataclasses import replace

from treatmentrx import TreatmentRxOrchestrator
from treatmentrx.arms import TREATMENT_ARMS
from treatmentrx.contracts import RegimeEstimate
from treatmentrx.decision.bma import BayesianModelAverager
from treatmentrx.demo_data import sample_ra_bundle
from treatmentrx.domain import RegimeType
from treatmentrx.data import DataLayer
from treatmentrx.estimation import EstimationLayer
from treatmentrx.scientific import (
    EstimandContractError,
    EvaluationPartitionContract,
    ScientificMode,
    ra_dtr_estimand,
)


class EstimandContractTests(unittest.TestCase):
    def test_ra_contract_is_stable_and_complete(self):
        first = ra_dtr_estimand(TREATMENT_ARMS)
        second = ra_dtr_estimand(TREATMENT_ARMS)
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(first.mode, ScientificMode.DTR_RESEARCH)
        self.assertIn(first.reference_action, first.treatment_strategies)
        self.assertGreater(first.horizon_days, 0)
        self.assertIn("fingerprint", first.as_dict())

    def test_estimand_mismatch_blocks_model_averaging(self):
        first = ra_dtr_estimand(TREATMENT_ARMS)
        changed = replace(first, horizon_days=180)
        with self.assertRaises(EstimandContractError):
            first.assert_compatible(changed)

    def test_reference_action_must_be_on_the_menu(self):
        contract = ra_dtr_estimand(TREATMENT_ARMS)
        with self.assertRaises(EstimandContractError):
            replace(contract, reference_action="not-an-arm")

    def test_model_averaging_rejects_different_estimands(self):
        def estimate(name, fingerprint):
            return RegimeEstimate(
                estimator=name,
                regime_type=RegimeType.SPTR,
                recommended_arm="a",
                q_values={"a": 0.7, "b": 0.6},
                policy_value=0.5,
                confidence_band=(0.4, 0.8),
                estimand_fingerprint=fingerprint,
            )

        with self.assertRaisesRegex(ValueError, "identical estimand"):
            BayesianModelAverager().aggregate(
                [estimate("Q-Pooled", "one"), estimate("dWOLS-Shared", "two")]
            )


class FingerprintCacheTests(unittest.TestCase):
    """The fingerprint is memoised, and `replace()` must not carry a stale one.

    The contract is `frozen=True`, so the hash is the same sixteen characters
    for the life of the object — but it was recomputed on every read, and each
    read runs `dataclasses.asdict`, `json.dumps` and a SHA-256. Six reads a
    request made it the most expensive thing in the pipeline once the cross-arm
    covariance was memoised.

    The cache is stashed under a name that is deliberately not a field. Had it
    been one, `dataclasses.replace` would have copied the old hash onto a
    contract whose content no longer matched it — a wrong fingerprint is worse
    than a slow one, because the estimand check exists to catch exactly that.
    """

    def setUp(self):
        self.contract = ra_dtr_estimand(TREATMENT_ARMS)

    def test_the_fingerprint_is_stable_across_reads(self):
        self.assertEqual(self.contract.fingerprint, self.contract.fingerprint)

    def test_a_replaced_contract_computes_its_own(self):
        """The hazard the non-field stash exists to avoid."""
        import dataclasses

        original = self.contract.fingerprint
        altered = dataclasses.replace(self.contract, outcome="a different outcome")
        self.assertNotEqual(
            altered.fingerprint,
            original,
            "the derived contract inherited the original's hash",
        )

    def test_the_cache_is_not_a_field(self):
        """Asserted structurally, because the test above would still pass if the
        cache were a field that `replace` happened to reset."""
        self.assertNotIn("_fingerprint_cache", self.contract.__dataclass_fields__)

    def test_the_memoised_value_is_the_one_it_replaced(self):
        """Recomputed the long way, so the cache cannot quietly return something
        else."""
        import hashlib
        import json

        payload = json.dumps(
            self.contract.as_dict(include_fingerprint=False), sort_keys=True
        )
        expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        self.assertEqual(self.contract.fingerprint, expected)


class PartitionContractTests(unittest.TestCase):
    def test_partitions_must_be_patient_disjoint(self):
        with self.assertRaises(EstimandContractError):
            EvaluationPartitionContract(
                training=frozenset({"p1", "p2"}),
                tuning=frozenset({"p2"}),
                calibration=frozenset({"p3"}),
                final_test=frozenset({"p4"}),
            )

    def test_final_test_must_be_locked(self):
        with self.assertRaises(EstimandContractError):
            EvaluationPartitionContract(
                training=frozenset({"p1"}),
                tuning=frozenset({"p2"}),
                calibration=frozenset({"p3"}),
                final_test=frozenset({"p4"}),
                final_test_locked=False,
            )


class DeployedPartitionTests(unittest.TestCase):
    """The contract has to describe the split that exists, not an aspiration.

    `EvaluationPartitionContract` was exported, tested against hand-built inputs,
    and never constructed from the pipeline — so the package's public surface
    advertised a locked final test while `training.py` had two lists.
    """

    @classmethod
    def setUpClass(cls):
        from treatmentrx.estimation import training

        cls.training_module = training
        cls.partition = training.evaluation_partition()

    def test_it_describes_the_split_the_models_are_actually_fit_on(self):
        fit = self.training_module.fitted()
        self.assertEqual(len(self.partition.training), len(fit.train))
        self.assertEqual(len(self.partition.calibration), len(fit.holdout))

    def test_no_patient_appears_in_two_roles(self):
        """The constructor enforces it; this checks the real ids satisfy it.

        A cohort whose `patient_index` were not unique would silently produce
        overlapping frozensets, and the disjointness check would start failing
        for a reason that has nothing to do with leakage.
        """
        self.assertFalse(self.partition.training & self.partition.calibration)

    def test_this_build_has_no_held_back_final_test(self):
        """An empty partition satisfies "locked" vacuously.

        That vacuous reading is what let the type describe a discipline the
        codebase was not following, so the absence is asserted rather than
        assumed to be obvious from a zero.
        """
        self.assertFalse(self.partition.has_final_test)
        self.assertTrue(self.partition.final_test_locked)
        self.assertEqual(self.partition.roles_in_use, ("training", "calibration"))

    def test_the_model_card_says_there_is_no_final_test(self):
        """A consumer comparing this card against a published model's would
        otherwise assume a held-back split exists."""
        from treatmentrx.service import RecommendationService

        block = RecommendationService().model_card()["known_limitations"][
            "evaluation_partition"
        ]
        self.assertFalse(block["has_final_test"])
        self.assertIn("no partition held back", block["what_it_means"])


class ModeBoundaryTests(unittest.TestCase):
    def test_every_serving_estimator_carries_the_state_estimand(self):
        """And therefore the BMA fingerprint check cannot fire from here.

        The layer stamps every result from one source, so the fingerprints are
        identical by construction on the serving path. That is what makes the
        check in `BayesianModelAverager.aggregate` a precondition on a public
        component rather than the thing keeping the ensemble coherent — the
        guard invariant 18 actually rests on is `EstimationLayer.estimators`
        matching `SERVING_ENSEMBLE`, asserted in `tests/test_layers.py`.
        """
        state = DataLayer().build_patient_state(sample_ra_bundle())
        estimates = EstimationLayer().estimate(state)
        self.assertTrue(state.estimand_contract)
        self.assertEqual(
            {estimate.estimand_fingerprint for estimate in estimates},
            {state.estimand_contract.fingerprint},
        )
        self.assertEqual(
            len({estimate.estimand_fingerprint for estimate in estimates}),
            1,
            "the serving path cannot produce a mismatch for the check to catch",
        )

    def test_recommendation_pins_mode_and_estimand(self):
        recommendation = TreatmentRxOrchestrator().run(sample_ra_bundle())
        self.assertEqual(recommendation.provenance["operating_mode"], "dtr_research")
        self.assertEqual(
            recommendation.audit_event["estimand_contract"]["fingerprint"],
            recommendation.provenance["estimand_contract"]["fingerprint"],
        )
        selection = recommendation.audit_event["selection_inference"]
        self.assertEqual(selection["unordered_pair_count"], 15)
        self.assertAlmostEqual(selection["per_pair_alpha"], 0.05 / 15)

    def test_trial_mode_cannot_use_patient_recommendation_workflow(self):
        with self.assertRaisesRegex(ValueError, "does not support operating mode"):
            TreatmentRxOrchestrator().run(
                sample_ra_bundle(), mode=ScientificMode.RANDOMIZED_TRIAL
            )


class DeclaredMenuTests(unittest.TestCase):
    """The menu the estimators score must be the one the contract declares.

    `state.feasible_arms` comes from `RADataContract.treatment_arms`, a class
    attribute hardcoded to `arms.TREATMENT_ARMS`. `DiseaseDefinition.treatment_arms`
    is carried, reported in `capability()`, and reaches neither Layer 1 nor the
    estimators. So a second disease could declare its own menu while these models
    scored the RA one — measured, patching only the contract's diagnosis check and
    the DAG registry served a **Breast Cancer** record a **rituximab**
    recommendation over the RA arm vocabulary.

    Invariant 34 says an unregistered disease must never reuse RA arms or models.
    That held on two fail-closed guards upstream rather than on anything asserting
    it where the models are used.
    """

    def test_the_three_declarations_agree_for_the_registered_disease(self):
        """Invariant 3 at this level. They agree today because all three read
        `arms.TREATMENT_ARMS`; nothing had checked that they must."""
        from treatmentrx.data.contract import RADataContract
        from treatmentrx.diseases import rheumatoid_arthritis_definition

        definition = rheumatoid_arthritis_definition()
        declared = set(definition.treatment_arms)
        estimand = set(definition.estimand_contracts[0].treatment_strategies)
        scored = set(RADataContract.treatment_arms)
        self.assertEqual(declared, estimand)
        self.assertEqual(declared, scored)
        self.assertEqual(declared, set(TREATMENT_ARMS))

    def test_a_contract_declaring_another_menu_is_refused(self):
        """The check, exercised directly: the estimators are fit on one arm
        vocabulary and must not be asked about another."""
        from dataclasses import replace

        state = DataLayer().build_patient_state(sample_ra_bundle())
        other = ra_dtr_estimand(
            ("continue-current", "endocrine-therapy", "CDK4/6-inhibitor")
        )
        mismatched = replace(state, estimand_contract=other)
        with self.assertRaises(ValueError) as raised:
            EstimationLayer().estimate(mismatched)
        message = str(raised.exception)
        self.assertIn("declares a different menu", message)
        self.assertIn("CDK4/6-inhibitor", message)

    def test_a_second_disease_cannot_reach_the_ra_models(self):
        """End to end, with the two upstream guards removed.

        This is the scenario that served `rituximab` to a breast-cancer record.
        Both guards are disabled deliberately, because the point is that
        invariant 34 must not rest on them alone.
        """
        import copy

        from treatmentrx.data import contract as contract_module
        from treatmentrx.data import dag as dag_module
        from treatmentrx.diseases import (
            DiseaseDefinition, DiseaseRegistry, DiseaseWorkflow,
            rheumatoid_arthritis_definition,
        )
        from treatmentrx.orchestrator import TreatmentRxOrchestrator

        arms = ("continue-current", "endocrine-therapy", "CDK4/6-inhibitor")

        def workflow():
            from treatmentrx.agent import AgentLayer
            from treatmentrx.decision import DecisionLayer
            from treatmentrx.estimation import training
            from treatmentrx.feedback import FeedbackLayer
            from treatmentrx.safety import SafetyLayer

            return DiseaseWorkflow(
                DataLayer(), EstimationLayer(), DecisionLayer(),
                SafetyLayer(), AgentLayer(), FeedbackLayer(), training,
            )

        registry = DiseaseRegistry((
            rheumatoid_arthritis_definition(),
            DiseaseDefinition(
                disease_id="other_disease", display_name="Other Disease",
                diagnosis_terms=("othercondition",), treatment_arms=arms,
                endpoint="n/a", dag="n/a", model_family="n/a",
                safety_policy="n/a", knowledge_base="n/a",
                workflow_factory=workflow,
                operating_modes=(ScientificMode.DTR_RESEARCH,),
                estimand_contracts=(ra_dtr_estimand(arms),),
            ),
        ))

        bundle = copy.deepcopy(sample_ra_bundle())
        for entry in bundle["entry"]:
            if entry["resource"].get("resourceType") == "Condition":
                entry["resource"]["code"] = {"text": "othercondition"}

        original_validate = contract_module.RADataContract.validate
        original_match = dag_module.CausalDAGRegistry.match

        def permissive(self, patient):
            report = original_validate(self, patient)
            kept = [
                issue for issue in report.issues
                if "rheumatoid arthritis diagnosis" not in issue.message
            ]
            return type(report)(
                report.disease, report.primary_endpoint, report.treatment_arms,
                kept, report.missing_variable_families,
            )

        try:
            contract_module.RADataContract.validate = permissive
            dag_module.CausalDAGRegistry.match = (
                lambda self, patient: self._dags["rheumatoid arthritis"]
            )
            with self.assertRaises(ValueError) as raised:
                TreatmentRxOrchestrator(registry).run(bundle)
        finally:
            contract_module.RADataContract.validate = original_validate
            dag_module.CausalDAGRegistry.match = original_match

        self.assertIn("declares a different menu", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
