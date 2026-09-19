"""The molecule vocabulary, and the relationships four literals kept by hand.

`arms.py` is the canonical arm vocabulary and invariant 3 says every layer must
agree on it. They do. The layer below — which molecules belong to an arm, and
which hazards they carry — was declared in four places, and *nothing compared
them*, so the one that had already fallen behind went unnoticed: `arms.py`
recognises three JAK molecules and `safety/feasible_set.py` hazard-classed the
same three, while the offer list carried one.

These assert the relationships rather than the contents. The curation is
illustrative and will be replaced; what must survive that replacement is that
the offer list, the hazard classes and the recognition vocabulary still line up.
"""

import unittest

from treatmentrx import formulary
from treatmentrx.arms import REFERENCE_ARM, TREATMENT_ARMS, normalize_arm


class VocabularyAgreementTests(unittest.TestCase):
    def test_every_offerable_molecule_is_recognisable(self):
        """The agent must be able to read back a line it proposed.

        A molecule it offers but `normalize_arm` cannot place would arrive in the
        next visit's history as `manual-review`, and the trajectory would record
        a decision the agent never made.
        """
        self.assertEqual(formulary.unrecognised_offerables(), ())

    def test_every_offerable_molecule_lands_on_its_own_arm(self):
        """Recognisable is not enough — it has to normalise to the *same* arm."""
        for molecule in formulary.MOLECULES:
            if not molecule.offerable:
                continue
            with self.subTest(molecule=molecule.name):
                self.assertEqual(normalize_arm(molecule.name), molecule.arm)

    def test_every_declared_molecule_belongs_to_a_real_arm(self):
        for molecule in formulary.MOLECULES:
            with self.subTest(molecule=molecule.name):
                self.assertIn(molecule.arm, TREATMENT_ARMS)

    def test_the_composite_filter_reads_the_formulary_not_its_own_list(self):
        """The drift this file exists to prevent, asserted from the other side."""
        from treatmentrx.safety import feasible_set

        self.assertEqual(
            set(feasible_set.JAK_DRUGS),
            set(formulary.hazard_tokens(formulary.HAZARD_JAK)),
        )
        self.assertEqual(
            set(feasible_set.HEPATOTOXIC_DRUGS),
            set(formulary.hazard_tokens(formulary.HAZARD_HEPATOTOXIC)),
        )

    def test_the_arm_level_rule_and_the_composite_filter_agree(self):
        """Two safety modules classed hepatic risk from two separate literals,
        neither a subset of the other. They agreed on every arm — by curation,
        not by construction. Now they are one declaration, and this is what
        would catch them parting again."""
        from treatmentrx.safety import rules
        from treatmentrx.safety.feasible_set import ALT_CEILING, FeasibleSet

        filter_ = FeasibleSet()
        for arm in TREATMENT_ARMS:
            if arm == REFERENCE_ARM:
                continue
            composites = formulary.composites_for(arm)
            gated = bool(composites) and all(
                filter_._unsafe_reason(action, 400.0, 90.0, False, [], arm) is not None
                for action in composites
            )
            with self.subTest(arm=arm, alt_ceiling=ALT_CEILING):
                self.assertEqual(
                    rules._is_hepatotoxic(arm),
                    gated,
                    "the arm-level rule and the composite filter disagree on hepatic risk",
                )

    def test_the_arm_level_rule_only_matches_canonical_arm_names(self):
        """Both readers of the old list substring-matched *molecule* spellings
        against a canonical arm name, so four of six tokens were unreachable.
        A molecule name must not be mistaken for an arm."""
        from treatmentrx.safety import rules

        for molecule in formulary.MOLECULES:
            if molecule.name in TREATMENT_ARMS:
                continue
            with self.subTest(molecule=molecule.name):
                self.assertFalse(rules._is_hepatotoxic(molecule.name))


class MenuTests(unittest.TestCase):
    def test_the_menu_is_derived_from_the_formulary(self):
        from treatmentrx.estimation.actions import ARM_CANDIDATES

        self.assertEqual(
            {arm: [a.label for a in options] for arm, options in ARM_CANDIDATES.items()},
            {arm: [a.label for a in options]
             for arm, options in formulary.arm_candidates().items()},
        )

    def test_every_advanced_arm_keeps_a_monotherapy_composite(self):
        """Invariant 15, now derivable instead of asserted in a comment.

        Listing a biologic only in MTX combination turns one methotrexate
        contraindication into a blocked recommendation for a patient who had a
        viable option.
        """
        for arm in TREATMENT_ARMS:
            if arm == REFERENCE_ARM:
                continue
            composites = formulary.composites_for(arm)
            with self.subTest(arm=arm):
                self.assertTrue(composites, "an arm with no composite cannot be offered")
                self.assertTrue(
                    any(not action.combination for action in composites),
                    "every advanced arm needs a composite that is not MTX-combination",
                )

    def test_a_non_offerable_molecule_contributes_no_regimen(self):
        for molecule in formulary.MOLECULES:
            if molecule.offerable:
                continue
            with self.subTest(molecule=molecule.name):
                self.assertEqual(molecule.regimens, ())

    def test_a_non_offerable_molecule_still_carries_its_hazards(self):
        """`leflunomide` is the case: recognised, hepatotoxic, never proposed.
        Dropping it from the hazard tokens would let it through as a background
        combination."""
        hepatotoxic = formulary.hazard_tokens(formulary.HAZARD_HEPATOTOXIC)
        self.assertIn("leflunomide", hepatotoxic)
        self.assertFalse(
            any(m.offerable for m in formulary.MOLECULES if m.name == "leflunomide")
        )

    def test_every_molecule_states_its_provenance(self):
        """The curation is illustrative and each entry says so itself, so a
        reader inspecting one does not have to find the module header."""
        for molecule in formulary.MOLECULES:
            with self.subTest(molecule=molecule.name):
                self.assertIn("not a clinical source", molecule.provenance)


class ArmNamingTests(unittest.TestCase):
    """Whether a within-class alternative is even a coherent idea for an arm.

    Derived from the arm name rather than declared, so a new arm cannot forget
    to say which kind it is.
    """

    def test_class_named_arms_are_the_ones_with_alternatives(self):
        self.assertFalse(formulary.is_molecule_named("TNF-inhibitor"))
        self.assertFalse(formulary.is_molecule_named("IL-6 inhibitor"))
        self.assertFalse(formulary.is_molecule_named("JAK-inhibitor"))

    def test_an_arm_named_for_its_drug_is_recognised_as_such(self):
        self.assertTrue(formulary.is_molecule_named("rituximab"))
        self.assertTrue(formulary.is_molecule_named("methotrexate-optimization"))

    def test_recognition_stays_broader_than_the_menu(self):
        """`arms.py` maps `hydroxychloroquine` to the methotrexate arm and
        `abatacept` to rituximab, because for reading a history "some csDMARD"
        and "some non-TNF advanced therapy" is the right granularity. Neither is
        a substitute *within* the arm's meaning, and the menu must not acquire
        them by being derived from the recognition vocabulary."""
        offerable = {m.name for m in formulary.MOLECULES if m.offerable}
        for token in ("hydroxychloroquine", "sulfasalazine", "abatacept"):
            with self.subTest(token=token):
                self.assertEqual(normalize_arm(token) in TREATMENT_ARMS, True)
                self.assertNotIn(token, offerable)


class BreadthTests(unittest.TestCase):
    """Breadth is a curation choice, and these make it a measured one.

    An arm offered as a single molecule is an arm a single drug allergy removes
    outright. That is not a property of the method and it is not inherent — it is
    how many regimens the formulary happens to carry.
    """

    @classmethod
    def setUpClass(cls):
        cls.breadth = formulary.breadth()

    def test_breadth_covers_every_treatment_arm(self):
        self.assertEqual(
            set(self.breadth),
            {arm for arm in TREATMENT_ARMS if arm != REFERENCE_ARM},
        )

    def test_every_widenable_arm_survives_a_single_molecule_allergy(self):
        """The property this menu exists to have, and the reason the regimens
        for `tofacitinib`, `baricitinib` and `sarilumab` were written.

        A class-named arm with one offerable molecule is an arm a single drug
        allergy removes outright, for a patient the same system would recognise
        as having taken another member of that class.
        """
        widenable = {
            arm: row for arm, row in self.breadth.items() if row["widening_is_possible"]
        }
        self.assertGreaterEqual(len(widenable), 3)
        for arm, row in widenable.items():
            with self.subTest(arm=arm):
                self.assertTrue(
                    row["survives_a_single_molecule_allergy"],
                    f"{arm} names a drug class and offers only "
                    f"{row['offerable_molecules']}",
                )

    def test_a_molecule_named_arm_is_not_scored_as_a_gap(self):
        """`rituximab` and `methotrexate-optimization` are named for the drug
        they are: swapping the molecule makes them a different arm. Reporting
        them as unwidened breadth would be a gap that cannot be closed, and the
        audit's denominator would be wrong rather than merely unflattering."""
        named = {arm for arm, row in self.breadth.items() if row["named_after_its_molecule"]}
        self.assertEqual(named, {"rituximab", "methotrexate-optimization"})
        for arm in named:
            with self.subTest(arm=arm):
                self.assertFalse(self.breadth[arm]["widening_is_possible"])

    def test_survival_follows_the_offer_count_not_the_declared_count(self):
        """The flag has to track the offer list, not the declared one, because
        the offer list is what the patient is left with."""
        for arm, row in self.breadth.items():
            with self.subTest(arm=arm):
                self.assertEqual(
                    row["survives_a_single_molecule_allergy"],
                    len(row["offerable_molecules"]) > 1,
                )

    def test_the_survival_flag_matches_what_the_pipeline_does(self):
        """Asserted end to end rather than from the counts, because the counts
        are the claim and the pipeline is the evidence."""
        from treatmentrx.feedback.audit import _with_allergy
        from treatmentrx.orchestrator import TreatmentRxOrchestrator

        orchestrator = TreatmentRxOrchestrator()
        for arm, row in self.breadth.items():
            for molecule in row["offerable_molecules"]:
                recommendation = orchestrator.run(_with_allergy(molecule))
                arm_survived = arm not in recommendation.audit_event["removed_arms"]
                with self.subTest(arm=arm, molecule=molecule):
                    self.assertEqual(arm_survived, row["survives_a_single_molecule_allergy"])


if __name__ == "__main__":
    unittest.main()
