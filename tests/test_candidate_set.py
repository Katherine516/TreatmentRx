"""The set of arms the data cannot separate — Layer 3's answer when it declines.

The agent abstains for most patients it sees, and the abstention is earned. What
was missing was anything to act on: the clinician still has to prescribe, and got
back a status. These tests pin the three properties that make the set usable —
it contains the leader, it uses the same rule the separation line reports, and it
never contradicts the rest of the card — plus the one that makes it safe: it does
not raise the recommend rate.
"""

import unittest

from treatmentrx.data import DataContractError, DataLayer
from treatmentrx.decision import DecisionLayer
from treatmentrx.demo_data import sample_ra_bundle
from treatmentrx.domain import RecommendationStatus
from treatmentrx.estimation import EstimationLayer
from treatmentrx.orchestrator import TreatmentRxOrchestrator
from treatmentrx.simulation.fhir_export import simulated_bundles


def _decisions(n=40, seed=606):
    data, estimation, decision = DataLayer(), EstimationLayer(), DecisionLayer()
    out = []
    for bundle in simulated_bundles(n, seed=seed):
        try:
            state = data.build_patient_state(bundle)
        except DataContractError:
            continue
        out.append(decision.decide(state, estimation.estimate(state)))
    return out


class CandidateSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.decisions = _decisions()

    def test_the_leader_is_always_in_the_set(self):
        for decision in self.decisions:
            with self.subTest(arms=decision.candidate_arms):
                leader = max(decision.q_values, key=decision.q_values.get)
                self.assertEqual(decision.candidate_arms[0], leader)

    def test_membership_is_the_rule_the_separation_line_reports(self):
        """An arm is in the set exactly when its own interval fails to exclude
        zero. If these ever disagree the card states two different verdicts about
        the same pair — the failure `_separation` was written to fix."""
        for decision in self.decisions:
            for arm, contrast in decision.candidate_contrasts.items():
                with self.subTest(arm=arm):
                    self.assertEqual(
                        arm in decision.candidate_arms,
                        not contrast.robustly_distinguishable,
                    )

    def test_a_singleton_set_means_every_alternative_was_excluded(self):
        singletons = [d for d in self.decisions if len(d.candidate_arms) == 1]
        self.assertTrue(singletons, "expected at least one clearly separated patient")
        for decision in singletons:
            self.assertTrue(
                all(c.robustly_distinguishable for c in decision.candidate_contrasts.values())
            )

    def test_the_set_narrows_the_menu_for_declined_patients(self):
        """The whole point. If the set were routinely the whole menu it would be
        an honest way of saying nothing."""
        declined = [d for d in self.decisions if d.status is RecommendationStatus.EQUIPOISE]
        self.assertTrue(declined)
        sizes = [len(d.candidate_arms) for d in declined]
        menu = len(declined[0].q_values)
        mean = sum(sizes) / len(sizes)
        self.assertLess(mean, menu * 0.6, f"mean set {mean:.2f} of {menu} is barely a narrowing")
        self.assertTrue(all(size < menu for size in sizes))

    def test_it_does_not_raise_the_recommend_rate(self):
        """The guard against the one change CLAUDE.md forbids.

        The set is what the agent says when it will *not* recommend. If adding it
        moved the abstention rate, it would have tuned the action bar — the thing
        the notes explicitly warn against — rather than adding information beside
        it.
        """
        declined = sum(1 for d in self.decisions if d.status is RecommendationStatus.EQUIPOISE)
        rate = declined / len(self.decisions)
        self.assertGreater(rate, 0.45, f"abstention rate {rate:.2f} moved; the bar was tuned")


class CardConsistencyTests(unittest.TestCase):
    """The card must not explain away an arm the model could not rule out."""

    @classmethod
    def setUpClass(cls):
        orchestrator = TreatmentRxOrchestrator()
        cls.recommendations = []
        for bundle in simulated_bundles(30, seed=606):
            try:
                cls.recommendations.append(orchestrator.run(bundle))
            except DataContractError:
                continue

    def test_why_not_never_names_an_arm_still_in_contention(self):
        """`WHY_NOT_REASONS` is hard-coded clinical prose on a model-derived gap.
        Printed for an arm inside the candidate set it made the card say "cannot
        separate these four" and then explain why two of them were wrong."""
        for recommendation in self.recommendations:
            candidates = set(recommendation.audit_event["candidate_arms"])
            card = recommendation.clinician_card
            if "Why not the alternatives:" not in card:
                continue
            clause = card.split("Why not the alternatives:", 1)[1].split("\n\n", 1)[0]
            for arm in candidates:
                with self.subTest(arm=arm):
                    self.assertNotIn(arm, clause)

    def test_the_card_never_lists_an_arm_the_safety_layer_removed(self):
        """A set printed for a clinician is read as a menu. It must not contain
        something Layer 4 already refused."""
        for recommendation in self.recommendations:
            card = recommendation.clinician_card
            if "Cannot separate:" not in card:
                continue
            listed = card.split("Cannot separate:", 1)[1].split(".", 1)[0]
            for arm in recommendation.audit_event.get("removed_arms", {}):
                with self.subTest(arm=arm):
                    self.assertNotIn(arm, listed)

    def test_a_declined_patient_is_given_something_to_act_on(self):
        declined = [
            r for r in self.recommendations if r.status is RecommendationStatus.EQUIPOISE
        ]
        self.assertTrue(declined)
        for recommendation in declined:
            with self.subTest(patient=recommendation.patient_hash):
                self.assertIn("Cannot separate:", recommendation.clinician_card)
                self.assertIn("not a recommendation", recommendation.clinician_card)

    def test_the_audit_event_carries_the_set_and_its_intervals(self):
        for recommendation in self.recommendations:
            audit = recommendation.audit_event
            self.assertEqual(
                audit["candidate_set_size"], len(audit["candidate_arms"])
            )
            for arm, block in audit["candidate_contrasts"].items():
                self.assertEqual(block["excluded"], arm not in audit["candidate_arms"])


class CandidateSetCoverageTests(unittest.TestCase):
    def test_the_set_contains_the_truly_optimal_arm(self):
        """The property that makes the set safe is not its size.

        Replicated over refits, per invariant 31: on a single fit every patient's
        set shares the same parameters, so one unlucky fit misses for everyone
        and the fraction measured is a property of that draw.
        """
        from treatmentrx.feedback.coverage import candidate_set_coverage

        result = candidate_set_coverage(replications=12)
        self.assertGreater(result["contains_optimal_arm"], 0.90)
        self.assertLess(result["mean_set_size"], result["arms_on_the_menu"])
        self.assertGreater(result["regret_reduction"], 0.5)


class MultiplicityTests(unittest.TestCase):
    """The divisor is a choice, so it has to carry its numbers."""

    @classmethod
    def setUpClass(cls):
        from treatmentrx.feedback.coverage import multiplicity_sweep

        cls.sweep = multiplicity_sweep(replications=6, patients=25)
        cls.levels = {row["divisor"]: row for row in cls.sweep["levels"]}

    def test_the_deployed_divisor_is_the_one_the_sweep_reports(self):
        from treatmentrx.arms import TREATMENT_ARMS
        from treatmentrx.estimation.inference import DEFAULT_ALPHA, simultaneous_alpha

        arms = len(TREATMENT_ARMS)
        self.assertEqual(self.sweep["deployed_divisor"], arms * (arms - 1) // 2)
        self.assertAlmostEqual(
            self.levels[self.sweep["deployed_divisor"]]["alpha"],
            round(DEFAULT_ALPHA / self.sweep["deployed_divisor"], 5),
        )
        self.assertAlmostEqual(
            simultaneous_alpha(arms), DEFAULT_ALPHA / self.sweep["deployed_divisor"]
        )

    def test_correcting_harder_widens_the_set_and_declines_more(self):
        """The trade the divisor is buying, asserted as a direction.

        A stricter level can only add arms to the set — an arm excluded at a
        wider interval is excluded at a narrower one — so both the set size and
        the decline rate must move monotonically with the divisor. If they ever
        do not, the correction is not doing what its name says.
        """
        ordered = [self.levels[d] for d in sorted(self.levels)]
        for looser, stricter in zip(ordered, ordered[1:]):
            with self.subTest(pair=(looser["divisor"], stricter["divisor"])):
                self.assertGreater(stricter["alpha"], 0.0)
                self.assertLess(stricter["alpha"], looser["alpha"])
                self.assertGreaterEqual(stricter["mean_set_size"], looser["mean_set_size"])
                self.assertGreaterEqual(stricter["decline_rate"], looser["decline_rate"])

    def test_even_the_uncorrected_level_over_covers(self):
        """Why the sweep exists. The conservatism is not coming from the
        multiplicity — the decision rule's variance bound already runs about
        1.27x the actual spread, so this correction stacks a second one on a
        first, and a reader deciding the divisor needs to see that."""
        self.assertGreater(self.levels[1]["contains_optimal_arm"], 0.95)


if __name__ == "__main__":
    unittest.main()
