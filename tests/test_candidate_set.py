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
        from treatmentrx.feedback.audit import _with_pregnancy

        orchestrator = TreatmentRxOrchestrator()
        cls.recommendations = []
        # The same patients with a contraindication injected, because a removed
        # arm is the only way to exercise the safety half of these rules.
        cls.pregnant = []
        for bundle in simulated_bundles(30, seed=606):
            try:
                cls.recommendations.append(orchestrator.run(bundle))
                cls.pregnant.append(orchestrator.run(_with_pregnancy(bundle)))
            except DataContractError:
                continue
        assert any(r.audit_event.get("removed_arms") for r in cls.pregnant)

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

    def test_why_not_never_explains_away_an_arm_safety_removed(self):
        """The same rule from the other side, and it is the worse half.

        `WHY_NOT_REASONS` is colour attached to a statistical exclusion. An arm
        Layer 4 removed was not excluded statistically — it was refused — so a
        sentence saying it narrowly lost on merit misdescribes why it is gone and
        reads as though it could be reconsidered. Nothing is lost by dropping it:
        the safety layer raises an `arm_removed` flag for every removal, so each
        one is already named on this card with the reason that applies.
        """
        for recommendation in self.pregnant:
            card = recommendation.clinician_card
            if "Why not the alternatives:" not in card:
                continue
            clause = card.split("Why not the alternatives:", 1)[1].split("\n\n", 1)[0]
            for arm in recommendation.audit_event.get("removed_arms", {}):
                with self.subTest(arm=arm):
                    self.assertNotIn(arm, clause)

    def test_a_recommendation_never_carries_declining_prose(self):
        """Invariant 35, in the block that was written to satisfy it.

        Status is decided on the top-two contrast alone, so a lower-scoring arm
        with a wider interval survives the exclusion test while the runner-up
        fails it — and then a RECOMMEND card rendered the declining wording:
        "Cannot separate: X, Y ... This is not a recommendation", four paragraphs
        under "recommend X". Measured 6 of 120 patients before the fix.
        """
        recommended = [
            r for r in self.recommendations
            if r.status is RecommendationStatus.RECOMMEND
        ]
        self.assertTrue(recommended)
        for recommendation in recommended:
            card = recommendation.clinician_card
            with self.subTest(patient=recommendation.patient_hash):
                for declining in (
                    "Cannot separate:",
                    "not a recommendation",
                    "has not been recommended",
                    "remains in contention",
                ):
                    self.assertNotIn(declining, card)

    def test_the_unexcluded_arms_still_reach_a_recommendation_card(self):
        """Suppressing the block would have been the wrong fix.

        The information is right and only the framing was wrong, so the arms the
        data cannot rule out must still be named beside a recommendation —
        hiding them makes the card more confident than the evidence.
        """
        shown = 0
        for recommendation in self.recommendations:
            if recommendation.status is not RecommendationStatus.RECOMMEND:
                continue
            others = [
                arm for arm in recommendation.audit_event["candidate_arms"]
                if arm != recommendation.recommended_arm
            ]
            if not others:
                continue
            shown += 1
            card = recommendation.clinician_card
            with self.subTest(patient=recommendation.patient_hash):
                self.assertIn("Not excluded:", card)
                for arm in others:
                    self.assertIn(arm, card.split("Not excluded:", 1)[1])
        self.assertTrue(shown, "no recommended patient had an unexcluded alternative")

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


class WhyNotIsModelDerivedTests(unittest.TestCase):
    """The why-not reason is the gap's own decomposition, not prose about the arm.

    `WHY_NOT_REASONS` was keyed on the arm alone, so every patient read the same
    sentence for a given arm while the gap printed beside it varied correctly —
    and `JAK-inhibitor`'s named organ function, which `BLIP_BASIS` does not
    contain. These pin the three properties that replace it: the reason names a
    term the model actually used, the decomposition reconstructs the gap it
    annotates, and the answer moves with the patient.
    """

    @classmethod
    def setUpClass(cls):
        from treatmentrx.estimation import training
        from treatmentrx.estimation.features import stage_index

        data, estimation, decision = DataLayer(), EstimationLayer(), DecisionLayer()
        n_stages = training.fitted().pooled.n_stages
        cls.entries = []
        cls.stages_covered = set()
        for bundle in simulated_bundles(60, seed=606):
            try:
                state = data.build_patient_state(bundle)
            except DataContractError:
                continue
            made = decision.decide(state, estimation.estimate(state))
            cls.stages_covered.add(stage_index(state.stages, n_stages))
            for entry in (made.explanation.why_not if made.explanation else []):
                cls.entries.append((made, entry))
        assert cls.entries

    def test_the_language_map_covers_the_blip_basis(self):
        """A basis term with no phrase reaches a clinician card as a bare
        variable name. The map is keyed on the covariate, so adding a modifier
        to `BLIP_BASIS` without a word for it is the failure to catch."""
        from treatmentrx.estimation.basis import BLIP_BASIS
        from treatmentrx.estimation.explainability import BLIP_TERM_LANGUAGE

        for name in BLIP_BASIS:
            with self.subTest(covariate=name):
                self.assertIn(name, BLIP_TERM_LANGUAGE)

    def test_the_reason_names_the_term_that_most_moves_the_gap(self):
        """Not the largest absolute term — the largest one working *for* the
        leader, because the question the card answers is why this arm lost."""
        from treatmentrx.estimation.explainability import BLIP_TERM_LANGUAGE

        checked = 0
        for _, entry in self.entries:
            positive = {k: v for k, v in entry.contributions.items() if v > 0}
            if not positive:
                continue
            checked += 1
            leading = max(positive.items(), key=lambda kv: kv[1])[0]
            with self.subTest(arm=entry.action):
                self.assertIn(
                    BLIP_TERM_LANGUAGE[leading],
                    entry.dominant_reason,
                    "the reason names a term other than the one driving the gap",
                )
        self.assertGreater(checked, 100)

    def test_the_contributions_reconstruct_the_gap_at_every_served_stage(self):
        """The scale guard, and it is the one that can fail loudly.

        Both sides are now the same weighted mean — `_pair_contrast` builds the
        gap from the members' contrasts on the BMA weights, and the decomposition
        averages their blips on those same weights — so the residual is the 4dp
        coefficient rounding rather than the members' disagreement: **0.0006**
        max here, against 0.0575 when a single member's psi was carried.

        The bar stays well above that because what it guards is a *scale*, not a
        rounding. `Q-Pooled` publishes a value-to-go stage psi and dividing it by
        the remaining horizon is what puts it on the gap's scale; remove that
        division and stage 1 goes to **0.0828** while the terminal block, horizon
        1, does not move at all. 0.01 sits ~16x above the healthy residual and
        ~8x below the regression, and the sweep covers every served stage because
        the terminal block is the one place the two members coincide anyway.
        """
        worst = 0.0
        for _, entry in self.entries:
            residual = abs(sum(entry.contributions.values()) - entry.q_gap)
            worst = max(worst, residual)
            with self.subTest(arm=entry.action):
                self.assertLess(
                    residual,
                    0.01,
                    "the decomposition is not on the same scale as the gap it explains",
                )
        # The two members coincide only at the terminal block, so a sweep that
        # reached one stage would not be testing the scale at all.
        self.assertGreater(
            len(self.stages_covered),
            1,
            f"only stage {self.stages_covered} reached; the horizon is 1 there",
        )
        self.assertLess(worst, 0.01)

    def test_the_reason_moves_with_the_patient(self):
        """The property the replaced prose structurally could not have.

        `WHY_NOT_REASONS` held exactly one sentence per arm. If a change ever
        collapses this back toward a constant, the card is asserting something
        about the arm rather than about the patient in front of it.
        """
        from collections import defaultdict

        per_arm = defaultdict(set)
        for _, entry in self.entries:
            per_arm[entry.action].add(entry.dominant_reason)
        self.assertTrue(per_arm)
        for arm, reasons in per_arm.items():
            with self.subTest(arm=arm):
                self.assertGreater(
                    len(reasons),
                    1,
                    "every patient reads the same reason for this arm",
                )

    def test_the_reference_arm_gap_is_the_leaders_own_blip(self):
        """`continue-current` is the reference, so the gap over it *is* the
        leader's blip — the quantity the attribution block already publishes.
        The two blocks are computed by different code from the same
        coefficients, and they agree exactly; if they ever stop, one of them has
        picked up a different source or a different basis."""
        checked = 0
        for decision, entry in self.entries:
            if entry.action != "continue-current":
                continue
            attribution = decision.explanation.attributions[0]
            if attribution.action != decision.selected.recommended_arm:
                continue
            checked += 1
            with self.subTest(patient=checked):
                self.assertAlmostEqual(
                    sum(entry.contributions.values()),
                    attribution.total_advantage,
                    places=3,
                )
        self.assertGreater(checked, 20)


if __name__ == "__main__":
    unittest.main()
