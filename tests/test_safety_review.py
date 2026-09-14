"""Blocking on an arm you did not recommend.

`SafetyLayer._status` blocks whenever the top-scored arm is infeasible, and it
reads that field regardless of whether Layer 3 actually recommended it. On an
equipoise decision `recommended_arm` is only the argmax — the layer has already
declared it indistinguishable from the others — so a contraindication on it
escalated a case where no recommendation existed to strike down. Measured over
two injected scenarios, 12 of 34 blocked cases were that.

The dangerous change here is the one these tests forbid: turning a *recommended*
arm's contraindication into a tidy list of alternatives. BLOCKED is the only
status that stops, and invariant 2 exists because a naming bug once removed an
arm and the runner-up was silently promoted.
"""

import copy
import unittest

from treatmentrx.data import DataLayer
from treatmentrx.decision import DecisionLayer
from treatmentrx.demo_data import sample_ra_bundle
from treatmentrx.domain import RecommendationStatus
from treatmentrx.estimation import EstimationLayer
from treatmentrx.safety import SafetyLayer
from treatmentrx.simulation.fhir_export import simulated_bundles


_DATA_LAYER = DataLayer()


def _contraindicate(bundle, code, value=True):
    """Attach an observation *inside* the patient's final stage.

    A fixed `effectiveDay` does not work on the simulated cohort, and getting it
    wrong is invisible: the injected contraindication silently fails to attach and
    the test then measures the fixture rather than the filter. Day 365 landed
    inside 11 of 20 final stages; one past the last recorded day landed inside
    **none**, because the temporal firewall drops anything after the decision
    point. So the day is read off the patient: build the state, take the final
    stage's start, and inject there.
    """
    day = _DATA_LAYER.build_patient_state(bundle).stages[-1].start_day
    bundle = copy.deepcopy(bundle)
    key = "valueBoolean" if isinstance(value, bool) else "valueQuantity"
    bundle["entry"].append(
        {
            "resource": {
                "resourceType": "Observation",
                "code": {"text": code},
                key: value if isinstance(value, bool) else {"value": value},
                "effectiveDay": day,
            }
        }
    )
    return bundle


def _pregnant(bundle):
    return _contraindicate(bundle, "pregnant", True)


def _hepatotoxic_risk(bundle):
    """High ALT removes the hepatotoxic arms — methotrexate and JAK."""
    return _contraindicate(bundle, "ALT", 180.0)


def _allergic_to(bundle, arm):
    bundle = copy.deepcopy(bundle)
    bundle["entry"].append(
        {"resource": {"resourceType": "AllergyIntolerance", "code": {"text": arm}}}
    )
    return bundle


def _safe_decisions(mutate, n=60, seed=606):
    data, estimation, decision, safety = (
        DataLayer(),
        EstimationLayer(),
        DecisionLayer(),
        SafetyLayer(),
    )
    out = []
    for bundle in simulated_bundles(n, seed=seed):
        from treatmentrx.data import DataContractError

        try:
            state = data.build_patient_state(mutate(bundle))
        except DataContractError:
            continue
        made = decision.decide(state, estimation.estimate(state))
        out.append((made, safety.apply(made, state)))
    return out


class BlockingIsPreservedTests(unittest.TestCase):
    """The half of the behaviour that must not move."""

    @classmethod
    def setUpClass(cls):
        cls.cases = _safe_decisions(lambda b: _allergic_to(b, "rituximab")) + _safe_decisions(
            _pregnant
        )

    def test_a_recommended_arm_that_is_infeasible_still_blocks(self):
        """Invariant 2. The model had a definite answer and safety struck it
        down; that is exactly the case a human must see, and the only status
        that stops is BLOCKED."""
        checked = 0
        for decision, safe in self.cases:
            if decision.status is not RecommendationStatus.RECOMMEND:
                continue
            if decision.recommended_arm in safe.feasible_arms:
                continue
            checked += 1
            self.assertEqual(
                safe.status,
                RecommendationStatus.BLOCKED,
                f"{decision.recommended_arm} was recommended, is infeasible, and did not block",
            )
        self.assertGreater(checked, 0, "no recommended-and-infeasible case in the fixture")

    def test_an_empty_feasible_set_always_blocks(self):
        for _, safe in self.cases:
            if safe.feasible_arms:
                continue
            self.assertEqual(safe.status, RecommendationStatus.BLOCKED)

    def test_the_contraindicated_arm_is_never_feasible(self):
        """The actual safety property, unchanged by any of this."""
        checked = 0
        for _, safe in _safe_decisions(_pregnant):
            self.assertNotIn("JAK-inhibitor", safe.feasible_arms)
            checked += 1
        self.assertGreater(checked, 0)

    def test_a_review_case_still_hides_every_removed_arm(self):
        for _, safe in _safe_decisions(_hepatotoxic_risk):
            for arm in safe.removed_arms:
                self.assertNotIn(arm, safe.feasible_arms)


    def test_a_recorded_allergy_still_hard_blocks(self):
        """The boundary of this change, stated rather than discovered.

        `rules.py` raises a `block`-severity flag when the top-scored arm
        conflicts with a recorded allergy, and it reads the same argmax
        `_status` used to. That is the same defect in a second place — but an
        allergy is the strongest contraindication the system records, blocking
        is its fail-safe direction, and moving two safety paths in one change is
        how a regression gets in. The arm itself is removed from the feasible set
        either way, so no patient is exposed; what is preserved is the halt.
        """
        blocked = [
            safe
            for decision, safe in _safe_decisions(lambda b: _allergic_to(b, "rituximab"))
            if decision.status is RecommendationStatus.EQUIPOISE
            and any(f.code == "allergy_contraindication" for f in safe.safety_flags)
        ]
        self.assertTrue(blocked, "no equipoise-with-allergy case in the fixture")
        for safe in blocked:
            self.assertEqual(safe.status, RecommendationStatus.BLOCKED)
            self.assertNotIn("rituximab", safe.feasible_arms)


class UndecidedContraindicationTests(unittest.TestCase):
    """The half that changes: no recommendation existed to block."""

    @classmethod
    def setUpClass(cls):
        # Two scenarios, because the branch has to be reachable by more than one
        # contraindication. Allergy is deliberately excluded — it blocks through a
        # `block`-severity rule rather than through `_status`, and that boundary is
        # pinned in `BlockingIsPreservedTests` rather than moved.
        cls.cases = [
            (decision, safe)
            for decision, safe in _safe_decisions(_pregnant)
            + _safe_decisions(_hepatotoxic_risk)
            if decision.status is RecommendationStatus.EQUIPOISE
            and decision.recommended_arm not in safe.feasible_arms
            and any(arm in set(safe.feasible_arms) for arm in decision.candidate_arms)
        ]

    def test_the_fixture_actually_contains_this_case(self):
        self.assertGreater(len(self.cases), 0)

    def test_it_routes_to_review_rather_than_blocking(self):
        for decision, safe in self.cases:
            self.assertEqual(
                safe.status,
                RecommendationStatus.REVIEW,
                f"decision was {decision.status.value}; nothing was recommended to block",
            )

    def test_no_arm_is_promoted(self):
        """The anti-substitution guard. Routing to review must not name a winner
        — that is the silent promotion invariant 2 was written about."""
        for _, safe in self.cases:
            self.assertNotEqual(safe.status, RecommendationStatus.RECOMMEND)
            surviving = [a for a in safe.decision.candidate_arms if a in set(safe.feasible_arms)]
            self.assertGreater(len(surviving), 0)

    def test_the_review_reports_no_recommended_arm(self):
        """`Recommendation.recommended_arm` must stay absent end to end. If a name
        appears there, something was promoted no matter what the status says."""
        from treatmentrx.orchestrator import TreatmentRxOrchestrator

        orchestrator = TreatmentRxOrchestrator()
        seen = 0
        for bundle in simulated_bundles(40, seed=606):
            recommendation = orchestrator.run(_pregnant(bundle))
            if recommendation.status is not RecommendationStatus.REVIEW:
                continue
            seen += 1
            self.assertIsNone(recommendation.recommended_arm)
        self.assertGreater(seen, 0)

    def test_the_removed_arm_is_named_not_just_dropped(self):
        for decision, safe in self.cases:
            removed = set(decision.candidate_arms) - set(safe.feasible_arms)
            self.assertTrue(removed)
            for arm in removed:
                self.assertIn(arm, safe.removed_arms, "a removal must carry its reason")

    def test_a_flag_says_why_this_was_not_blocked(self):
        for _, safe in self.cases:
            self.assertTrue(
                any(
                    flag.code == "contraindication_without_recommendation"
                    for flag in safe.safety_flags
                ),
                "the reason for not blocking has to be on the record",
            )

    def test_nothing_is_re_ranked(self):
        """Safety may delete from the set; it may not reorder it."""
        for decision, safe in self.cases:
            surviving = [a for a in decision.candidate_arms if a in set(safe.feasible_arms)]
            order = {arm: i for i, arm in enumerate(decision.candidate_arms)}
            self.assertEqual(surviving, sorted(surviving, key=order.get))


class ReviewNarrativeTests(unittest.TestCase):
    def test_the_card_does_not_blame_out_of_distribution(self):
        """REVIEW previously had one cause — an out-of-distribution patient — and
        the card says so in prose. A safety-driven review has a different reason
        and must not inherit that sentence."""
        from treatmentrx.orchestrator import TreatmentRxOrchestrator

        orchestrator = TreatmentRxOrchestrator()
        seen = 0
        for bundle in simulated_bundles(40, seed=606):
            recommendation = orchestrator.run(_pregnant(bundle))
            if recommendation.status is not RecommendationStatus.REVIEW:
                continue
            seen += 1
            # The prose lives in the patient summary, not the card — checking
            # the card passed vacuously and hid the defect.
            self.assertNotIn(
                "outside the patient patterns", recommendation.patient_summary
            )
            self.assertIn("not safe to use", recommendation.patient_summary)
            # Assert the property, not a literal: the set line has a different
            # phrasing when safety prunes it to a single survivor, and that
            # branch is exactly the one that must not read as a recommendation.
            card = recommendation.clinician_card.lower()
            self.assertIn("nothing has been substituted", card)
            self.assertTrue(
                "cannot separate:" in card or "remains in contention" in card,
                "the review card must still report what is left in contention",
            )
            self.assertIn("removed from this set by the safety layer", card)
        self.assertGreater(seen, 0, "no safety-driven review case produced")


class CardHonestyTests(unittest.TestCase):
    """No block of the card may contradict the status at the top of it."""

    @classmethod
    def setUpClass(cls):
        from treatmentrx.orchestrator import TreatmentRxOrchestrator

        orchestrator = TreatmentRxOrchestrator()
        cls.cards = {}
        for mutate in (_pregnant, _hepatotoxic_risk, lambda b: _allergic_to(b, "rituximab")):
            for bundle in simulated_bundles(30, seed=606):
                recommendation = orchestrator.run(mutate(bundle))
                cls.cards.setdefault(recommendation.status, recommendation)

    def test_a_blocked_card_never_claims_it_was_not_blocked(self):
        """The regression this class exists for.

        The safety flag raised for an undecided contraindication used to say
        "routed to review ... not blocked". `apply()` can still upgrade to
        BLOCKED afterwards when a rule raises a block-severity flag — an allergy
        does — and the card then read "blocked" in its heading and "not blocked"
        in its body. A flag reports what it observed; the status says itself.
        """
        blocked = self.cards.get(RecommendationStatus.BLOCKED)
        self.assertIsNotNone(blocked, "no blocked case in the fixture")
        card = blocked.clinician_card.lower()
        self.assertIn("blocked", card)
        self.assertNotIn("not blocked", card)
        self.assertNotIn("routed to review", card)

    def test_a_blocked_card_gives_the_reviewer_something_to_review(self):
        """BLOCKED escalates to a person by definition, and used to hand them one
        line while the candidate set, the contrast and the removals sat unused."""
        blocked = self.cards[RecommendationStatus.BLOCKED]
        card = blocked.clinician_card
        self.assertGreater(len(card.split("\n\n")), 3, "still a one-liner")
        self.assertIn("Removed by the safety layer:", card)
        self.assertIn("not a list of alternatives", card)

    def test_no_card_offers_an_arm_the_safety_layer_removed(self):
        for status, recommendation in self.cards.items():
            removed = set(recommendation.audit_event.get("removed_arms", {}))
            if not removed:
                continue
            with self.subTest(status=status.value):
                self.assertIsNone(
                    recommendation.recommended_arm
                    if recommendation.recommended_arm in removed
                    else None
                )

    def test_an_attribution_for_a_removed_arm_says_it_was_removed(self):
        """Its subject is the top-scored arm, which on a review card is sometimes
        the arm safety just took away. Unqualified, it reads as advocacy for
        something the patient must not receive."""
        for status, recommendation in self.cards.items():
            removed = set(recommendation.audit_event.get("removed_arms", {}))
            top = recommendation.top_scored_arm
            if top not in removed:
                continue
            with self.subTest(status=status.value):
                for block in recommendation.clinician_card.split("\n\n"):
                    if block.startswith("Estimated advantage"):
                        self.assertIn("removed by the safety layer", block)



class DelayedToxicitySubjectTests(unittest.TestCase):
    """A warn flag may not name an arm nobody recommended.

    `_delayed_toxicity` read `decision.recommended_arm` unconditionally and said
    "before continuing" — but on an equipoise decision that field is the argmax
    and Layer 3 has already declared it inseparable from the candidate set.
    Invariant 2's defect, in a rule rather than a status.
    """

    @classmethod
    def setUpClass(cls):
        from treatmentrx.data.contract import DataContractError
        from treatmentrx.orchestrator import TreatmentRxOrchestrator
        from treatmentrx.simulation.fhir_export import simulated_bundles

        orchestrator = TreatmentRxOrchestrator()
        cls.flagged = []
        for bundle in simulated_bundles(120, seed=991):
            try:
                recommendation = orchestrator.run(bundle)
            except DataContractError:
                continue
            for flag in recommendation.safety_flags:
                if flag.code == "delayed_toxicity_accumulation":
                    cls.flagged.append((recommendation, flag))

    def test_the_flag_fires_somewhere(self):
        self.assertTrue(self.flagged, "no patient triggered the rule; fixture changed")

    def test_a_named_arm_is_always_the_published_one(self):
        for recommendation, flag in self.flagged:
            if flag.affected_arm is None:
                continue
            with self.subTest(arm=flag.affected_arm):
                self.assertEqual(flag.affected_arm, recommendation.recommended_arm)

    def test_an_undecided_patient_gets_an_observation_not_an_instruction(self):
        """"before continuing" asserts an intent that does not exist yet."""
        for recommendation, flag in self.flagged:
            if recommendation.recommended_arm is not None:
                continue
            with self.subTest(status=recommendation.status.value):
                self.assertIsNone(flag.affected_arm)
                self.assertNotIn("before continuing", flag.message)
                self.assertIn("No arm has been recommended", flag.message)

    def test_the_warning_is_not_suppressed_by_a_benign_argmax(self):
        """Gating on the argmax lost the warning for a set that still held risk.

        The trajectory evidence — cumulative hepatotoxic exposure plus a rising
        ALT — does not depend on which arm happens to score highest, so neither
        should the flag.
        """
        undecided = [r for r, f in self.flagged if r.recommended_arm is None]
        self.assertGreater(
            len(undecided),
            1,
            "the argmax gate should no longer be suppressing these warnings",
        )

if __name__ == "__main__":
    unittest.main()
