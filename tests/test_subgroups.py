"""Who the agent abstains on, and whether the decomposition means anything.

The pooled abstention rate hides a large spread across disease-activity
tertiles. These tests pin the two things that make the stratified version
readable: that the counterfactual actually isolates precision from signal, and
that a stratum too small to support a rate says so instead of printing one.
"""

import unittest

from treatmentrx.feedback.subgroups import (
    MIN_CELL,
    Cell,
    PatientOutcome,
    score_patients,
    subgroup_report,
)


def _outcome(difference, standard_error, true_gap=0.05, exact=False, **features):
    """A patient whose `abstained` flag is *derived* from the separation rule.

    Hand-labelling it invites the fixture to disagree with the rule it is meant
    to exercise — the first version of this passed `abstained=True` for a
    contrast of 0.05 at SE 0.02, which the rule separates at 0.049, and the
    identity test would have been vacuous had the values gone the other way.
    """
    from treatmentrx.estimation.inference import SANDWICH_INFLATION

    base = {"das28": 4.5, "crp": 30.0, "anti_ccp": 0.0, "prior_tnf": 0.0, "egfr": 90.0, "alt": 30.0}
    base.update(features)
    half_width = 1.96 * standard_error  # what a 95% sandwich interval would give
    bar = half_width if exact else half_width * SANDWICH_INFLATION
    return PatientOutcome(
        features=base,
        stage_index=2,
        abstained=abs(difference) <= bar,
        difference=difference,
        standard_error=standard_error,
        half_width=half_width,
        exact=exact,
        true_gap=true_gap,
    )


class CounterfactualTests(unittest.TestCase):
    """`abstain_rate_at_pooled_se` has to move for precision and only precision."""

    def test_substituting_a_smaller_standard_error_can_only_help(self):
        cell = Cell("test", [_outcome(0.05, 0.03) for _ in range(10)])
        self.assertGreaterEqual(
            cell.abstain_rate, cell.counterfactual_abstain_rate(0.005)
        )

    def test_substituting_the_cells_own_error_reproduces_its_rate(self):
        """The identity case. If this drifts, the decomposition is not additive."""
        outcomes = [
            _outcome(0.08, 0.02),
            _outcome(0.01, 0.02),
            _outcome(0.04, 0.02),
        ]
        cell = Cell("test", outcomes)
        self.assertAlmostEqual(
            cell.counterfactual_abstain_rate(0.02), cell.abstain_rate, places=9
        )

    def test_precision_excess_is_zero_when_the_cell_has_average_precision(self):
        cell = Cell("test", [_outcome(0.03, 0.02) for _ in range(5)])
        self.assertAlmostEqual(cell.precision_excess(0.02), 0.0, places=9)

    def test_an_exact_interval_is_not_charged_the_inflation(self):
        """`conservative=True` intervals have already paid the correction.

        Charging them again would make the counterfactual claim a precision
        penalty that the decision layer never applied.
        """
        difference, error = 0.05, 0.024
        sandwich = Cell("sandwich", [_outcome(difference, error)])
        exact = Cell("exact", [_outcome(difference, error, exact=True)])
        self.assertGreaterEqual(
            sandwich.counterfactual_abstain_rate(error),
            exact.counterfactual_abstain_rate(error),
        )


class SmallCellTests(unittest.TestCase):
    def test_an_axis_with_one_usable_stratum_declines_a_verdict(self):
        """Two patients in a cell is a tally, not a rate."""
        report = subgroup_report(n_patients=6, seed=11)
        for name, block in report["axes"].items():
            with self.subTest(axis=name):
                usable = [s for s in block["strata"] if not s["underpowered"]]
                if len(usable) < 2:
                    self.assertFalse(block["verdict"]["resolved"])

    def test_cells_below_the_threshold_are_flagged(self):
        cell = Cell("tiny", [_outcome(0.03, 0.02)])
        self.assertTrue(cell.as_dict(0.02)["underpowered"])
        big = Cell("big", [_outcome(0.03, 0.02) for _ in range(MIN_CELL)])
        self.assertFalse(big.as_dict(0.02)["underpowered"])


class StratifiedAbstentionTests(unittest.TestCase):
    """The findings themselves, asserted as relationships rather than constants."""

    @classmethod
    def setUpClass(cls):
        cls.outcomes = score_patients(n_patients=240)
        cls.report = subgroup_report(n_patients=240)

    def test_every_stratum_is_covered_exactly_once_per_axis(self):
        total = self.report["pooled"]["n"]
        for name, block in self.report["axes"].items():
            with self.subTest(axis=name):
                self.assertEqual(sum(s["n"] for s in block["strata"]), total)

    def test_abstention_is_not_uniform_across_disease_activity(self):
        """The headline rate hides the spread this whole module exists to show."""
        strata = self.report["axes"]["das28"]["strata"]
        rates = [s["abstain_rate"] for s in strata]
        self.assertGreater(
            max(rates) - min(rates),
            0.20,
            f"expected a wide spread across das28 tertiles, got {rates}",
        )

    def test_the_most_abstaining_stratum_has_the_closest_arms(self):
        """Abstention has to track the truth, stratum by stratum and not just
        on average. If the group declined most often were the group with the
        *widest* true gaps, the agent would be declining for the wrong reason."""
        for name in ("das28", "anti_ccp"):
            with self.subTest(axis=name):
                strata = [
                    s for s in self.report["axes"][name]["strata"] if not s["underpowered"]
                ]
                worst = max(strata, key=lambda s: s["abstain_rate"])
                best = min(strata, key=lambda s: s["abstain_rate"])
                self.assertLess(
                    worst["mean_true_gap"],
                    best["mean_true_gap"],
                    f"{name}: {worst['stratum']} abstains most but has wider true gaps",
                )

    def test_abstention_is_earned_within_every_usable_stratum(self):
        """`cli audit` establishes this pooled. Pooled can be true while a
        subgroup has it backwards, which is exactly what a subgroup analysis is
        for."""
        for name, block in self.report["axes"].items():
            for stratum in block["strata"]:
                if stratum["underpowered"]:
                    continue
                with self.subTest(axis=name, stratum=stratum["stratum"]):
                    members = self._members(name, stratum["stratum"])
                    declined = [o.true_gap for o in members if o.abstained]
                    recommended = [o.true_gap for o in members if not o.abstained]
                    if len(declined) < 5 or len(recommended) < 5:
                        continue
                    self.assertLess(
                        sum(declined) / len(declined),
                        sum(recommended) / len(recommended),
                        "declined patients should have closer arms than recommended ones",
                    )

    def _members(self, axis, stratum_name):
        from treatmentrx.feedback.subgroups import _axes

        for cell in _axes(self.outcomes)[axis]:
            if cell.name == stratum_name:
                return cell.outcomes
        raise AssertionError(f"no stratum {stratum_name!r} on axis {axis!r}")

    def test_the_report_does_not_call_itself_a_fairness_audit(self):
        """There are no protected attributes in this cohort to audit.

        Naming it one would be the exact failure this repo keeps removing: a
        label claiming more than the measurement supports. `fairness_clean` on
        the validation ladder is a separate and still-unmet criterion.
        """
        note = self.report["note"].lower()
        self.assertIn("not a fairness audit", note)


if __name__ == "__main__":
    unittest.main()
