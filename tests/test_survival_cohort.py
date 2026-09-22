"""The survival generator, checked against its own declared truth.

`ra_cohort` earns its place by exporting `TRUE_BLIPS` and letting every
estimator be scored against it. This file is the same discipline one step
earlier: before any estimator exists, does the generator produce what it says?
A generator that grades its own homework is worth nothing, so the closed-form
oracle is checked against the sampler, and the declared hazard ratios are
checked against the times that come out.
"""

from __future__ import annotations

import collections
import math
import random
import statistics
import unittest

from treatmentrx.simulation import survival_cohort as sc


def _patients(n, seed):
    rng = random.Random(seed)
    return [sc.sample_baseline_features(rng) for _ in range(n)]


class ClosedFormTests(unittest.TestCase):
    """The oracle is arithmetic, so the arithmetic has to be right.

    `expected_months` is the Weibull mean and `oracle_action_value` sums it over
    lines. If it disagreed with what `sample_progression_months` actually draws,
    every regret measured against the oracle would carry that error and nothing
    downstream would show it.
    """

    FEATURES = {
        "biomarker_std": 0.3,
        "marker_positive": 1.0,
        "prior_line": 0.0,
        "acquired_resistance": 0.0,
    }

    def test_the_mean_matches_the_sampler(self):
        rng = random.Random(1)
        for arm in sc.SURVIVAL_ARMS:
            closed = sc.expected_months(self.FEATURES, arm)
            sampled = statistics.mean(
                sc.sample_progression_months(self.FEATURES, arm, rng)
                for _ in range(30_000)
            )
            with self.subTest(arm=arm):
                self.assertLess(
                    abs(sampled - closed) / closed,
                    0.02,
                    f"{arm}: closed form {closed:.3f} against sampled {sampled:.3f}",
                )

    def test_the_declared_hazard_ratio_is_recoverable_from_the_times(self):
        """Under PH, `E[T_a] / E[T_ref] = exp(-tau_a / shape)`.

        This is what makes the cohort usable: an estimator can in principle
        recover `psi`, so a failure to do so is the estimator's rather than the
        fixture's.
        """
        rng = random.Random(2)
        reference = statistics.mean(
            sc.sample_progression_months(self.FEATURES, sc.SURVIVAL_REFERENCE_ARM, rng)
            for _ in range(40_000)
        )
        for arm in sc.SURVIVAL_ARMS:
            if arm == sc.SURVIVAL_REFERENCE_ARM:
                continue
            sampled = statistics.mean(
                sc.sample_progression_months(self.FEATURES, arm, rng)
                for _ in range(40_000)
            )
            recovered = -sc.WEIBULL_SHAPE * math.log(sampled / reference)
            declared = sc.true_log_hazard_ratio(arm, self.FEATURES)
            with self.subTest(arm=arm):
                self.assertAlmostEqual(recovered, declared, delta=0.05)

    def test_the_reference_arm_carries_no_effect(self):
        """What identifies every other arm."""
        for features in _patients(50, seed=3):
            with self.subTest(features=features["biomarker_std"]):
                self.assertEqual(
                    sc.true_log_hazard_ratio(sc.SURVIVAL_REFERENCE_ARM, features), 0.0
                )

    def test_the_oracle_matches_a_simulation_of_itself(self):
        """Closed form against rollout, on the **same** patients.

        Comparing against a fresh patient draw reads about 1% apart, which is two
        samples rather than bias — on matched patients it is 0.06%, well inside
        Monte Carlo error. Worth pinning, because a biased oracle is the one
        error that makes every regret in a future study wrong by a constant.
        """
        patients = _patients(400, seed=4)
        closed = statistics.mean(
            sc.oracle_value(f, sc.DEFAULT_LINES) for f in patients
        )
        policy = sc.oracle_policy()
        rng = random.Random(5)
        simulated = []
        for features in patients:
            total = 0.0
            reps = 20
            for _ in range(reps):
                state, elapsed = dict(features), 0.0
                for line in range(sc.DEFAULT_LINES):
                    arm = policy(state, line)
                    elapsed += sc.sample_progression_months(state, arm, rng)
                    state = sc.advance_line(state, arm)
                total += elapsed
            simulated.append(total / reps)
        self.assertLess(abs(statistics.mean(simulated) - closed) / closed, 0.02)


class SequentialStructureTests(unittest.TestCase):
    """The property the generator exists to have, and the one it lacked.

    A first version advanced only `prior_line`, which does not depend on the
    arm, so the line-1 choice had no effect on the line-2 state. The myopic rule
    then matched the backward-induction optimum on **0 of 3,000** patients: the
    problem had decomposed into three independent choices and there was nothing
    for a DTR estimator to find. `RESISTANCE_INDUCING_ARMS` is the repair.
    """

    @classmethod
    def setUpClass(cls):
        cls.patients = _patients(1500, seed=11)

    def test_the_transition_depends_on_the_arm(self):
        """Directly: two arms must not lead to the same next state."""
        features = self.patients[0]
        potent = sc.advance_line(features, "arm-a")
        other = sc.advance_line(features, "arm-c")
        self.assertNotEqual(potent, other)
        self.assertGreater(
            potent["acquired_resistance"], other["acquired_resistance"]
        )

    def test_the_myopic_rule_does_not_reproduce_the_oracle(self):
        """The regression guard for the 0-of-3000 bug.

        If this ever returns to agreeing everywhere, the delayed effect has gone
        and the cohort has stopped being a sequential problem — whatever else
        still passes.
        """
        disagreements = sum(
            1
            for f in self.patients
            if sc.myopic_arm(f) != sc.oracle_arm(f, sc.DEFAULT_LINES)
        )
        share = disagreements / len(self.patients)
        self.assertGreater(
            share, 0.25, f"the myopic rule matches the oracle on {1 - share:.1%} of patients"
        )

    def test_the_potent_arm_is_saved_for_last(self):
        """The specific structure, not just that something is sequential.

        `arm-a` has the strongest single-line effect and leaves resistance
        behind, so the optimum avoids it while there is a future to pay and
        takes it once there is not.
        """
        early = collections.Counter(
            sc.oracle_arm(f, sc.DEFAULT_LINES) for f in self.patients
        )
        final = collections.Counter(sc.oracle_arm(f, 1) for f in self.patients)
        early_share = early["arm-a"] / len(self.patients)
        final_share = final["arm-a"] / len(self.patients)
        self.assertLess(early_share, 0.05, f"arm-a taken early {early_share:.1%}")
        self.assertGreater(final_share, 0.4, f"arm-a taken last {final_share:.1%}")

    def test_the_oracle_beats_the_myopic_rule_and_every_fixed_arm(self):
        """Measured in months, because "sequential" without a margin is a claim."""
        oracle = sc.rollout_value(sc.oracle_policy(), n=2500, with_attrition=False)
        myopic = sc.rollout_value(
            lambda f, i: sc.myopic_arm(f), n=2500, with_attrition=False
        )
        self.assertGreater(oracle, myopic + 1.0)
        for arm in sc.SURVIVAL_ARMS:
            fixed = sc.rollout_value(
                lambda f, i, a=arm: a, n=2000, with_attrition=False
            )
            with self.subTest(arm=arm):
                self.assertGreater(oracle, fixed)

    def test_the_delayed_cost_is_invisible_to_the_blip_basis(self):
        """Why the myopic rule loses rather than merely differing.

        `acquired_resistance` is charged on the prognostic surface and is not in
        `HAZARD_BASIS`, so an estimator that recovers every `psi` exactly still
        has to look ahead. That mirrors ALT in `ra_cohort`, which is in the
        treatment-free basis and not the blip basis.
        """
        self.assertNotIn("acquired_resistance", sc.HAZARD_BASIS)
        self.assertIn("acquired_resistance", sc._PROGNOSTIC)
        resistant = {
            "biomarker_std": 0.0,
            "marker_positive": 0.0,
            "prior_line": 1.0,
            "acquired_resistance": 1.0,
        }
        naive = dict(resistant, acquired_resistance=0.0)
        for arm in sc.SURVIVAL_ARMS:
            with self.subTest(arm=arm):
                # The blip is identical; only the prognostic surface moves.
                self.assertEqual(
                    sc.true_log_hazard_ratio(arm, resistant),
                    sc.true_log_hazard_ratio(arm, naive),
                )
                self.assertGreater(
                    sc.prognostic_log_hazard(resistant),
                    sc.prognostic_log_hazard(naive),
                )


class ConfoundingStructureTests(unittest.TestCase):
    """The property without which double robustness cannot be measured here.

    A treatment-free surface can only be *wrong in the way that matters* if
    there is a variable it can omit which moves assignment **and** the outcome
    and is not in the blip basis. Omitting a `HAZARD_BASIS` term instead does
    not create confounding — it destroys the blip's own identification, because
    `A * h(X)` becomes the only X-varying column and the blip terms absorb the
    prognosis.

    Measured on the version of this file before `performance_status` existed,
    no such variable was present: assignment was a softmax over
    `true_log_hazard_ratio` alone, so every confounder was in `HAZARD_BASIS`.
    That is why this is pinned rather than assumed.
    """

    BASE = {
        "biomarker_std": 0.0,
        "marker_positive": 0.0,
        "prior_line": 0.0,
        "acquired_resistance": 0.0,
        "performance_status": 0.0,
    }

    def _assignment_shift(self, name, low, high):
        first = sc.assignment_probabilities(dict(self.BASE, **{name: low}))
        second = sc.assignment_probabilities(dict(self.BASE, **{name: high}))
        return max(abs(first[arm] - second[arm]) for arm in sc.SURVIVAL_ARMS)

    def _outcome_shift(self, name, low, high):
        return abs(
            sc.prognostic_log_hazard(dict(self.BASE, **{name: high}))
            - sc.prognostic_log_hazard(dict(self.BASE, **{name: low}))
        )

    def test_there_is_a_confounder_outside_the_blip_basis(self):
        """All three conditions at once, which is what makes it omittable."""
        self.assertNotIn("performance_status", sc.HAZARD_BASIS)
        self.assertGreater(self._assignment_shift("performance_status", 0.0, 1.0), 0.05)
        self.assertGreater(self._outcome_shift("performance_status", 0.0, 1.0), 0.20)

    def test_the_confounding_is_arm_specific(self):
        """A shift common to every arm cancels in the softmax and confounds
        nothing, so the caution has to differ across arms or the covariate is
        prognostic only — which is the case this fixture was missing."""
        caution = set(sc._PRESCRIBING_CAUTION.values())
        self.assertGreater(len(caution), 1, "one caution for every arm confounds nothing")
        frail = sc.assignment_probabilities(dict(self.BASE, performance_status=1.0))
        fit = sc.assignment_probabilities(dict(self.BASE, performance_status=0.0))
        self.assertLess(
            frail["arm-a"], fit["arm-a"],
            "the aggressive arm must be avoided in frail patients",
        )

    def test_acquired_resistance_is_prognostic_only(self):
        """The negative control, and the reason a second covariate was needed.

        It moves the outcome and is outside `HAZARD_BASIS`, so it looks like a
        confounder — but it does not touch assignment, so omitting it from a
        surface costs efficiency and creates no bias for a weight to remove.
        """
        self.assertNotIn("acquired_resistance", sc.HAZARD_BASIS)
        self.assertGreater(self._outcome_shift("acquired_resistance", 0.0, 1.0), 0.5)
        self.assertEqual(self._assignment_shift("acquired_resistance", 0.0, 1.0), 0.0)

    def test_every_other_confounder_is_in_the_blip_basis(self):
        """Which is why it had to be added rather than found: the remaining
        variables that move assignment are exactly the blip basis terms."""
        for name, low, high in (("biomarker_std", -1.0, 1.0),
                                ("marker_positive", 0.0, 1.0),
                                ("prior_line", 0.0, 2.0)):
            with self.subTest(name=name):
                self.assertGreater(self._assignment_shift(name, low, high), 0.0)
                self.assertIn(name, sc.HAZARD_BASIS)


class RiskSetSelectionTests(unittest.TestCase):
    """Attrition selects *who reaches a later line*, not only when they are seen.

    Invariant 76. A slow progressor reaches the study horizon before ever
    starting line 2, so the later lines over-represent fast progressors — and
    fast means a high `biomarker_std`, which is a blip-basis term. Inverse
    weighting repairs observed event times within a line; it cannot repair which
    patients have a row in that line at all. This is a property of the
    generator, so it is pinned here rather than described in a docstring.

    Line 1's shift is the control and is *noise*, not an exact zero: turning
    censoring off lets more lines run per patient, so the RNG stream diverges
    and later patients draw different baselines. Measured, it falls +0.0199 at
    3x1200, +0.0051 at 4x2000, +0.0003 at 8x3000 while lines 2 and 3 stay put —
    which is what a quantity that is truly zero looks like next to two that are
    not.
    """

    SEEDS = (71, 72, 73, 74)
    SIZE = 2000

    @classmethod
    def setUpClass(cls):
        saved = (sc.STUDY_HORIZON_MONTHS, sc.COMPETING_RISK_RATE,
                 sc.LOSS_TO_FOLLOWUP_RATE)
        try:
            cls.censored = cls._profile()
            sc.STUDY_HORIZON_MONTHS = 1e9
            sc.COMPETING_RISK_RATE = 1e-12
            sc.LOSS_TO_FOLLOWUP_RATE = 1e-12
            cls.uncensored = cls._profile()
        finally:
            (sc.STUDY_HORIZON_MONTHS, sc.COMPETING_RISK_RATE,
             sc.LOSS_TO_FOLLOWUP_RATE) = saved
        cls.lines = sorted(cls.uncensored)

    @classmethod
    def _profile(cls):
        by_line = {}
        for seed in cls.SEEDS:
            for trajectory in sc.generate_survival_cohort(cls.SIZE, seed=seed):
                for stage in trajectory.stages:
                    by_line.setdefault(stage.line, []).append(
                        stage.features["biomarker_std"]
                    )
        return by_line

    def _shift(self, line):
        return (statistics.mean(self.censored[line])
                - statistics.mean(self.uncensored[line]))

    def test_the_lines_are_numbered_from_one(self):
        """The indexing this test reads, asserted rather than assumed.

        Filtering on `line == 0` selects nothing, and a fit on nothing returns
        zeros that look like a measurement — which is exactly how invariant 76's
        first attempt at a line-1 control produced a number (2.26, the sum of
        the true parameter magnitudes) rather than a comparison.
        """
        self.assertEqual(self.lines, [1, 2, 3])

    def test_the_first_line_is_not_selected(self):
        """Every patient has one, so there is nothing to select on."""
        self.assertLess(abs(self._shift(1)), 0.03, f"line 1 shifted {self._shift(1):+.4f}")
        self.assertEqual(len(self.censored[1]), len(self.uncensored[1]))

    def test_later_lines_are_selected_toward_fast_progressors(self):
        """The property no weight on observed event times can repair."""
        for line, floor in ((2, 0.05), (3, 0.10)):
            with self.subTest(line=line):
                shift = self._shift(line)
                self.assertGreater(
                    shift, floor,
                    f"line {line} biomarker shift {shift:+.4f} — has the attrition gone?",
                )
                self.assertLess(
                    len(self.censored[line]), len(self.uncensored[line]),
                    "no rows were lost to attrition",
                )

    def test_the_selection_grows_with_the_line(self):
        """Later lines sit further past the horizon, so more of the slow tail is
        gone. A flat profile would mean something other than attrition."""
        shifts = [self._shift(line) for line in self.lines]
        self.assertEqual(shifts, sorted(shifts), f"shifts {shifts} are not monotone")


class CohortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cohort = sc.generate_survival_cohort(300, seed=21)

    def test_it_is_reproducible_from_the_seed(self):
        again = sc.generate_survival_cohort(300, seed=21)
        self.assertEqual(
            [(s.arm, s.months, s.cause) for t in self.cohort for s in t.stages],
            [(s.arm, s.months, s.cause) for t in again for s in t.stages],
        )

    def test_all_three_ways_of_ending_follow_up_occur(self):
        """A fixture where one of them never happens cannot exercise the
        distinction the module is built around."""
        causes = collections.Counter(s.cause for t in self.cohort for s in t.stages)
        for cause in ("progression", "competing_death", "censored"):
            with self.subTest(cause=cause):
                self.assertGreater(causes[cause], 0, f"{cause} never occurs")

    def test_the_three_terminal_causes_partition_the_cohort(self):
        """`progression_observed` and `censored` are conveniences and neither
        stands in for the other — a competing death is in neither."""
        progressed = sum(t.progression_observed for t in self.cohort)
        censored = sum(t.censored for t in self.cohort)
        competing = sum(t.terminal_cause == "competing_death" for t in self.cohort)
        self.assertEqual(progressed + censored + competing, len(self.cohort))
        self.assertGreater(competing, 0, "the conflation this split exists for")

    def test_only_progression_continues_the_sequence(self):
        for trajectory in self.cohort:
            for stage in trajectory.stages[:-1]:
                with self.subTest(patient=trajectory.patient_index, line=stage.line):
                    self.assertEqual(stage.cause, "progression")

    def test_positivity_holds_for_every_patient(self):
        """Without this the cohort is a positivity violation dressed as
        confounding, and inverse-probability weights would be unbounded."""
        for features in _patients(500, seed=31):
            probabilities = sc.assignment_probabilities(features)
            with self.subTest(features=features["biomarker_std"]):
                self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=9)
                self.assertGreaterEqual(min(probabilities.values()), sc._ASSIGNMENT_FLOOR)

    def test_assignment_is_confounded_rather_than_random(self):
        """If it were uniform the cohort would need no propensity model at all,
        and would not test what it exists to test."""
        spreads = [
            max(sc.assignment_probabilities(f).values())
            - min(sc.assignment_probabilities(f).values())
            for f in _patients(200, seed=41)
        ]
        self.assertGreater(statistics.mean(spreads), 0.05)

    def test_the_recorded_propensity_is_the_one_used(self):
        for trajectory in self.cohort:
            for stage in trajectory.stages:
                expected = sc.assignment_probabilities(stage.features)[stage.arm]
                with self.subTest(patient=trajectory.patient_index, line=stage.line):
                    self.assertAlmostEqual(stage.propensity, expected, places=9)

    def test_no_line_starts_after_the_horizon(self):
        for trajectory in self.cohort:
            for stage in trajectory.stages:
                with self.subTest(patient=trajectory.patient_index, line=stage.line):
                    self.assertLess(stage.entry_month, sc.STUDY_HORIZON_MONTHS)

    def test_attrition_shortens_what_is_observed(self):
        """`rollout_value` reports what a patient accrues; `oracle_value` reports
        the value of the decision. The gap is retention, not error — invariant
        43's distinction on this scale — so the ordering must hold."""
        with_attrition = sc.rollout_value(sc.oracle_policy(), n=2000)
        without = sc.rollout_value(sc.oracle_policy(), n=2000, with_attrition=False)
        self.assertLess(with_attrition, without)


if __name__ == "__main__":
    unittest.main()
