"""Does the pipeline survive data it was not fit on?

Two things these tests protect. The shift mechanism must not perturb the default
cohort by so much as a bit — every seeded result in the repo depends on it. And
the transfer claims themselves are asserted as relationships (this holds, that
one moves) rather than as constants, because the numbers will drift.
"""

import unittest

from treatmentrx.feedback.transfer import (
    SITES,
    _abstention,
    _decisions_at_site,
    _site_patient_grid,
    evaluate_site,
    transfer_coverage,
)
from treatmentrx.simulation.ra_cohort import CohortShift, generate_ra_cohort


def _digest(cohort):
    return [
        (stage.arm, round(stage.outcome, 12), round(stage.features["das28"], 12))
        for trajectory in cohort
        for stage in trajectory.stages
    ]


class ShiftMechanismTests(unittest.TestCase):
    def test_an_absent_shift_changes_nothing(self):
        """Determinism is a hard constraint: a given commit must always produce
        the same recommendation. Adding a shift parameter must be invisible when
        it is not used."""
        baseline = _digest(generate_ra_cohort(200, 7))
        self.assertEqual(baseline, _digest(generate_ra_cohort(200, 7, shift=None)))
        self.assertEqual(baseline, _digest(generate_ra_cohort(200, 7, shift=CohortShift())))

    def test_each_knob_moves_only_what_it_names(self):
        base = generate_ra_cohort(300, 11)

        def mean(cohort, key):
            values = [s.features[key] for t in cohort for s in t.stages]
            return sum(values) / len(values)

        sicker = generate_ra_cohort(300, 11, shift=CohortShift(das28_range=(6.0, 9.0)))
        self.assertGreater(mean(sicker, "das28"), mean(base, "das28") + 1.0)

        sero = generate_ra_cohort(300, 11, shift=CohortShift(anti_ccp_rate=0.05))
        self.assertLess(mean(sero, "anti_ccp"), 0.25)

    def test_a_prescribing_tilt_moves_the_arm_mix(self):
        base = generate_ra_cohort(400, 13)
        tilted = generate_ra_cohort(
            400, 13, shift=CohortShift(assignment_tilt=(("TNF-inhibitor", 2.5),))
        )

        def share(cohort, arm):
            arms = [s.arm for t in cohort for s in t.stages]
            return arms.count(arm) / len(arms)

        self.assertGreater(share(tilted, "TNF-inhibitor"), share(base, "TNF-inhibitor") + 0.10)

    def test_a_dropout_shift_moves_retention(self):
        def stages_per_patient(cohort):
            return sum(len(t.stages) for t in cohort) / len(cohort)

        heavy = generate_ra_cohort(300, 17, shift=CohortShift(dropout_shift=1.5))
        light = generate_ra_cohort(300, 17, shift=CohortShift(dropout_shift=-1.5))
        self.assertLess(stages_per_patient(heavy), stages_per_patient(light))

    def test_only_blip_scale_touches_the_estimand(self):
        """The separation the whole study rests on. Case mix, prescribing and
        retention must leave TRUE_BLIPS alone, or a degradation cannot be
        attributed to the pipeline rather than to the model being wrong."""
        for shift in SITES:
            with self.subTest(site=shift.name):
                self.assertEqual(shift.shifts_the_estimand, shift.blip_scale != 1.0)
        preserving = [s for s in SITES if not s.shifts_the_estimand]
        self.assertGreaterEqual(len(preserving), 5)
        self.assertTrue(any(s.shifts_the_estimand for s in SITES), "the bound case must exist")


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = evaluate_site(SITES[0], n=200, rollouts=600)

    def test_the_baseline_site_reproduces_the_training_process(self):
        """Site zero is the same generating process, so it is the control: if it
        does not look like the numbers everything else in the repo reports, the
        harness is measuring the wrong thing."""
        self.assertLess(self.baseline.calibration_error, 0.02)
        self.assertGreater(self.baseline.ipw_improvement, 0.0)
        # Two corrections move this band and both are recorded rather than
        # tuned away. The family-wise all-pairs level is deliberately more
        # conservative than the former pointwise top-two threshold and pushed it
        # up; keeping the dWOLS cross-arm covariance instead of adding the two
        # arms' variances as if independent then took ~11 points back off, by
        # removing width that was never a safety margin — the contrast SE ran
        # 1.24x the estimator's actual spread and now runs 0.95x.
        self.assertTrue(0.55 < self.baseline.abstention_rate < 0.85)

    def test_the_data_contract_accepts_every_site(self):
        """A site whose patients Layer 1 refuses is a transfer failure of a more
        basic kind than a wide interval, and it would silently shrink every
        other number here."""
        for shift in SITES:
            with self.subTest(site=shift.name):
                _, rejected = _decisions_at_site(shift, 60, 4242)
                self.assertEqual(rejected, 0)

    def test_the_policy_beats_local_practice_at_every_site(self):
        for shift in SITES:
            with self.subTest(site=shift.name):
                result = evaluate_site(shift, n=150, rollouts=600)
                self.assertGreater(
                    result.rollout_improvement,
                    0.0,
                    f"{shift.name}: agent {result.rollout_agent} vs local "
                    f"behaviour {result.rollout_behaviour}",
                )

    def test_abstention_is_the_fragile_number(self):
        """The headline finding. The baseline rate is population-specific,
        and a deployment reading it as a property of the method will be wrong by
        tens of points."""
        rates = []
        for shift in SITES:
            if shift.shifts_the_estimand:
                continue
            rows, _ = _decisions_at_site(shift, 200, 4242)
            rates.append(_abstention(rows))
        self.assertGreater(max(rates) - min(rates), 0.20, f"rates were {rates}")

    def test_calibration_detects_an_estimand_shift_that_policy_value_misses(self):
        """The sharpest result, and the one a deployment should act on.

        Scaling every blip preserves the *ordering* of the arms, so the
        recommendation stays right and the policy value stays good while the
        numbers attached to it stop meaning anything. Calibration is what sees
        it — so a monitor that watches policy value alone is blind to exactly
        the shift that makes the reported Q-values wrong.
        """
        shifted = next(s for s in SITES if s.shifts_the_estimand)
        result = evaluate_site(shifted, n=200, rollouts=600)
        self.assertGreater(result.rollout_improvement, 0.0, "the policy should survive")
        self.assertGreater(
            result.calibration_error,
            5 * self.baseline.calibration_error,
            "calibration should register the estimand shift the policy misses",
        )


class TransferCoverageTests(unittest.TestCase):
    def test_coverage_is_replicated_over_fits_not_over_patients(self):
        """A single fit cannot produce coverage.

        Every patient's interval is built from the same parameters, so their
        errors are correlated and the fraction covered is a property of the one
        draw taken. Measured that way the baseline site reads 87% against the 98%
        `cli coverage` reports for the same rule. This checks the replicated
        version lands where the established study does.
        """
        results = transfer_coverage(SITES[:1], patients_per_site=6, replications=15)
        baseline = results[SITES[0].name]
        self.assertGreater(baseline.coverage, 0.90)

    def test_the_estimand_shifted_site_reports_no_coverage(self):
        """Its truth comes from an oracle that no longer describes the site. A
        number that looks like coverage but is scored against the wrong target is
        worse than no number."""
        shifted = tuple(s for s in SITES if s.shifts_the_estimand)
        self.assertEqual(transfer_coverage(shifted, patients_per_site=4, replications=3), {})

    def test_the_patient_grid_samples_the_site_it_names(self):
        sick = _site_patient_grid(
            CohortShift(name="sick", das28_range=(7.0, 9.0)), 8, 99
        )
        mild = _site_patient_grid(
            CohortShift(name="mild", das28_range=(2.5, 4.0)), 8, 99
        )
        self.assertEqual(len(sick), 8)
        self.assertGreater(
            sum(f["das28"] for f in sick) / 8, sum(f["das28"] for f in mild) / 8
        )


if __name__ == "__main__":
    unittest.main()
