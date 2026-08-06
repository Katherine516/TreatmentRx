"""Informative dropout, and what handling it correctly is worth.

The generator's true dropout hazard is available here, but only ever as an
answer key. The estimator never sees it — `CensoringModel` fits the hazard from
the covariates and the observed response, which is what an analyst would have.
"""

import unittest

from treatmentrx.estimation.censoring import CensoringModel
from treatmentrx.estimation.q_learning import QLearningModel
from treatmentrx.simulation.ra_cohort import (
    HIGH_BURDEN_ARMS,
    REFERENCE_ARM,
    TREATMENT_ARMS,
    TRUE_BLIPS,
    dropout_probability,
    generate_ra_cohort,
)

_COHORT = generate_ra_cohort(600, seed=7)


def _blip_error(model: QLearningModel) -> float:
    """Total absolute error of the terminal-stage blip intercepts."""
    terminal = model.n_stages - 1
    return sum(
        abs(model.blip_parameters(arm, terminal)["intercept"] - TRUE_BLIPS[arm][0])
        for arm in TREATMENT_ARMS
        if arm != REFERENCE_ARM
    )


class DropoutMechanismTests(unittest.TestCase):
    def test_trajectories_end_early(self):
        self.assertTrue(any(t.censored for t in _COHORT))
        self.assertTrue(any(t.n_observed < 3 for t in _COHORT))
        self.assertTrue(any(t.n_observed == 3 for t in _COHORT))

    def test_dropout_is_informative(self):
        """Patients who leave were responding worse — that is what biases things."""
        left = [t.stages[0].outcome for t in _COHORT if t.censored]
        stayed = [t.stages[0].outcome for t in _COHORT if not t.censored]
        self.assertLess(sum(left) / len(left), sum(stayed) / len(stayed))

    def test_dropout_is_differential_by_arm(self):
        """Burdensome arms are abandoned unless they are clearly working."""
        features = dict(_COHORT[0].stages[0].features)
        burdensome = sorted(HIGH_BURDEN_ARMS)[0]
        poor, good = 0.35, 0.85
        self.assertGreater(
            dropout_probability(features, poor, burdensome),
            dropout_probability(features, poor, REFERENCE_ARM),
        )
        # ...and the gap narrows sharply when the therapy is working.
        burden_gap_poor = dropout_probability(features, poor, burdensome) - dropout_probability(
            features, poor, REFERENCE_ARM
        )
        burden_gap_good = dropout_probability(features, good, burdensome) - dropout_probability(
            features, good, REFERENCE_ARM
        )
        self.assertGreater(burden_gap_poor, burden_gap_good)

    def test_timing_is_irregular_and_severity_driven(self):
        intervals = [
            (s.interval_days, s.features["das28"])
            for t in _COHORT
            for s in t.stages
            if s.interval_days is not None
        ]
        self.assertGreater(len({days for days, _ in intervals}), 20)
        sicker = [d for d, das28 in intervals if das28 > 6.0]
        milder = [d for d, das28 in intervals if das28 < 4.5]
        self.assertLess(sum(sicker) / len(sicker), sum(milder) / len(milder))


class CensoringModelTests(unittest.TestCase):
    def test_estimated_survival_tracks_the_true_hazard(self):
        """Fit from observables only, checked against the generator's answer key."""
        model = CensoringModel(_COHORT)
        errors = [
            abs(estimate - stage.uncensored_probability)
            for trajectory in _COHORT
            for stage, estimate in zip(trajectory.stages, model.survival_probabilities(trajectory))
        ]
        self.assertLess(sum(errors) / len(errors), 0.05)

    def test_weights_are_stabilised_and_bounded(self):
        model = CensoringModel(_COHORT)
        weights = [w for trajectory in _COHORT for w in model.weights(trajectory)]
        self.assertGreater(min(weights), 0.0)
        self.assertLess(max(weights), 10.0)
        # Stabilised weights average near one; unstabilised ones would not.
        self.assertAlmostEqual(sum(weights) / len(weights), 1.0, delta=0.2)

    def test_unlikely_returners_stand_in_for_those_who_left(self):
        """A patient who was unlikely to come back carries more than their own weight.

        That is the whole mechanism: the observed survivors represent the
        censored patients who resembled them.
        """
        model = CensoringModel(_COHORT)
        eligible = [t for t in _COHORT if t.n_observed >= 2]
        by_survival = sorted(eligible, key=lambda t: model.survival_probabilities(t)[1])
        unlikely, likely = by_survival[0], by_survival[-1]
        self.assertGreater(
            model.row_weight(unlikely, 0, needs_next=True),
            model.row_weight(likely, 0, needs_next=True),
        )
        self.assertGreater(model.row_weight(unlikely, 0, needs_next=True), 1.0)


class CensoredRowHandlingTests(unittest.TestCase):
    def test_a_censored_last_stage_is_not_a_terminal_decision(self):
        """The defect this handling exists to prevent.

        Treating a dropout's final visit as a terminal decision sets its target
        to the observed outcome alone — telling the model that the future is
        worth nothing for exactly the patients who left. Only the earlier-stage
        blocks are corrupted (a terminal-stage block never carries a
        pseudo-outcome), so that is where the damage shows.

        The *direction* is not asserted: it depends on which patients happened to
        be censored. The magnitude is the robust part, and it is comparable to
        the size of the blips themselves.
        """
        correct = QLearningModel(_COHORT, share_blip=False)
        defective = QLearningModel(_COHORT, share_blip=False, treat_censored_as_terminal=True)

        displacement = sum(
            abs(
                correct.blip_parameters(arm, 0)["intercept"]
                - defective.blip_parameters(arm, 0)["intercept"]
            )
            for arm in TREATMENT_ARMS
            if arm != REFERENCE_ARM
        )
        magnitude = sum(abs(TRUE_BLIPS[arm][0]) for arm in TREATMENT_ARMS if arm != REFERENCE_ARM)
        self.assertGreater(displacement, magnitude / 3.0)

    def test_the_defect_leaves_terminal_stage_blocks_alone(self):
        """Which is why it went unnoticed: the terminal blips look fine."""
        correct = QLearningModel(_COHORT, share_blip=False)
        defective = QLearningModel(_COHORT, share_blip=False, treat_censored_as_terminal=True)
        self.assertAlmostEqual(_blip_error(correct), _blip_error(defective), delta=0.05)

    def test_ipcw_does_not_make_things_worse(self):
        """Honest framing: the weights are a small correction here, not a rescue.

        With the censored rows handled and an outcome model that conditions on
        the covariates driving dropout, most of the bias is already gone. The
        weights matter when neither of those holds, which is the real-data case.
        """
        weighted = QLearningModel(_COHORT, share_blip=False, use_ipcw=True)
        unweighted = QLearningModel(_COHORT, share_blip=False, use_ipcw=False)
        self.assertLessEqual(_blip_error(weighted), _blip_error(unweighted) + 0.01)

    def test_ipcw_can_be_turned_off(self):
        model = QLearningModel(_COHORT, use_ipcw=False)
        self.assertIsNone(model.censoring)


if __name__ == "__main__":
    unittest.main()
