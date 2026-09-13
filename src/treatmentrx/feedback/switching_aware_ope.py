"""v5.1 #3 (L6) — this patient's trajectory, and how far it deviated.

**What this used to be, and why it is gone.** It reported an
`iptw_policy_value`: the mean of this patient's observed outcomes, reweighted by
an "inverse probability of receiving the treatment as assigned" defined as
`adherence x 0.5^switched x 0.7^rescue`. Three things were wrong with it, and
measuring the cohort settled all three.

*The estimand did not exist.* Inverse-probability-of-treatment weighting recovers
a population policy value by correcting for confounded assignment. Applied to a
single patient's own three or four observed outcomes there is no counterfactual
being recovered — their outcomes are what they were. Reweighting them produces a
number that is not the patient's experience and not a policy value.

*Two of the three factors were dead.* Over 728 stage-rows from 200 simulated
patients, `adherence` was identically 1.000 (min = median = max) and
`rescue_therapy` fired exactly **zero** times. So `0.7` was never exercised by
anything but the hand-written demo, and `adherence` contributed nothing. The
entire weight reduced to `2.0 if switched else 1.0` — one hand-set constant.

*And that constant moved a reported number.* Switching fires on 471 of 728 rows,
so the reweighting shifted the value by 0.024 on average and up to **0.112**
against outcomes that sit around 0.64. An invented constant moving a published
quantity by eleven points is the failure this repo keeps removing.

Fitting the deviation model instead of hand-setting it would have fixed the
constant and left the first problem untouched, so the number is gone rather than
improved. What replaces it is the set of facts it was standing in front of: the
patient's observed mean, and a plain count of how far their trajectory departed
from what was prescribed. Those are descriptive, they are labelled descriptive,
and a reader can see exactly what they are.

`model_policy_value` stays. It is the estimator's held-out score, it is
model-level, and it must never be averaged with anything on this page —
invariant 14.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from treatmentrx.contracts import RegimeEstimate
from treatmentrx.domain import StageRecord


@dataclass(frozen=True)
class OPEResult:
    """This patient's observed trajectory. Descriptive, not an estimate.

    `observed_mean_outcome` is the arithmetic mean of the outcomes actually
    recorded — no weighting, no adjustment, no claim to be a policy value. With
    two to four visits per patient there is nothing an interval could be built
    on, and `n_stages` is reported so a reader can see that for themselves.
    """

    observed_mean_outcome: float
    n_stages: int
    stages_switched: int
    stages_with_rescue: int
    min_adherence: float
    note: str
    model_policy_value: float = 0.0
    deviation_flags: tuple[str, ...] = field(default=())


class SwitchingAwareOPE:
    """Summarises the trajectory in front of us, and says so.

    The name is kept because `FeedbackReceipt.ope` is a published field, but the
    docstring is the honest description: this is switching-*aware* in that it
    counts deviations, and it is not off-policy evaluation. The real off-policy
    evaluation is model-level and lives in `offline_evaluation.evaluate_policy`,
    measured on held-out patients.
    """

    def evaluate(self, stages: list[StageRecord], selected: RegimeEstimate) -> OPEResult:
        # The trailing open stage is the visit being decided now.  Its outcome is
        # the endpoint's UNKNOWN fallback, not an observation, and must never enter
        # a quantity labelled as observed.
        observed = [stage for stage in stages if stage.end_day is not None]
        if not observed:
            return OPEResult(
                observed_mean_outcome=0.0,
                n_stages=0,
                stages_switched=0,
                stages_with_rescue=0,
                min_adherence=1.0,
                model_policy_value=selected.policy_value,
                note="no completed stages with observed outcomes to summarise",
            )

        switched = sum(1 for s in observed if s.switching is not None and s.switching.switched)
        rescued = sum(
            1 for s in observed if s.switching is not None and s.switching.rescue_therapy
        )
        adherences = [
            s.switching.adherence for s in observed if s.switching is not None
        ]
        min_adherence = min(adherences) if adherences else 1.0

        flags = []
        if switched:
            flags.append(f"switched_at_{switched}_of_{len(observed)}_stages")
        if rescued:
            flags.append(f"rescue_therapy_at_{rescued}_stages")
        if min_adherence < 0.8:
            flags.append(f"adherence_below_0.8_(min_{min_adherence:.2f})")

        return OPEResult(
            observed_mean_outcome=round(sum(s.outcome for s in observed) / len(observed), 3),
            n_stages=len(observed),
            stages_switched=switched,
            stages_with_rescue=rescued,
            min_adherence=round(min_adherence, 3),
            model_policy_value=selected.policy_value,
            deviation_flags=tuple(flags),
            note=(
                "Descriptive summary of THIS patient's observed trajectory: the "
                "unweighted mean of their recorded outcomes and a count of the "
                "deviations from prescribed therapy. It is not a policy value and "
                "carries no interval — there are only "
                f"{len(observed)} completed visits. `model_policy_value` is the estimator's "
                "held-out score, is model-level, and must not be combined with "
                "anything else here."
            ),
        )


__all__ = ["OPEResult", "SwitchingAwareOPE"]
