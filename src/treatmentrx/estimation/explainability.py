"""v5.1 #9 (L3) — Model-level explainability.

Faithful, model-derived explanations: blip attributions, why-not table, local
counterfactuals, and assumption sensitivity. Layer 5 *renders* these in
language; it never invents them.
"""

from __future__ import annotations

import math

from treatmentrx.estimation.basis import BLIP_BASIS, blip_basis
from treatmentrx.estimation.features import model_features
from treatmentrx.contracts import RegimeEstimate
from treatmentrx.domain import (
    AssumptionSensitivity,
    BlipAttribution,
    CounterfactualProbe,
    ExplanationBundle,
    StageRecord,
    WhyNotEntry,
)


#: Clinical language for each blip-basis term, keyed on the **covariate** and not
#: on the arm. That swap is the whole point. The replaced `WHY_NOT_REASONS` was
#: keyed on the arm alone, so every patient got the same sentence for a given arm
#: while the gap beside it varied correctly — and one of those sentences, "organ-
#: function / safety profile reduces net benefit" for `JAK-inhibitor`, named a
#: quantity `BLIP_BASIS` does not contain. Here the model chooses which term
#: dominates and with what sign; this map only supplies the words.
#:
#: A covariate with no entry falls back to its own identifier rather than
#: borrowing a neighbour's sentence, and `tests/test_candidate_set.py` asserts the
#: map covers `BLIP_BASIS` — a basis term added without a phrase would otherwise
#: reach a clinician card as a bare variable name.
BLIP_TERM_LANGUAGE = {
    "intercept": "this arm's baseline effect",
    "das28_std": "disease activity",
    "anti_ccp": "anti-CCP status",
    "prior_tnf": "prior TNF exposure",
}

#: A negative term is shown as an offset only when it is worth a reader's
#: attention next to the leading one. Purely presentational — `contributions`
#: always carries every term.
_OFFSET_SHARE = 0.2


def _e_value(effect: float, outcome_sd: float) -> float:
    """VanderWeele-Ding E-value for a standardised continuous effect.

    `RR ~ exp(0.91 * d)` is their published approximation for a continuous
    outcome; the bound is then `RR + sqrt(RR (RR - 1))`. A zero effect gives
    exactly 1.0, which is the honest reading — nothing has to be explained away.
    """
    if effect <= 0.0 or outcome_sd <= 0.0:
        return 1.0
    risk_ratio = math.exp(0.91 * (effect / outcome_sd))
    return round(risk_ratio + math.sqrt(max(risk_ratio * (risk_ratio - 1.0), 0.0)), 3)


class ModelExplainer:
    def explain(
        self,
        selected: RegimeEstimate,
        candidates: list[RegimeEstimate],
        stages: list[StageRecord],
        contrast=None,
        arm_contrasts=None,
    ) -> ExplanationBundle:
        """`contrast` is the decision's own top-two interval, passed in.

        It used to be absent, and `_sensitivity` built its own quantity out of
        `selected.q_values` instead — the same incoherence invariant 16 removed
        from the interval itself, where a number beside the decision described
        something the decision did not use. The caller already has the contrast
        when it calls this; there was never a reason to reconstruct a worse one.

        `arm_contrasts` is the same argument for the why-not table: the decision
        layer already holds a model-averaged interval for the leader against
        *every* arm, because that is what the candidate set is built from.
        """
        return ExplanationBundle(
            attributions=self._attributions(selected, stages),
            why_not=self._why_not(selected, stages, arm_contrasts),
            counterfactuals=self._counterfactuals(selected, stages),
            sensitivity=self._sensitivity(contrast),
        )

    def _attributions(self, selected: RegimeEstimate, stages: list[StageRecord]) -> list[BlipAttribution]:
        """Decompose the recommended arm's blip into its per-covariate terms.

        `total_advantage` is literally psi . h(X) — the model's estimate of how
        much better this arm is than continuing current therapy for this
        patient. Each contribution is one term of that sum, so the parts add up
        to the whole and the explanation cannot drift from the model.
        """
        basis = dict(zip(BLIP_BASIS, blip_basis(model_features(stages))))
        prefix = f"psi:{selected.recommended_arm}:"
        contributions: dict[str, float] = {}
        for key, coefficient in selected.coefficients.items():
            if not key.startswith(prefix):
                continue
            if not isinstance(coefficient, (int, float)) or isinstance(coefficient, bool):
                continue
            covariate = key[len(prefix):]
            if covariate in basis:
                contributions[covariate] = round(coefficient * basis[covariate], 4)
        total = round(sum(contributions.values()), 4)
        return [
            BlipAttribution(
                action=selected.recommended_arm,
                contributions=dict(sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)),
                total_advantage=total,
            )
        ]

    def _why_not(
        self,
        selected: RegimeEstimate,
        stages: list[StageRecord],
        arm_contrasts=None,
    ) -> list[WhyNotEntry]:
        """Why each non-recommended arm lost, decomposed from the model that ranked it.

        **The reason used to be prose keyed on the arm, and it could not vary
        with the patient.** `WHY_NOT_REASONS` held one sentence per arm, so two
        patients with opposite covariates read the same explanation while the gap
        printed beside it moved correctly. The module docstring above calls this
        table model-derived and says Layer 5 never invents it; the gap was, the
        sentence was not. Measured over 120 patients, **every** card carried one.

        One of those sentences was worse than generic. `JAK-inhibitor` printed
        "organ-function / safety profile reduces net benefit" — on **20 of 120**
        cards — and `BLIP_BASIS` is `(intercept, das28_std, anti_ccp, prior_tnf)`,
        which contains no organ-function term at all. The card asserted a
        mechanism the model has no parameter for. Worse, once the entries were
        restricted to arms the safety layer did *not* remove, that safety-flavoured
        sentence printed only for patients whose organ function had cleared every
        rule on the same card.

        What replaces it is the gap's own decomposition. Each arm's blip is
        one-vs-reference over `BLIP_BASIS`, so the contrast between the leader and
        any arm is `(psi_leader - psi_arm) . h(X)` and splits term by term. Those
        terms are read from `selected.coefficients` — the same place
        `_attributions` reads them, so both blocks on the card describe the same
        member, the one `attribution_source` names. The reference arm carries no
        psi and needs none: the gap over continuing current therapy *is* the
        leader's blip, and the two blocks agree to the last decimal there.

        `q_gap` stays the decision's own model-averaged contrast (invariant 54),
        so it still matches the separation line. The decomposition is one
        member's, so it reconstructs that gap to within the two members'
        disagreement rather than exactly: over 600 entries from 120 patients the
        residual runs mean **0.0100**, median 0.0072, max **0.0575**, against
        gaps that reach 0.306. Reporting the decomposed total instead would make
        the two numbers on the card disagree, which is the defect invariant 54
        removed.

        **That residual is a disagreement, not a scale error, and only because
        the source is dWOLS.** `Q-Pooled` publishes a stage psi from a
        value-to-go fit and does not divide it by the remaining horizon, while
        `q_gap` is per-remaining-visit — so a decomposition sourced from it would
        be off by that horizon wherever the horizon is not 1. Measured at
        `stage_index` 1, Q-Pooled's decomposition runs **1.54x to 3.49x** the gap
        it claims to explain, against dWOLS tracking it closely; at the terminal
        block, where the horizon is 1, both agree. `attribution_source` is
        dWOLS-Shared for all 120 patients on the deployed fit, so nothing served
        crosses scales today — but the BMA margin that decides it is 0.003
        (invariant 56), so `tests/test_candidate_set.py` asserts the
        reconstruction at every served stage rather than trusting that margin.
        That is a guard, not a repair: the repair belongs where the scale is
        known, in what `coefficient_summary` publishes.

        Ordered by the gap printed, because the renderer shows the first two.
        """
        arm_contrasts = arm_contrasts or {}
        basis = dict(zip(BLIP_BASIS, blip_basis(model_features(stages))))
        leader = selected.recommended_arm
        leader_psi = self._blip_parameters(selected, leader)
        best = selected.q_values[leader]

        ranked: list[tuple[float, str, dict[str, float]]] = []
        for action, value in selected.q_values.items():
            if action == leader:
                continue
            test = arm_contrasts.get(action)
            gap = test.difference if test is not None else best - value
            arm_psi = self._blip_parameters(selected, action)
            contributions = {
                name: round((leader_psi.get(name, 0.0) - arm_psi.get(name, 0.0)) * weight, 4)
                for name, weight in basis.items()
            }
            ranked.append((gap, action, contributions))

        return [
            WhyNotEntry(
                action=action,
                q_gap=round(gap, 3),
                dominant_reason=self._dominant_reason(contributions),
                contributions=dict(
                    sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
                ),
            )
            for gap, action, contributions in sorted(ranked)
        ]

    @staticmethod
    def _blip_parameters(selected: RegimeEstimate, action: str) -> dict[str, float]:
        """This arm's psi, as the estimate carries it.

        Empty for the reference arm, which has no blip by construction rather
        than by omission — subtracting nothing is the right answer there.
        """
        prefix = f"psi:{action}:"
        return {
            key[len(prefix):]: value
            for key, value in selected.coefficients.items()
            if key.startswith(prefix)
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        }

    @staticmethod
    def _dominant_reason(contributions: dict[str, float]) -> str:
        """The term that most moves the gap toward the leader, named and signed.

        The largest *positive* term, not the largest absolute one: the question
        the card is answering is why this arm lost, and a term working in the
        arm's favour does not answer it. It is shown as an offset instead, when
        it is big enough next to the leading term to change how that term reads —
        a leading contribution of +0.136 against a gap of 0.086 is confusing
        until the -0.079 pulling the other way is named too.

        Falls back to the largest absolute term when nothing is positive, which
        the served path does not reach — `_why_not` entries are printed only for
        arms an interval excluded, so the gap is positive — but a caller
        constructing an estimate by hand can.
        """
        terms = {name: value for name, value in contributions.items() if value}
        if not terms:
            return "no blip term separates these arms for this patient"

        name, value = max(terms.items(), key=lambda kv: kv[1])
        if value <= 0:
            name, value = max(terms.items(), key=lambda kv: abs(kv[1]))
            return f"{BLIP_TERM_LANGUAGE.get(name, name)} {value:+.3f}, and none favours the leader"

        reason = f"{BLIP_TERM_LANGUAGE.get(name, name)} {value:+.3f}"
        offset, offset_value = min(terms.items(), key=lambda kv: kv[1])
        if offset_value < 0 and abs(offset_value) >= _OFFSET_SHARE * value:
            reason += f", partly offset by {BLIP_TERM_LANGUAGE.get(offset, offset)} {offset_value:+.3f}"
        return reason

    def _counterfactuals(self, selected: RegimeEstimate, stages: list[StageRecord]) -> list[CounterfactualProbe]:
        latest = stages[-1]
        crp = latest.features.get("crp")
        probes: list[CounterfactualProbe] = []
        ordered = sorted(selected.q_values.values(), reverse=True)
        gap = ordered[0] - ordered[1] if len(ordered) > 1 else 1.0
        if isinstance(crp, (int, float)) and not isinstance(crp, bool):
            flips = float(crp) > 20 and gap < 0.05
            probes.append(
                CounterfactualProbe(
                    covariate="crp",
                    perturbation="CRP halved (lower inflammation)",
                    recommendation_changes=flips,
                    note=(
                        "Lower CRP narrows the inflammatory case for escalation; with a small Q-gap the "
                        "recommendation could shift toward continuity." if flips
                        else "Recommendation is stable to a moderate CRP reduction."
                    ),
                )
            )
        return probes

    def _sensitivity(self, contrast) -> AssumptionSensitivity:
        """E-value for the contrast the decision reports. See `AssumptionSensitivity`.

        The outcome here is a bounded response score, not a risk, so the E-value
        goes through VanderWeele and Ding's approximation for continuous
        outcomes: standardise the contrast by the outcome's own spread, take
        `RR ~ exp(0.91 * d)`, then `E = RR + sqrt(RR (RR - 1))`.

        Two numbers rather than one, and the second is the one to quote. The
        point estimate's E-value says how strong confounding would have to be to
        move the *estimate* to null; the interval's says how strong it would have
        to be to make the data unable to exclude null, which is the question a
        reader is actually asking. Where the agent abstains the interval already
        contains zero, so that number is 1.0 — correctly, and for the first time.

        What this is not: a claim about the *four named unadjusted confounders*
        specifically. It is a bound on any single unmeasured confounder's
        strength, and the approximation assumes the contrast is between two arms
        on a comparable scale. On this cohort the generating process has no age,
        gender, steroid or comorbidity effect, so the true E-value question has
        no bite here — it is a property of the basis that would matter on real
        data, which is exactly what the model card says about those four nodes.
        """
        from treatmentrx.estimation import training

        outcome_sd = training.holdout_outcome_sd()
        if contrast is None or outcome_sd <= 0.0:
            return AssumptionSensitivity(
                e_value=1.0,
                e_value_for_interval=1.0,
                contrast=0.0,
                outcome_sd=round(outcome_sd, 4),
                note=(
                    "No contrast was available for this patient, so no bound on "
                    "unmeasured confounding is reported. This is an absent "
                    "measurement, not a finding of robustness."
                ),
            )

        # The confidence limit nearest the null. When the interval spans zero
        # this is zero, and the bound below correctly collapses to 1.0.
        nearest_null = 0.0
        if contrast.lower > 0.0:
            nearest_null = contrast.lower
        elif contrast.upper < 0.0:
            nearest_null = contrast.upper

        point = _e_value(abs(contrast.difference), outcome_sd)
        limit = _e_value(abs(nearest_null), outcome_sd)
        return AssumptionSensitivity(
            e_value=point,
            e_value_for_interval=limit,
            contrast=round(contrast.difference, 4),
            outcome_sd=round(outcome_sd, 4),
            note=(
                f"To explain away a contrast of {contrast.difference:+.4f}, an "
                f"unmeasured confounder would need associations of about "
                f"RR={point} with both the arm given and the outcome. To make the "
                f"interval unable to exclude zero it would need RR={limit}"
                + (
                    " — the interval already contains zero, so no confounding is "
                    "required and this recommendation rests on separation the "
                    "data does not have."
                    if limit <= 1.0
                    else "."
                )
                + " Standardised by a held-out outcome spread of "
                f"{outcome_sd:.4f} via the VanderWeele-Ding approximation for "
                "continuous outcomes."
            ),
        )
