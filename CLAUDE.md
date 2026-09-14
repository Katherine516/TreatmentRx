# CLAUDE.md

TreatmentRx: a research-stage clinical decision-support agent for sequential
treatment decisions in rheumatoid arthritis. **Research scaffolding, not a
medical device.** Nothing here is clinically validated; the synthetic cohort is a
test fixture, not evidence.

## Commands

```bash
PYTHONPATH=src python3 -m unittest discover -s tests    # 489 tests, ~6 min
PYTHONPATH=src python3 -m treatmentrx.cli demo          # one patient end to end
PYTHONPATH=src python3 -m treatmentrx.cli evaluate      # estimator scorecard
PYTHONPATH=src python3 -m treatmentrx.cli stability     # k-fold + seed sweep (~10s)
PYTHONPATH=src python3 -m treatmentrx.cli inference     # sandwich vs bootstrap (~35s)
PYTHONPATH=src python3 -m treatmentrx.cli audit         # layer-by-layer evaluation (~2s)
PYTHONPATH=src python3 -m treatmentrx.cli coverage      # do the 95% intervals cover? (~76s)
PYTHONPATH=src python3 -m treatmentrx.cli coverage --candidate-set  # does the set hold the best arm? (~40s)
PYTHONPATH=src python3 -m treatmentrx.cli coverage --multiplicity   # what the all-pairs correction costs (~60s)
PYTHONPATH=src python3 -m treatmentrx.cli coverage --stages         # coverage at every stage, not only terminal (~40s)
PYTHONPATH=src python3 -m treatmentrx.cli misspecification  # estimator robustness (~15s)
PYTHONPATH=src python3 -m treatmentrx.cli misspecification --omitted-modifier  # blip basis too small (~4m)
PYTHONPATH=src python3 -m treatmentrx.cli misspecification --extra-modifier    # blip basis too big (~40s)
PYTHONPATH=src python3 -m treatmentrx.cli power          # data needed vs abstention (~23s)
PYTHONPATH=src python3 -m treatmentrx.cli subgroups      # who does it abstain on? (~4s)
PYTHONPATH=src python3 -m treatmentrx.cli transfer       # fit at site A, score at site B (~50s)
PYTHONPATH=src python3 -m treatmentrx.cli specification  # is the blip basis missing a modifier? (~3s)
PYTHONPATH=src python3 -m treatmentrx.cli serve           # HTTP on 127.0.0.1:8371, stdlib only
```

```bash
curl -s localhost:8371/model | python3 -m json.tool          # what it was fit on, what it won't say
curl -s -XPOST localhost:8371/recommend -d @bundle.json      # a FHIR bundle in, a Recommendation out
```

## Hard constraints

- **Python 3.9 runtime.** Every module starts with `from __future__ import
  annotations`; keep it. No `match`, no PEP 604 unions outside annotations, no
  `X | Y` in `isinstance`.
- **Zero third-party dependencies.** `dependencies = []` is deliberate. Linear
  algebra lives in `estimation/linalg.py`. Do not reach for numpy/scipy without
  the user asking — the constraint is what keeps the prototype auditable and
  installable anywhere.
- **Determinism.** The cohort is seeded (`training.COHORT_SEED`) and models are
  fit lazily once per process. A given commit must always produce the same
  recommendation for the same patient. The only wall-clock value anywhere is the
  audit-event timestamp, and `test_orchestrator.py` enforces that.

## Input data

`docs/INPUT_DATA.md` specifies what a record must contain — the FHIR subset the
adapter reads, the observation codes with their units and plausible ranges, which
six of them actually reach the estimators, the rejection rules, and the separate
and much larger contract a *training* cohort has to satisfy. Read it before
changing `data/fhir.py`, `data/contract.py`, `data/units.py` or either basis in
`estimation/basis.py`; they have to agree and the doc is where that agreement is
written down.

Three things a real deployment must set or supply, each with a seam and a
measured default: the **endpoint** (`data/endpoints.py` — what a stage's outcome
*is*), the **units** (`data/units.py` — checked, converted, or rejected), and
the **propensity** (`estimation/propensity.py` — fitted, never the generator's).

## Layout

One package, six layers, one direction:

```text
treatmentrx/
  domain.py      clinical objects (PatientRecord, StageRecord, SafetyFlag, ...)
  contracts.py   the six layer handoffs (PatientState, RegimeEstimate, Decision,
                 SafeDecision, Recommendation, FeedbackReceipt)
  arms.py        the canonical treatment-arm vocabulary
  simulation/    the cohort with known blips, plus a FHIR exporter so
                 simulated patients re-enter through Layer 1
  data/          Layer 1  → PatientState
  estimation/    Layer 2  → RegimeEstimate[]
  decision/      Layer 3  → Decision
  safety/        Layer 4  → SafeDecision
  agent/         Layer 5  → Recommendation
  feedback/      Layer 6  → FeedbackReceipt
  orchestrator/  sequencing only, no clinical logic
  diseases.py    registered disease definitions; unsupported diagnoses fail closed
  service.py     HTTP transport only, no clinical logic
```

There was previously a second package (`precisionrx_agent`) implementing the same
agent. It is gone. If you are tempted to add a parallel path, don't — the fork
produced a real clinical divergence, and the notes below are the scar tissue.

## Invariants — do not break these silently

1. **Safety runs before explanation.** `SafetyLayer` and `safety/rules.py` are
   code, not prompts. An LLM layer may render a block; it may never lift one.
2. **The safety layer never substitutes an arm.** If the top-scored arm is
   infeasible, the case is blocked and routed to review with the reason stated.
   Silently promoting the runner-up once turned an arm-naming mismatch into what
   read as a clinical judgement.

   One exception, and it is not substitution. `_status` used to block whenever
   `decision.recommended_arm` was infeasible, reading that field regardless of
   `decision.status` — but on an equipoise decision it is only the argmax, and
   Layer 3 has already declared it indistinguishable from the rest of the
   candidate set. There was no recommendation for a contraindication to strike
   down. Measured over injected scenarios, **12 of 34** blocked cases were that.
   Those now route to **REVIEW** carrying the surviving candidates, under the
   flag `contraindication_without_recommendation`. Nothing is re-ranked, nothing
   is promoted, and `recommended_arm` stays `None` — `tests/test_safety_review.py`
   asserts that end to end, because a name appearing there means something was
   promoted whatever the status says.

   Every other path keeps BLOCKED, and that is deliberate: BLOCKED is the only
   status that *stops*, and the bug invariant 2 is about — a naming mismatch
   silently removing an arm — must still halt rather than produce a tidy list of
   alternatives. So a recommended arm being infeasible blocks, and an empty
   surviving set blocks. The allergy rule in `safety/rules.py` has the same
   argmax defect and is deliberately **not** changed: a recorded allergy is the
   strongest contraindication here, blocking is its fail-safe direction, and
   moving two safety paths at once is how a regression gets in. The arm is
   removed from the feasible set either way, so no patient is exposed; what is
   preserved is the halt. That boundary is pinned by a test rather than left to
   memory.

   `cli audit` reports `contraindication_routing` so the four branches are
   visible: on 60 simulated patients with pregnancy injected, 51 leader-feasible,
   2 blocked on a recommended arm, 7 routed to review, 0 blocked for want of a
   survivor.
3. **One arm vocabulary**: `arms.py`. The data contract, the estimators, the
   composite action space and the feasible-set filter must all agree.
4. **One type per layer handoff.** `MethodResult`/`RegimeEstimate` and
   `PatientStage`/`StageRecord` were once duplicate pairs, and the conversion
   between them silently dropped the timing, belief, switching and competing-risk
   annotations before the estimators saw them. Do not reintroduce a parallel type.
5. **Memory never moves a number, and the snapshot is deep.**
   `agent/memory.apply_memory` deep-copies `statistical_output` and raises
   `MemoryInfluenceError` if `q_values`, `policy_value`, `confidence_band`,
   `recommended_arm` or `safety_status` changed. The copy must stay deep: with a
   shallow `dict()` the snapshot holds the *same* `q_values` object the bundle
   does, so `q_values["TNF-inhibitor"] = 0.99` passed the check silently. The
   guard only held because `apply_memory` happens to replace containers rather
   than edit them, and that stops being true the moment a real memory component
   gets a handle on the bundle. Memory shapes narrative and retrieval only.
6. **Leakage guards raise.** A temporal-firewall violation raises `LeakageError`
   out of `DataLayer.build_patient_state`. Never downgrade it to a diagnostic.
7. **Estimators see only the training split, and `training.py` is the only
   thing that fits one.** Never fit on `fitted().holdout`, and never fit an
   estimator ad hoc in a layer that consumes one — but also never cache a model
   anywhere else. `dwols.fitted_model()` kept its own `_MODEL` global, so
   `training.fitted().dwols` (scored, joint-bootstrapped, measured by every
   study) and the model that actually *served* patients were different objects.
   They agreed only because both were fit from the same default cohort;
   `reset()` cleared one of them, so after any change to a training constant
   half the serving ensemble was stale. Nothing failed loudly — what it did was
   flatten the standard error's response to sample size from n^-0.45 to n^-0.15,
   which reads like a floor in the method. `cli power` is the regression guard.
   "Once per process" also has to survive threads, now that `service.py` runs one
   per request: `fitted()` double-checks under `_FIT_LOCK`, because unguarded,
   eight concurrent cold callers each ran the whole fit. That failure is silent
   by construction — the fit is deterministic, so the discarded ensembles agreed
   with the kept one. Keep it a plain `Lock`: if `_fit_all` ever re-enters
   `fitted()` it should deadlock loudly rather than quietly fit twice again.
8. **Calibration is measured on held-out data.** Scoring a patient's outcomes
   against themselves yields ECE ≈ 0 for everyone and makes the validation-ladder
   gate vacuous. That was a real bug; do not reintroduce it.
9. **Q-values live on one scale.** Q-learning's `raw_q` is a value-to-go;
   `q_values()` divides by the remaining horizon so every estimator reports
   expected response per remaining visit. Model averaging across estimators is
   only meaningful because of this. `predict_outcome` uses the terminal-stage
   parameters and is the single-stage quantity calibration is measured on.
10. **Equipoise needs both conditions.** The care goal sets the clinically
    meaningful difference; the contrast interval decides whether the data can
    resolve one that size. Neither alone is sufficient.
11. **A censored patient's last visit is not a terminal decision.** Their future
    is unobserved, not absent. Those rows are excluded from the earlier-stage
    regressions and the patients who did return carry their weight
    (`use_ipcw`). Counting them as terminal tells the model the future is
    worthless for exactly the patients who left, and it corrupts every
    non-terminal block; `treat_censored_as_terminal=True` reproduces it as a
    comparator and nothing else should ever set it.
12. **`non_regularity` always uses the sandwich.** It decides the bootstrap's
    resample size, so reading it off an already-attached bootstrap makes a refit
    depend on its own previous output and stop being reproducible.
13. **The data contract is a gate, not a report.** `DataLayer` raises
    `DataContractError` on any error-severity issue before running anything that
    assumes a usable record. Impossible values (`PLAUSIBLE_RANGES`) are errors:
    a DAS28 of -5 is a corrupt record, and silently modelling it moves the
    recommendation with nothing to show for it.
14. **Model-level and patient-level quantities never mix.** Estimands and the
    held-out policy value describe the *policy* and are measured on held-out
    patients; `SwitchingAwareOPE` summarises *this patient's* trajectory. An
    earlier build multiplied one by the other and averaged them together, which
    produced a number that was neither. Two later recurrences are fixed:
    `CompetingRiskEndpoint` scaled the held-out policy value by this patient's
    event incidence (it now annotates `coefficients` and leaves the value alone),
    and the validation ladder read a patient's OPE effective sample size to
    decide a deployment question (it now reads `training.deployment_readiness`).
    When you need a patient-level number, put it in `coefficients`, not on top of
    a measured population quantity.
15. **Every advanced arm keeps a monotherapy composite.** Listing biologics only
    in MTX combination turns one methotrexate contraindication into a blocked
    recommendation for a patient who had a viable option.
16. **The interval describes the quantity the decision uses.** The decision is
    made on the model-averaged Q-values, so `DecisionLayer._contrast` centres the
    interval on the averaged contrast and bounds its variance by the weighted sum
    of the component standard errors. Reporting the widest single estimator's
    interval instead made the difference and the decision come from different
    models and added the selection's own variability: measured end to end that
    covered 78%, worse than any component. `DecisionLayer.estimators` and
    `EstimationLayer.estimators` are separate tuples and must hold the same
    membership as `training.SERVING_ENSEMBLE`; `tests/test_layers.py` checks it,
    because if they drift the interval stops describing the Q-values beside it.
17. **Covariance is measured when it can be, bounded when it cannot — and the
    bound stays the default.** `training.enable_joint_inference()` refits every
    estimator on the *same* resamples, which is the only way to see how they
    co-vary; independent resamples would destroy exactly what is being measured.
    It is correct and it is off, on measurement rather than caution:
    `coverage.joint_replicate_sweep` shows its coverage climbing 87% -> 93% as
    the draw count goes 25 -> 200 *while the interval widens*, which is a
    percentile stabilising rather than a method changing, and at 200 replicates
    it still reaches only 85% at the worst patient against the bound's 96%, for
    1.15x the narrowness. An interval that silently degrades is worse than one
    that is visibly wide.

18. **Only estimators that estimate the same quantity may be averaged.**
    `training.SERVING_ENSEMBLE` is the membership and carries the measurements
    behind it. All four estimators stay fitted and scored — `cli evaluate`,
    `stability`, `misspecification` and `coverage` need the endpoints to compare
    against — but a model whose parameter is a stage-pooled compromise cannot be
    averaged with stage-resolved ones and then reported with an interval. Adding
    a member means checking its terminal-stage bias first;
    `tests/test_coverage.py` pins the current four.

19. **A study of the serving ensemble derives its membership from
    `SERVING_ENSEMBLE`, never from a local list.** This bit twice in one sitting.
    `decision_rule_coverage` and `joint_rule_coverage` each kept their own map of
    estimator names and filtered it against the serving tuple; when the serving
    Q-learning model became `Q-Pooled`, neither map contained it, both filters
    collapsed to dWOLS alone, and both studies reported one estimator's coverage
    under the ensemble's name. `cli stability` had the same shape and was
    scoring three estimators of which one served. `coverage._serving_models` is
    now the single place that maps a method name to a fitted model, it raises on
    a member it cannot build, and `tests/test_coverage.py` checks that every
    study covers exactly `SERVING_ENSEMBLE`. A silent collapse to a subset is the
    failure mode to design against — the numbers stay plausible.

20. **Shared and stage-specific are one axis, not two models.** `Q-Pooled`
    parameterises it as `psi_{j,a} = psibar_a + delta_{j,a}` with only the
    deviations penalized, so `pooling_ridge` interpolates: infinity is the shared
    fit, zero is the stage-specific one. It beats both endpoints — terminal
    parameter error 0.037 against 0.047 and 0.098, and lower total blip error
    than either at every curvature — because it borrows strength across stages
    without inheriting the shared fit's stage bias. `DEFAULT_POOLING_RIDGE`
    carries the sweep that set it; the plateau over [0.5, 2.0] is flat.
21. **`conservative=True` means the correction is already paid.** An interval so
    marked is exempt from `SANDWICH_INFLATION`; widening it again charges twice
    and pushes borderline cases into equipoise for no statistical reason.
22. **A separation that depends on the interval method is not a separation.**
    The sandwich is measurably too narrow and it drives equipoise, a clinical
    output. `ContrastTest.robustly_distinguishable` accepts a verdict only if it
    survives the interval widening by `SANDWICH_INFLATION`; a bootstrap interval
    is already honest and is exempt. Read that property, never `distinguishable`,
    when deciding.
23. **`linalg` is hot and hand-optimised; keep it exact.** `matmul` accumulates
    by row, `weighted_least_squares` skips structural zeros, `quadratic_form`
    walks only the support of its vector, and `sandwich_product` computes the
    half of A·B·A that symmetry does not give for free. `tests/test_linalg.py`
    pins each against its textbook form — a rewrite that is subtly wrong would
    otherwise shift every standard error in the system without failing anything.
24. **Weighting constants are measured, not chosen.** `DEFAULT_BLIP_RIDGE`,
    `USE_VISIT_INTENSITY` and `SANDWICH_INFLATION` each carry the numbers that
    set them in a comment beside them. Changing one means re-running the command
    that produced those numbers, not re-deciding by feel.

25. **A gate that cannot close is not a gate.** The SILENT rung read
    `ope_stable = effective_sample_size > 0` off the patient's own OPE, which is
    at least 1 for anyone with a visit, so it passed for every patient ever
    scored. Each criterion in `ValidationLadder._silent_blockers` must be able to
    fail on its own, and `tests/test_layers.py` checks each in isolation.
    `live_data` is False in this build and stays False: a held-out split of the
    cohort the model was fit on cannot establish stability *on live data*,
    however good the numbers look.

    The upper rungs had the same defect in both directions, and
    `_criteria` now fixes it: `metrics.get(key, default)` cannot tell "measured
    and failing" from "nobody measured this", and a reader deciding whether to
    advance needs those worded differently. `fairness_clean` defaulted to False,
    so every assessment ever made printed "fairness checks not clean" — which
    reads as a failed check, when nothing in this codebase computes fairness at
    all. `safety_events` defaulted to **0, which passes**: an unmeasured safety
    criterion reading as satisfied, on the one rung whose purpose is catching
    harm before it reaches anyone. An absent key is now its own blocker, and
    `tests/test_layers.py` checks both that unmeasured and failing produce
    different text and that every upper-rung criterion can still pass when it is
    actually measured.

26. **Governance defaults route to a human, never to the model.**
    `OverrideRouter`'s unmatched-reason default was POSSIBLE_MISSPECIFICATION —
    the only channel with model influence — so every reason the keyword list
    could not read defaulted into the one place that trains the policy. Unknown
    now routes to SAFETY_REVIEW, and `influences_model` requires the channel to
    have been *positively* identified. Short keyword tokens match whole words:
    `"ui"` is a substring of "guideline", "requires" and "quality", which made
    GUIDELINE_CONFLICT unreachable by its own name.

27. **A number a reader will act on cannot be a heuristic wearing a
    statistic's name.** `confidence_band` is the cluster-robust interval for the
    recommended arm's blip and nothing else. `BeliefAwareAdjuster` used to add
    `0.5 * belief.uncertainty` to each end, turning a measured (0.781, 0.837)
    into (0.593, 1.000) — seven times wider, clipped at the ceiling, and driven
    by a filter whose "uncertainty" comes from using the `PROXIES` mixing weights
    as precisions. Belief uncertainty is reported through the
    `uncertain_disease_activity_belief` flag and the coefficients, where it can
    be read for what it is.

    The selected top-two contrast is also not an ordinary pointwise interval:
    the serving rule applies a Bonferroni family-wise alpha over all 15
    unordered pairs of six arms. The action threshold and every candidate-set
    exclusion must use that same simultaneous level.

28. **Zero is a measurement.** Never write `_numeric(...) or default`. An eGFR
    of 0 is the most extreme patient the plausible range admits, and truthiness
    turned it into the healthy default of 90, so the out-of-support warning never
    fired for them. Use `_numeric_or`, which branches on `None`.

29. **An oracle the agent beats is not an oracle.** Layer 3's regret is measured
    against `oracle_arm`, the backward-induction optimum under the generating
    process. `optimal_arm` is the per-visit blip argmax and is *not* optimal for
    a sequential problem: it scores 2.137 against 2.155 for the oracle and 2.155
    for the fitted Q-Shared policy, so using it as the reference charged the
    agent regret against a rule it beats and rewarded whichever estimator was
    most myopic. The certainty-equivalent oracle is a near-optimal reference, not
    a proven bound — it clears the myopic rule by +0.020 on every seed but
    Q-Shared by only +0.0016 against a paired sd of 0.0040, so a small negative
    regret means "indistinguishable from optimal", not "bug".

30. **An argmax over indistinguishable numbers is not a selection.** Four call
    sites took `max(..., key=ipw_policy_value)` over three held-out values whose
    intervals overlap. `training.best_score()` promotes the leader only when its
    interval clears the runner-up's and otherwise breaks the tie by
    `INTERPRETABILITY_ORDER`, with `ranking_is_resolved()` reporting which
    happened. Every estimator's *gain over the behaviour policy* does exclude
    zero — that part is real; it is the ordering between them that is not.

31. **Coverage is replicated over fits, never over patients on one fit.** The
    obvious version of an off-distribution coverage check — take the deployed
    model, run it over a site's patients, count how many intervals contain the
    truth — is not coverage at all. Every patient's interval is built from the
    *same* fitted parameters, so their errors are correlated: one unlucky fit
    misses for everybody at once, and the fraction measured is a property of the
    single draw taken. Measured that way the training site read **87%** against
    the **98%** `cli coverage` reports for the same rule, and the whole gap was
    the method of measurement. `transfer.transfer_coverage` replicates over
    site-A refits and lands at 98.8% with SE/spread 1.27, matching the
    established study exactly. The refit loop is shared across sites because the
    fit does not depend on where it is evaluated — that is what keeps it
    affordable.

32. **The candidate set is what the agent says when it will not recommend, and
    it must never become a way to recommend more.** Layer 3 declines for most
    patients — 67% at the training population, 57-89% across `cli transfer`
    sites, 97% for seronegative patients — and that abstention is earned. What it
    used to produce was a status and nothing to act on, while the clinician still
    had to prescribe. `Decision.candidate_arms` is the leader plus every arm
    whose model-averaged interval fails to exclude zero: the same
    `robustly_distinguishable` rule the separation line reports, asked of every
    arm rather than only the runner-up. Measured over declined patients it
    averages **2.7 of 6** arms, and choosing the worst arm in it rather than the
    worst on the whole menu cuts worst-case regret **0.206 -> 0.047, a 77%
    reduction**. Replicated over 40 refits on the coverage grid the set contains
    the truly optimal arm in **240/240** draws (mean size 1.97, regret reduction
    88%). Do not read 240/240 as a guarantee; at that draw count the miss rate is
    bounded near 1% — and note that set containment is a different property from
    the interval's nominal 95%, with a naturally higher rate, so the two are not
    comparable.

    **The level is simultaneous, not pointwise.** The leader and the comparator
    are selected from the same estimates used for inference, so a pointwise 95%
    interval does not account for that search; `inference.simultaneous_alpha`
    applies a Bonferroni family-wise alpha over all 15 unordered pairs of six
    arms, which is what makes this an all-pairs confidence set rather than a
    collection of pointwise ones. That correction took abstention from 58% to
    78%; keeping the dWOLS cross-arm covariance (invariant 38) then brought it
    back to 67%, by removing width that was an error rather than a margin.

    **What the divisor costs is measured** (`cli coverage --multiplicity`, 480
    patient-draws over 12 refits, fresh patients each time):

    | family | alpha | contains best | misses | mean set | declines |
    | --- | --- | --- | --- | --- | --- |
    | pointwise, no correction | 0.0500 | 99.17% | 4 | 1.61 | 46.7% |
    | leader vs each (5 comparisons) | 0.0100 | 99.17% | 4 | 1.86 | 55.0% |
    | **all unordered pairs (15, deployed)** | 0.0033 | **99.58%** | 2 | 1.99 | **58.8%** |

    All-pairs buys **two avoided misses per 480 patients** for **12 points of
    extra abstention**; the /5 level buys nothing at all over pointwise here
    while costing eight. Read containment carefully: it sits above 99% at every
    level because the set always holds the leader and the leader is the true best
    arm 91% of the time — it is not a 95%-calibrated quantity, and comparing it
    to the interval's nominal was the error in an earlier reading of this table.
    The *interval* is calibrated, at 95.0% with SE/spread 1.04 since the dWOLS
    covariance was kept, so this correction now sits on an honest interval rather
    than stacking on an inflated one. Do not use the six-patient grid to
    compare levels: there all three contain the true best arm in 240/240 draws
    and the fixture, not the method, is what you would be reporting.

    It stays at 15, and not because of that table: the leader is *selected* by
    looking at every arm, so the honest family is the one the search ranged over
    rather than the five comparisons that survive into the report. Narrowing it
    would make the agent recommend more often — the direction these notes warn
    about — so change it only while saying that is what you are doing.

    It has **one definition**, and the reason is a bug it already caused:
    `coverage.candidate_set_coverage` hard-coded 0.05 while Layer 3 emitted at
    0.05/15, so the study reported mean set size 1.67 and 93% regret reduction
    for a rule nobody deployed. The numbers stayed entirely plausible — they just
    described a different object. Both now call the same helper.

    Three things it is not. It is **not a recommendation** — `status` and the
    action bar are untouched, and `tests/test_candidate_set.py` pins the
    abstention rate so a later change cannot collapse the set toward one arm and
    call it an improvement. It is **not pre-safety** on the card: Layer 4's
    removals are filtered out and named, which is not the arm substitution
    invariant 2 forbids because nothing is re-ranked or promoted. And it is not
    free of the rest of the card: `_why_not` now reports only arms *outside* the
    set, because hard-coded prose explaining away an arm the model could not
    exclude is the card asserting a clinical judgement the model never made.

33. **`service.py` is transport, and the model card is not optional.** The HTTP
    layer decides nothing: `tests/test_service.py` asserts the served
    recommendation equals the orchestrator's for the same bundle, because a
    second decision path is the failure this file opens with. It binds to
    loopback, caps the body, and returns no traceback to a caller — a stack
    trace quotes field values and field values are PHI. `GET /model` exists
    because the agent abstains on ~67% of patients, and a consumer that
    reads `equipoise` without knowing that reads a failure instead of a measured
    statement about sample size. The card also names the four unadjusted
    confounders, the measured interval coverage, and that only six covariates
    reach the estimators.
34. **No cross-disease fallback.** `DiseaseRegistry` must resolve exactly one
    registered definition before Layer 1 runs. A missing disease workflow is a
    typed error; it must never reuse RA arms, models, safety rules or evidence.

35. **A card block may not contradict the status at the top of it, and BLOCKED
   is not an excuse to say nothing.** Two defects, both found by rendering all
   four statuses side by side rather than by reading the code.

   The safety flag for an undecided contraindication said "routed to review ...
   not blocked". `apply()` upgrades to BLOCKED whenever a rule raises a
   block-severity flag — an allergy does — so the card read *blocked* in its
   heading and *not blocked* in its body. A flag reports what it observed; the
   status is decided after it and says itself. Never write a routing outcome
   into a flag message.

   And BLOCKED rendered as one interpolated line, which is the worst
   information-per-need ratio in the system: the one status that escalates to a
   person by definition handed that person the least, while the candidate set,
   the contrast and the removals sat computed and unused. `blocked_card` now
   carries them — labelled *context for the review, not a list of alternatives*,
   because a blocked case has no recommendation and the card must not read as
   though it does. The attribution block says so too when its subject is the arm
   safety just removed; unqualified it read as advocacy for something the patient
   must not receive.

   `tests/test_layers.py`'s memory guard was asserted against the arm name memory
   mentions, which stopped distinguishing "memory leaked in" from "safety said
   why the arm went" the moment removals appeared on the card. It now asserts
   against the memory sections themselves, which is what it was always about.

36. **A rate is meaningless without the denominator it was computed over.**
   `audit_decision` reported `oracle_arm_rate` and `mean_regret_vs_oracle` as
   bare numbers. When `recommended_arm` stopped being populated for undecided
   patients — correct behaviour — the loop's `continue` silently took the
   denominator from 120 to 43, and the section went to a perfect **1.0 / 0.0 /
   0.0** while `myopic_oracle_agreement_rate` hit 1.0 against an invariant that
   says the myopic rule is a *different* rule. Nothing improved; the question
   changed underneath the name, and the guarding test (`oracle_arm_rate > 0.7`)
   waved it through.

   There are two questions and they now have two names. `when_it_commits` scores
   the arm the agent published (35 patients, 100%, regret 0.0 — trivially high,
   because it commits only when the gap is large). `if_forced_to_commit` scores
   `top_scored_arm` over everyone (120 patients, 92.5%, mean regret 0.0003, max
   0.0101) and is the ranking itself. Both carry `patients`.

   `abstention_price` then prices the system's defining behaviour instead of
   asserting it, over the 85 declined patients: taking the model's own top arm
   costs **0.0005** mean, the worst arm in the candidate set **0.0464**, the
   worst arm on the menu **0.2058**. That last pair is the retrospective case for
   the candidate set — handing back a bare status was, in regret terms, roughly
   4.3x worse than handing back the set. That ratio was 6x before the all-pairs
   correction widened the set; a wider set is safer to be inside and less
   decisive to choose within, and both halves of that show up here.

   `tests/test_coverage.py` asserts the forced denominator covers every scored
   patient and that the committed subset is strictly smaller, because equal
   denominators mean the two blocks have collapsed into one.


37. **A gate criterion that cannot reach its own threshold is decoration.**
   `UncertaintyDecomposer._ood_score` carried four terms and one of them could
   never fire. It added `max(rms(encoded_state.vector[:32]) - 0.85, 0)`, but the
   encoder is a deterministic summariser of nine features each normalised into
   [0, 1] and then tiled, so that root-mean-square is bounded well below 0.85 by
   construction — measured over 121 patients it ran **0.374 to 0.664** and never
   once crossed. The term contributed exactly zero to every score the system has
   ever produced.

   It is removed rather than recalibrated, because "vector energy" was never a
   distributional statistic: the vector is a fixed function of the same clinical
   features the other three terms already read, so its magnitude says nothing
   about being *out of distribution* that they do not. A trained encoder would be
   a different argument; this one is not trained and its docstring says so.

   What remains is three one-sided range checks, each normalised so that crossing
   its own threshold alone reaches 1.0 and trips the 0.75 review gate — and each
   threshold sits inside what the data contract admits (`das28 >= 9.25` of 10.0,
   `crp >= 140` of 500, `egfr <= 7.5` of 200). `tests/test_workflow.py` fires each
   one in isolation, which is the property invariant 25 is about.

   **That left nothing reading `EncodedState.vector`,** and the open question of
   whether the encoder earned its place is now closed: `GRUBaselineEncoder` is
   deleted. Once the out-of-distribution term went, the only surviving use of its
   256-entry output was its own name and length in one audit line. Its
   `feature_map` was a copy of `HandcraftedFeatureEncoder`'s, its 224-entry tail
   held exactly **7** distinct values by construction, and it re-ran the
   handcrafted encoder internally — so that ran twice per request. Measured:
   **177us of the 211us** this layer spent encoding, and encoding was **29% of
   `build_patient_state`**, which now costs **561us against 723us**.

   The argument for keeping it was that it held the shape of the planned `z_t`
   interface. It did not: `EncodedState` holds that shape and
   `HandcraftedFeatureEncoder` already returns one. The wrapper was holding a
   seat the thing it wrapped was already holding. `PatientState.features` and
   `feature_names` come from that single encode now, and
   `tests/test_data_layer.py` asserts both reach the state and are read — which
   is the property that made deleting the other one correct.


38. **Ignoring a covariance you can compute is not conservatism, it is lost
   precision.** `_dwols_contrast` added the two arms' variances as if
   independent. They are not: every `ArmFit` is one-vs-reference, so any two arms
   share every reference-arm row and move together — measured over 60 refits the
   correlation runs **+0.20 to +0.51**. The contrast SE ran **1.24x** the
   estimator's actual sampling spread; with the cross term it runs **0.95x**.

   `ArmFit.cross_covariance` is the usual M-estimator sandwich with a cross meat
   term, `A_a^-1 (sum_i s_a,i s_b,i') A_b^-1`, summed over clusters present in
   *both* fits. A patient who received only one of the two arms scores zero in
   the other fit, so the whole cross term comes from the shared reference rows —
   which is exactly the mechanism that correlates them. It carries the same
   small-cluster correction `sandwich_covariance` applies, as the geometric mean
   `sqrt(scale_a * scale_b)`; without that it does not reduce to the variance
   when the two fits coincide, and `Var(x - x)` came out at a small positive
   residual instead of zero. `tests/test_estimators.py` pins that identity.

   The effect is system-wide, and all of it is recovered width rather than
   loosened standards: decision-rule coverage **98% -> 95.0%** (exactly nominal),
   SE/spread **1.27 -> 1.04**, abstention **78% -> 67%**, transfer-site coverage
   **97-100% -> 95-97%**. An interval that is too wide for a reason you can
   remove is not erring on the safe side; it is declining to answer questions the
   data can answer. The deployed facade and `coverage._dwols_contrast` both call
   `contrast_standard_error`, and a test checks they agree — invariant 19's shape.

39. **A per-decision value is not a sequential regime's value.**
   `evaluate_policy` walks stage-rows independently, keeps the rows where the
   policy agreed with the arm actually given, and weights by *that stage's*
   propensity. For a DTR that is a per-decision quantity. A patient who deviated
   at stage 1 still contributed their stage-2 row, from a history the regime
   would never have produced, and the weight is a single-stage probability rather
   than the cumulative product the sequential estimand requires.

   `sequential_policy_value` computes the regime's own value, and reporting both
   is the point — they answer different questions and the denominators differ by
   a factor of three:

   | | value | matched rows | ESS | max weight |
   | --- | --- | --- | --- | --- |
   | per-decision (`ipw_policy_value`) | 0.7405 | 88 | **75.5** | 14.7 |
   | sequential (the regime) | 0.8263 | 44 | **14.6** | 94.3 |

   Three facts a reader needs. The gap (**0.086**) is larger than the whole
   claimed gain over the behaviour policy (0.066). The sequential effective
   sample falls **below `MIN_OPE_EFFECTIVE_SAMPLE`**, so by this repo's own
   standard the regime's value is *not identified* at n=280 — `identified` is a
   field, and the scorecard says so in a note rather than leaving a reader to
   infer it from a small number. And only **3 of 120** holdout trajectories
   follow the regime to the end (34, then 7, then 3).

   That is not a defect in the estimator; it is what a three-stage regime costs
   in an observational cohort this size, and it was invisible because nothing
   computed it. `ipw_policy_value` still drives the BMA weights and
   `best_score()` — which is defensible for a per-decision comparison against a
   per-decision `behaviour_value` — but it must never be described as the value
   of the regime.

   **Every IPW number now carries its weights.** `WeightDiagnostics` reports the
   maximum weight, the mean, the share of mass in the heaviest row, and the share
   of rows pinned to `_PROPENSITY_FLOOR`. A value of 0.74 at ESS 75 looks the
   same whether the weights are flat or whether three rows carry a third of the
   mass, and the two estimators differ exactly there: per-decision efficiency
   0.86 with 3% of mass in its heaviest row and positivity fine; sequential
   efficiency **0.33**, **16.7%** of the mass in one row, 9% of rows at the
   floor, `positivity_ok` **False**.


40. **Coverage has two axes and only one of them was being swept.**
   `PATIENT_GRID` exists because coverage at one covariate point is not coverage:
   the estimand and its standard error both move over the covariate space. The
   stage index has exactly that property and was pinned at `n_stages - 1` by nine
   call sites. The terminal block is also the most flattering stage to measure
   at — there is no future left, so a value-to-go blip *is* a single-visit blip
   and both serving estimators target the same quantity — so the one stage
   measured was the one where the ensemble cannot be incoherent.

   `coverage.decision_rule_stage_sweep` sweeps it, at n=280 over 40 refits:

   | stage | coverage | worst | SE/spread | SE/within | bias spread | served |
   | --- | --- | --- | --- | --- | --- | --- |
   | 0 | **77.5%** | 37.5% | **0.64** | **1.13** | **0.0171** | no |
   | 1 | 96.7% | 95.0% | 1.07 | 1.19 | 0.0055 | yes |
   | terminal | 95.0% | 92.5% | 1.04 | 1.05 | 0.0024 | yes |

   **Stage 0 fails on centring, not width, and an earlier version of this
   invariant said the opposite.** `se_to_sd_ratio` is measured about each
   patient's *truth*, so a bias that differs between patients enters its
   denominator — 0.64 reads as an interval a third too narrow. About their own
   means the same estimates give **1.13**: that interval is slightly wide. The
   individual biases run **-0.031 to +0.015** against a sampling spread near
   0.013 and they differ in sign, so the pooled -0.009 cancels them away.
   `se_to_within_sd_ratio` and `bias_dispersion` exist so the two cannot be
   confused again, and a test asserts the width ratio stays above 1 at every
   stage — if it ever drops, that is a different defect wanting a different fix,
   because widening cannot repair a centre.

   The mechanism is invariant 18's, at a stage nobody was sweeping. The ensemble
   averages dWOLS's single-visit blip with `Q-Pooled`'s value-to-go divided by
   the remaining horizon; those coincide *exactly* at a terminal block and
   nowhere else, so the expected bias is about half their gap — predicted -0.029
   for the two seronegative patients against -0.027 and -0.031 measured. The
   horizon rescaling shrinks the gap toward the terminal block without closing
   it. Each member is nearly unbiased for its *own* estimand at every stage,
   which is exactly the problem.

   It has never cost anything because `SERVED_STAGE_INDICES` is `(1, 2)`:
   `DataLayer.build_patient_state` appends the *pending* visit, so `stage_index`
   is at least 1 for anyone the pipeline sees. Measured over 240 audit bundles
   the split is 22 at stage 1 and 218 at the terminal stage, none at stage 0. The
   row is reported with `served: False` rather than dropped, because what keeps
   it out of reach is Layer 1's stage bookkeeping and not a property of the
   estimator, and `tests/test_coverage.py` re-derives the served set from the
   pipeline rather than trusting the constant.

   **The truth has to carry the horizon division.** `sandwich_contrast` divides
   the value-to-go contrast by the remaining stages so its numbers sit on the
   same per-remaining-visit scale as `q_values` and as dWOLS's single-visit blip.
   Scored against the *undivided* value-to-go the sweep reads 0% at stage 0 and
   13% at stage 1 — arithmetic, not an estimator, and it looked like a much
   larger finding until the scale was checked. Two scale mistakes in one study,
   in opposite directions: that one made the defect look catastrophic, and
   reading `se_to_sd_ratio` as a width ratio then made it look like the wrong
   kind of defect. Measure the denominator before believing the quotient.

41. **The gate on a sequential regime reads the regime's own effective sample.**
   `deployment_readiness()` reported only `ope_effective_sample_size` — the
   *per-decision* one, 75.5 on the deployed holdout, comfortably over
   `MIN_OPE_EFFECTIVE_SAMPLE`. What advances a rung is a three-stage regime whose
   own value is estimated on 44 rows at an effective sample of **14.6**. Gating a
   sequential deployment on the per-decision number is the easier question
   wearing the harder one's name, which is invariant 25 one level up.

   Both are now reported and both are gated, so the SILENT rung carries two
   blockers rather than one. That matters because the other one was structural:
   `live_data` is False by construction, and it was the *only* thing holding the
   gate shut. Flip it — which a first real cohort does — and the gate would have
   opened on a regime whose value is not identified.
   `tests/test_layers.py` strips `live_data` to check the statistical criteria
   can still close the gate on their own.

   **The regime's value now carries an interval**, a trajectory-clustered
   percentile bootstrap: 0.8263 (0.765 to 0.867). `identified` compares an
   effective sample against a threshold and returns a boolean, which tells a
   reader the number is untrustworthy without saying by how much — and the
   natural thing to do with a bare point estimate is quote it. The bootstrap
   holds the policy fixed, so it is the precision of the evaluation and not the
   variability of the fit, and at this effective sample a percentile interval is
   itself unvalidated: `identified` is reported beside it, not replaced by it.
   `PolicyScore.notes` was populated and never emitted by `as_dict`, so the line
   saying the regime's value is not identified reached nobody; it is emitted now.

42. **A contract nobody constructs describes a discipline nobody keeps.**
   `EvaluationPartitionContract` was exported, unit-tested against hand-built
   inputs, and never constructed from the pipeline — so the package's public
   surface advertised a locked final test that did not exist. An empty
   `final_test` satisfies `final_test_locked=True` vacuously, which is precisely
   the reading that let it look like a discipline.

   `training.evaluation_partition()` now builds the partition this build has:
   **280 training, 120 evaluation, no tuning split, no final test**, and
   `has_final_test` reports the absence on the model card. The evaluation split
   does four jobs — it sets the model-averaging weights, selects the serving
   estimator through `best_score()`, supplies the calibration the validation
   ladder gates on, and is the held-out policy value quoted on the card. Each is
   defensible alone; together they mean nothing untouched remains to check the
   headline numbers against. The Monte Carlo studies do draw fresh cohorts
   (`coverage` at `base_seed` 9_700, independent of `COHORT_SEED`), so the tuned
   constants are not fitted to this split — but fresh seeds do not give back a
   test set.

   **And a three-way split is not the remedy at this cohort size.**
   `power.final_test_feasibility` (reported by `cli power`) prices it, and the
   two quantities disagree:

   | held back | evaluation ESS | final-test ESS | final identifies per-decision | identifies the regime |
   | --- | --- | --- | --- | --- |
   | none (today) | 73.0 / 14.0 | — | — | — |
   | 25% | 52.7 / 10.5 | 20.4 / 5.4 | no | no |
   | 40% | 42.9 / 8.2 | 30.3 / 6.2 | yes | no |
   | 50% | 33.8 / 7.4 | 39.4 / 7.5 | yes | no |

   A final test that identifies the *per-decision* value does exist at 40-50%
   held back — bought by taking the evaluation split from 73 to 43, and the
   per-decision value is not what the agent deploys. For the **regime's own**
   value no split works: 5.4 to 7.5 against `MIN_OPE_EFFECTIVE_SAMPLE` of 30, so
   the confirmation would itself be unidentified, which is the reading
   `identified` exists to refuse. Keeping today's evaluation precision *and*
   adding an identified final test needs roughly **565** trajectories for the
   per-decision value and **1,258** for the regime's — against 400 today, and
   next to `cli power`'s 1,490 for 30% abstention. Both say the same thing about
   this cohort. Reported rather than acted on: `COHORT_SIZE` is a stated choice
   near the low end and raising it moves the headline abstention rate, which is a
   separate decision.

   The same defect in a different place: `BayesianModelAverager.aggregate`'s
   estimand-fingerprint check cannot fire from the serving path, because
   `EstimationLayer.estimate` stamps every result from the *same*
   `state.estimand_contract`. It is a precondition on a public component, not the
   thing keeping the ensemble coherent — that is `EstimationLayer.estimators`
   matching `SERVING_ENSEMBLE` — and its docstring now says so rather than
   letting a reader mistake it for a live guard.

43. **Pure inverse weighting was the reason the regime's value was unidentified,
   and it was fixable.** `sequential_policy_value` discards a trajectory at its
   first deviation, so five of 120 holdout trajectories carry the whole estimate
   and the heaviest carries 35% of the weight. That is what inverse weighting
   costs over three decisions with six arms — and the fitted Q-functions the
   agent already serves from can supply the value where a trajectory leaves the
   regime's path.

   `sequential_dr_value` is the standard doubly-robust backward recursion
   (Murphy 2001; Bang & Robins 2005; Jiang & Li 2016):

       V_{T+1} = 0
       V_t     = Q(x_t, d(x_t))
                 + 1{a_t = d(x_t)} / pi_t * (y_t + V_{t+1} - Q(x_t, a_t))

   Where the observed arm matches, the residual is inverse-weighted in; where it
   does not, the indicator is zero and the trajectory contributes the model's own
   `Q(x_t, d(x_t))`. **Every trajectory contributes** and the cumulative
   propensity product never appears as an explicit weight. Measured on the
   deployed holdout against a known truth of **2.2938**:

   | estimator | value | SE | 95% interval | trajectories | ESS |
   | --- | --- | --- | --- | --- | --- |
   | sequential DR (AIPW) | 2.3278 | 0.0673 | (2.196, 2.460) | **120** | **120** |
   | sequential IPW (Hajek) | 2.2864 | 0.2345 | (1.713, 2.542) | **5** | **3.9** |

   Both cover the truth; the DR interval is **3.1x narrower**, its heaviest
   trajectory carries **1.5-1.7%** of the estimate against the IPW estimate's
   **16.7%** — `max_influence_share` and `WeightDiagnostics.top_share` are
   deliberately the same quantity so the two are comparable — and both serving
   estimators' DR values land within **0.07 and 0.52 standard errors** of their
   own oracle.

   **Which truth, and this is the trap.** The augmenting Q-model is fit with
   IPCW, so it targets the value-to-go *had the patient stayed in care*:
   `rollout_value(policy, dropout=False)` = 2.2938. `oracle_rollout_value`, the
   figure reported beside every other policy value, is what a patient actually
   accrues once dropout is simulated — 2.1469. The 0.147 between them is
   retention, not error, and scoring the DR estimate against the smaller one
   charges it for a gap it is not estimating. `training.oracle_uncensored_value`
   exists so the comparison is made against the right one; that is invariant 14
   in a place where both quantities are model-level and only the follow-up
   assumption differs.

   **It does not retire the IPW gate, and that is the statistically interesting
   part.** The DR estimate says the regime's value *is* identified — under the
   outcome model. The IPW estimate is the only thing here that could falsify that
   model, and at an effective sample of 14.6 it has no power to. A doubly-robust
   estimate whose inverse-weighted check cannot fail is a model-based estimate
   wearing a robustness label. So the SILENT blocker now reads "not identified
   *without leaning on the outcome model*" and names the DR value alongside,
   rather than reporting only what is unknown.

   One honesty note on the augmentation: `QLearningModel.raw_q` is fit by
   backward induction under its own `max`, so it is Q^d only for the policy that
   is that model's own greedy rule. dWOLS's policy borrows `Q-Pooled`'s value
   function, which is legitimate — double robustness does not require the
   augmenting model to be Q^d — but costs efficiency and shifts the estimator's
   weight onto the propensity model. `augmentation_model` names which was used.
   dWOLS's own `raw_q` is a single-visit outcome and must never be passed here.

44. **"Identified under the outcome model" is a caveat until someone prices it.**
   Invariant 43 ends by saying the doubly-robust estimate leans on the Q-model
   and that the inverse-weighted check has no power to falsify it. That is true
   and it is not enough: a reader cannot act on it. `sequential_dr_sensitivity`
   turns it into a number by asking how wrong the model would have to be.

   Misspecification is parameterised the way the rest of this repo already bends
   an estimand — every estimated blip scaled by `1 + gamma`, treatment-free
   surface untouched — and the **regime under evaluation is held fixed**, so this
   measures error in the value estimate rather than quietly scoring a different
   policy. `cli evaluate` reports it under `outcome_model_sensitivity`.

   Measured on the deployed holdout, against a behaviour policy worth **1.9056**
   on the same value-to-go scale:

   | gamma | DR value | gain over behaviour | separated |
   | --- | --- | --- | --- |
   | +0.00 | 2.2742 | +0.3686 | yes |
   | +0.25 | 2.1951 | +0.2895 | yes |
   | +0.40 | 2.1476 | +0.2420 | yes |
   | **+0.434** | — | — | **tipping point** |
   | +0.50 | 2.1159 | +0.2103 | no |
   | +1.00 | 1.9575 | +0.0519 | no |

   **The claim survives the model over-stating every treatment effect by 43%.**
   The tipping point is bisected rather than read off that grid, so adding a row
   for legibility cannot move it.

   **Where it lands is the point.** A 50% proportional blip error is the same
   magnitude as the estimand shift `cli transfer` applies in its 1.5x row — the
   row where calibration rises to 0.051 against 0.002-0.006 everywhere else
   *while policy value gets better*. So the misspecification that would overturn
   this claim is one the deployment monitor already detects, and it detects it
   through calibration rather than value. That is the same finding as the
   estimand-shift row, arrived at from the other end, and it is the argument for
   why calibration is the monitor to watch.

   Two caveats on that comparison, and neither is small. The transfer row shifts
   the *truth* while the model stays put; gamma shifts the *model* while the
   truth stays put. Both are a 1.5x mismatch and calibration reads the gap either
   way, but the direction is not identical and the magnitudes need not match.
   And the benchmark is a simulation rollout, like every other oracle here —
   `training.behaviour_uncensored_value()`, on the value-to-go scale, because
   `PolicyScore.behaviour_value` is the per-decision mean and comparing the DR
   estimate against *that* would be invariant 14 with a fresh pair of quantities.
   The ratio between them, 2.83, is a horizon and not an improvement.

   A second reading comes almost free: the standard error is minimised **near**
   gamma = 0 (0.067 against 0.15 at either end), so a badly wrong Q-model costs
   the augmentation precision as well as centre. *Near*, not at — the minimum is
   exactly at 0 only when the augmenting model is the evaluated policy's own, and
   for the deployed pairing it sits at +0.1. That is invariant 43's borrowing
   cost showing up in a second place.

45. **The E-value was a heuristic wearing a statistic's name, on a served field.**
   `AssumptionSensitivity` claimed to say how strong unmeasured confounding
   would have to be to overturn a recommendation. What it computed was
   `rr = q_values[0] / q_values[1]` — a ratio of two *nearly equal bounded
   means* — pushed through the E-value formula. It never referenced confounding,
   the four unadjusted confounders, or anything but the model's own point
   estimates, and because the top two Q-values are close by construction the
   ratio was numerically unstable in exactly the regime the agent operates in.

   Measured over 60 patients it returned **1.000 to 1.617, with 11 under 1.1**.
   A published E-value of 1.0 asserts that *no* unmeasured confounding is needed
   to explain an effect away — a strong claim, reached by arithmetic that could
   not support it, and shipped inside the served `Recommendation`. That is
   invariant 27's pattern in a place nothing rendered, so nothing caught it.

   It now uses VanderWeele and Ding's approximation for continuous outcomes:
   standardise the contrast by the held-out outcome spread (0.1531),
   `RR ~ exp(0.91 d)`, `E = RR + sqrt(RR (RR - 1))`. Two numbers are reported and
   **the second is the one to quote**:

   | status | n | E (point) | E (interval) | interval bound = 1.0 |
   | --- | --- | --- | --- | --- |
   | equipoise | 43 | 1.579 | **1.000** | 40/43 |
   | recommend | 17 | 1.999 | 1.261 | 0/17 |

   `e_value_for_interval` is the bound on the confidence limit nearest the null —
   "could confounding make this indistinguishable from no difference" — and it is
   **1.0 exactly when the interval already contains zero**, because then nothing
   is needed. So 1.0 now carries meaning instead of being an artifact, and it
   lands on the abstaining patients rather than at random.

   Two things it is not. It is not a statement about the four named unadjusted
   confounders in particular — it bounds any single one — and on this cohort the
   generating process has no age, gender, steroid or comorbidity effect, so the
   question has no bite here and is a property of the basis that would matter on
   real data. And the spread is a **population** quantity from the held-out
   split: one patient's contrast over the cohort's spread is Cohen's d, which is
   the intended side of invariant 14, not the forbidden one.

   The contrast is now **passed in** from the decision layer rather than rebuilt
   from `q_values`. The caller already had it; reconstructing a worse quantity
   beside the decision is the incoherence invariant 16 removed from the interval,
   one layer over.

46. **The leader was chosen twice, two different ways, and one of them used a
   display quantity.** `q_values` is clamped to `[Q_FLOOR, Q_CEILING]` and
   rounded to three decimals before anything downstream sees it. Both steps are
   many-to-one. A patient whose predicted response saturates the ceiling had two
   arms collapse to 0.99 — measured, Q-Pooled 0.9897/0.9995 and dWOLS
   0.9898/1.0013, both ceiling-clipped — `max` fell through to dictionary order,
   and `DecisionLayer._contrast` then re-sorted the same clamped dict and picked
   its own top pair. The result was a **negative contrast for the decision's own
   leader**: the interval printed beside the recommendation saying the runner-up
   was better. **6 of 120 patients, 5%**, with 8.3% saturating the clamp on at
   least one arm.

   The information was never lost, only discarded. `QLearningModel.recommend` and
   `DWOLSModel.recommend` already read the unclamped value-to-go; it was the
   facades that re-derived a ranking from their own display dict. Three changes,
   all of them "choose the leader once": the facades take `recommended_arm` from
   the model, `BayesianModelAverager` breaks a clamped tie by the members'
   weighted votes rather than by dictionary order, and `DecisionLayer._ranked`
   puts `selected.recommended_arm` first instead of re-sorting.

   **It improved the ranking without touching the rate**, which is the shape a
   fix should have here:

   | | before | after |
   | --- | --- | --- |
   | status distribution | 35 / 85 | unchanged |
   | oracle-arm rate, forced | 0.9083 | **0.9250** |
   | mean regret | 0.0011 | **0.0003** |
   | max regret | 0.0406 | **0.0101** |

   Max regret falling fourfold is the tell: the agent's worst mistakes *were* the
   clamp artefacts. Abstention did not move, so this is recommending better
   rather than recommending more — the direction invariant 32 warns about.

   One case survives and is a different thing. Two members genuinely disagree
   (Q-Pooled 0.7275 against dWOLS 0.7426 for the same pair), the averaged values
   tie at 0.736 — mid-range, nowhere near the clamp — and the weighted vote
   breaks it. The resulting -0.0016 contrast honestly reports that disagreement
   and routes to equipoise. `tests/test_layers.py` allows exactly one such case
   and no clamp artefacts.

## What is real vs. still a placeholder

Real: the four estimators, the cohort and its known blips, informative-dropout
handling and IPCW, held-out IPW policy evaluation — both per-decision and
sequential, each with a trajectory-clustered bootstrap interval and its weight
diagnostics — calibration, cluster-robust standard errors including the dWOLS
cross-arm covariance, m-out-of-n bootstrap intervals,
contrast tests, cross-validated stability, blip attributions, the
backward-induction oracle used for regret, the safety sweep (20 labelled cases,
recall *and* precision), and the causal DAG's identifiability check —
`minimal_backdoor_set` derives the adjustment set from the edge list rather than
declaring it, `satisfies_backdoor` implements the backdoor criterion with
descendant exclusion and collider handling, and `_ra_v1` passes
`adjustment_set=()` so the graph is the only thing that fills it.

`identified` reads True for every patient the pipeline actually sees, and that is
not a check that cannot fail: strip the observations and it returns False naming
the missing adjusters. It is constant because the data contract already rejects
records without DAS28 or CRP, so by the time the DAG is consulted the adjusters
are present by construction. `_has_adjuster` treating any medication history as
evidence for `prior_biologic_exposure` is deliberate — "no prior biologic" is a
value of that variable, not a missing one, and requiring a biologic to appear
blocked every csDMARD-only patient for having been treated conservatively.

Still deliberately simple, and labelled as such in-module:
`HandcraftedFeatureEncoder` (nine clinical features normalised and tiled — not a
learned representation, and it does not claim to be; the `GRUBaselineEncoder`
that wrapped it is deleted, see invariant 37),
`SemanticKnowledgeBase` (five hard-coded passages, whitespace-token retrieval),
`WHY_NOT_REASONS` (hard-coded clinical prose attached to a model-derived Q-gap —
the one place Layer 5 asserts something the model did not produce). The E-value
is no longer on this list: it was a heuristic and is now the VanderWeele-Ding
bound on the decision's own contrast, with its approximation stated (invariant
45).

**`SwitchingAwareOPE` no longer reports an estimate, and the reason generalises.**
It used to publish an `iptw_policy_value`: this patient's observed outcomes
reweighted by `adherence x 0.5^switched x 0.7^rescue`. Measuring the cohort
killed all three parts. Over 728 stage-rows from 200 patients, `adherence` was
identically 1.000 (min = median = max) and `rescue_therapy` fired **zero** times,
so two factors were dead and the whole weight reduced to `2.0 if switched else
1.0` — one hand-set constant, which moved the published number by 0.024 on
average and up to **0.112** against outcomes near 0.64. And the estimand did not
exist: IPTW recovers a *population* policy value by correcting confounded
assignment, so applied to one patient's own three or four outcomes there is no
counterfactual to recover. Fitting the deviation model would have fixed the
constant and left that untouched. What it reports now is descriptive and labelled
descriptive — `observed_mean_outcome`, `n_stages`, `stages_switched`,
`stages_with_rescue`, `min_adherence`, `deviation_flags` — plus
`model_policy_value`, which is real, held-out and still must not be combined with
any of them.

`service.py` is real as transport and deliberately unfinished as infrastructure:
`http.server` is single-process, has no TLS and no authentication, which is why
it binds to loopback and warns when told not to. Those gaps are in `LIMITATIONS`
and on `GET /health` rather than left for a reader to infer. Do not close them by
adding a dependency — a research prototype that needs an ASGI server to start is
a different artifact.

`StageRecord` no longer carries `visit_weight` or `censoring_weight`, and
`PatientState` no longer carries `tailoring_variables`. All three were Layer 1
heuristics that named a statistical role they did not have. The weights were
never read by anything statistical (the real weighting is fitted in Layer 2,
`estimation/censoring.py` and `estimation/visit_intensity.py`). The tailoring
variables were scored as `sqrt(variance) + abs(latest)` on unstandardised
features, which ranked by unit size — `egfr` (82) over `crp` (28) over `das28`
(5.2) — and skipped booleans, so neither effect modifier in the blip basis could
ever be selected; the clinician card named four "drivers" while the attribution
three lines below credited `anti_ccp`, which the selector could not have chosen.
Selection now happens where the model is, in
`estimation.features.top_tailoring_variables`, ranked by `|psi_k * h_k(X)|`. Do
not reintroduce a Layer 1 quantity that no estimator consumes.

`AdaptiveRegimeSelector` is gone. It ran on every request, its `RegimeAssignment`
was passed to every estimator and read by none (`RegimeEstimate.regime_type`
comes from the estimator's class), and its BIC proxy was degenerate: the
stage-specific term was `sum((outcome - stage.outcome) ** 2)` over the same
sequence twice, identically zero, so one side of the comparison was a constant.
The shared-versus-stage-specific question is not a routing decision at all: it
is a shrinkage parameter, and `Q-Pooled` fits it as one.

**Measured coverage of the nominal 95% interval** (`cli coverage`, 60
replications, n=280 — the size the deployed models are fit on — swept over the
six-patient grid in `feedback/coverage.py`, each contrasted on their own true
top-two arms):

| interval | pooled | demo patient | worst patient | SE/spread | bias |
| --- | --- | --- | --- | --- | --- |
| sandwich, stage-specific | 91% | 92% | 85% | 0.89 | -0.002 |
| sandwich, shared blip | 28% | 74% | 0% | 0.38 | +0.083 |
| **decision rule, correlation bound** | **95%** | — | 93% | 1.04 | 0.000 |
| decision rule, joint bootstrap (200 replicates) | 93% | — | 85% | 0.94 | +0.001 |

The study used to run at one patient and n=250, and that patient was the best
case for two of the three methods. Do not narrow the grid back.

Read `se_to_sd_ratio` rather than the coverage tally when replications are few —
it is a quotient of two means, not a proportion of a few dozen Bernoulli draws.

**How the decision rule got from 74% to 98%, in two steps.** It was never a
width problem. The old averaged interval's SE ran 1.12-1.52x the actual spread
everywhere and coverage still fell to 45% at its worst patient, because the
centre was wrong.

*Step one — drop the shared blip out of the average (74% -> 97%).* A shared psi
is one parameter standing in for three stages of delayed effect, so its terminal
contrast is a stage-pooled compromise: measured against the true value-to-go
contrast it runs -0.075 at stage 0, +0.009 at stage 1 and +0.067 at the terminal
stage, where 91% of the patients the pipeline is asked about sit — a property of
`simulated_bundles`, which exports whole trajectories so a patient always presents
at their last visit, rather than of a clinic. Averaging that with stage-resolved
estimators
produced an ensemble centred between two parameters, which no weighting scheme
repairs. It cost nothing on the decision side to remove: true rollout value
2.1467 -> 2.1473, oracle-arm agreement 0.888 -> 0.900.

*Step two — replace the stage-specific member with the partially pooled one
(97% -> 98%).* Dropping the shared blip had thrown away the stable end of a
bias-variance axis. `Q-Pooled` makes that axis continuous and beats both
endpoints on terminal parameter recovery (0.037 against 0.047 stage-specific and
0.098 shared) and on total blip error at every curvature. See
`training.SERVING_ENSEMBLE` and `DEFAULT_POOLING_RIDGE`.

**Joint inference stays off, and the question is now closed.** Sweeping the
bootstrap replicate count settles what an earlier 92% could not distinguish:

| joint bootstrap | pooled | worst patient | SE/spread | width |
| --- | --- | --- | --- | --- |
| 25 replicates | 87% | 75% | 0.96 | 0.0607 |
| 50 replicates | 91% | 80% | 0.96 | 0.0638 |
| 100 replicates | 92% | 80% | 0.95 | 0.0653 |
| 200 replicates | 93% | 85% | 0.94 | 0.0662 |

Coverage climbs with the draw count *and the interval widens as it does* — the
signature of a percentile estimate stabilising, not of the method changing.
SE/spread sits at ~0.95 throughout, so the standard error was always honest; the
2.5th and 97.5th quantiles from 25 draws were not.

At 200 replicates it reaches 93% pooled, within Monte Carlo error of nominal
(120 draws, MC error 2.3 points) — but 85% at the worst patient against the
bound's 96%, and only **1.15x narrower**, not the 1.43x measured before. That
earlier ratio came from the collapsed-ensemble bug: dWOLS alone has a smaller
parameter space and a correspondingly narrower interval. Trading 11 points of
worst-patient coverage for 15% of width is the wrong direction for a clinical
output, so the bound is the default and this is settled rather than open.

**What the extra abstention costs, and why it is right.** Equipoise went from 41
to 85 of 120 audit patients: 77 from dropping the biased contrast, up to 99 with
the all-pairs correction, then back down as the dWOLS cross-arm covariance
removed width that was never a safety margin. The old contrast was biased *away*
from zero, so it manufactured separation: patients it called separable had a true
top-two gap of 0.019, against 0.026 now, while the recommended group's gap rose
from 0.054 to 0.081. The agent recommends about half as often and is right about it more often.
Do not tune this back by widening the action bar.

**Abstention is a sample-size choice, and `cli power` prices it.** The same 240
patients, the ensemble refit at each cohort size:

| train n | abstains | mean contrast SE | mean \|contrast\| | mean z |
| --- | --- | --- | --- | --- |
| 98 | 80% | 0.0261 | 0.045 | 1.73 |
| 196 | 86% | 0.0200 | 0.037 | 1.87 |
| **280 (deployed)** | **67%** | 0.0174 | 0.040 | 2.29 |
| 560 | 40% | 0.0124 | 0.044 | 3.54 |
| 1120 | 33% | 0.0092 | 0.045 | 4.86 |

The contrast itself is flat; only the precision moves. The standard error shrinks
at n^-0.43 against the n^-0.50 a correctly specified estimator earns. Under the
simultaneous all-pairs rule, roughly 1,490 training trajectories would bring
abstention to 30% (an extrapolation beyond the measured range). `COHORT_SIZE =
400` is therefore a deliberate choice near the low end, not a tuned one — raise
it and the agent recommends more, which is a statement about data, not about the
method.

This table used to score only 60 patients, too few for a stable headline
abstention estimate. The reference evaluation now uses 240 patients and must be
regenerated whenever the decision threshold or its multiplicity correction
changes. Do not lower `power.DEFAULT_PATIENTS` back.

**Does any of it survive data it was not fit on? `cli transfer` asks.** Every
other number in this repo is measured on the process the estimators were built
against, and their blip basis *is* that process's basis. `curvature` and
`blip_modifier` bend that one cohort along two known axes; this fits at site A
and scores at site B. The shift is confined to things that leave the estimand
alone — case mix, prescribing habits, retention — because a bent estimand is
already measured and nothing survives it. n=400 per site, coverage replicated
over 30 site-A refits:

| site | abstains | coverage | SE/spread | ECE | IPW | local practice | rollout gain |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline (same process) | 68% | 97% | 1.11 | 0.004 | 0.734 | 0.639 | +0.386 |
| sicker, more seronegative | **89%** | 95% | 1.04 | 0.004 | 0.600 | 0.527 | +0.372 |
| milder, mostly seropositive | **57%** | 96% | 1.03 | 0.003 | 0.846 | 0.753 | +0.437 |
| TNF-first, rituximab-averse | 65% | 95% | 1.04 | 0.004 | 0.726 | 0.647 | +0.386 |
| heavy attrition | 75% | 97% | 1.10 | 0.005 | 0.748 | 0.669 | +0.469 |
| near-complete follow-up | 74% | 95% | 0.99 | 0.006 | 0.720 | 0.621 | +0.395 |
| combined: all three | 84% | 95% | 1.08 | 0.002 | 0.617 | 0.555 | +0.346 |
| **estimand shifted: blips 1.5x** | 70% | — | — | **0.051** | 0.824 | 0.695 | +0.538 |

Three of the four headline claims transfer and one does not.

*The policy transfers.* It beats local practice on rollout value at 7/7
estimand-preserving sites, by +0.346 to +0.469, including the site where
clinicians already prescribe TNF-first. *Coverage transfers*, holding 95-97% —
at nominal now that the dWOLS cross-arm covariance is kept rather than dropped.
*Calibration transfers*, 0.002-0.006 against 0.004 at home. Layer 1 rejected
**0** records at every site.

*Abstention does not.* It runs 57% to 89% against the 68% on the model card —
a 32-point swing driven by population changes, and in the direction that matters:
the sicker, more seronegative site is the one where the agent goes nearly silent.
That number describes a population at least as much as a method, and a
deployment that reads 68% as a property of the tool will be wrong by tens of
points. This is `cli subgroups` again at the population level, and it is the
single most portable finding here.

**The estimand-shifted row is the one to read twice.** Scaling every blip by 1.5
leaves the *ordering* of the arms intact, so the recommendation stays right and
the policy value stays good — rollout gain +0.538, the best in the table — while
every reported Q-value is wrong by half. Calibration catches it at 0.051, an
order of magnitude above every other row. **Policy value cannot detect an
estimand shift and calibration can**, so a deployment monitor that watches
value alone is blind to exactly the failure that makes the numbers on the
clinician card meaningless.

None of this advances the validation ladder. A second simulation is not live
data, `live_data` stays False, and no row here is evidence the agent would work
in a clinic. What it establishes is narrower: which claim is fragile, and in
what order they break.

**Abstention is not uniform, and `cli subgroups` says who absorbs it.** The
pooled rate hides a 58-point spread. Strata are the non-intercept terms of
`BLIP_BASIS` — the covariates the true effect actually varies over — at n=240:

| stratum | n | abstains | true gap | contrast SE | at pooled SE |
| --- | --- | --- | --- | --- | --- |
| anti-CCP negative | 90 | **97%** | 0.034 | 0.0179 | 97% |
| anti-CCP positive | 150 | 53% | 0.051 | 0.0169 | 74% |
| TNF naive | 158 | 76% | 0.038 | 0.0166 | 98% |
| prior TNF exposure | 82 | 57% | 0.056 | 0.0186 | 52% |
| das28 < 3.85 | 80 | 74% | 0.045 | 0.0167 | 78% |
| das28 3.85-5.89 | 80 | **39%** | 0.056 | 0.0155 | 78% |
| das28 >= 5.89 | 80 | **96%** | 0.032 | 0.0195 | 93% |

Two things to read here, and they point opposite ways. *Abstention is earned in
every stratum* — within each cell the declined patients have closer arms than the
recommended ones, which `tests/test_subgroups.py` asserts cell by cell rather
than pooled, and the most-abstaining stratum on each axis is the one with the
smallest true gap. But *precision is not evenly distributed*: the last column
holds each patient's own contrast and substitutes the pooled standard error, and
the difference is the part of that cell's abstention that is the model knowing
less rather than the arms being closer. In this run, low-activity and prior-TNF
strata carry about 2.5 and 2.4 points of excess abstention relative to pooled
precision.

Seronegative patients are the case a deployment has to be told about: 97%
abstention, and *none* of it is precision. Rituximab's blip carries `+0.14 *
anti_ccp`, so removing seropositivity removes the main thing separating the arms.
The agent is close to silent for 37% of this population and it is right to be.
Do not read that as a defect to tune out.

Inference has two paths and they are not interchangeable. The sandwich
(`sandwich_contrast`) is the default: cheap, exact at a stage-specific terminal
block, caveated everywhere else. The m-out-of-n bootstrap (`fit_bootstrap`,
`treatmentrx.cli inference`) re-runs the whole procedure per replicate and is
the honest answer wherever backward induction is involved — which with a shared
blip is every stage. It measures ~20% wider here. Once attached, `contrast()`
prefers it automatically.

Measured and stated rather than assumed: IPCW is a ~5% correction on this
generating process, and inverse-intensity weighting is slightly *negative*
(0.284 against 0.274 of total blip error), because the outcome model is correctly
specified and conditions on the covariates that drive both dropout and visit
frequency. Both are implemented and validated; intensity weighting is off by
default for that reason. Do not quietly re-tune the simulation to make either
look larger.

**A wrong blip basis is the failure real data has, and nothing survives it.**
`curvature` bends the *nuisance* surface, which double robustness exists to
survive. `blip_modifier` bends the **estimand**: the true effect varies over
standardised CRP, which is in the treatment-free basis and deliberately not in
`BLIP_BASIS`, so the estimators can adjust for it as a confounder and still
cannot represent it as an effect modifier. `cli misspecification
--omitted-modifier`, 25 replications at n=280 over a CRP-varying grid:

| blip modifier | pooled coverage | worst patient | bias | max patient bias | SE/spread |
| --- | --- | --- | --- | --- | --- |
| 0 (correct) | 100% | 100% | +0.003 | 0.006 | 1.14 |
| 0.05 | 48% | 0% | -0.035 | 0.087 | 0.44 |
| 0.10 | 35% | 0% | -0.058 | 0.138 | 0.35 |
| 0.20 | 29% | 0% | -0.125 | 0.279 | 0.35 |

The interval does not widen to absorb it — SE/spread *falls* — because a standard
error computed under the wrong basis has no way to see a bias in the estimand.
Every other robustness result in this repo is conditional on the blip basis being
right, and this is what that assumption is worth. There is no estimator-side
defence; the only defence is getting the basis right, which is a clinical
question rather than a statistical one.

**The specification test runs on every fit, not on request.** It costs 0.18s on
the deployed split and a caveat nobody runs is a caveat nobody sees. A flag
propagates three ways and changes no decision:

* `deployment_readiness()["blip_basis_unflagged"]` -> a **validation-ladder
  blocker**. A model whose contrasts are biased by an unknown amount does not
  advance a rung.
* an `Uncertainty` flag, `blip_basis_may_omit:<covariate>`, on every patient.
* a `CAVEAT` block on the clinician card, directly under the separation line it
  undercuts.

`BLOCK_ON_FLAGGED_BASIS` is False and that is a stated policy: the test is a
falsification test with an actionable fix — add the covariate and refit — not a
permanent property of the data, and converting a diagnostic into a silent
behaviour change is the pattern this repo keeps removing. Flip it for a
deployment that would rather withhold than caveat.

**`cli specification` is the part that transfers.** The study above compares
against `true_blip` and so exists only in simulation. The specification test
augments the blip block with a named candidate and tests whether its coefficient
is zero, which an analyst can run on their own data. Its operating
characteristics are measured, not assumed:

| blip modifier | detected (10 seeds) | mean max\|z\| for crp_std |
| --- | --- | --- |
| 0 | 0/10 | 1.42 |
| 0.02 | 0/10 | 2.78 |
| 0.05 | 10/10 | 5.68 |
| 0.10 | 10/10 | 10.95 |
| 0.20 | 10/10 | 19.70 |

False positives over 30 null cohorts: **0/30** against a nominal 5%. Getting
there took two corrections, both found by measuring rather than reasoning. The
first working version rejected on **33%** of null cohorts: it corrected
multiplicity over arms while running covariates-times-arms tests, and took the
sandwich at face value where `cli coverage` measures it at 0.86-0.90 of the
actual spread. Both are in `rejection_threshold`.

**A candidate modifier must be pre-treatment.** Pointing the test at
`alt_excess` rejects at z=4.35 on a cohort with no ALT effect modification at
all, and the statistic does not respond to the modifier that *is* present. ALT is
caused by the previous arm — mean 62.2 after a hepatotoxic arm against 27.9 after
any other — so interacting it with the current arm measures mediation. Half the
spurious statistic is differential censoring (dropout off takes 4.15 to 2.12) and
weighting fixes that; nothing fixes the mediation. `EXCLUDED_CANDIDATES` records
it.

**The near-flag on `das28_squared` is real and acting on it would be wrong.**
It tests at max|z| = **3.367** against a **3.669** threshold — 92% of the way to
firing, on the same covariate where `cli subgroups` finds 96% abstention and the
worst standard error in the cohort. That is not a coincidence and the mechanism is
not noise: the estimators target the blip, which is linear in `das28_std` by
construction, but the quantity the *decision* uses is the value-to-go contrast,
and backward induction composes the blip with a `max` over arms, which is curved
even when the blip is not. `cli misspecification --extra-modifier` prices adding
the term, at n=240 over three patient seeds:

| quantity | change |
| --- | --- |
| contrast interval width | **+13%** |
| pooled abstention | **+2.9 points** |
| top das28 tertile abstention | **-6.2 points** (on every seed) |
| bottom das28 tertile abstention | **+13.7 points** |
| total error on the four true parameters | -0.013 of 0.190, over twenty coefficients |

So the term does help the stratum the near-flag points at, consistently — and it
buys those six points by paying fourteen in the bottom tertile and three pooled,
while widening every interval in the system by 13%. Recovery of the parameters
that actually exist barely moves, so the gain is not coming from better
estimation. **Keep the four-term basis.** The falsification test's margin is
documented rather than acted on, and this is the counterexample to the obvious
misreading of `--omitted-modifier`: that if omitting a modifier is unrecoverable,
you should add every candidate.

The basis swap that study uses is a **test fixture, not a seam**
(`misspecification.estimator_blip_basis` patches the six modules that bind the
basis at import time). Do not promote it to configuration. That the estimator
basis *is* the generating basis here is exactly what makes every robustness result
in this repo conditional on the basis being right, and `--omitted-modifier` is the
measurement of what that assumption is worth.

`generate_ra_cohort(..., curvature=)` bends the treatment-free surface beyond
what the estimators' linear basis can represent, leaving the blips — and so the
estimand — untouched. It is 0 by default; every other result in the repo assumes
the correctly specified cohort. `cli misspecification` uses it to show the
estimators trading off as designed: dWOLS is the most accurate when the nuisance
model is right and degrades the most, the shared-blip fit the reverse. That
trade-off is the justification for averaging them rather than picking one.

When you touch one of these, either make it real or keep the docstring honest
about what it is not. The value of this codebase is that a reader can tell the
difference.

## Conventions

- Docstrings explain *why* a method was chosen and what it does not guarantee —
  match that register. Comments earn their place by explaining a clinical or
  statistical reason, not by restating the code.
- Dataclasses are `frozen=True`; construct new ones with `replace()` or
  `StageRecord(**(record.__dict__ | {...}))` rather than mutating.
- Tests assert on quantities (parameter recovery, policy value, calibration,
  standard errors), not on plumbing. When behaviour changes, prefer making a test
  data-driven over hard-coding the new answer.
- Anything that refits from scratch (stability, bootstrap) is a CLI command, not
  something on the inference path.
