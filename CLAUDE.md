# CLAUDE.md

TreatmentRx: a research-stage clinical decision-support agent for sequential
treatment decisions in rheumatoid arthritis. **Research scaffolding, not a
medical device.** Nothing here is clinically validated; the synthetic cohort is a
test fixture, not evidence.

## Commands

```bash
PYTHONPATH=src python3 -m unittest discover -s tests    # 722 tests, ~7 min
PYTHONPATH=src python3 -m treatmentrx.cli demo          # one patient end to end
PYTHONPATH=src python3 -m treatmentrx.cli evaluate      # estimator scorecard
PYTHONPATH=src python3 -m treatmentrx.cli stability     # k-fold + seed sweep (~10s)
PYTHONPATH=src python3 -m treatmentrx.cli inference     # sandwich vs bootstrap (~35s)
PYTHONPATH=src python3 -m treatmentrx.cli audit         # layer-by-layer evaluation (~5s)
PYTHONPATH=src python3 -m treatmentrx.cli coverage      # do the 95% intervals cover? (~76s)
PYTHONPATH=src python3 -m treatmentrx.cli coverage --candidate-set  # does the set hold the best arm? (~40s)
PYTHONPATH=src python3 -m treatmentrx.cli coverage --multiplicity   # what the all-pairs correction costs (~60s)
PYTHONPATH=src python3 -m treatmentrx.cli coverage --stages         # coverage at every stage, not only terminal (~40s)
PYTHONPATH=src python3 -m treatmentrx.cli misspecification  # estimator robustness (~15s)
PYTHONPATH=src python3 -m treatmentrx.cli misspecification --omitted-modifier  # blip basis too small (~4m)
PYTHONPATH=src python3 -m treatmentrx.cli misspecification --extra-modifier    # blip basis too big (~40s)
PYTHONPATH=src python3 -m treatmentrx.cli power          # data needed vs abstention (~85s, 4 cohort draws per size)
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
  formulary.py   the molecules each arm may be prescribed as, and their hazards
  simulation/    the cohort with known blips, plus a FHIR exporter so
                 simulated patients re-enter through Layer 1;
                 `survival_cohort.py` is the same contract for a
                 time-to-event endpoint, with `estimation/survival_dwols.py`
                 the estimator scored against it; both wired to nothing yet
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

### Where to look first

The invariants below are one defect each, in the order each was found. That is
the right order for the record and the wrong one for a reader about to change
something — `q_values` alone has bitten **fourteen** times, spread from 5 to 68.
So this is the other index: the thing you are touching, and every invariant that
has already gone wrong on it. Read the row before the diff, not after.

| touching | read |
| --- | --- |
| `q_values`, and anything that ranks from them | 5, 9, 32, 40, 45, 46, 53, 54, 55, 56, 58, 68, 71, 73, 79 |
| `recommended_arm` / `top_scored_arm` | 2, 5, 36, 46, 48, 59, 79 |
| what the card decomposes | 32, 54, 56, 57, 58, 60, 71 |
| standard errors and the covariance | 38, 58, 65, 66, 68, 74 |
| `linalg` | 23, 38, 66, 67, 68, 72, 74, 77 |
| the serving ensemble | 16, 18, 19, 42, 46, 53, 56, 71 |
| the candidate set and the abstention rate | 21, 22, 24, 32, 55, 77 |
| the arm and molecule vocabularies | 3, 62, 63 |
| Layer 1 ingestion and the contract | 6, 13, 51, 64 |
| policy value and off-policy evaluation | 30, 39, 41, 42, 43 |
| the validation gate | 14, 25, 41, 79, 80 |
| the blip basis | 57, 68, 72 |
| the safety layer | 1, 35, 54, 61, 70 |
| `cli audit` and the layer sections | 2, 33, 36, 62, 65, 66, 69 |

It reaches **56 of 80**. The rest are one-offs — a single
component, found once, unlikely to be what you are holding — and they are not
listed here because a row of one is not an index, it is a search result.
`tests/test_docs.py` derives this table from the invariant bodies and fails when
it goes stale, which is the only thing that makes an index worth having.

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
   composite action space and the feasible-set filter must all agree. **And one
   molecule vocabulary**: `formulary.py` — see invariant 62, which is this same
   rule a layer down, where four literals had already drifted.
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
6. **Leakage guards raise — all of them, which was not true until invariant 69.**
   Any violation from `LeakageTestSuite` raises `LeakageError` out of
   `DataLayer.build_patient_state`. Never downgrade one to a diagnostic. It used
   to read `temporal_firewall_passed` alone while the suite ran four checks, so
   the other three annotated the record and it was served.
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
    patients — 66% at the training population, 54-89% across `cli transfer`
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
    back to 67%, by removing width that was an error rather than a margin; keeping
    the care-goal bar off the clamped `q_values` (invariant 53) then took it to
    65%.

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
    set — and outside `removed_arms` (invariant 54) — because hard-coded prose
    explaining away an arm the model could not exclude, or one a rule refused
    rather than an interval, is the card asserting a clinical judgement the model
    never made.

33. **`service.py` is transport, and the model card is not optional.** The HTTP
    layer decides nothing: `tests/test_service.py` asserts the served
    recommendation equals the orchestrator's for the same bundle, because a
    second decision path is the failure this file opens with. It binds to
    loopback, caps the body, and returns no traceback to a caller — a stack
    trace quotes field values and field values are PHI. `GET /model` exists
    because the agent abstains on ~66% of patients, and a consumer that
    reads `equipoise` without knowing that reads a failure instead of a measured
    statement about sample size. The card also names the four unadjusted
    confounders, the measured interval coverage, and that only six covariates
    reach the estimators.

    **A hardcoded number on the card drifts from the study it cites.** The
    abstention block carried prose saying the pooled rate does not transfer —
    correct, and there since `cli transfer` was written — beside a single
    machine-readable field, `pooled_rate: 0.66`. So the structure invited
    exactly the reading the strings warn against: a consumer reading numbers saw
    a point estimate and nothing else. `population_range` (0.54-0.89, `cli
    transfer`) and `stratum_range` (0.38-0.97, `cli subgroups`) are now fields
    rather than sentences.

    Writing them found the drift they exist to prevent: `not_uniform` claimed
    "39% to 97%" where the study measures **37.5% to 96.7%**. One digit, and
    harmless on its own — what was not harmless is that nothing connected the
    card to the study, so it could drift arbitrarily far. `tests/test_service.py`
    asserts the published ranges *bracket* what `subgroup_report()` actually
    returns, which is the property; pinning the digits would need updating in
    two places on any legitimate change and is how this went stale.

    **A third range, and the one the card was missing entirely.** Three commands
    report a pooled rate for the deployed fit and they disagree — `cli power`
    0.663, `cli subgroups` 0.683, `cli audit` 0.692. That 3-point spread is
    patient sampling on a rate this size (about 0.03) and none of them is wrong.
    The spread worth knowing is four times larger and had never been measured:
    redraw the *training cohort* at the same size and the rate runs **0.563 to
    0.688** (invariant 78). `training_draw_range` carries it, and
    `why_the_published_figures_differ` says which spread is which, so a reader
    who runs two commands and gets two numbers finds the reason on the card
    rather than concluding one is a bug. A test asserts that range stays wider
    than patient-sampling noise — if it ever narrows to that width it has
    stopped describing the fit, which is the only thing it is for.
34. **No cross-disease fallback.** `DiseaseRegistry` must resolve exactly one
    registered definition before Layer 1 runs. A missing disease workflow is a
    typed error; it must never reuse RA arms, models, safety rules or evidence.

    **That guarantee rested on two upstream guards and nothing at the point of
    use, which was measured by trying it.** Registering a second disease and
    walking the pipeline: the registry routes correctly and the contract fails
    closed on the diagnosis, then `data/dag.py` fails closed on the graph — both
    right. But patch only those two and a record reading **Breast Cancer** is
    served a **rituximab** recommendation, scored over the **RA arm vocabulary**,
    while the definition declared an entirely different menu.

    The mechanism is that `DiseaseDefinition.treatment_arms` is decorative on
    the serving path. `state.feasible_arms` comes from
    `RADataContract.treatment_arms`, a class attribute hardcoded to
    `arms.TREATMENT_ARMS`; the definition's own tuple is carried, reported in
    `capability()`, and reaches neither Layer 1 nor the estimators. The
    `EstimandContract` does carry the declared menu and is stamped on every
    estimate, and nothing compared it to what was being scored — invariant 3's
    rule left to curation, and invariant 42's shape on a field that *is*
    constructed. Today all three agree only because all three read
    `arms.TREATMENT_ARMS`.

    `EstimationLayer.estimate` now refuses a menu the contract does not declare,
    which is one comparison at the place the models are used.
    `tests/test_scientific_contracts.py` asserts the three declarations agree,
    that a mismatched contract raises, and that the breast-cancer case is
    refused **with both upstream guards deliberately disabled** — because the
    point is that this invariant must not rest on them alone.

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

   **The fix reached one of the two branches, and invariant 70 found the other.**
   `SafetyLayer._status` carries the lesson in a comment beside its REVIEW
   branch; the BLOCKED branch fifteen lines below still said *"Routed to
   clinical review rather than substituting the next-best arm."* on the status
   that **stops**. The sweep that now guards it reads every flag the pipeline
   can raise rather than any one message.

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
   the arm the agent published (41 patients, 100%, regret 0.0 — trivially high,
   because it commits only when the gap is large). `if_forced_to_commit` scores
   `top_scored_arm` over everyone (120 patients, 92.5%, mean regret 0.0003, max
   0.0101) and is the ranking itself. Both carry `patients`.

   `abstention_price` then prices the system's defining behaviour instead of
   asserting it, over the 79 declined patients: taking the model's own top arm
   costs **0.0005** mean, the worst arm in the candidate set **0.0499**, the
   worst arm on the menu **0.2101**. That last pair is the retrospective case for
   the candidate set — handing back a bare status was, in regret terms, roughly
   4.2x worse than handing back the set. That ratio was 6x before the all-pairs
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

   **That identity held by luck per arm until invariant 68, and the test was too
   narrow to see it.** It checked `rituximab`, which returned exactly 0.0, while
   `JAK-inhibitor` was returning **8.065e-10** at the same moment. Two causes,
   both now removed. `Cov(beta, beta)` was recomputed through two plain
   `matmul`s while `Var(beta)` came from `sandwich_product`, which exploits
   symmetry and orders the same sums differently — `cross_covariance` returns
   `self.covariance` outright when handed itself, because that *is* the
   identity. And `contrast_standard_error` assembled the variance from
   `blip_standard_error ** 2`, squaring a square root, which is not exact; it
   now takes one loading vector through three quadratic forms and one final
   square root. A residual near 1e-19 in a variance becomes 1e-9 in a standard
   error, which is why a nine-place tolerance hid it. The test sweeps every arm
   and asserts **exact** zero.

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
   next to `cli power`'s 1,430 for 30% abstention. Both say the same thing about
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

47. **A keyword index that cannot match its own vocabulary, on a served field.**
   Six commits had all come from the statistical core; the first pass over Layer
   5 found this. `SemanticKnowledgeBase.retrieve` split the query on whitespace,
   and the arm vocabulary is hyphenated while the knowledge-base keys are not —
   so `TNF-inhibitor` never matched `tnf inadequate response`. Measured:
   **`continue-current`, `methotrexate-optimization`, `TNF-inhibitor` and
   `JAK-inhibitor` retrieved zero passages from their own name**, four of six
   arms. Only `IL-6 inhibitor` (it contains a space) and `rituximab` (no hyphen)
   worked. This is invariant 26 with the sign reversed: there a short token
   matched too much, here a compound token matched nothing.

   The second defect was hidden behind the first. The query is the recommended
   arm *plus* the history summary, every match scored one point, and `sort` is
   stable — so ties fell back to the order passages appear in `_KNOWLEDGE_BASE`.
   The demo patient is recommended **rituximab**, the knowledge base contains a
   rituximab passage, and the two retrieved were about TNF response and
   methotrexate. A card printing `Evidence:` under a recommendation was citing
   passages about something else, and `Recommendation.evidence` is served.

   Tokenise on words, and separate the **subject** — what the evidence is
   supposed to be *for* — from the history around it. A subject match outranks a
   history match; remaining ties break by knowledge-base order, arbitrary but
   deterministic and auditable. Every arm that has a passage now retrieves it,
   and `continue-current` correctly retrieves none because none exists.

   It is still five hard-coded passages and still not a vector store. What
   changed is that the keyword index matches keywords — the docstring was honest
   about the *method* and silent about the fact that it did not work.

48. **A warn flag may not name an arm nobody recommended.**
   `SafetyRules._delayed_toxicity` read `decision.recommended_arm`
   unconditionally and phrased itself as "monitor ... before continuing" — but
   on an equipoise decision that field is only the argmax, and Layer 3 has
   already declared it inseparable from the candidate set. So the flag named an
   arm nobody had recommended and attributed an intent to continue it. That is
   invariant 2's defect in a rule rather than a status, and being a *warn* is
   why it survived: it stopped nothing, so nothing caught it.

   Gating on the argmax also **suppressed** the warning. The trigger required the
   argmax to be hepatotoxic, so an undecided patient whose argmax happened to be
   benign lost the warning even though the candidate set still held hepatotoxic
   arms — the trajectory evidence is identical either way. Measured over 120
   patients the flag fired **once**; it now fires **eight** times, all on
   undecided patients, and names an arm only when one was published.

   The trajectory observation is the same in both cases; what changes is the
   claim. With a recommendation it is about that arm. Without one it is about the
   arms still under consideration and says so, with `affected_arm` left `None` —
   because a name appearing there is how invariant 2 detects a promotion.

49. **Layer 1's headline metric measured the predicate it used as truth.**
   Running the whole workflow and reading each layer's audit side by side, Layer
   1 was the only one whose every metric sat at its ceiling — 40/40, 1.0, 1.0,
   1.0. Three of those are round-trip identities and honest. The fourth was not.

   `switch_detection_recall` asked whether *any* stage was flagged for a patient
   whose arm changed. `SwitchingCapture`'s third condition **is** "the arm
   changed", so the metric tested the same predicate it used as ground truth and
   read 1.0 by construction — invariant 25's defect wearing an accuracy figure.
   It compounded that three ways: `any()` meant flagging the *wrong* stage
   counted as a hit; the denominator counted patients rather than stages; and it
   excluded every patient who never switched, which is the only place a false
   positive could appear.

   **The near-miss is worth recording.** Measuring per-stage precision against
   "the arm changed" gives 0.836, with 15 of 18 never-switching patients
   flagged — which reads as a badly over-firing detector. It is not.
   `switched` is the union of four conditions: a recorded discontinuation
   reason, a dispensed name differing from the order, an arm change, and a
   loss-of-response note. Only the third has ground truth here, so precision
   against it measures the wrong thing. That was the fourth time in this sitting
   that a plausible number turned out to be answering a different question, and
   the check that caught it was reading the detector before believing the metric.

   What the audit reports now is what is knowable: `arm_change_always_flagged`
   (1.0, **labelled a wiring check rather than an accuracy figure**),
   `switches_beyond_arm_change` (the share resting on conditions the simulator
   cannot verify — 11%, 10 of 92 stages), and the two dead seams as explicit
   zeros the way `SwitchingAwareOPE` reports its own: **no stage has a dispensed
   name differing from the order**, so `SwitchingRecord.realized` only ever
   echoes `assigned` and the ITT / per-protocol / as-treated distinction has no
   input from that field (`FHIRAdapter._supply_records` already said so); and
   `adherence` takes **one** distinct value across 153 stages, so the
   days-covered path never runs.

   Both zeros are properties of the fixture rather than of the code — a bundle
   carrying `MedicationDispense` resources would move them — and saying which is
   the difference between a gap and a defect.

50. **The randomized-trial module's 95% interval was an 86% one at small n.**
   `RandomizedTrialPrecisionAnalyzer._fit` computes HC2 standard errors — the
   right choice for a randomized contrast — then took a **normal** critical
   value, while computing, storing and reporting `residual_degrees_of_freedom`
   and not using them. A standard error estimated from the same small sample as
   the effect is itself uncertain, and the normal quantile ignores that.

   Measured with the module's own `simulate_precision_power` under the null:

   | n | unadjusted | adjusted | after the fix |
   | --- | --- | --- | --- |
   | **8** (its accepted minimum) | **0.101** | **0.138** | 0.053 / 0.064 |
   | 12 | 0.060 | 0.086 | 0.045 / 0.049 |
   | 40 | 0.058 | 0.058 | 0.048 / 0.052 |

   The *adjusted* analysis was the worse of the two, which is the tell: it
   spends a further degree of freedom that a normal quantile cannot see, so the
   precision feature the module exists to demonstrate was the one paying most.

   `inference.student_t_critical_value` is the fix, solved by bisection on the
   exact t CDF through a regularized incomplete beta written out in full — this
   package has no third-party dependencies and a Cornish-Fisher expansion is
   0.25% low at five degrees of freedom, which is the end of the range this is
   for. It matches published tables to four decimals from df=1 to df=1000 and
   `tests/test_inference.py` pins it there.

   **`_required_sample_size` had the same inconsistency from the other side.**
   It planned with a normal quantile for a test that now uses t, so at small N it
   promised power the test would not deliver. Validated against
   `simulate_precision_power` at the N it returns, measured power ran 0.747 to
   0.834 against a target of 0.80 with the miss at the smallest N — 46, where
   t(43) is 2.017 against z of 1.960. It now solves by substitution with the
   analyzer's own critical value. The correction is properly df-dependent — 4% of
   N at N≈50, 0.3% at N≈675 — and too small for a 1200-replicate simulation to
   resolve against the noise a variance estimated from the pilot introduces, so
   the claim is internal consistency rather than a measured power gain.

51. **A field that cannot take its other value is not a check.**
   `PrognosticScore.within_validated_domain` was assigned `True` at the only
   place a score is constructed and could be nothing else, because
   `FrozenPrognosticModel._validate_domain` **raises** on a categorical
   mismatch. So it restated "you got a score at all", and a consumer branching
   on it wrote dead code — invariant 25 on a public contract field.

   What was missing is the numeric half. The domain constrained disease,
   modality, endpoint, horizon, site and platform, and said nothing about the
   *range of feature values* the artifact was fit on. A frozen linear score
   applied to a CRP of 400 when it was derived on 0-50 is extrapolating, and
   that is the classic way an external biomarker fails.

   `BiomarkerDomain.validated_ranges` supplies it, and the split mirrors what
   the main model already does: a categorical mismatch raises the way
   `data/contract.PLAUSIBLE_RANGES` raises on an impossible value, while an
   extreme-but-admissible patient is flagged the way
   `safety/rules._out_of_support` flags one. `out_of_range_features` names which
   covariate left the range rather than handing back a bare False, and an
   artifact declaring **no** ranges leaves the question unjudged rather than
   asserting safety — `capability()["declares_validated_ranges"]` says which,
   because "does not know" and "is fine" must not read the same.

52. **`satisfies_backdoor` is right; `minimal_backdoor_set` was not minimal.**
   The DAG is what licenses calling any of this causal, so it was checked
   against graphs with known answers rather than against the one it ships with.

   **The criterion holds.** Plain confounder, mediator (a descendant, correctly
   refused), **M-bias** — `T <- U1 -> M <- U2 -> Y`, where the empty set is valid
   and `{M}` is not, because conditioning on the collider *opens* a path that was
   closed — and a **descendant of a collider**, which opens it just the same.
   Those last two are what separate a real implementation from a plausible one,
   and `path_is_blocked` gets both. The claim in this file was earned.

   **The minimiser was not.** It walked candidates in sorted order and added one
   whenever the set so far did not yet block, without ever testing whether *that
   node* helped — so a redundant node got in by sorting first. On `T <- Z -> Y`
   plus `T <- Z <- W -> Y`, where `{Z}` blocks both paths alone, it returned
   `{W, Z}`. It also carried a `without = ...` local computed and never read,
   which is what a half-finished condition leaves behind.

   It happened to be right where it is used: every backdoor path in the RA graph
   is a single confounder with arrows into both treatment and outcome, so nothing
   substitutes for anything and the greedy order cannot matter. **The deployed
   adjustment set did not change** — `minimal_backdoor_set` still agrees with
   brute force on the RA graph, and a test asserts that rather than assuming it.

   It matters anyway, because `_split_by_what_the_model_carries` derives
   `unmodelled_confounders` from this, and that list is what the model card
   reports as *unadjusted residual confounding*. A spurious entry there claims a
   failure that never happened — the mirror of invariant 25, a check reporting a
   problem it does not have rather than missing one it does.

   Now two passes: cover until every path is blocked, then **prune** — drop any
   node whose removal still leaves the set blocking, which is the definition of
   irreducible and the step that was missing. Both iterate in sorted order so the
   result stays deterministic. *Irreducible*, not *minimum* cardinality: the
   latter is NP-hard in general and the docstring says which one is on offer
   rather than implying the stronger guarantee.

53. **The fourth place the leader was re-derived from a display quantity.**
   Invariant 46 fixed three — the estimator facades, `BayesianModelAverager`, and
   `DecisionLayer._ranked`. `GoalConditionedThresholds.decide` was the fourth and
   it was missed: it sorted `q_values`, which is clamped to
   `[Q_FLOOR, Q_CEILING]` and rounded to 3dp, and took the top-two difference.

   Measured over 120 patients, **every one of the nine whose predicted response
   saturated the ceiling had a top-two gap of exactly 0.0000**. Arms the model
   distinguishes, reported on the clinician card as "Q-gap over the next-best arm
   is 0.000" and failing the care-goal action bar for an arithmetic reason.

   It also makes invariant 10 exact rather than approximate. The care goal sets
   how large a difference is worth acting on and the interval decides whether the
   data can resolve *a difference that size*; computed from different quantities,
   "that size" was ambiguous. `DecisionLayer` now builds the contrast first and
   hands `decide` its own difference, so both conditions read one number and a
   test asserts they agree.

   Signed, not absolute. The contrast can be negative in the honest case where
   the two serving members disagree and the weighted vote picks the leader — a
   leader scoring *below* its comparator should fail the bar directly rather than
   clear it on magnitude and be caught downstream by the interval condition.

   **This one moves the rate, which is the direction to be careful about.**
   Abstention went 71% to 66% on the audit fixture and 67% to 65% on `cli power`,
   because nine patients were being held at equipoise by a clamp rather than by
   the data. The evidence that the recovered decisions are earned, and it is the
   check invariant 32 asks for:

   | | before | after |
   | --- | --- | --- |
   | `when_it_commits` | 35 patients | **41 patients** |
   | oracle-arm rate when it commits | 1.000 | **1.000** |
   | max regret when it commits | 0.0000 | **0.0000** |
   | true gap, recommended | 0.0811 | 0.0700 |
   | true gap, declined | 0.0257 | 0.0273 |
   | `abstention_is_earned` | True | True |

   Every one of the six recovered commitments is the oracle-optimal arm at zero
   regret. The recommended group's mean true gap falls because the recovered
   patients sit lower in it than the ones already there, and the gap between the
   two groups narrows from 3.2x to 2.6x while staying clearly separated — which
   is what recovering suppressed decisions looks like, as against lowering a bar.
   `cli power`'s extrapolated target moves 1,490 to 1,373 trajectories for 30%
   abstention (and to 1,430 after invariant 55).

54. **The candidate set is not empty on a recommendation, and three card blocks
   did not know that.** Found the way invariant 35 was — rendering all four
   statuses side by side rather than reading the code. Status is decided on the
   top-two contrast alone, so a *lower-scoring* arm with a wider interval can
   survive the exclusion test while the runner-up fails it. Measured over 120
   patients, **6 (5%)** were recommended with a two-arm candidate set, and their
   cards read "recommend methotrexate-optimization" in the headline and, four
   paragraphs down, "Cannot separate: methotrexate-optimization, rituximab ...
   **This is not a recommendation**".

   **The information is right and only the framing was wrong**, which the oracle
   settles. On all four of the clearest cases the leader and the surviving arm
   have a true value of **1.0000 each** — genuinely tied at the optimum — while
   the comparator the separation line names is worth 0.96-0.99. The set is
   correct, the recommendation is correct (oracle arm, zero regret), and what was
   missing was a sentence saying how both hold at once. So the block is re-worded
   per status as `Not excluded:` rather than suppressed: hiding it would make the
   card more confident than the evidence, which is the direction invariant 32
   warns about. The block's own rate does not move.

   It stays reachable after invariant 55, and the reason is worth keeping: the
   smallest *difference* is not the smallest z. Measured over 240 patients, 3 of
   71 recommendations have a comparator at +0.049 with SE 0.0144 (z 3.41,
   separable) while a second arm at +0.050 with SE 0.0176 (z 2.86) cannot be
   excluded. RECOMMEND does **not** mean "separated from every arm", and this
   block is what says so.

   **Two more places the same discipline was missing.**

   `WhyNotEntry.q_gap` was `q_values[best] - q_values[action]` — invariant 46's
   display quantity in a fifth place, and the card is where it showed:
   "TNF-inhibitor (gap 0.000)" one line under "Separation: ... over TNF-inhibitor
   is +0.051 — separable at this sample size". **4 of 120** printed exactly 0.000
   under a separable verdict; **29** disagreed with the separation line by any
   amount, by up to **0.065**. The decision layer already holds a model-averaged
   interval for the leader against *every* arm — it is what the candidate set is
   built from — so `candidate_contrasts` is computed first and handed to
   `ModelExplainer.explain`, exactly as `contrast` already was for the E-value.
   The two numbers now agree by construction (max residual 0.0005, which is
   `q_gap`'s third decimal). Entries are ordered by the gap they print, because
   the renderer shows the first two.

   `_why_not` filtered on the candidate set alone, so an arm **Layer 4 removed**
   still collected a model reason: with pregnancy injected, **79 of 240** cards
   carried one, and the words were the wrong kind of wrong — "why not
   JAK-inhibitor — organ-function / safety profile reduces net benefit" for an
   arm the same card reports as contraindicated in pregnancy. A contraindicated
   arm is not an option that lost on merit. Nothing is lost by dropping them: the
   safety layer raises an `arm_removed` flag for every removal, so each is
   already named with the reason that applies.

   Two smaller ones, from the same pass. `patient_summary` appended " ...because
   the options are close" when `goal_decision.act` was False, and
   `DecisionLayer._status` returns RECOMMEND only *after* `act` is True — so the
   hedge could not fire, on any patient, ever (41/41). It is removed rather than
   rewired: the 51 patients who clear the care-goal bar and fail the interval
   condition are the ones it was written for, and they already get the EQUIPOISE
   summary, which says the options are close in its first sentence. And
   `blocked_card` carried the separation interval without `_basis_caveat`, the
   one caveat that undercuts it — the reviewer got the number without its
   qualifier. Latent on this build (`blip_basis_unflagged` is True), so it is
   closed by inspection and pinned by a test that injects the flag.

55. **The arm the recommendation was justified against was chosen by dictionary
   order.** The last place invariant 46's display quantity reached a clinical
   output, and it was invariant 54's residue. `DecisionLayer._contrast` took the
   pair from `_ranked()[1]` — the argmax over the **clamped, rounded** non-leader
   `q_values` — so for a patient whose response saturates the ceiling three arms
   sit at 0.99 and a stable sort settled it. Measured over 120 patients: **5**
   had a tied comparator, and for **4** the arm dictionary order picked was
   separable (+0.051, z 3.1) while the genuinely closest arm was not (+0.018).

   The comparator is now the minimum of the contrasts `_candidate_set` already
   computes, and the pair is not computed here at all — building the runner-up's
   interval twice was how the separation line and the set could report different
   verdicts about the same arms. Selecting the minimum is a search over all five
   comparisons, which is exactly the family `_simultaneous_alpha` corrects for
   (invariant 32), so no level changes. It also makes invariant 10 exact in the
   remaining place: the care-goal bar now judges the gap to the *nearest*
   competitor, which is the only thing "large enough to be worth acting on" can
   mean.

   **It moves the rate, and every check says the right way:**

   | | before | after |
   | --- | --- | --- |
   | status distribution | 41 / 79 | **37 / 83** |
   | oracle-arm rate when it commits | 1.000 | 1.000 |
   | max regret when it commits | 0.0000 | 0.0000 |
   | `if_forced_to_commit` (the ranking) | 0.925 / 0.0003 / 0.0101 | unchanged |
   | true gap, recommended | 0.0700 | **0.0775** |
   | true gap, declined | 0.0273 | **0.0260** |
   | worst arm in the candidate set | 0.0499 | **0.0475** |
   | `abstention_is_earned` | True | True |

   The gap between the two groups widens from 2.56x to 2.98x: the four patients
   it now declines are exactly the ones whose alternative is **truly tied at the
   optimum** (both arms worth 1.0000), so abstaining there costs no regret and
   the set it hands back holds two optimal arms. The ranking is untouched —
   `if_forced_to_commit` does not move a digit — so this is recommending *less
   often and about larger gaps*, not ranking differently.

   **It does not make RECOMMEND mean "separated from every arm",** and that is
   the easy thing to assume. The smallest difference is not the smallest z; see
   invariant 54's closing paragraph and `_not_excluded`, which is the card block
   that reports the difference. Taking the z-minimum instead would hand the
   care-goal bar the *larger* of two near-identical gaps, which is the permissive
   direction.

56. **A number attributed to "the model" when two models are named above it.**
   `q_values` are averaged and psi is not — dWOLS's is a single-visit blip and
   Q-Pooled's a stage psi from a value-to-go fit, so summing them term by term
   mixes the scales invariant 9 keeps apart. `BayesianModelAverager` therefore
   carries the **dominant member's** coefficients, and the card renders that
   decomposition directly under a line reading "model-averaged over 2 estimators
   (weights Q-Pooled 0.50, dWOLS-Shared 0.50)".

   Which member is dominant turns on a weight margin of **0.003** on the deployed
   fit (0.4985 / 0.5015), while the two members' blips for the same patient
   differ by up to **0.041** — the size of the contrast the decision reports. A
   hair's-breadth change in the BMA weights would swap the published attribution
   by more than the effect it explains.

   Not repaired by averaging, which the scales forbade at the time: recorded and
   named. `attribution_source` is stamped in `aggregate`, carried on the audit
   event, and rendered — "+0.244 (dWOLS-Shared's blip, not the ensemble
   average)". **Invariant 58 removed that obstacle and invariant 60 does the
   averaging**, so the margin no longer decides anything; what survives from here
   is `attribution_source` itself, which now reads `BMA Ensemble`.

   **And it is what Layer 5's audit should have been measuring.** That section
   read 1.0 / 0.0 / 0 on every metric, the tell invariant 49 describes, and two
   of the three could not have read anything else.

   `attribution_sums_to_advantage` tested its own arithmetic: `_attributions`
   sets `total = round(sum(contributions.values()), 4)` from parts already
   rounded to 4dp, so the residual is **5.6e-17** — float noise, not evidence.
   Kept as `attribution_parts_sum_to_total` and *labelled a wiring check*, with
   the real question asked separately: `attribution_matches_its_source_model`
   recomputes psi . h(X) from the fitted model rather than reading the estimate's
   own coefficients back to themselves. It discriminates **261x** — 0.000158
   against the member it names (the 4dp floor), 0.041182 against the other
   serving member — so it can fail.

   `phi_leaks_into_narrative` scanned 30 cards for the patient hash while the
   only two sections carrying patient-specific free text — `Continuity:` and
   `Recorded patient preferences:`, both fed from episodic memory — were empty
   for every one of them, because the loop scores fresh simulated patients. The
   guard swept cards that structurally could not contain what it looked for:
   invariant 36's denominator defect on top of invariant 25's. It works — record
   one episodic item whose free text carries the hash and it fires — so the
   memory path is now exercised and the denominator reported beside the count.

   Three details that are the difference between the fix and the appearance of
   one. The **0 of 30** is measured and printed rather than argued in a
   docstring, the way invariant 49 reports its dead seams. The fixture fills
   **both** free-text routes — `preference` renders under `Recorded patient
   preferences:`, `outcome_summary` is interpolated into `Continuity:`, and
   `override_reason` is stored and never rendered — because a fixture exercising
   one of two routes is the same narrowing one level down; a test pins that
   inventory. And `cards_scanned` counts the fixture card, because `leaks` is
   summed over it: a count and a denominator taken over different populations is
   the defect being fixed, reproduced inside the fix.

   **`history_summary` is the suspect that does not reach the card**, and it is
   worth recording as a negative because it looks like it should. Layer 1 builds
   it from the stage list and it reaches Layer 5 through `provenance`, but the
   only thing that reads it is the retrieval query; the card renders `care_goal`
   and `top_tailoring_vars` from the patient context and nothing else. It *is*
   returned in the served provenance block, which is a different question from
   this one — that block goes back to the caller who supplied the record.

   One more number that was wrong in the same register: the identity check
   reported `round(residual, 12)`, which renders 5.6e-17 as an exact **0.0**.
   An exact zero asserts an identity the arithmetic does not have and hides the
   one magnitude that shows the check is a wiring check, so it is reported to
   three significant digits instead.

57. **The card's reason for ruling an arm out was prose keyed to the arm, and it
   named a parameter the model does not have.** `explainability.py` opens by
   calling its explanations "faithful, model-derived" and saying Layer 5 "never
   invents them". The *gap* was model-derived — invariant 54 made it the
   decision's own averaged contrast. The sentence beside it was not.

   `WHY_NOT_REASONS` held one string per arm, so the explanation could not vary
   with the patient while the number beside it varied correctly. It printed on
   **120 of 120** cards. And one entry was worse than generic: `JAK-inhibitor`
   read "organ-function / safety profile reduces net benefit" on **20 of 120**
   cards, while `BLIP_BASIS` is `(intercept, das28_std, anti_ccp, prior_tnf)` —
   **there is no organ-function term**, so the card asserted a mechanism the
   model has no parameter for. Invariant 54 had already stopped these entries
   printing for arms Layer 4 removed, which sharpened it: that safety-flavoured
   sentence printed *only* for patients whose organ function cleared every rule
   on the same card.

   The replacement is the gap's own decomposition. Every blip is one-vs-reference
   over `BLIP_BASIS`, so the leader-versus-arm contrast is
   `(psi_leader - psi_arm) . h(X)` and splits term by term. `BLIP_TERM_LANGUAGE`
   is keyed on the **covariate**, not the arm: the model picks which term
   dominates and with what sign, and the map only supplies words. The reason
   names the largest term working *for* the leader — not the largest absolute
   one, since a term in the arm's favour does not explain why it lost — and names
   the offsetting term when one is large enough to change how the first reads.

   Measured over 120 patients, 600 entries: the dominant reason takes **19 to 55
   distinct values per arm** where the old table had exactly one, and all four
   basis terms get named (anti-CCP 41%, baseline 32%, disease activity 18%, prior
   TNF 9%). The terms are read from `selected.coefficients`, the same place
   `_attributions` reads them, so both blocks describe the member
   `attribution_source` names — and for the reference arm they agree **exactly**
   (max difference 0.000000 over 120), because the gap over continuing current
   therapy *is* the leader's blip.

   **`q_gap` stays the averaged contrast**, so it still matches the separation
   line (invariant 54). The decomposition was one member's when this landed, so
   it reconstructed that gap only to within the members' disagreement — mean
   **0.0100**, max **0.0575**, against gaps reaching 0.306. Invariant 60 averages
   the blips too and takes that to **0.00018**.

   **That residual is a disagreement, and it would be a scale error if the source
   flipped.** `Q-Pooled` publishes a stage psi from a value-to-go fit *undivided
   by the remaining horizon*, while `q_gap` is per-remaining-visit — invariant 9's
   two scales. Measured at `stage_index` 1, a Q-Pooled-sourced decomposition runs
   **1.54x to 3.49x** the gap it claims to explain; at the terminal block, horizon
   1, both members agree. `attribution_source` is dWOLS-Shared for all 120
   patients on the deployed fit, so nothing served crosses scales today — but the
   BMA margin deciding it is **0.003** (invariant 56). So the reconstruction is
   asserted at every served stage rather than assumed, and the guard closes:
   forced onto Q-Pooled it fires at stage 1 (0.1434 against a 0.1 bar) and stays
   quiet at the terminal block, which is the right discrimination.

   **That guard is not the repair**, and the repair is invariant 58. The guard
   stays — it is what would catch the next way these two scales drift apart — and
   its bar tightened from 0.1 to **0.01** once invariant 60 made the healthy
   residual a rounding.

   What remains hand-written is a four-entry vocabulary of clinical words for the
   four basis terms, and a test asserts it covers `BLIP_BASIS` so a new modifier
   cannot reach a card as a bare variable name.

58. **A model published its point estimate and its standard error on two
   different scales.** Invariant 57 guarded this; this is the fix.
   `QLearningModel.blip_standard_error` divides by the remaining horizon and its
   docstring says "on the per-remaining-visit scale". `blip_parameters` does not,
   because it is the value-to-go parameter the model actually estimates. Both are
   right. What was wrong is that `coefficient_summary` — the audit-facing view
   that becomes `RegimeEstimate.coefficients` — published the *undivided* blip,
   and the only thing that consumes those psi keys is the explanation layer,
   which decomposes them into the per-covariate terms the clinician card prints
   **under a gap taken from `q_values`**. `q_values` are value-to-go divided by
   the remaining horizon (invariant 9). So the card would have shown a
   decomposition and a gap differing by exactly that horizon.

   **The division is exact, which is what makes it a repair and not a fudge.**
   `psi_a . h(X)` *is* `raw_q(a) - raw_q(reference)` by construction, so dividing
   by the horizon gives precisely `q_values[a] - q_values[reference]` — verified
   to the 3dp `q_values` are rounded to, at every stage.

   Measured, forcing `attribution_source` onto Q-Pooled:

   | | before | after |
   | --- | --- | --- |
   | worst residual, `stage_index` 1 | **0.1434** | **0.0560** |
   | worst ratio to the gap, `stage_index` 1 | **3.49x** | **1.90x** |
   | worst residual, terminal block | 0.0290 | 0.0290 |
   | invariant 57's guard at stage 1 | fires | quiet |

   What remains at stage 1 is the two members' genuine disagreement, the same
   thing the deployed dWOLS path shows (max 0.0575). The terminal block does not
   move because its horizon is 1, which is also why this was invisible: it is the
   stage `build_patient_state` puts almost every patient at (invariant 40).

   **Nothing served changed**, and that is asserted rather than assumed — cards,
   audit events and why-not entries over 40 patients hash identically before and
   after, because `attribution_source` is dWOLS-Shared throughout and dWOLS's
   blip is single-visit with no horizon to divide by.

   Three things deliberately not rescaled. `blip_parameters` stays the
   value-to-go parameter: `cli coverage`, `cli misspecification` and the
   parameter-recovery tests compare it against the generating process's blips,
   and dividing it would break the comparison it exists for. `beta:` stays raw —
   it is the treatment-free surface, not a contrast, nothing renders it beside a
   per-visit quantity, and nothing in the package reads it. And
   `top_tailoring_variables` ranks by `|psi_k * h_k(X)|`, which a positive
   constant cannot reorder, so it reads the accessor and is unaffected either way.

   `audit._attribution_against_its_model` divides too. It is the audit's
   *independent* recomputation, so it has to target the scale that is published;
   without the division it would have measured the rescaling rather than the
   model. `tests/test_estimators.py` pins the identity at every stage and asserts
   the horizon is greater than 1 somewhere, because a division by 1 everywhere
   would make the other two assertions vacuous — which is precisely the shape
   that hid this.

59. **Layer 6's audit measured two patients and never looked at what the layer
   is for.** The section reported the estimands, the rung and two OPE tripwires.
   `estimands_are_model_level` compared one patient against one other, so a
   patient-dependent estimand had to disagree on exactly that pair to be caught —
   invariant 36's denominator problem at n=2. It sweeps 60 now and reports
   `distinct_value_sets` beside the verdict; measured, 1 set over 60 patients.

   **What was missing entirely is the separation the layer's own docstring opens
   with.** Layer 6 keeps three tracks, and the rule is that an abstention must
   not reach Track B: there is no policy action to evaluate, and the diagnostic
   top-scored arm must never enter as though it were a recommendation. That is
   invariant 2's shape one layer down — a name appearing where nothing was
   recommended — and since the agent abstains on most patients the denominator is
   large and the property can genuinely fail. Nothing checked it. Measured over
   120 patients: 120 observational rows, **33 on the OPE track against 33
   recommendations**, and 87 abstentions all carrying `clinician-usual-care`.

   Both halves are counted separately and each can fail on its own, which
   `tests/test_coverage.py` asserts by injecting both regressions: promoting the
   top-scored arm into a policy action moves
   `abstentions_carrying_the_top_scored_arm` to 19 while
   `ope_rows_not_the_published_arm` stays 0, and smuggling abstentions onto Track
   B does the reverse (`ope_rows` 30 against 11 recommendations). One counter
   standing in for two properties is how a partial regression passes.

   **The rung now carries its blockers.** `validation_rung: silent` beside
   `retraining_allowed: false` told a reader the gate was shut and nothing about
   why, while `ValidationStatus.blockers` sat populated and unemitted — invariant
   41's defect in the section whose subject is the gate.

   Three metrics here are **regression tripwires, not measurements**, and are
   grouped and labelled: `estimands_are_distinct`, `ope_is_patient_level` and
   `ope_is_descriptive_only` are structural assertions guarding defects that were
   fixed and could silently return. None has a denominator, because a structural
   assertion does not have one. The test that covered them read
   `assertTrue(metrics["estimands_are_model_level"])`, which became vacuous the
   moment that metric grew a denominator and turned into a dict — every dict is
   truthy. It asserts the fields now.

   One caught in the writing, worth recording because it is the same defect one
   level in: the first version scored `n` patients and then ran
   `sample_ra_bundle()` once more for the validation status, which appended a row
   to every track — so it reported **61** rows against a denominator of **60**.
   The extra run is gone and the last swept patient is used instead.

60. **A 0.003 weight margin decided which model two card blocks described, and
   invariant 58 is what made averaging possible.** Invariant 56 found the margin
   and chose to *name* the winner rather than average, because the two members
   parameterise psi differently: dWOLS's is a single-visit blip and `Q-Pooled`
   published an undivided value-to-go stage psi, so summing them term by term
   mixed the scales invariant 9 keeps apart. That reasoning was right at the
   time. Invariant 58 put both on the per-remaining-visit scale, and the
   objection went with it.

   Averaging is not merely now-possible, it is the **right** quantity.
   `DecisionLayer._pair_contrast` builds the gap the card prints as exactly the
   BMA-weighted mean of the members' contrasts, so the decomposition beside it
   should be the weighted mean of their blips — which is invariant 16's
   principle ("the interval describes the quantity the decision uses") applied to
   the decomposition rather than the interval. Measured over 300 entries:

   | decomposition | mean residual vs `q_gap` | max |
   | --- | --- | --- |
   | dominant member (before) | 0.00983 | **0.05590** |
   | BMA-weighted (now) | 0.00005 | **0.00018** |

   0.00018 is the 4dp coefficient rounding floor, and it holds at **both served
   stages** — the members coincide only at the terminal block, so an average that
   worked there and nowhere else would not be this.

   **It reaches three card-facing things, not one.** The attribution block and
   the why-not decomposition both read `selected.coefficients`. So does
   `top_tailoring_variables`, whose docstring calls itself "exactly the
   decomposition `ModelExplainer` already reports" — a coherence claim that only
   holds while both read the same psi. Measured over 120 patients, the two
   members disagreed on a printed driver magnitude by up to **0.147**
   (`anti_ccp` at -0.016 against +0.131, *opposite signs* on the card) and on the
   **order** of the drivers for **11**. `aggregate` takes `features` now and
   ranks off the averaged blip.

   `attribution_source` becomes `BMA Ensemble` and the card says so: the line
   read "(dWOLS-Shared's blip, not the ensemble average)" and now reads "(the
   weighted average of the estimators above)". It is kept rather than dropped
   because the line above names two models and a reader is owed which object this
   is — the qualifier is a confirmation now instead of a caveat. A single-member
   source is still handled and still named as itself.

   **The faithfulness check got stronger.** `_attribution_against_its_model`
   averages the *fitted* models the same way, so it remains independent of the
   coefficients the estimate carries. It used to discriminate 261x against the
   one other member; it now discriminates **376x against Q-Pooled and 372x
   against dWOLS** — it says "this is the ensemble", not merely "this is not the
   other one". Worst residual against the object it names: **0.00011**.

   Non-psi coefficients still come from the dominant member. They are not
   decomposed onto the card, `beta:` is a treatment-free surface rather than a
   contrast, and averaging them is a question nothing here is asking. Weights are
   renormalised over the members that actually carry each key, matching what
   `_pair_contrast` does when an estimator cannot produce a contrast — a member
   that does not report an arm must not be read as reporting zero for it.

61. **A safety sweep that scored which arms went, never why.** Layer 4 was the
   one audit section whose perfect pair — recall 1.000 and precision 1.000 over
   20 labelled cases — turned out to be *earned*: the expectation sets are
   hand-written literals declared independently of `feasible_set.py`, and the
   safe levels (ALT 25/100, eGFR 90/45, non-pregnant) give precision a real
   denominator. What was missing is a different question.

   `_RENAL_HEPATIC_ARMS` and `_TERATOGENIC_ARMS` are **the same pair**, so all
   three physiological conditions remove the same two arms. The sweep read
   `removed_arms` for its keys and dropped the values — and the values are the
   reasons. So the counts cannot distinguish a filter that read the right
   observation from one that read the wrong one and removed the same pair.

   Injected, both regressions pass the old metrics untouched:

   | injected defect | recall | precision | reason rate |
   | --- | --- | --- | --- |
   | none | 1.000 | 1.000 | 1.000 |
   | ALT branch fires, names the renal reason | **1.000** | **1.000** | **0.857** |
   | every reason collapsed to one string | **1.000** | **1.000** | **0.000** |

   The first is the one to read. It produces a clinician card saying
   *"JAK inhibitor unsafe with eGFR < 30"* for a patient whose eGFR is **90** and
   whose ALT is **400** — a false explanation of why an arm was withdrawn, on the
   layer whose whole claim is that it is code rather than prose.

   Each case now carries the substring its removals must name — `ALT`, `eGFR`,
   `pregnan`, or the allergen token, which `allergy conflict: <token>` always
   carries — and `removal_reason_names_the_condition` scores it over the 14
   labelled removals. `tests/test_coverage.py` asserts **both** that the check
   catches the crossed wire *and* that recall and precision stay at 1.000
   through it, because the point is not that the new metric works but that the
   old ones are blind to this.

   The reasons were already distinct and already served — `SafetyLayer` raises an
   `arm_removed` flag carrying each one, and invariant 54 relies on that when it
   drops model prose for arms Layer 4 removed. Nothing had ever checked they were
   the *right* reasons.

62. **The arm vocabulary is one file; the molecule vocabulary was four.**
   Invariant 3 holds — every layer agrees on the six arm names. The layer below
   it did not. *Which molecules belong to an arm, and which hazards they carry*
   was declared in four places, and nothing compared them:

   | | declares | for |
   | --- | --- | --- |
   | `arms.ARM_SYNONYMS` | 5 TNF, 2 IL-6, 3 JAK molecules + class tokens | reading a history |
   | `estimation.actions.ARM_CANDIDATES` | 2 TNF, 1 IL-6, **1 JAK** | what may be proposed |
   | `safety.feasible_set.{JAK_DRUGS, HEPATOTOXIC_DRUGS}` | 3 JAK, 3 hepatotoxic tokens | composite filter |
   | `safety.rules.HEPATOTOXIC_TOKENS` | 6 tokens | arm-level warning |

   **Two of them agreed and the third had fallen behind.** `arms.py` recognises
   `upadacitinib`, `tofacitinib` and `baricitinib`; `feasible_set.py`
   hazard-classes the same three; the menu offered **one**. So an upadacitinib
   allergy removes the whole JAK arm from a patient the agent would recognise as
   having taken baricitinib — invariant 15's failure mode one level below where
   invariant 15 guards it.

   **The two hepatic lists were not subsets of each other** (`mtx` only in one,
   the three JAK molecules only in the other) and still classed every arm the
   same way. That agreement was curation, not construction, and nothing would
   have caught it parting.

   `formulary.py` is now the single declaration and the other three derive from
   it. Three vocabularies stay **deliberately distinct**, because collapsing them
   would be wrong: *recognition* is broadest and must place a drug the agent
   would never propose; *offer* is a curated subset and a formulary decision;
   *hazard* covers everything offerable and is matched by **spellings** rather
   than molecules — `mtx` because a composite carries `combination="MTX"`.
   `offerable=False` makes "known, hazard-classed, never proposed" a declared
   state rather than an accident, which is what `leflunomide` had always been.

   **A dead list in a safety rule, found on the way.**
   `rules.HEPATOTOXIC_TOKENS` was molecule spellings substring-matched against a
   **canonical arm name** — in both places that read it, including
   `stage.treatment`, which Layer 1 maps through `normalize_arm`. So
   `tofacitinib`, `baricitinib`, `upadacitinib` and `leflunomide` **could never
   match in either**; only `methotrexate` and `jak` did any work. The list read
   as though it broadened the rule and did not. An arm-level question now reads
   arm-level membership, `arms_with_hazard`. *I checked one call site first and
   was wrong about the second — the exception from the missing import is what
   sent me to look, and `stage.treatment` had to be measured rather than assumed.*

   **Nothing served changed, asserted rather than assumed.** 25 cohort cards,
   every safety path (high ALT, low eGFR, pregnancy, eight allergy tokens), the
   feasible composite sets and the whole safety audit hash **identically** before
   and after. The derived menu reproduces the old literal exactly.

   **Breadth is now measured instead of accidental.** `cli audit` reports
   `formulary_breadth` and `GET /model` carries the version. An arm offered as
   one molecule is an arm one allergy removes outright — a property of the
   curation, not of the method. Invariant 63 acts on it.

   Every entry carries `provenance` saying the curation is illustrative, per
   molecule rather than in a header, so a reader inspecting one does not have to
   go looking.

63. **"1 of 5 arms survives a single allergy" was the wrong denominator.**
   Invariant 62 published that number and left the decision to a human. Asking
   what the number should be first is what made it actionable, and it was two
   questions rather than one.

   **Two of those five arms cannot be widened at all.**
   `methotrexate-optimization` and `rituximab` are named after their only
   molecule: swapping it makes them a different arm. `TNF-inhibitor`,
   `IL-6 inhibitor` and `JAK-inhibitor` name a *class*, and a class has members.
   `is_molecule_named` derives that from the arm name rather than declaring it,
   so a new arm cannot forget to say which kind it is, and the audit's
   denominator is the three arms widening is coherent for. Scored over all five,
   the old figure reported three-fifths of a gap where two-fifths of it did not
   exist.

   **On the other three the narrowness was curation, and it is closed.**
   `arms.py` already recognised `tofacitinib`, `baricitinib` and `sarilumab`, and
   the hazard classes already covered the JAK pair — the menu was the only place
   they were missing. Writing their regimens takes JAK from 1 offerable molecule
   to 3 and IL-6 from 1 to 2, so **3 of 3** widenable arms now survive a
   single-molecule allergy where **1 of 3** did.

   | | before | after |
   | --- | --- | --- |
   | IL-6 inhibitor survives a tocilizumab allergy | no | **yes** |
   | JAK-inhibitor survives an upadacitinib allergy | no | **yes** |
   | TNF-inhibitor survives either of its two | yes | yes |
   | composites offered (IL-6 / JAK) | 2 / 1 | **3 / 3** |

   **It changes feasibility and nothing else, asserted rather than assumed.**
   Statuses, recommendations, Q-values and removals over 40 patients with no
   allergy hash **identically** before and after — a wider menu cannot move a
   decision, because the arm-level Q-values never saw the composites. What moves
   is only which arms survive a contraindication, and only in the direction
   invariant 15 wants.

   The organ-function paths still remove the *whole* widened arm: pregnancy, ALT
   400 and eGFR 12 each take all three JAK composites with the reason naming the
   condition, because the hazard is declared on the class and every member
   carries it. IL-6 correctly drops from 3 composites to 2 under pregnancy — the
   MTX-combination goes and the monotherapy survives, which is invariant 15
   working rather than an exception to it.

   **What is deliberately still narrow.** `TNF-inhibitor` gains nothing here:
   `arms.py` recognises three more TNF molecules, but the arm already survives
   and adding them buys no measured property. `hydroxychloroquine`,
   `sulfasalazine` and `abatacept` stay recognised and unoffered — they map to an
   arm for *reading a history*, where "some csDMARD" is the right granularity,
   and none is a substitute within that arm's meaning. `leflunomide` stays
   declared, hepatotoxic and unoffered. The recognition vocabulary must stay
   broader than the menu, and a test asserts the menu does not acquire those
   three by being derived from it.

   The three regimens are ordinary RA dosing and, like every other entry, carry
   `provenance` marking the curation illustrative rather than sourced. This is
   the one commit in this sequence that adds clinical content, and what keeps it
   honest is that the content is labelled, the effect is measured on both sides,
   and the property it buys is pinned by a test that fails if the menu narrows
   back.

   **The safety sweep caught two of its own labels going stale, which is the
   fixture working.** `_ALLERGY_CASES` states its rule in a comment — *a
   drug-level allergy removes only that molecule's composites, and the arm
   survives if another molecule in it does, so the arm-level expectation is
   empty* — and `tocilizumab` and `upadacitinib` were listed non-empty. That was
   the rule failing to apply rather than an exception to it: those arms had one
   offerable molecule, so the only composite was the allergen's. Widening made
   the rule reach them, recall fell to **0.857** naming both as misses, and both
   moved to the empty expectation every other drug-level allergy already had.

   `rituximab` and `methotrexate` stay non-empty and are not exceptions either —
   their arms are named after the molecule, so removing it *is* removing the arm.
   The rule is now uniform with no special cases, which it was not before.

   **The guarantee that did not move** is the composite one, and it is the one
   that matters: the allergen never survives into a feasible action, verified for
   all eight tokens. `tests/test_workflow.py` asserted the old narrowness in a
   test *name* — `test_a_drug_level_allergy_still_removes_the_arm` — and now
   asserts the molecule goes, the arm survives on `sarilumab`, and the allergen
   appears nowhere.

   *Found by the full suite, not by the module subset I ran first: I checked the
   safety modules at the refactor stage and then changed behaviour, re-running
   only the formulary and docs tests before the full run.*

64. **The identification certificate read a different vocabulary from the model
   it certifies.** `data/dag.py` opens by saying the previous version asked
   whether the *bundle mentioned* an adjuster, and that what matters is whether
   the estimator conditions on the variable. `_has_adjuster` still did not ask
   that. It hand-listed observation codes, and one entry had drifted:
   `baseline_disease_activity` was satisfied by `cdai`, `sdai` or `haq_di` —
   none of which produces a DAS28 — while `MODELLED_BY`, the map declaring node
   → basis term, sat twenty lines above it, unused by the check.

   Measured on the demo record:

   | record | identified | das28 the estimators read | status |
   | --- | --- | --- | --- |
   | complete | yes | 5.200 | recommend |
   | **HAQ-DI present, DAS28 absent** | **yes** | **5.000** | **recommend** |
   | ESR instead of CRP | no | 5.200 | blocked |
   | RF instead of anti-CCP | no | 5.200 | blocked |

   5.000 is `FEATURE_DEFAULTS["das28"]`, and row two is the case `data/dag.py`'s
   own docstring names as the one this check exists to catch: *"a patient with no
   disease-activity measurement is running the estimators on a default, and the
   effect genuinely is not identified for them."* It was certified and served a
   recommendation.

   **Nothing else in the system had anything to say about that record.**
   `estimation/features.py` carried a comment beside `FEATURE_DEFAULTS` reading
   "the data contract is what flags genuinely missing families" — it does not.
   The contract checks variable *families* and grades a missing one a **warning**,
   and the families are deliberately broader than the covariates: `haq_di`
   satisfies `disease_activity`, `esr` satisfies `inflammation`,
   `rheumatoid_factor` satisfies `serostatus`. Each of those is an ordinary RA
   record, not a corrupt one. Measured, **1 of 6** such records draws any contract
   diagnostic at all. Two modules each believed the other held this.

   `OBSERVED_AS` is now the single declaration of which record key each adjuster
   is read from, and `_has_adjuster` asks the narrow question — *will
   `estimation.features` find a usable value under that key, or fall back to a
   default* — with the same numeric test `numeric_feature` applies, so a DAS28
   recorded as the string "high" is not a measurement however present it looks.
   `validate` takes the stage history rather than `patient.observations` and
   reads the treatment off it, because two arguments describing one stage are two
   things that can disagree. The dead `patient.demographics` branch is gone; no
   node in `adjustment_set` was ever demographic.

   **Nothing served changed, asserted rather than assumed.** Statuses,
   recommendations, Q-values, removals, safety flags and cards over 40 cohort
   patients hash identically before and after — the simulator writes all three
   covariates for every patient, which is *why* `identified` was constant and is
   the thing the old paragraph in this file got wrong.

   **The audit now scores the gate instead of round-tripping well-formed
   records.** Every other Layer 1 metric is an identity on a record the fixture
   built correctly, so the only thing they can report is that the fixture is
   correct. `identification_matches_the_features` is scored on six records built
   to fail, and the truth is a **perturbation**, not a comparison against
   `OBSERVED_AS`: two records differing in one observation, and if the covariate
   the estimators read does not move when the record does, the model is not
   reading the record. That is the docstring's own phrasing measured, and it owes
   nothing to the map under test.

   Comparing the value against `FEATURE_DEFAULTS` instead would not work, and the
   reason is invariant 28 one covariate over: `anti_ccp` defaults to 0.0 and a
   recorded *negative* is also 0.0, so "never tested" and "tested negative" would
   be one number. Refusing the second would abstain on every seronegative patient
   for having been tested. That row is the precision half's whole point.

   Recall and precision are separate for Layer 4's reason — a certificate that
   refused every record would score perfect recall — and both denominators are
   real: 4 unreadable records, 2 readable. Injected, the discrimination is clean:

   | check | recall | precision |
   | --- | --- | --- |
   | deployed | 1.000 | 1.000 |
   | **the old code list** | **0.500** | 1.000 |
   | certifies everything | 0.000 | 1.000 |
   | refuses everything | 1.000 | **0.000** |

   `tests/test_coverage.py` asserts both that the metric catches the old check
   *and* that every other Layer 1 number reads identically either way, because
   the finding is not that the new metric works — it is that nothing else here
   could see the defect. `tests/test_data_layer.py` pins the property itself
   against `model_features` rather than against `OBSERVED_AS`; three of its seven
   assertions fail on the old implementation and four pass, which is right,
   because the old code list happened to get `crp` and `anti_ccp` correct.

   **What is deliberately not changed is the blocking behaviour.** A record with
   an ESR and no CRP is still refused, and that is correct — the estimators have
   no ESR term, so the value they use is a default whatever the contract's family
   check says. What was wrong is that Layer 1 said nothing about it, and that is
   now a reported number rather than a surprise three layers down. Widening the
   contract's families to reject these records would be the larger change and it
   trades a card carrying context for the reviewer (invariant 35) against a typed
   error at the gate (invariant 13); it is left alone rather than decided in
   passing.

65. **A quantity that does not depend on the patient was being computed per
   patient, and it was most of the request.** Profiling a warm request rather
   than reasoning about it: 9.37ms, of which **Layer 3 was 6.62ms**, and inside
   it one call dominated.

   `ArmFit.cross_covariance` **takes no features**. It walks the clusters two
   fits share and returns `A_a^-1 (sum s_a s_b') A_b^-1` — a property of the
   *fit*, fixed once per process. The patient enters afterwards, in the
   quadratic form against `blip_basis(features)`. So the same 10x10 matrix was
   rebuilt for every patient forever, at **1.31ms a call and four calls a
   request — 56% of the request**. `EstimandContract.fingerprint` was the same
   shape on a `frozen=True` dataclass: six reads a request, each running
   `dataclasses.asdict` over the whole contract, `json.dumps` and a SHA-256, to
   return the same sixteen characters.

   | | before | after |
   | --- | --- | --- |
   | warm request | 9.37 ms | **3.17 ms** |
   | Layer 3 `decide` | 6.62 ms | **1.15 ms** |
   | Layer 2 `estimate` | 0.68 ms | 0.43 ms |
   | `cli audit` | 6.8 s | **5.1 s** |
   | `as_dict` calls per request | 10 | 4 |

   **Nothing served changed, asserted rather than assumed.** Statuses,
   recommended and top-scored arms, Q-values, the candidate set, every arm
   contrast and its interval, removals, flags and cards over 40 cohort patients
   hash identically before and after.

   **This is not the cache invariant 7 forbids, and the distinction is the whole
   point.** That defect was a second *model* — `dwols.fitted_model()` kept its
   own `_MODEL` global, fit from its own cohort, so the object the studies
   measured and the object that served patients were different and silently
   diverged. Nothing here caches a fit. The memo lives on the `DWOLSModel` that
   *owns* both `ArmFit`s, they are written once in `__init__` and never mutated
   anywhere in the package, and `refit` constructs a whole new model — so a
   bootstrap replicate starts with an empty cache and cannot read a parent's. If
   it could, every replicate would report the deployed fit's covariance and the
   joint interval would stop responding to resampling, which is invariant 7's
   failure mode exactly. `tests/test_estimators.py` asserts both halves: that
   every contrast still equals one built from a fresh matrix, and that a replica
   starts empty.

   The fingerprint memo is stashed under a name that is **deliberately not a
   field**. `dataclasses.replace` rebuilds through `__init__`, so a derived
   contract computes its own; had it been a field, `replace` would have copied
   the old hash onto a contract whose content no longer matched it — and the
   estimand fingerprint exists to catch precisely that. A wrong fingerprint is
   worse than a slow one, so `tests/test_scientific_contracts.py` asserts a
   replaced contract differs *and* that the attribute is not a field, because
   the first test alone would pass on a field that `replace` happened to reset.

   **The suite barely benefits, and measuring *that* needed call counts rather
   than a stopwatch.** The expectation going in was that a 3x faster request
   would reach the 394s suite, since every study loop scores patients. Timing it
   could not answer the question: the same unchanged suite ran **271s and 387s**
   on this machine, about 40% run-to-run variance, which swamps anything being
   looked for. So the memo was counted instead — how many `cross_covariance`
   calls it removes, which is deterministic:

   | workload | calls before | after | removed |
   | --- | --- | --- | --- |
   | `cli audit` — ~300 patients through **one** fit | **1340** | **16** | 1324 (~1.7s) |
   | `decision_rule_coverage(8)` — **8 refits**, few patients each | 48 | 24 | 24 (~0.03s) |

   That is the entire mechanism in two rows. Work that scores many patients
   against one fitted model amortises **84x**; work that refits gets a fresh
   cache each time and has nothing to amortise. The suite is **refit-bound**:
   measured per module, `test_robustness` (86s) and `test_coverage` (84s) are
   43% of it on their own, and with `test_transfer`, `test_inference` and
   `test_bootstrap` roughly 58% is Monte Carlo studies whose cost is the *fit*.
   So the suite keeps most of its 6.5 minutes and that is structural, not a
   shortfall in the memo.

   **Splitting those studies into a second suite was considered and rejected.**
   It would take the everyday run to roughly 165s, and it is the obvious thing to
   do. It is also exactly backwards here: invariant 63 is the scar from running a
   module subset before a behaviour change, and the tests that would become
   optional are the ones that catch statistical regressions. A 6.5-minute suite
   that always runs is safer than a 2.7-minute one plus a 4-minute one somebody
   forgets. Do not split it without a stronger argument than speed.

   **Do not price a change here with a stopwatch.** The 40% variance above is the
   standing reason; this file's other timings are single runs on a quiet machine
   and should be read as magnitudes, not measurements.

   What is deliberately *not* optimised further: `as_dict` still runs four times
   a request at about 0.8ms, and the quadratic form only ever reads the 4x4 blip
   sub-block of that 10x10 matrix. Both are real and neither is worth the
   aliasing risk of handing out a cached mutable dict, or the subtlety of a
   matrix whose shape no longer matches what `cross_covariance` documents, for
   under a millisecond in a research prototype whose request is already 3ms.

66. **The same matrix was inverted twice, fifteen lines apart, and one of them
   was 98x98.** Invariant 65 took the per-request redundancy; this is the same
   defect on the fit path, which is where the remaining time actually was —
   `cli audit` and every Monte Carlo study refit through here.

   `sandwich_covariance` took the normal matrix X'WX and inverted it. Both
   callers that need the inverse for anything else then inverted it again
   themselves:

   * `QLearningModel._fit` factors X'WX once for the fixed point — there is a
     comment above it saying exactly why, *"only the pseudo-outcomes do... which
     is what makes the bootstrap's hundreds of refits affordable"* — and then
     handed the **un-inverted** matrix to `_sandwich`, which redid it. At 98
     parameters that is a **277ms** Gauss-Jordan, repeated.
   * `ArmFit._fit` let `sandwich_covariance` compute the bread, discarded it,
     and rebuilt the identical 10x10 for `_bread` on the next line.

   The parameter is the **bread** now, not the matrix. That makes the
   redundancy structurally impossible rather than merely fixed: there is one
   object to pass, so a caller cannot hold the inverse and hand over the
   original. `specification.py`, the one caller with no other use for it,
   inverts at its own call site.

   Exact, from counting rather than timing (see invariant 65 on why):

   | inversion | before | after |
   | --- | --- | --- |
   | 98x98 | 2 | **1** |
   | 78x78 | 2 | **1** |
   | 38x38 | 2 | **1** |
   | 10x10, five arm fits | 10 | **5** |
   | 11x11, `specification` | 15 | 15 |

   Priced with dense inversions measured at 277ms / 116ms / 13ms / 0.3ms, that
   is about **407ms removed from every full fit**, and one 98x98 plus five 10x10
   from every study replication that computes a covariance — roughly 2.2s of an
   8-replication `decision_rule_coverage`.

   **Nothing moved, and this is the change where that mattered most.** Every
   standard error in the system flows from these matrices. The fitted
   covariances — all five dWOLS arm fits, their breads, and the Q-learning
   models' — hash to `6170c6241c15` **before and after**, at twelve decimal
   places, and the served output over 40 patients hashes unchanged too.

   The risk the signature creates is a caller passing X'WX where the bread
   belongs, which would fail *silently*: a plausible covariance built from the
   wrong matrix, shifting every interval without raising anything. So
   `tests/test_estimators.py` pins `sandwich_covariance` against
   `A^-1 (sum_g s_g s_g') A^-1 . G/(G-1)` computed the long way on a case small
   enough to check by hand — invariant 23's rule applied to the argument that
   changed — and asserts it *disagrees* when handed the un-inverted matrix. A
   second test counts inversions per fit, because the identity would still hold
   if the redundancy came back.

   **Two honest notes on the measurement.** A *diagonal* 98x98 inverts in 5ms,
   not 277ms, because `linalg.inverse` skips zero factors — the first attempt to
   price this used a diagonal matrix and understated it fifty-fold. And the
   wall-clock here was useless again: `training.fitted()` timed 5.20s, 4.00s and
   15.64s across runs of code that differed by one line. The counts are the
   evidence; the milliseconds are a price list.

67. **Half of every dot product in the propensity fit was the same dot product,
   and the sparse solver that would have fixed the rest was already there with
   no callers.** `PropensityModel._fit` is 44% of a cold fit. Two findings, and
   the second is the one worth reading.

   **The redundant half.** Each IRLS pass computes every row's linear predictor
   inside `_probabilities`, then the arm loop recomputes it:
   `eta = linalg.dot(row, beta)` where `beta is coefficients[arm]` — which this
   pass has *not yet replaced* for the arm being fit, so it is bit-for-bit the
   `scores[arm]` already in hand. Counted: **409,690 dot products, of which
   204,845 were repeats** — exactly `773 rows x 5 arms x 53 iterations x 2`.
   `_probabilities` is split into `_linear_scores` and `_softmax`, the pass keeps
   both halves, and the count halves to 204,845.

   **The duplicate very nearly committed.** The remaining cost was
   `weighted_least_squares`, which re-derives the design's sparsity pattern on
   every call — 773 rows x 265 calls = **204,845 reconstructions of a pattern
   fixed at construction**, worth 29% of the fit's least-squares time and
   bit-identical when hoisted. A `weighted_least_squares_sparse` was written to
   take pre-sparsified rows. `linalg` already contained
   **`sparse_weighted_least_squares`**, character-for-character the same
   accumulation, with a per-coefficient ridge vector as a superset — and
   **zero callers anywhere in the package**. A hand-optimised solver sitting
   unused, and therefore unpinned, in the one module invariant 23 says must be
   kept exact.

   It was caught by listing the module's functions after the edit, not before.
   The lesson is invariant 3's at a different granularity: *read the module you
   are adding to*. A near-duplicate under a transposed name would have been two
   implementations of the normal equations, and the drift would have moved
   standard errors without failing anything.

   The new function is deleted. `weighted_least_squares` is now the **dense
   front door** — it sparsifies and delegates — so there is one implementation,
   the existing tests reach it for the first time, and the propensity fit
   sparsifies once and calls it directly.

   **Nothing moved, at three levels.** The propensity coefficients hash to
   `50c99a25254b`, the fitted covariances to `6170c6241c15`, and the served
   output over 40 patients to `b94e28ce8b93` — all identical before and after,
   and 53 iterations to convergence either way. That matters here because these
   coefficients are the denominator of every IPW weight in the system.

   **A stale `eta` would not raise; it would converge somewhere else.** So the
   guard is not the identity but the *solution*: the multinomial score equation
   `sum_i x_i (1{a_i = arm} - p_i(arm)) ~ 0`, which is a property of the fit and
   owes nothing to how the pass is arranged. At the deployed fit it is **0.0119**
   over 773 rows; with `eta` forced to zero it is **359.85**, a 30,000x
   discrimination. A second test asserts the fit never takes the dense front
   door, because that call is what carries the sparsity cost, and
   `tests/test_linalg.py` now pins the sparse solver directly — including the
   per-coefficient ridge vector, which the dense door cannot reach and nothing
   had ever exercised.

   **What is deliberately not touched, and this is now measured rather than
   deferred.** The fit takes 53 IRLS passes at tolerance 1e-4 under a
   block-diagonal Hessian approximation whose docstring says it "converges more
   slowly and to the same place". Two levers were considered and both are
   declined on evidence.

   *The tolerance is not over-tight.* Capping the pass count and watching what
   actually reads the propensity — the held-out IPW value that feeds the
   estimands, the BMA weights and `best_score()`:

   | cap | max \|coef - final\| | IPW value | change |
   | --- | --- | --- | --- |
   | 12 | 8.1e-02 | 0.738700 | -8e-04 |
   | 20 | 3.2e-02 | 0.739200 | -3e-04 |
   | 30 | 9.8e-03 | 0.739400 | -1e-04 |
   | 40 | 2.7e-03 | 0.739500 | **0** |
   | 53 (converged) | 0 | 0.739500 | 0 |

   It is flat to six decimals by pass 40, so the last 13 passes buy nothing — but
   stopping at 30 moves the **fourth** decimal, which the scorecard prints. The
   setting is right and loosening it would change a reported number for 0.5s.

   *A full Newton step is the only remaining lever, and it is the wrong trade.*
   It would reach the same fixed point in fewer passes, so unlike truncation it
   would not move the coefficients meaningfully — at a drift of 1e-4 the table
   above says the value does not move at all. The prize is roughly **1.2s off a
   once-per-process fit**. Against that: a multinomial Newton step needs the
   cross-arm Hessian blocks the block-diagonal approximation exists to avoid, in
   the module invariant 23 says must be kept exact, and its failure mode is the
   dangerous one — a subtly wrong Hessian converges *fast* to the wrong place,
   where the current iteration's failure mode is merely being slow. That is
   invariant 23's argument, and the smallest payoff left in the repo is not what
   buys an exception to it.

   **Where the propensity actually reaches, since that decides the risk.**
   `ArmFit._propensities` fits its own linear probability model per arm pair —
   *"Only used to form weights, so a simple model is adequate"* — so the
   multinomial `PropensityModel` never touches dWOLS's weights. It reaches
   `Recommendation.estimands` through `training.holdout_estimands` (cached: 13ms
   once, 0ms after), plus the held-out policy values, the DR estimate, the OPE
   effective samples and the validation blockers. It does **not** reach the
   recommended arm, the Q-values, the contrast, the candidate set, the
   abstention rate or the clinician card. An earlier note here implied it moved
   what the agent recommends; it does not.

68. **Gauss-Jordan on a matrix that was symmetric positive definite all along.**
   The last item in the 65-67 sequence and the only one that is *not*
   arithmetic-preserving, so it is reported differently: what moved is measured
   rather than asserted to be zero.

   Every matrix this package inverts is a normal-equations matrix `X'WX` plus a
   positive ridge — measured, all **23** built during a full fit are symmetric
   and positive definite. That licenses Cholesky, which does not pivot and does
   not need to for such a matrix. Three steps of about n^3/6 (factor, invert the
   triangular factor, `A^-1 = (L^-1)'(L^-1)` with only the upper triangle
   computed) against the elimination's ~2n^3 on the augmented `[A | I]`.

   Interleaved A/B in one process, medians of nine, so machine drift hits both
   arms equally:

   | n | count per fit | Gauss-Jordan | Cholesky | ratio | removed |
   | --- | --- | --- | --- | --- | --- |
   | 10 | 5 | 0.63 ms | 0.31 ms | 2.00x | 1.6 ms |
   | 11 | 15 | 0.81 ms | 0.40 ms | 2.01x | 6.1 ms |
   | 38 | 1 | 21.90 ms | 9.49 ms | 2.31x | 12.4 ms |
   | **78** | 1 | 76.62 ms | 75.58 ms | **1.01x** | 1.0 ms |
   | 98 | 1 | 399.84 ms | 146.37 ms | **2.73x** | 253.5 ms |

   **275ms off a full fit**, and about 255ms off every study replication that
   computes a covariance. The n=78 row is the interesting one: that matrix is
   **82.5% zeros** and `gauss_jordan_inverse` skips zero multipliers, so there is
   nothing to win. Invariant 23's hand-optimisation is doing real work, and it
   wins outright on the sparsest block.

   **It is not more accurate, and that was worth checking rather than
   assuming.** The first framing of this was "moves numbers toward the truth",
   which the measurement did not support: residuals of `max |A A^-1 - I|` on the
   real matrices run 1.8e-15 to 4.9e-14 for Cholesky against 1.8e-15 to 3.8e-14
   for the elimination — better at n=38, slightly worse at n=78 and n=98, a wash.
   The case for this is speed and nothing else.

   **What moved, measured at full precision.** Refitting the whole ensemble both
   ways and comparing every fitted covariance and bread entry unrounded: worst
   absolute difference **2.767e-15**, on an entry of 0.0112. The worst *relative*
   difference reads 3.1e-09 and is meaningless — it sits on an entry that is
   itself near zero.

   **What that reaches: nothing reported.** The fitted covariances agree to 12
   decimal places (hash `6170c6241c15`), the served output over 40 patients is
   identical including full card text (`b94e28ce8b93`), and over 120 audit
   patients every figure is unchanged — status distribution 37/83, oracle-arm
   rate 0.925, mean and max regret 0.0003 / 0.0101, true gaps 0.026 / 0.0775,
   ECE 0.011, worst blip parameter error 0.0251, and `das28_squared` still at
   **3.367**, the near-flag this file tracks at 92% of its threshold. A 2.8e-15
   perturbation cannot survive `q_values` being rounded to three decimals, and
   it does not.

   That is the standard a change like this has to meet, and it is weaker than
   the one invariants 65-67 met. **If any of those numbers had moved, the right
   answer would have been to revert**, because a faster inverse is not worth an
   argument about which abstention rate is correct.

   `linalg.inverse` tries Cholesky and falls back to `gauss_jordan_inverse`,
   which stays because the function's contract is general — an indefinite matrix
   must still get a correct answer rather than a silent `None`.
   `tests/test_linalg.py` asserts the two agree, that the refusal happens on an
   indefinite and on a negative-definite matrix, that the fallback still
   reproduces the identity, and that a normal matrix takes the fast path.

   **It broke one test, and the test was right to break.**
   `test_an_arm_against_itself_has_no_spread` pins invariant 38's `Var(x - x) = 0`
   and failed at 5.46e-10. The reflex is to loosen a tolerance; measuring first
   showed the opposite. Under the *old* inverse, `JAK-inhibitor` already returned
   **8.065e-10** — larger than the failure — and three of five arms returned
   exactly zero. The identity was holding per arm by accident, and the test
   checked one arm and had picked a lucky one. Cholesky did not break it; it
   reshuffled which arms were lucky.

   The repair is in invariant 38 and makes the identity exact for every arm
   under both inverses: `cross_covariance` returns `self.covariance` when handed
   itself, and `contrast_standard_error` stops squaring a square root. Neither
   moved a served number — the covariances still hash `6170c6241c15` and the
   served output `b94e28ce8b93`. **A failing test on a numerical change is
   evidence to read, not a threshold to adjust**, and here it was pointing at a
   defect older than the change that surfaced it.

69. **"Assertions, not warnings" — one of four raised, and the only one that
   could fire was measuring the wrong thing.** `data/leakage.py` opens with the
   strongest claim in the package: *"These are assertions, not warnings: a
   cohort build that puts any post-decision information into H_j fails."*
   Measured over 60 cohort patients, **none of its four checks has ever fired**,
   and three of them cannot:

   | check | can it fire from the pipeline? | what happened if it did |
   | --- | --- | --- |
   | `temporal_firewall` | **no — structurally** | raised |
   | `immortal_time` | **no** | nothing |
   | `outcome_not_in_features` | yes | **nothing** |
   | `timestamp_monotonic` | **no** | nothing |

   `build_patient_state` read `temporal_firewall_passed` and nothing else, so the
   rest were computed, appended to `violations`, and reduced to a
   warning-severity diagnostic — and **three of the four booleans were written
   and never read anywhere in the package.**

   **The firewall cannot fire**, and the demonstration is the useful part.
   `StageHistoryBuilder._features_until` admits only observations at
   `days_from_baseline <= start_day`; the firewall then asks whether any
   observation for a present feature exists at or before that day, which is true
   by construction. Measured, every feature's earliest source is at most 0 days
   relative to its own decision day. Moving the demo patient's CRP to day 9999
   does **not** trip it — the value drops out of the feature map and the patient
   is blocked by invariant 64's adjuster check instead, a different mechanism
   reporting a different reason. `immortal_time` and `timestamp_monotonic` are
   guarded by `FHIRAdapter.parse_bundle` sorting medications and by the contract
   raising on unsorted starts before either runs.

   **That is not a reason to delete them.** Each re-verifies a property
   something upstream enforces, and the firewall reads `stage.features` and
   `patient.observations` as two independently produced objects — change
   `_features_until` to a window, or to last-value-wins regardless of date, and
   it fires. They are invariant 59's structural tripwires, and the defect was
   that nothing said so and two thirds of them could not stop anything. All
   three raise now, `cli audit` reports them under `leakage_assertions` with the
   count that is expected to be zero and the reason it is, and the module
   docstring names the upstream guarantee each one guards.

   **The fourth is deleted rather than repaired.** `_outcome_in_features`
   flagged any feature whose *name* contained `outcome`. It could not do the job
   its name claims: `_features_until` has already excluded everything after the
   decision, so anything it saw was pre-decision by construction, and the one
   thing it could catch is a legitimately recorded *past* outcome — history, not
   leakage. Measured, it was also the only check here that could fire, and
   firing it changed nothing: a record carrying an observation coded `outcome`
   was served a **recommendation** with the violation filed in a diagnostic
   nobody reads. A name match was never a leakage statistic, which is why this
   is deleted the way invariant 37 deleted the out-of-distribution vector term
   rather than recalibrating it.

   The property it gestured at is real and is held elsewhere: a stage's outcome
   must not be computable from its own covariates, and `data/endpoints.py` keeps
   the windows strictly disjoint — baseline at `days <= start_day`, attained at
   `start_day < days <= end_day`, and `None` for an open stage, because *"the
   patient is standing at the decision and the outcome has not happened"*.
   Measured over 153 stages, no observation feeds both. Re-checking that in
   `leakage.py` would have been vacuous — restating the two window conditions
   and intersecting them is `set() & set()` — which is the trap that made
   deleting the check the honest answer rather than replacing it.

   `tests/test_data_layer.py` fires each of the three on its own and asserts the
   others stay clean, because a report that says a record leaked without saying
   *which* guarantee broke names three different repairs. The pipeline test
   breaks the **guarantee** rather than the check — reversing stage order trips
   only `_timestamp_monotonic`, which the contract cannot see because it checks
   medications — and reverting the one-line `passed` change makes it fail, which
   is the only thing that shows the change was the one that mattered. A test
   named `test_a_leaking_record_cannot_produce_a_patient_state` never called
   `build_patient_state`; it does now, under a name that says what it does.

70. **Two card lines that misstated what they described.** Found by running the
   whole workflow and rendering all four statuses side by side — the method that
   found invariants 35 and 54, and the one that keeps working because a card is
   read, not executed.

   **The interval's level was truncated.** Both renderings computed
   `int((1 - alpha) * 100)`. The deployed alpha is `0.05/15`, so the interval is
   a **99.6667%** one and the clinician card printed **99%**:

   | arms | pairs | alpha | true level | printed | z used | z from the label |
   | --- | --- | --- | --- | --- | --- | --- |
   | 4 | 6 | 0.00833 | 99.167% | **99%** | 2.638 | 2.576 |
   | 5 | 10 | 0.00500 | 99.500% | **99%** | 2.807 | 2.576 |
   | **6 (deployed)** | 15 | 0.00333 | **99.667%** | **99%** | **2.935** | **2.576** |
   | 7 | 21 | 0.00238 | 99.762% | **99%** | 3.038 | 2.576 |

   One label for four different corrections, printed directly beside a sentence
   saying the alpha is Bonferroni-adjusted over every unordered arm pair — so the
   number could not identify the object the line beside it was explaining. And
   truncation errs the **unsafe** way for anyone who reconstructs the interval:
   at six arms the printed 99% implies `z = 2.576` against the 2.935 actually
   used, a **12.2% narrower** interval than the one drawn. Nothing pinned it;
   no test mentioned a level.

   `inference.confidence_label` is the one renderer now, at `%.4g` — enough to
   separate 99.17 from 99.5 from 99.67, while an ordinary pointwise interval
   still reads "95" rather than "95.00". `tests/test_inference.py` asserts every
   family size gets a distinct label *and* that deriving a critical value from
   the printed level lands on the one the interval used, which is the property
   truncation broke rather than the formatting.

   **The BLOCKED flag promised a routing that does not happen**, which is
   invariant 35's defect in the sibling branch of the function whose comment
   records the lesson. `SafetyLayer._status` returns BLOCKED with a flag reading
   *"Routed to clinical review rather than substituting the next-best arm."* —
   so the card said "BLOCKED — no treatment is being suggested" in its heading
   and "routed to clinical review" four lines down. BLOCKED stops; REVIEW
   routes; they are different outcomes. Measured with pregnancy injected, **both
   of the two blocked cards in 60** carried it, as does every allergy block.

   What survives is what the flag observed, plus *"No arm has been
   substituted"* — invariant 2's guarantee, which is a statement about this
   layer's own action rather than about the status.

   **The guard is a sweep, not a spot check**, because the defect is not in any
   one message: writing an outcome into a flag reads naturally while being
   wrong. `tests/test_safety_review.py` raises every flag the pipeline can —
   16 bundles across five injected conditions, reaching all six codes — and
   asserts no message contains `routed`, `blocked`, `not blocked`, `escalat` or
   `equipoise`. It also asserts the sweep reached at least five codes, because
   otherwise it passes by finding nothing. `review` is deliberately absent from
   that list: *"manual review is required"* is legitimate on the out-of-support
   **warn**, which recommends an action without claiming the status took it.

   Nothing else moved — the two changes are a label and a sentence, and no
   number, status or arm differs.

71. **Three dead classes, and the sixth place a display quantity reached a served
   field.** Two passes: delete what nothing constructs, then go looking for the
   *sibling* of each fix that landed on one call site. Invariants 69 and 70 were
   both recurrences — a rule applied to one branch and not the one beside it — so
   the second pass is the one that paid.

   **Deleted, and each was worse than unused.**

   `POMDPInterface` (`data/belief.py`) was "a deliberate seam ... so swapping in
   a real POMDP solver is not a rewrite", which is the argument invariant 37
   rejected for `GRUBaselineEncoder`, failing the same way: zero constructions
   anywhere. A seam is a thing something passes through; this had nothing on
   either side. Its fallback also read `float(latest.outcome)` as the belief's
   activity, and at the open decision point `outcome` is the `UNKNOWN` sentinel.

   `PolicyValueSelector` (`estimation/estimators.py`) picked "the most
   interpretable estimator among those it cannot separate" — `best_score()`'s job
   (invariant 30). With no callers its copy of the tie-break had drifted out of
   sight, and the two no longer described the same preference:

   | `PolicyValueSelector` | `training.INTERPRETABILITY_ORDER` |
   | --- | --- |
   | 0 Q-Shared + Penalized | 0 dWOLS-Shared |
   | 1 dWOLS-Shared | 1 Q-Pooled |
   | 2 Bayesian Hierarchical Q | 2 Stage-Specific Q-learning |
   | 3 Stage-Specific Q-learning | 3 Q-Shared + Penalized |
   | 4 Survival Forest DTR | |

   Every shared name disagrees on position, and the reversal is the one that
   matters: it ranked `Q-Shared + Penalized` **first** where the live order ranks
   it last. That is the shared-blip fit invariant 18 keeps out of the ensemble
   and `cli coverage` measures at **28% pooled, 0% at the worst patient**. It
   also named two estimators this package has never contained and omitted
   `Q-Pooled`, a `SERVING_ENSEMBLE` member, which `.get(name, 99)` would have
   ranked behind every phantom. Had anything called it, it would have selected
   the estimator with the worst measured coverage in the repo.
   `tests/test_estimators.py` now pins the surviving order against the fitted set
   in both directions, because the `99` default is what makes that drift silent.

   `OverrideValidator` (`feedback/override_governance.py`) returned
   `record.outcome_confirmed_clinician is True` — one of the three conjuncts
   `OverrideRouter.route` already requires for `influences_model`, and so a
   strictly more permissive answer to the same question. It would have called an
   outcome-confirmed override a modelling signal even when the reason routed to
   SAFETY_REVIEW, which is what invariant 26 exists to refuse. A second, weaker
   copy of a safety-relevant predicate is the thing to delete rather than leave
   for someone to find and use.

   **The sibling: `_counterfactuals`.** `q_values` are clamped and rounded, and
   invariants 46, 53, 54 and 55 removed a ranking re-derived from them in five
   places. The sixth sat in the same function signature as two that were fixed —
   `explain()` hands `contrast` to `_sensitivity` and `arm_contrasts` to
   `_why_not`, and `_counterfactuals(selected, stages)` got neither and sorted
   the dict. It decides `recommendation_changes` on `gap < 0.05`, so the clamp
   reached a **boolean** rather than a rounding.

   Measured over 120 patients: the two gaps differ by up to **0.0648**, more than
   the threshold itself, and **2** patients got a different verdict.
   `ExplanationBundle.counterfactuals` is served on the `Recommendation` and
   nothing renders it — invariant 45's position exactly, a quantity a consumer
   can read built from arithmetic that could not support it, in a place no card
   happened to print.

   **It sharpens invariant 53 rather than repeating it.** That one measured nine
   patients whose top-two gap was *exactly* 0.0000, which is what happens when
   **both** leading arms saturate. Here only the leader does: both disagreeing
   patients have `rituximab` at exactly 0.99 with the runner-up at 0.989 and
   0.980, so the display gap is **compressed** to 0.0010 and 0.0100 rather than
   zeroed — against contrasts of 0.0658 and 0.0715, a factor of seven on the
   *same* pair of arms. Compression alone crosses a threshold; the gap never has
   to reach zero. The first version of the guarding test asserted the exact-zero
   case and found none, which is how the distinction got measured.

   **Checked and deliberately left.** `power.py` passes the literal `"Q-Pooled"`
   to `evaluate_policy`, but `model = fit.pooled` on the line above, so the
   string labels the model actually handed over rather than filtering a
   membership — not invariant 19's collapse. And
   `visit_intensity.expected_interval` ends `return self.mean_interval or 90.0`,
   invariant 28's forbidden shape: `mean_interval` is 0.0 until `_fit` sets it,
   so the truthiness is standing in for "not computed". On the deployed cohort it
   reads 125.34 with `fitted` True, so the branch is unreachable, on a feature
   `USE_VISIT_INTENSITY` turns off by default — doubly out of reach, and a real
   0 would be a division by zero rather than a clinical value.

72. **Three method names claimed an agent, and there is none.** The last of the
   three items this file had been carrying as deferred. The other two are
   resolved as amendments rather than new entries — invariant 67 now carries the
   measured case for leaving the propensity's IRLS alone, and the `cli
   specification` sections carry `das28_squared`'s null distribution and a
   correction to the false-positive rate.

   `AgentLayer` exposed `run_agents` and called `_safety_agent` and
   `_guideline_agent`. What those named is a `"; ".join` over the safety flags
   and a lookup in a five-passage keyword index. **This package has never
   contained a model call.** That is invariants 27, 45 and 51's rule — a name may
   not claim more than the arithmetic supports — in the one file a reader opens
   expecting to find an agent, and it had gone unexamined because those three are
   about *numbers* wearing a statistic's name and these are only strings.

   `compose`, `_safety_section`, `_evidence_section`, and `AgentLayer` has a
   class docstring for the first time, saying what the layer is and is not.

   **The package name stays, and that is the judgement rather than an
   oversight.** `agent/` is a layer in a decision-support *agent*, ordinary usage
   for a clinical DSS and not a claim about how the text is produced. Renaming it
   touches 31 import sites and would churn seventy invariants that say "Layer 5",
   to fix something the three methods were already saying wrong. The overstatement
   was local; the fix is local.

   **What is pinned is the claim, not the names.** A test that asserts a method
   is not called `_safety_agent` is the plumbing this repo's conventions warn
   about. The substantive property is that no model call can appear, so
   `tests/test_docs.py` walks every module, resolves each top-level import, and
   asserts every one is `treatmentrx` or the standard library — which also closes
   a gap that had nothing to do with naming: `dependencies = []` is a **hard
   constraint** in this file and `pyproject.toml` declared it, but nothing
   checked it. Declaring is not checking, and a package that imports only the
   standard library cannot acquire an LLM SDK quietly. The sweep asserts it
   reached `linalg.py`, `dwols.py`, `rationale.py` and `contract.py`, because a
   path change would otherwise make it pass by parsing nothing.

73. **A survival generator, and the version of it that had nothing to learn.**
   Extending to an oncology workflow needs a time-to-event endpoint before it
   needs a disease: `data/endpoints.py` maps a stage to a bounded response on
   `[0, 1]` and `q_values` are clamped to `[Q_FLOOR, Q_CEILING]`, so overall and
   progression-free survival are a different **estimand** rather than a
   different endpoint. `simulation/survival_cohort.py` is `ra_cohort`'s contract
   on that scale — known truth, an oracle, and a way to check both.

   Weibull proportional hazards with the treatment effect as a per-arm
   log-hazard ratio, `h(t | x, a) = h_0(t) exp(g(x) + tau_a(x))`, which is the
   survival analogue of a blip and carries the same identification device: the
   reference arm's `tau` is zero by construction. Weibull because it inverts in
   closed form, so a seeded draw is exact, **and** because its mean is closed
   form, so the oracle is arithmetic rather than Monte Carlo and a regret
   measured against it does not carry the oracle's own sampling error.

   **The first version was not a sequential problem, and the measurement is
   what said so.** Everything checked out — the closed-form mean matched the
   sampler to 0.1%, the declared `tau` was recoverable from the times to 0.001,
   positivity held — and the myopic rule matched the backward-induction optimum
   on **0 of 3,000 patients**. `advance_line` had advanced only `prior_line`,
   which does not depend on the arm, so the line-1 choice had no effect on the
   line-2 state and the problem decomposed into three independent choices. A
   generator whose optimum a greedy rule reproduces exactly has nothing for a
   DTR estimator to find, and none of the other four checks could see it.

   The repair is the delayed effect `ra_cohort` gets from ALT:
   `RESISTANCE_INDUCING_ARMS` leaves `acquired_resistance` behind, charged on
   the **prognostic** surface and deliberately absent from `HAZARD_BASIS` — so
   an estimator that recovers every `psi` exactly still has to look ahead.
   Measured after:

   | | before | after |
   | --- | --- | --- |
   | myopic disagrees with the oracle | **0 / 3,000** | **1,965 / 3,000** |
   | `arm-a` chosen with three lines left | 66% | **0%** |
   | `arm-a` chosen with one line left | 66% | **67%** |
   | oracle over myopic | — | **+6.9 months** |

   That last row is the shape to want: the potent arm is saved for last, and
   "always `arm-a`" becomes the **worst** fixed policy (32.1 months against 47.1
   for "always `arm-c`") despite having the strongest single-line effect.
   `tests/test_survival_cohort.py` pins it, and reverting the one repair fails
   four of its five structural tests with *"the myopic rule matches the oracle
   on 100.0% of patients"*.

   **Two smaller things the sanity checks caught.** The closed-form oracle and a
   simulated rollout read about 1% apart, which is two different patient samples
   rather than bias — on matched patients it is 0.06%, or 0.09 standard errors,
   and the test uses matched patients so it cannot drift into measuring the
   wrong thing. And a `fully_observed` property returned True for a **competing
   death**, conflating "follow-up ended in an event" with "the event of interest
   was observed" — the exact conflation the module's docstring claims to avoid,
   differing on 61 of 200 line-endings. It is now `progression_observed` and
   `censored`, neither standing in for the other, with `terminal_cause` for the
   three-way question.

   **It is deliberately not a disease, and not wired to anything.** The arms are
   `reference`, `arm-a`, `arm-b`, `arm-c`. Naming them after breast or brain
   tumour regimens would make a hazard model with chosen numbers look like a
   clinical one, which is invariant 72's rule applied before the fact rather
   than after. Nothing imports it, no disease definition uses it, and the
   estimators cannot consume it — `EstimationLayer` scores a bounded response
   and would have to change before any of this reaches a recommendation. What
   exists is the ground truth a survival estimator would be scored against, and
   the checks that say it is worth scoring against.

74. **The survival estimator, and a generator that could not ask the question it
   is named for.** `survival_dwols.py` ports `dwols.py` onto the log-hazard
   scale: `log T` is linear in the covariates under Weibull PH, so the same
   `weighted_least_squares`, the same `sandwich_covariance`, the same
   `|A - pi|` weight, against an estimand on a different scale. The shape comes
   from the Gumbel spread of the residuals rather than being supplied, and
   `psi = -k * coefficient`. Recovery works: shape **1.3949** against a true
   1.4, and every blip parameter back inside its own sampling spread.

   **The first word in its docstring is "doubly-robust", so that was measured
   rather than inherited — and the first three attempts to measure it were all
   wrong in instructive ways.**

   *Crippling the treatment-free surface to an intercept* took the error from
   0.669 to 3.789 and the propensity did not rescue it, which read as the
   property being absent. It is not a misspecification at all. `HAZARD_BASIS`
   is a **subset** of the prognostic covariates, so stripping the surface
   leaves `A * h(X)` as the only X-varying columns and the blip terms absorb
   the prognosis outright. That is non-identification, and no weight repairs
   it. **I reported "not doubly robust" to the user on the strength of it, and
   that was wrong.**

   *Then the shape.* `psi = -k * coefficient` and `k` comes from the residual
   spread, so a wrong surface inflates the residuals and drags `k` down — an
   attractive mechanism, because it would be a defect no weighting could reach.
   Measured, the shape moves 1.4034 to 1.3657, **2.7%**, while the error moves
   2.3x. The bias is in the coefficient, not in the conversion. Refuted.

   *What was actually wrong is the cohort.* Double robustness survives a wrong
   *treatment-free surface*, so measuring it needs a variable the surface can
   omit which moves assignment **and** the outcome and is **not** in the blip
   basis. No such variable existed. `assignment_probabilities` was a softmax
   over `true_log_hazard_ratio` alone, so every confounder was in
   `HAZARD_BASIS`; `acquired_resistance` sits outside it and moves the outcome
   by 0.75 and assignment by **exactly 0.0000**. Every misspecification that
   could be written was either no confounding at all or not a surface question,
   which is why the measured gain sat near 2% however wrong the surface was
   made. That is invariant 73's defect one level up — a fixture missing the
   structure the thing built against it needs — and it was found the same way,
   by measuring rather than by reading.

   `performance_status` is the repair, and it is confounding by indication: a
   frail patient does worse whatever is given and is steered away from the
   aggressive arm. The caution is **arm-specific**, because a shift common to
   every arm cancels in the softmax and confounds nothing. With it, six seeds
   at n=3000:

   | treatment-free surface | no weight | fitted propensity | the true propensity |
   | --- | --- | --- | --- |
   | correct | 0.675 | 0.661 | 0.671 |
   | omits the confounder | 0.712 | 0.699 | **0.671** |

   A correct propensity takes the wrong surface to **0.671**, the number the
   correct surface gives, and the fitted propensity recovers 36% of the damage —
   the gap being the price of `_propensities` being a linear probability model
   standing in for a softmax.

   **That is directional and not a clean demonstration, which is worth saying
   because the table reads like one.** The damage is 0.037 on a base of 0.675,
   and scored as a *bias* over eight seeds instead of as absolute error the
   omitted-confounder-with-true-propensity arm comes out **below** the
   correctly-specified one, 0.308 against 0.326 — noise, not a wrong surface
   beating a right one. The firm evidence is the mechanism rather than the
   outcome: `|A - pi|` satisfies `pi w(1,X) = (1-pi) w(0,X)` pointwise and
   measurably balances the covariates, and that is what the tests pin.

   **The larger finding is a bias that has nothing to do with any of this**, and
   it was only visible once bias was separated from spread — invariant 40's
   distinction, and mean absolute error hides it because the sampling spread
   dominates at this cohort size. With a correct surface *and* the generator's
   own propensity the estimator still carries a total |bias| of **0.307** over
   the twelve blip parameters, which is an order of magnitude larger than the
   0.037 the double-robustness demonstration repairs. The cause is stated in the
   module's own docstring as an assumption: *"censoring is independent of the
   covariates"*. It is not. Two of the three ways follow-up ends do satisfy it —
   loss to follow-up and the competing risk are exponential with constant rates,
   independent of the arm by construction. Administrative censoring is
   `horizon - entry_month`, and entry month is the sum of the earlier lines'
   durations, which depend on the covariates and the arms taken. Measured over
   20 seeds at n=3000, as total |bias|:

   | | total \|bias\| | worst |
   | --- | --- | --- |
   | no censoring at all | **0.106** | 0.027 |
   | deployed, IPCW on | 0.307 | 0.058 |
   | deployed, IPCW **off** | **0.515** | 0.124 |

   So **two thirds of the deployed bias is censoring** and the marginal
   Kaplan-Meier correction removes about 40% of it. This invariant then said a
   covariate-dependent censoring model was the repair. **Invariant 76 measured
   that and it is not**, so read the two together: what is wrong here is the
   named remedy, not the diagnosis. The docstring's claim that the independence
   was *"right for the generator"* was the defect, and it stands corrected.

   **Five seeds said IPCW made the bias worse, and three eight-seed blocks all
   said better.** `_pooled_bias` takes the absolute value of a mean, which its
   own sampling noise inflates upward, so at small seed counts the two arms are
   biased by different amounts and the comparison can flip sign: 0.439 -> 0.510
   at five seeds against 0.489 -> 0.326, 0.653 -> 0.466 and 0.640 -> 0.474 at
   eight. `_WEIGHTING_SEEDS` is eight for that reason and the comment says
   which measurement set it. A test that flips on the seed block is worse than
   no test, and the first version of this one flipped.

   **What is pinned and what is recorded.** The tests pin recovery, the sign
   convention (`psi = -k * coefficient`, asserted term by term — an argmax on a
   hazard recommends the worst arm while looking reasonable), that IPCW removes
   bias, and the **mechanism** of the double robustness: that `|A - pi|`
   balances the covariates, worst imbalance **0.0725 -> 0.0157**. The rescue
   itself is recorded in the docstring rather than pinned, because 0.037 on a
   base of 0.675 needs more seeds than this suite can spend, and saying so is
   better than a test that passes for the wrong reason.
   `tests/test_survival_cohort.py` pins the confounder's three conditions and
   keeps `acquired_resistance` as the negative control; zeroing
   `_PRESCRIBING_CAUTION` fails exactly those two tests and leaves the control
   passing.

   **Still wired to nothing.** `EstimationLayer` scores a bounded response and
   would have to change before any of this reaches a recommendation. The arms
   stay `reference`, `arm-a`, `arm-b`, `arm-c` for invariant 73's reason.

75. **dWOLS's weight does earn its place — and the study measuring that was
   scored against a truth a quarter of the cohort no longer had.** Invariant 74
   ended by noting the survival generator could not express the misspecification
   double robustness survives, because every confounder sat in the blip basis.
   `ra_cohort.assignment_probabilities` scores arms from `blip_basis(features)`
   **exactly**, so the deployed cohort has the same shape, and nothing had ever
   measured whether the `|A - pi|` weight earns its place here. Asking that is
   what found the rest.

   **It is not the same dead end**, and the difference is why the question is
   answerable. Splitting the two bases: `das28_std`, `anti_ccp` and `prior_tnf`
   drive assignment *and* the outcome; `crp_std` and `alt_excess` are prognostic
   only. So the surface can only *omit* a variable that creates no confounding —
   invariant 74's trap — but `curvature` does not omit, it **mis-shapes**, and it
   mis-shapes over `das28_std`, which is a confounder. A linear surface is then
   genuinely wrong about a variable that drives assignment, and the weight has
   something to do.

   **It does it.** 24 seeds at n=400, scored as total |bias| over the 20 blip
   parameters, because confounding is a systematic shift and mean absolute error
   at this cohort size is mostly sampling spread — invariant 40's distinction,
   and the lesson invariant 74 learned by measuring it the wrong way first:

   | curvature | no weight (OLS) | fitted (deployed) | the true propensity | the weight buys |
   | --- | --- | --- | --- | --- |
   | 0.00 | 0.0306 | 0.0296 | 0.0310 | +0.0010 |
   | 0.05 | 0.4402 | **0.2667** | 0.2113 | **39%** |
   | 0.10 | 0.5132 | **0.3653** | 0.2717 | **29%** |
   | 0.15 | 0.5929 | **0.4328** | 0.3301 | **27%** |

   At curvature 0 it buys nothing, which is the control: with nothing
   misspecified there is nothing for double robustness to survive, and a weight
   that *helped* there would be doing something other than what it claims. The
   bias lands on `das28_std` — the confounder the curvature bends over — exactly
   as the mechanism predicts: OLS carries +0.0866 on IL-6's `das28_std` where
   the true propensity carries +0.0182.

   **The rescue is partial, and chasing why is what found the real defect.**
   With the generator's *own* propensity the `|A - pi|` weight balances any
   function of X, including the omitted `severity^2`, so double robustness
   predicts the bias should vanish rather than shrink. It shrinks. Sweeping n at
   curvature 0.10 with the true propensity: **0.2717 at n=400, 0.2697 at 800,
   0.2633 at 1600, 0.2716 at 3200** — flat, so not finite-sample.

   **`expected_outcome` clamps the sum.** It returns
   `clamp(treatment_free_value(X, curvature) + true_blip(a, X), 0, 1)`, so once
   `curvature * severity^2` drives the baseline against a bound *both* arms
   saturate together and the contrast between them collapses. The declared blip
   is untouched; the realised one is not. Measured on real stage covariates
   (3 cohorts of 400), against `misspecification.DEFAULT_CURVATURES`:

   | curvature | rows clamped | arm-pairs whose blip drifts | worst drift | baseline outside [0,1] |
   | --- | --- | --- | --- | --- |
   | 0.00 | 3.8% | 1.1% | 0.0836 | 0.0% |
   | 0.05 | 18.9% | 13.4% | 0.2294 | 9.9% |
   | 0.10 | 26.0% | 21.3% | 0.2336 | 16.7% |
   | 0.15 | 29.1% | **25.0%** | 0.2357 | 20.7% |

   So `_blip_error` compares fitted parameters against `TRUE_BLIPS` at
   curvatures where a quarter of arm-pairs no longer have that blip. The
   correctly specified row is clean, which is why nothing caught it.

   **The conclusion survives; one detail does not.** Rescored against the
   contrast the cohort actually has — `E[Y|X,a] - E[Y|X,ref]` per patient, which
   stays well defined under the clamp although no parameter vector describes it
   — over the study's own six seeds:

   | | Q-Shared | Stage-Specific | dWOLS |
   | --- | --- | --- | --- |
   | vs `TRUE_BLIPS`, curvature 0 | 0.5785 | 0.2852 | **0.1702** |
   | vs `TRUE_BLIPS`, curvature 0.15 | 0.9434 | **0.5936** | 0.6191 |
   | vs the realised blip, curvature 0 | 0.0552 | 0.0166 | **0.0095** |
   | vs the realised blip, curvature 0.15 | 0.0766 | 0.0414 | **0.0373** |

   dWOLS is the most accurate at curvature 0 and degrades the most under **both**
   yardsticks, so `preferred_estimator` and the case for averaging rather than
   picking are unaffected. What does not survive is the **crossover**: today's
   scoring has dWOLS falling behind the stage-specific fit at 0.15 (0.6191
   against 0.5936), and under the honest yardstick it leads at every curvature.
   "Neither dominates" is a weaker reading of this cohort than the numbers
   support.

   **The clamp is not removed**, and that is the judgement rather than an
   oversight. It is what keeps the outcome a bounded response, every other
   result in the repo is measured through it, and re-tuning a generator so a
   study reads better is the move this file keeps warning against. What is fixed
   is the claim: `treatment_free_value`'s docstring said curvature "changes only
   this nuisance function; the blips stay linear, so the estimand is untouched",
   and it now carries the table above.

   **The test guarding that claim was a tautology.**
   `test_curvature_leaves_the_estimand_untouched` asserted
   `true_blip(...) == true_blip(...)` — the same call twice, on a function that
   takes no curvature argument. `x == x`, which cannot fail, is invariant 25's
   defect in the one test whose name claimed the property being refuted. It is
   now two tests: one asserting `true_blip`'s *signature* carries no curvature,
   which is the real reason the declared blip is fixed, and one pinning the
   realised drift in both directions — negligible at 0, above 15% of arm-pairs
   at 0.15 — so the docs cannot go stale again. `DoubleRobustnessTests` pins the
   weight's effect and records, as a test rather than a comment, that `crp_std`
   and `alt_excess` are the only omittable covariates and neither moves
   assignment.

76. **A deferred item is a promise about what to build next, and invariant 74's
   was wrong.** That invariant diagnosed the survival estimator's 0.307 bias
   correctly — two thirds is censoring, the marginal Kaplan-Meier removes about
   40%, administrative censoring is `horizon - entry_month` and so depends on
   the covariates and the arms taken. Then it named a remedy it had not tried:
   *"a covariate-dependent censoring model is the repair."* Trying it is the
   only reason that is known to be false.

   The remaining horizon is **observable** — you know when a patient entered a
   line and when the study ends — so conditioning on it needs nothing a real
   analyst lacks, and it is the exact quantity the dependence runs through.
   Fitting the censoring curve within strata of it, 20 seeds at n=3000 with a
   correct surface and the generator's own propensity:

   | censoring model | total \|bias\| | worst |
   | --- | --- | --- |
   | marginal Kaplan-Meier (deployed) | 0.3073 | 0.0577 |
   | stratified on remaining horizon, 5 strata | **0.3033** | 0.0636 |
   | stratified on remaining horizon, 10 strata | **0.3085** | 0.0658 |
   | *(floor: no censoring at all)* | *0.1060* | *0.0270* |

   It buys **2% of the 0.20** censoring costs, and ten strata is worse than
   five. Whatever this is, it is not a censoring curve wanting more covariates.

   **There are two things in the residual and neither is one.**

   *Within a line, the complete case is truncated and reweighting cannot undo
   it.* `_rows_for` keeps only rows that progressed; a patient whose progression
   time exceeds their remaining horizon is dropped, which is likelier the lower
   their hazard, so the kept rows are short-time-selected in a
   covariate-dependent way. Inverse weighting repairs *random* censoring by
   upweighting comparable survivors and has nothing to upweight when the
   truncation is administrative. Measured over 12 seeds at n=3000, scoring the
   three parameters one line can identify, censoring costs **+0.0822 on line 1
   alone** — where every patient enters at month 0 and the horizon is the same
   60 months for all. **So it is not an entry-time effect**, which is what
   invariant 74 assumed and what made the stratification look promising.

   *Across lines, the risk set itself is selected.* A slow progressor reaches
   the horizon before ever starting line 2, so later lines over-represent fast
   progressors — high `biomarker_std`, a blip-basis term. 8 cohorts of 3000:

   | line | rows, censored | rows, uncensored | mean `biomarker_std` shift |
   | --- | --- | --- | --- |
   | 1 | 24,000 | 24,000 | **+0.0003** |
   | 2 | 18,096 | 24,000 | **+0.0805** |
   | 3 | 14,573 | 24,000 | **+0.1564** |

   The two are consistent rather than in tension: line 1's *times* are truncated
   while its *patients* are not selected, because everyone has a line 1. They
   stack — censoring costs +0.1357 at line 3 against +0.0822 at line 1.

   So the direction is a **censored-data likelihood** — an AFT fit that admits
   right-censored rows instead of discarding them, or Buckley-James imputation —
   **plus** an inverse-probability-of-being-at-risk weight for the sequential
   selection. Both are larger than swapping a nuisance model. Still not
   attempted, but the next person is now pointed at the right two things.

   **The first attempt at the line-1 control was an empty fit reporting a
   number.** `stage.line` is 1-indexed; the filter asked for `line == 0`,
   selected nothing, `_fit` refused a row count below the parameter count,
   `blip_parameters` returned zeros, and the "bias" came out at **2.2600** —
   which is exactly the sum of |true parameter| over the scored terms, as an
   empty fit must. It was about to be written down as *"line 1 is clean and
   censoring changes nothing there"*, on two empty fits agreeing with each
   other. The correctly indexed run says the opposite. `tests/test_survival_cohort.py`
   asserts the lines are numbered from one, because that is the assumption the
   rest of the measurement rests on, and it is cheap to state and expensive to
   get wrong.

77. **Invariant 76's deferred item, done — and its second half measured and
   refused.** That invariant named two repairs for the survival estimator's
   censoring bias: a censored-data likelihood for the within-line truncation,
   and an inverse-probability-of-being-at-risk weight for the across-line
   selection. Building both is how one of them turned out to be neither
   necessary nor helpful.

   **Buckley-James is now the default.** A row whose follow-up ended without
   progression is not missing at random — it is right-censored, and what is
   known is that the event came *later* than the time recorded. So the row is
   kept and its response replaced by `x'beta + E[e | e > e_i]`, with the residual
   distribution estimated by Kaplan-Meier and the fit iterated to convergence.
   It reuses `weighted_least_squares` and adds no optimiser to the module
   invariant 23 says to keep exact; a parametric Weibull AFT likelihood would be
   more efficient and would need Newton with a Hessian, and that trade is the
   reason for the choice. There is no censoring weight on the deployed path —
   imputing *and* weighting would count censoring twice — so
   `use_buckley_james=False` keeps the complete-case-plus-IPCW fit as a measured
   comparator, the way `treat_censored_as_terminal` does on the RA side. Twenty
   refits at n=2000:

   | | complete case + IPCW | Buckley-James |
   | --- | --- | --- |
   | total absolute bias | 0.4023 | **0.2978** |
   | shape error against a true 1.4 | 0.0104 | **0.0030** |
   | worst SE / actual spread | 0.91 | **1.03** |

   **The at-risk weight makes it worse**, 0.2122 to 0.2220 on the measurement
   that set this up, and the reason is the useful part: given `X_j` the line-j
   outcome is independent of how the patient got there, so selection on X alone
   does not bias a correctly specified regression — it only moves the covariate
   distribution, and the weight buys variance for nothing. Invariant 76's
   across-line table is therefore **real and harmless**: the shift is there,
   and it is not what the bias was. What *is* the bias is the within-line
   truncation, which is worse at later lines because less horizon remains, and
   that is what Buckley-James addresses. Stratifying its residual Kaplan-Meier
   on the remaining horizon buys a further 0.6% and is not taken either.

   **Three things went wrong in the building, and each is the kind this file
   exists to record.**

   *The shape was computed from the wrong residuals.* `psi = -k * coefficient`,
   so the shape multiplies every published parameter. The first integration took
   the residual Kaplan-Meier against the **imputed** responses, which moves every
   censored row and reads **1.343** against a true 1.4 — a 4% scale error on
   everything. A censoring indicator only means something against the time
   actually observed, so the shape reads the **recorded** residuals and the
   sandwich reads the imputed ones, because the first describes the error
   distribution and the second the estimating equation that was solved. With
   that split it reads **1.403**, better than the complete case's 1.410. The
   defect looked like a weakness of the method until it was localised.

   *The standard error is not the least-squares sandwich*, and shipping it as
   though it were would be invariant 27's pattern on a published quantity. The
   imputation carries uncertainty a variance computed as though every response
   had been observed cannot see. Measured the way `SANDWICH_INFLATION` was — 20
   refits, reported SE against the estimator's own spread — it runs 0.98 / 0.76
   / 0.84 by arm, mean 0.86. `_BJ_SE_INFLATION = 1.35` is set at the
   conservative end, as that constant is, because an interval too narrow
   *somewhere* is not repaired by being right on average. After it: 1.03 to
   1.32, no arm understating.

   *The obvious implementation was quadratic.* Computing each censored row's
   tail expectation by scanning the jumps is O(n^2) and measured **10x slower
   than the complete-case fit** at n=2000 — 8.76s against 0.89s, enough to
   matter to a suite this file lives in. Suffix sums over the jumps plus a
   bisect take it to 1.82s and the output is bit-identical, which is the only
   thing that makes the rewrite safe to keep.

   **The calibration is recorded, not asserted, and finding out why is the last
   item.** `test_the_standard_error_does_not_understate` passed at six seeds and
   should not have: `pstdev` over six refits reads the spread at **0.039** where
   twenty read **0.102**, so the comparison passes whatever the constant says.
   It was caught by a second test written against the same numbers disagreeing
   with the first. What the suite pins now is the **wiring** — that the
   inflation is exactly `_BJ_SE_INFLATION` times the uninflated sandwich, and
   that the comparator path does not carry it — and the calibration lives in the
   constant's comment where `cli coverage`'s numbers live. A six-seed spread is
   not something to assert on, and this file already says so about coverage
   tallies.

   **What remains.** Buckley-James closes about 40% of the gap to the
   no-censoring floor, not all of it, because its own assumption — censoring
   independent of the residual given the covariates — is not exactly true when
   the remaining horizon depends on history the covariates do not carry. Still
   wired to nothing: `EstimationLayer` scores a bounded response.

78. **`cli power` refit once per size and scored 240 patients against that one
   fit.** Invariant 31 says coverage is replicated over fits and never over
   patients on one fit, because every patient's interval is built from the same
   fitted parameters and one unlucky draw misses for everybody at once. The
   abstention curve has exactly that property and nobody had applied the rule to
   it: `power_curve` set `COHORT_SIZE`, called `reset()`, and scored every
   patient against the single ensemble that came back.

   **The published table was already saying so and it read as a finding.**
   79% at train 98, **85%** at 196, 66% at 280 — non-monotone at the low end,
   which invites reading it as more data making the agent *less* decisive at
   small n. Replicated over four training-cohort draws it is monotone
   throughout, so that was one fit:

   | train n | published (1 draw) | mean of 4 draws | spread | the draws |
   | --- | --- | --- | --- | --- |
   | 98 | 79% | 79.3% | **0.263** | .792 .833 .642 .904 |
   | 196 | **85%** | **68.9%** | **0.271** | .854 .629 .583 .688 |
   | **280 (deployed)** | **66%** | **62.9%** | 0.125 | .663 .563 .604 .688 |
   | 560 | 40% | 49.1% | 0.229 | .396 .488 .454 .625 |
   | 1120 | 33% | 36.8% | 0.133 | .325 .329 .358 .458 |

   **Scoring more patients cannot fix this, which is the whole point.** Binomial
   noise on a rate near 0.63 over 240 patients is about **0.032**; the observed
   spreads run 0.125 to 0.271, an order of magnitude larger. So the term that
   dominates is the one that was not being reported, and
   `DEFAULT_PATIENTS = 240` — raised once already for stability — could not have
   helped.

   **Two headline numbers move, and both now carry a range.** The shrinkage
   exponent goes 0.433 to **0.413** (0.388 to 0.433 across draws), still
   comfortably the n^-0.5 story. The trajectories needed for 30% abstention go
   1,430 to **1,641**, with draws spanning **1,273 to 2,164** — both are fitted
   to the *shape* of a curve whose every point was one fit, so `across_draws`
   reports what a redraw does to them. The point estimate alone was the defect.

   **It makes an existing guard stronger rather than weaker**, which is worth
   recording because the reflex with a noisier measurement is the opposite.
   `test_the_standard_error_responds_to_sample_size` asserts the exponent clears
   0.35 — the regression guard for the stale-dWOLS defect that read 0.15. On a
   three-size pilot a single draw read **0.309**, under the bar; the averaged
   curve read 0.393. Aggregating over draws is what keeps that test measuring
   the estimator rather than the draw.

   `power_curves` returns the unaggregated curves and `average_curves` folds
   them, so the per-draw rates survive onto every point as `equipoise_rates` and
   `equipoise_spread`. The sweep now restores **`COHORT_SEED` as well as**
   `COHORT_SIZE`, and a test asserts both: a restored size with a drifted seed
   would leave everything after it scoring against a cohort nobody chose, and
   that is the kind of leak a module-global sweep makes easy.

79. **Two SHADOW-gate criteria, built — and neither can be honestly satisfied by
   a simulation.** `live_data` is the loud blocker on the validation ladder and
   it is not the interesting one: it is a hardcoded `False` stating a fact, and
   no code clears it. Asking what happens *after* it flips is what pays.
   `deployment_readiness()` supplies **15 keys, every one for SILENT**. Every
   criterion above it reads *not measured* — `safety_events` and `concordance`
   at SHADOW, three more at ADVISORY, one at PRAGMATIC_TRIAL. So a real cohort
   arriving tomorrow would clear SILENT and stall immediately, not on a failure
   but because **the codebase can only execute the first step of the process it
   describes.** Both SHADOW criteria are now built, and building them is how
   their limits got measured rather than assumed.

   **`safety_guarantee_violations`** is invariant 2's guarantee, measured over
   the labelled safety cases: a published arm that safety removed, a name on a
   status that did not recommend, or a RECOMMEND beside a block flag. Three
   conditions counted separately, because one counter standing in for three is
   how a partial regression passes (invariant 59). It reads **0**, and the
   evidence that it is a check rather than decoration is that injecting the
   original defect — promote the runner-up when the leader is infeasible —
   takes it to **1**, naming the case and the reason.

   **Its first denominator was eight times too large, which is invariant 56
   inside the fix for something else.** "Cases where something was removed"
   reads 8 of 20; the number that matters is cases where *the arm the agent
   would publish* was removed, because that is the only situation where the
   layer must choose between halting and promoting. That is **1**: only an
   allergy to the leader removes it, since the physiological paths all
   contraindicate `JAK-inhibitor` and `methotrexate-optimization`, which are
   never this record's leader. A denominator of one is thin and is reported as
   such rather than smoothed.

   **And that denominator is not invariant under the regression it scales.**
   `top_scored_arm` is set from `safe.decision.recommended_arm`
   (`agent/__init__.py`), which is the field a promotion defect corrupts — so
   injected, it reads 0 where the healthy build reads 1. The violation count is
   the signal and the denominator is context. Deriving the leader from
   `q_values` would make it stable and is exactly the display-quantity
   re-derivation invariant 46 removed from five places, so it is not done.

   **`concordance` was not implementable, and the reason is structural.**
   `build_patient_state` appends an *open* decision point, so `stages[-1]`
   carries the sentinel `'current decision point'` and the record holds no
   clinician answer to the question the agent is asked. A first attempt to
   measure agreement read **0/120**, which was that sentinel and not a finding.
   `trajectory_to_bundle(..., through_stage=j)` is the repair: the record as it
   stood before decision `j`, with the arm taken at `j` withheld. The default
   path is byte-identical (`bec237a1e27b`), and a test asserts the truncated
   history is exactly the stages before `j` with the answer absent.

   Measured over 70 re-scored decisions, concordance is **0.129**. That is the
   design working, not a failure, and the decomposition says why:
   **`agent_never_chose` covers 35 of the 70** — `continue-current` 18,
   `JAK-inhibitor` 11, `TNF-inhibitor` 6 — arms the behaviour policy used and
   the agent never proposes, which can never agree. So the rate is structurally
   capped near 0.5, which is exactly where the gate's bar sits.

   **Neither is wired to its gate, and that is the finding rather than a
   shortfall.** `safety_events` means no patient was harmed while the model ran
   beside clinicians; what was built measures whether this layer's guarantee
   held on simulated records. `concordance` means agreement with clinicians;
   the reference here is the behaviour policy the agent is built to beat
   (rollout 2.156 against 1.770), so a `>= 0.5` bar against it would block a
   correct system — invariant 29's argument ("an oracle the agent beats is not
   an oracle") applied to a concordance threshold instead of a regret baseline.
   Wiring either would let the gate read satisfied on the strength of a
   simulation, which is the rung-skip invariant 25 exists to refuse. Tests
   assert both keys stay **absent** from `deployment_readiness()` and that the
   SHADOW gate keeps blocking on them.

80. **The card said the gate was shut and not that three of four gates have
   nothing behind them — and deriving that turned up two live defects.**
   Invariant 79 established that the ladder above SILENT is uninstrumented.
   `validation.instrumentation` now reports it: **1 of 4 rungs** measured, with
   the missing keys named per rung. It is derived by running the real gate
   rather than listing literals, because a card figure that parts company with
   its source is invariant 33's defect and this is a four-line summary of a
   module that changes.

   **The first derivation was wrong in a way worth recording.** It inferred
   "measured" from the blockers — count those saying *not measured* and compare
   with the total. That cannot see a criterion which is measured **and
   passing**, because a passing criterion produces no blocker at all. Supplying
   `safety_events: 0` left SHADOW reading uninstrumented, which a test asserting
   the report was derived caught immediately. The repair is one declaration:
   `_RUNG_CRITERIA` and `_SILENT_KEYS` are read by `_blockers`, which turns them
   into gate messages, and by `instrumentation_coverage`, which asks how many
   are supplied. Two readers, one list — invariant 3's rule on the ladder.

   **`blip_basis_unflagged` defaulted to passing.** `m.get("blip_basis_unflagged",
   True)`: an **absent** key read as "the basis is fine". That is invariant 25's
   defect — `metrics.get(key, default)` conflating "measured and passing" with
   "nobody measured this" — in the one branch of this module its fix did not
   reach, and defaulting the unsafe way. What sharpens it is the neighbours:
   `calibration_passed` two lines above and `live_data` two lines below both
   default to **blocking**, so this was the only SILENT criterion whose silence
   was taken for a pass. Latent today, because the specification test runs on
   every fit and always supplies the key — and latent is exactly how the
   original one survived.

   **Seven existing tests broke, and every one was right to.** Their "satisfied"
   fixtures omitted `blip_basis_unflagged` and passed *because* of the unsafe
   default, which is the clearest possible evidence it was load-bearing. They
   set it explicitly now.

   **And `_SILENT_KEYS` was one short.** `ope_improvement_lower` gates and was
   not declared, which surfaced only because
   `test_a_missing_measurement_is_a_blocker_not_a_pass` stopped asserting the
   literal `5` and started asserting `len(_SILENT_KEYS)`. A hardcoded count
   moves silently when a criterion is added; a derived one does not. That is
   this repo's own convention — *prefer making a test data-driven over
   hard-coding the new answer* — paying for itself in the same sitting.

   **`docs/INPUT_DATA.md` §4 is the other half**: what a first *live* cohort
   must supply, with every figure read from the code so `tests/test_docs.py`
   fails when the two part. It states the order of magnitude honestly —
   **~1,258 trajectories** for an identified final test and **~1,641 (1,273 to
   2,164)** for 30% abstention, against 400 today, so a first cohort of a few
   hundred does not clear the statistical gate however clean it is. It names the
   six SILENT keys, the four unadjusted confounders (`age`,
   `comorbidity_burden`, `gender`, `steroid_use`) that must be ingested **and
   placed in a basis** before the backdoor criterion holds, and the two SHADOW
   criteria that are built, measured and deliberately withheld. A test asserts
   that last part is still true — if someone wires one in, the document becomes
   wrong in the direction that matters, describing a gate that had quietly
   opened.

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
declaring it and returns an *irreducible* set, `satisfies_backdoor` implements
the backdoor criterion with descendant exclusion and collider handling — checked
against M-bias and descendant-of-collider graphs, not just the one it ships with
(invariant 52) — and `_ra_v1` passes `adjustment_set=()` so the graph is the
only thing that fills it.

`identified` reads True for every patient the pipeline actually sees, and that is
not a check that cannot fail: withhold a covariate and it returns False naming
the missing adjuster. **It is constant because the simulator always writes a
DAS28, a CRP and an anti-CCP — not because the contract requires them.** An
earlier version of this paragraph said the contract rejects records without
DAS28 or CRP; it does not. Missing variable families are *warnings* there, and
the families are broader than the covariates, so a HAQ-DI, an ESR or a
rheumatoid factor each satisfies the contract while leaving the estimators on a
default. Those records reach Layer 3 and are **blocked**, and `cli audit` now
scores that rather than asserting it — see invariant 64. `_has_adjuster`
treating any medication history as evidence for `prior_biologic_exposure` is
deliberate — "no prior biologic" is a value of that variable, not a missing one,
and requiring a biologic to appear blocked every csDMARD-only patient for having
been treated conservatively.

Still deliberately simple, and labelled as such in-module:
`HandcraftedFeatureEncoder` (nine clinical features normalised and tiled — not a
learned representation, and it does not claim to be; the `GRUBaselineEncoder`
that wrapped it is deleted, see invariant 37),
`SemanticKnowledgeBase` (five hard-coded passages, word-token keyword retrieval
— not a vector store, though it does now match its own arm vocabulary, see
invariant 47),
`BLIP_TERM_LANGUAGE` (four clinical phrases, one per blip-basis term — all that
is left of `WHY_NOT_REASONS`, which is gone: the reason an arm was ruled out is
now the gap's own per-covariate decomposition, keyed on the covariate the model
chose rather than on the arm, see invariant 57). The E-value
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

**Carry-forward is unbounded, its age is discarded, and on this cohort it costs
nothing — all three measured.** `StageHistoryBuilder._features_until` takes every
observation recorded up to the decision day, last one wins, with no recency
bound; `StageRecord.features` is then a plain `{name: value}` map, so the
measurement date is gone and nothing downstream could flag a stale covariate if
it wanted to. Nothing in `data/` or `safety/` mentions recency at all.

Measured over 60 simulated patients, **every covariate the estimators read is 0
days old** — min, median and max. That is the simulator writing a fresh
observation at every visit, not a property of the code: the demo bundle in every
test already carries an `anti_CCP` drawn at baseline and **365 days old** at the
decision point.

Priced by removing the most recent DAS28, CRP or anti-CCP so the previous value
carries forward — a median of **122 days** old, which is a lab not redrawn this
visit — it changed **0 of 60** statuses and **0 of 60** ranked arms, for each of
the three. The reason is worth keeping rather than the number: `das28` moves
0.551 between visits against a cohort spread of 1.828, which is real movement,
but it carries little weight in the blip; `anti_ccp` carries the most and
**never changes** (visit-to-visit delta exactly 0.000), correctly, because
serostatus is stable. So the zero is a property of the blip basis as much as of
the fixture, and on a cohort where disease activity drove the effect harder it
would not hold. Reported rather than acted on, and the adapter *does* sort
observations by date (`data/fhir.py`), so "most recent" is not decided by bundle
order — that was checked before it was assumed.

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
patients, the ensemble refit at each cohort size and at **four training-cohort
draws per size** (invariant 78 — this used to be one, and one is not a curve):

| train n | abstains (mean of 4 draws) | spread across draws | mean contrast SE | mean \|contrast\| |
| --- | --- | --- | --- | --- |
| 98 | 79.3% | **0.263** | 0.0260 | 0.045 |
| 196 | 68.9% | **0.271** | 0.0200 | 0.037 |
| **280 (deployed)** | **62.9%** | **0.125** | 0.0174 | 0.039 |
| 560 | 49.1% | 0.229 | 0.0124 | 0.044 |
| 1120 | 36.8% | 0.133 | 0.0092 | 0.044 |

The contrast itself is flat; only the precision moves. The standard error shrinks
at n^-0.41 (range 0.388-0.433 across draws) against the n^-0.50 a correctly
specified estimator earns. Under the simultaneous all-pairs rule, roughly
**1,641 training trajectories (1,273 to 2,164 across draws)** would bring
abstention to 30% — an extrapolation beyond the measured range, and now one that
says how much a single draw moves it. `COHORT_SIZE = 400` is therefore a
deliberate choice near the low end, not a tuned one — raise it and the agent
recommends more, which is a statement about data, not about the method.

**Read the shape, not any one point.** The spread at a fixed size runs 0.125 to
0.271, so the deployed 62.9% is a mean over draws that individually read 0.563
to 0.688. The previously published single-draw figures — 79 / 85 / 66 / 40 / 33 —
were non-monotone at the low end, which read as more data making the agent
*less* decisive; replicated, the curve is monotone throughout.

This table used to score only 60 patients, too few for a stable headline
abstention estimate. The reference evaluation now uses 240 patients and four
draws, and must be regenerated whenever the decision threshold or its
multiplicity correction changes. Do not lower `power.DEFAULT_PATIENTS` or
`power.DEFAULT_REPLICATES` back.

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
| baseline (same process) | 67% | 97% | 1.11 | 0.004 | 0.734 | 0.639 | +0.386 |
| sicker, more seronegative | **89%** | 95% | 1.04 | 0.004 | 0.600 | 0.527 | +0.372 |
| milder, mostly seropositive | **54%** | 96% | 1.03 | 0.003 | 0.846 | 0.753 | +0.437 |
| TNF-first, rituximab-averse | 63% | 95% | 1.04 | 0.004 | 0.726 | 0.647 | +0.386 |
| heavy attrition | 73% | 97% | 1.10 | 0.005 | 0.748 | 0.669 | +0.469 |
| near-complete follow-up | 74% | 95% | 0.99 | 0.006 | 0.720 | 0.621 | +0.395 |
| combined: all three | 84% | 95% | 1.08 | 0.002 | 0.617 | 0.555 | +0.346 |
| **estimand shifted: blips 1.5x** | 69% | — | — | **0.051** | 0.824 | 0.695 | +0.538 |

Three of the four headline claims transfer and one does not.

*The policy transfers.* It beats local practice on rollout value at 7/7
estimand-preserving sites, by +0.346 to +0.469, including the site where
clinicians already prescribe TNF-first. *Coverage transfers*, holding 95-97% —
at nominal now that the dWOLS cross-arm covariance is kept rather than dropped.
*Calibration transfers*, 0.002-0.006 against 0.004 at home. Layer 1 rejected
**0** records at every site.

*Abstention does not.* It runs 54% to 89% against the 67% on the model card —
a 35-point swing driven by population changes, and in the direction that matters:
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
pooled rate hides a 59-point spread (68.3% pooled). Strata are the non-intercept terms of
`BLIP_BASIS` — the covariates the true effect actually varies over — at n=240:

| stratum | n | abstains | true gap | contrast SE | at pooled SE |
| --- | --- | --- | --- | --- | --- |
| anti-CCP negative | 90 | **97%** | 0.034 | 0.0179 | 97% |
| anti-CCP positive | 150 | 51% | 0.051 | 0.0169 | 74% |
| TNF naive | 158 | 75% | 0.038 | 0.0166 | 98% |
| prior TNF exposure | 82 | 55% | 0.056 | 0.0185 | 52% |
| das28 < 3.85 | 80 | 71% | 0.045 | 0.0168 | 78% |
| das28 3.85-5.89 | 80 | **38%** | 0.056 | 0.0155 | 78% |
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

False positives over null cohorts: **2 of 60**, which is 3.3% against a nominal
5%. This said **0/30** and that figure was a single seed block: seeds 7000-7029
give 0/30 and 9900-9929 give 2/30. The rate is *near nominal*, which is what a
calibrated test should read — a different and weaker claim than zero, and the
tests span both blocks now so neither can be picked. Getting there took two
corrections, both found by measuring rather than reasoning. The
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
worst standard error in the cohort.

**Read that 92% against this candidate's own null distribution, not against the
threshold alone**, because the two obvious readings are both wrong. Measured over
60 null cohorts:

| candidate | null mean \|z\| | sd | deployed | percentile of its own null |
| --- | --- | --- | --- | --- |
| `crp_std` | 1.475 / 1.601 | 0.48 / 0.60 | 2.302 | 87-93 |
| `egfr_std` | 1.816 / 1.725 | 0.64 / 0.68 | 2.726 | 90-93 |
| **`das28_squared`** | **1.905 / 2.013** | 0.65 / 0.83 | **3.367** | **90-100** |

It is *not* "where this candidate always sits": the null mean is 1.905, so 3.367
is about two standard deviations above it and above every draw in one block of
30. And it is *not* a clean covariate-specific tail either — **the deployed seed
reads at the 87th to 93rd percentile for the other two candidates as well**, so
part of the elevation is a cohort that runs high across the board.

What the max-composition mechanism does explain is the **baseline**: this
candidate's null mean is 1.905 against 1.475 for `crp_std`, about 0.43 of z, and
that gap is the curvature backward induction creates. It is not the 1.46 that
separates 1.905 from 3.367. So the margin decomposes into three parts — a
structurally higher baseline for a curved candidate, a generally high-reading
cohort, and a genuine excess on top — and "92% of the way to firing" invites
reading all of it as the third. That is not a coincidence and the mechanism is
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
what the estimators' linear basis can represent. It leaves the **declared**
blips untouched — `true_blip` never reads curvature — but **not the realised
estimand**, which this paragraph used to claim and invariant 75 measures:
`expected_outcome` clamps the sum to [0, 1], so at the study's own grid up to
25% of arm-pairs have a contrast that is no longer `TRUE_BLIPS`. It is 0 by
default; every other result in the repo assumes the correctly specified cohort.
`cli misspecification` uses it to show the estimators trading off as designed:
dWOLS is the most accurate when the nuisance model is right and degrades the
most, the shared-blip fit the reverse. **That conclusion survives being rescored
against the contrast the cohort actually has** (invariant 75), so the trade-off
is still the justification for averaging them rather than picking one — but the
apparent crossover at curvature 0.15 does not survive, and is a clamp artifact.

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
