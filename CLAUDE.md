# CLAUDE.md

TreatmentRx: a research-stage clinical decision-support agent for sequential
treatment decisions in rheumatoid arthritis. **Research scaffolding, not a
medical device.** Nothing here is clinically validated; the synthetic cohort is a
test fixture, not evidence.

## Commands

```bash
PYTHONPATH=src python3 -m unittest discover -s tests    # full suite, ~16s
PYTHONPATH=src python3 -m treatmentrx.cli demo          # one patient end to end
PYTHONPATH=src python3 -m treatmentrx.cli evaluate      # estimator scorecard
PYTHONPATH=src python3 -m treatmentrx.cli stability     # k-fold + seed sweep (~15s)
PYTHONPATH=src python3 -m treatmentrx.cli inference     # sandwich vs bootstrap (~35s)
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
3. **One arm vocabulary**: `arms.py`. The data contract, the estimators, the
   composite action space and the feasible-set filter must all agree.
4. **One type per layer handoff.** `MethodResult`/`RegimeEstimate` and
   `PatientStage`/`StageRecord` were once duplicate pairs, and the conversion
   between them silently dropped the timing, belief, switching and competing-risk
   annotations before the estimators saw them. Do not reintroduce a parallel type.
5. **Memory never moves a number.** `agent/memory.apply_memory` snapshots
   `statistical_output` and raises `MemoryInfluenceError` if `q_values`,
   `policy_value`, `confidence_band`, `recommended_arm` or `safety_status`
   changed. Memory shapes narrative and retrieval only.
6. **Leakage guards raise.** A temporal-firewall violation raises `LeakageError`
   out of `DataLayer.build_patient_state`. Never downgrade it to a diagnostic.
7. **Estimators see only the training split.** Everything is fit through
   `estimation/training.py`. Never fit on `fitted().holdout`, and never fit an
   estimator ad hoc in a layer that consumes one.
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
    produced a number that was neither.
15. **Every advanced arm keeps a monotherapy composite.** Listing biologics only
    in MTX combination turns one methotrexate contraindication into a blocked
    recommendation for a patient who had a viable option.

## What is real vs. still a placeholder

Real: the three estimators, the cohort and its known blips, informative-dropout
handling and IPCW, held-out IPW policy evaluation, calibration, cluster-robust
standard errors, m-out-of-n bootstrap intervals, contrast tests, cross-validated
stability, blip attributions.

Still deliberately simple, and labelled as such in-module: `GRUBaselineEncoder`
(a deterministic summariser, not a trained GRU), `CausalDAGRegistry` (hand-listed
adjustment sets, no formal identifiability), `SemanticKnowledgeBase` (five
hard-coded passages), the E-value in the sensitivity report, the regime
selector's BIC proxy, and `IPCWHandler`'s visit weights (heuristic — the cohort
now generates severity-driven visit spacing, so there is ground truth to fit
against that is not yet used).

Inference has two paths and they are not interchangeable. The sandwich
(`sandwich_contrast`) is the default: cheap, exact at a stage-specific terminal
block, caveated everywhere else. The m-out-of-n bootstrap (`fit_bootstrap`,
`treatmentrx.cli inference`) re-runs the whole procedure per replicate and is
the honest answer wherever backward induction is involved — which with a shared
blip is every stage. It measures ~20% wider here. Once attached, `contrast()`
prefers it automatically.

Measured and stated rather than assumed: IPCW is a ~5% correction on this
generating process, because the outcome model is correctly specified and
conditions on the covariates that drive dropout. `estimation/censoring.py` says
so in its docstring. Do not quietly re-tune the simulation to make it look
larger.

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
