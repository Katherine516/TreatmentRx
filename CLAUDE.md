# CLAUDE.md

TreatmentRx / PrecisionRx: a research-stage clinical decision-support agent for
sequential treatment decisions in rheumatoid arthritis. **Research scaffolding,
not a medical device.** Nothing here is clinically validated; the synthetic
cohort is a test fixture, not evidence.

## Commands

```bash
PYTHONPATH=src python3 -m unittest discover -s tests   # full suite, ~3s
PYTHONPATH=src python3 -m precisionrx_agent.cli demo       # one patient end to end
PYTHONPATH=src python3 -m precisionrx_agent.cli evaluate   # estimator scorecard
```

## Hard constraints

- **Python 3.9 runtime.** `pyproject.toml` says `>=3.9` because that is the
  interpreter this is developed on. Every module starts with
  `from __future__ import annotations`; keep it. No `match`, no PEP 604 unions
  outside annotations, no `X | Y` in `isinstance`.
- **Zero third-party dependencies.** `dependencies = []` is deliberate. Linear
  algebra lives in `layer4_estimation/linalg.py`. Do not reach for numpy/scipy
  without the user asking — the constraint is what keeps the prototype
  auditable and installable anywhere.
- **Determinism.** The cohort is seeded (`training.COHORT_SEED`) and models are
  fit lazily once per process. A given commit must always produce the same
  recommendation for the same patient. The only wall-clock value anywhere is the
  audit-event timestamp.

## Layout

Two packages implement overlapping versions of the same agent:

- `src/precisionrx_agent/` — the live pipeline. Numbered layers
  (`layer1_ingestion` … `layer11_feedback`) plus `shared/`, `simulation/`.
  Entry point: `pipeline.PrecisionRxAgent.recommend`.
- `src/treatmentrx/` — a six-layer facade over typed contracts
  (`contracts.py`), reached via `TreatmentRxOrchestrator.run`. It reuses Layer 1,
  the Layer 4 estimators and Layer 9 memory, but **reimplements** decision,
  safety and rationale generation.

**This fork is a known liability, not a design.** The two paths can diverge
clinically — they already did once: the data contract and the estimators used
different arm names, so the facade's safety layer classified the top-scored arm
as infeasible and silently substituted the runner-up. `shared/arms.py` is now
the single arm vocabulary and
`tests/test_v5_architecture.py::test_both_pipelines_agree_on_the_recommended_arm`
guards it. Prefer consolidating over adding logic to both; never add clinical
logic to only one.

Top-level modules in `precisionrx_agent/` (`safety.py`, `models.py`, `fhir.py`, …)
are one-line `import *` shims for the layer packages. Import from the layer
module in new code.

## Invariants — do not break these silently

1. **Safety runs before explanation.** `SafetyGate` / `SafetyLayer` are code, not
   prompts. An LLM layer may render a block; it may never lift one.
2. **Memory never moves a number.** `layer9_memory_rag.apply_memory` snapshots
   `statistical_output` and raises `MemoryInfluenceError` if any of
   `q_values`, `policy_value`, `confidence_band`, `recommended_action`,
   `safety_status` changed. Memory shapes narrative and retrieval only.
3. **Leakage guards raise.** `LeakageTestSuite` failing a temporal firewall check
   raises `LeakageError` out of `recommend()`. Do not downgrade it to a warning.
4. **Estimators see only the training split.** Everything is fit through
   `layer4_estimation/training.py`. Never fit on `fitted().holdout`, and never
   fit an estimator ad hoc in a layer that consumes one.
5. **Calibration is measured on held-out data.** Scoring a patient's outcomes
   against themselves yields ECE ≈ 0 for everyone and makes the validation-ladder
   gate vacuous. That was a real bug; do not reintroduce it.
6. **One arm vocabulary**: `shared/arms.py`. The data contract, the estimators,
   the composite action space and the feasible-set filter must all agree.
7. **Q-values live on one scale.** Q-learning's `raw_q` is a value-to-go;
   `q_values()` divides by the remaining horizon so every estimator reports
   expected response per remaining visit. Model averaging across estimators is
   only meaningful because of this. `predict_outcome` uses the terminal-stage
   parameters and is the single-stage quantity calibration is measured on.

## What is real vs. still a placeholder

Real: the three Layer 4 estimators, the synthetic cohort and its known blips,
held-out IPW policy evaluation, calibration, blip attributions in
`layer4_estimation/explainability.py`.

Still deliberately simple, and labelled as such in-module: confidence bands
(support-size heuristic, not standard errors), `GRUBaselineEncoder` (a
deterministic summariser, not a trained GRU), `CausalDAGRegistry` (hand-listed
adjustment sets, no formal identifiability), `SemanticKnowledgeBase` (five
hard-coded passages, not a real RAG index), the E-value in the sensitivity
report, and the regime selector's BIC proxy.

When you touch one of these, either make it real or keep the docstring honest
about what it is not. The value of this codebase is that a reader can tell the
difference.

## Conventions

- Docstrings explain *why* a method was chosen and what it does not guarantee —
  match that register. Comments earn their place by explaining a clinical or
  statistical reason, not by restating the code.
- Dataclasses in `shared/models.py` are `frozen=True`; construct new ones with
  `replace()` or `StageRecord(**(record.__dict__ | {...}))` rather than mutating.
- Tests assert on quantities (recovery, policy value, calibration), not on
  plumbing. When behaviour changes, prefer making a test data-driven over
  hard-coding the new answer.
