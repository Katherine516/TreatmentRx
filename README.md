# PrecisionRx Agent Prototype

This repository contains a first-step implementation of the healthcare AI agent described in
`Healthcare Agent Architecture.html`.

The current build is a runnable Phase 1 spine:

- FHIR-like patient bundle ingestion
- RA MVP data contract validation
- Longitudinal stage construction for multi-stage treatment regimes
- Simple inverse visit intensity and censoring weights
- RA Causal DAG v1.0 identifiability scaffold and deterministic causal path text
- Handcrafted and GRU-compatible baseline patient-state encoders
- Tailoring variable extraction
- Adaptive regime selector with clinical-rule overlay
- Q-shared style treatment scoring with policy-value comparison
- Bayesian model averaging across Q-shared, dWOLS-shared, and stage-specific estimators
- Four-type uncertainty decomposition and ECE-style calibration reporting
- Synchronous safety gate before explanation generation
- PHI-minimized LLM context bundle
- Deterministic clinician and patient narratives for offline/demo use
- CLI demo and unit tests

### v5.1 Clinical Realism additions

The pipeline now folds the ten clinical-realism additions (plus a three-tier memory
architecture and a prospective validation ladder) into the existing layers:

1. **Treatment timing model** (`layer1_ingestion/timing.py`) — irregular intervals,
   per-drug response/toxicity windows, inverse-intensity visit weights.
2. **Competing risks** (`layer1_ingestion/competing_risks.py`,
   `layer4_estimation/competing_risk_outcomes.py`) — typed `(event, time)` outcomes;
   progression-free endpoint adjustment.
3. **Switching & rescue** (`layer1_ingestion/switching.py`,
   `layer11_feedback/estimands.py`, `switching_aware_ope.py`) — realized-vs-assigned
   capture; ITT / per-protocol / as-treated estimands; IPTW off-policy evaluation.
4. **Multi-action / combination** (`layer4_estimation/actions.py`,
   `layer8_safety/feasible_set.py`) — composite `{drug,dose,route,timing,combination}`
   actions over a curated candidate set; composite-aware safety filter.
5. **Partial observability** (`layer1_ingestion/belief.py`) — filtered belief over
   latent disease activity with a POMDP-ready seam; belief-aware uncertainty.
6. **Dynamic treatment goals** (`CareGoal`, `layer4_estimation/goal_conditioned.py`) —
   goal/phase becomes part of state; goal-conditioned thresholds and framing.
7. **Override governance** (`layer11_feedback/override_governance.py`) — four review
   channels; only outcome-validated overrides influence the model.
8. **Prospective validation ladder** (`layer11_feedback/validation_ladder.py`) —
   silent → shadow → advisory → pragmatic trial, gated.
9. **Model explainability** (`layer4_estimation/explainability.py`) — blip
   attributions, why-not table, counterfactual probes, assumption sensitivity.
10. **Leakage / immortal-time guards** (`layer1_ingestion/leakage.py`) — temporal
    firewall + immortal-time detector + CI leakage suite (hard assertions).

**Memory** (`layer9_memory_rag/memory.py`) — working / episodic (structured,
patient-keyed) / semantic (RAG knowledge base) tiers with a hard influence boundary:
`apply_memory` asserts memory never moves a statistical quantity.

This is research software scaffolding, not a medical device. The statistical modules are
deliberately simple placeholders where clinical validation, real estimators, and audited
guideline knowledge bases must be added before any clinical use.

### Estimation layer (real statistics)

The Layer 4 estimators are fit, not hand-tuned. All three are fit once per
process on the same seeded training split and scored on the same held-out split:

| Module | Method | What it is |
| --- | --- | --- |
| `simulation/ra_cohort.py` | — | Multi-arm, two-stage synthetic RA cohort with *known* per-arm blip functions, confounded treatment assignment, and a delayed hepatotoxicity cost |
| `layer4_estimation/q_learning.py` | Q-Shared + Penalized | Ridge-penalized Q-learning, blip parameters shared across stages, fit by backward induction on pseudo-outcomes |
| `layer4_estimation/q_learning.py` | Stage-Specific Q-learning | Same machinery with an independent blip per stage |
| `layer4_estimation/dwols.py` | dWOLS-Shared | Doubly-robust weighted blip regression, each arm fit one-vs-reference |
| `layer11_feedback/offline_evaluation.py` | — | Held-out IPW (Hajek) policy value, arm agreement, and calibration |
| `layer4_estimation/training.py` | — | Owns the split, the fits, and the scorecard |

`MethodResult.policy_value` is the estimator's **held-out IPW policy value**, so
Bayesian model averaging weights estimators by out-of-sample performance rather
than by self-report. Calibration is likewise measured on held-out patients — a
patient's own outcomes are never scored against themselves.

What the tests actually assert (`tests/test_estimators.py`): every arm's blip is
recovered under confounding; the adjusted estimate beats the naive contrast;
backward induction beats a myopic fit on the delayed-toxicity process; and every
learned policy beats the clinician policy that generated the data.

## Quick Start

```bash
PYTHONPATH=src python3 -m precisionrx_agent.cli demo
```

```bash
PYTHONPATH=src python3 -m precisionrx_agent.cli evaluate
```

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Research Proposal

The simplified research proposal is available at `docs/research_proposal.md`. It covers
research objectives, significance, agent architecture, technical approach, agent functions,
and summary.

## Current Architecture

```text
FHIR-like Bundle
  -> FHIRAdapter
  -> RADataContract
  -> StageHistoryBuilder
  -> VisitAligner + IPCWHandler + VariableSelector
  -> RA DAG Registry + baseline encoders
  -> AdaptiveRegimeSelector
  -> QSharedEstimator + dWOLSSharedEstimator + StageSpecificQEstimator
  -> BayesianModelAverager + UncertaintyDecomposer + CalibrationEvaluator
  -> SafetyGate
  -> ContextBundleBuilder
  -> RationaleGenerator
```

## v5 Six-Layer Facade

The v5 build spec is implemented as a facade package whose shape mirrors the
architecture:

```text
treatmentrx.contracts      # typed boundary objects
treatmentrx.data           # Layer 1: FHIR -> PatientState
treatmentrx.estimation     # Layer 2: PatientState -> RegimeEstimate[]
treatmentrx.decision       # Layer 3: BMA/uncertainty -> Decision
treatmentrx.safety         # Layer 4: feasible-set filter -> SafeDecision
treatmentrx.agent          # Layer 5: context/RAG/memory -> Recommendation
treatmentrx.feedback       # Layer 6: Track A / Track B receipt
treatmentrx.orchestrator   # thin run() entrypoint
```

Run the v5 path from Python:

```python
from precisionrx_agent.demo_data import sample_ra_bundle
from treatmentrx import TreatmentRxOrchestrator

recommendation = TreatmentRxOrchestrator().run(sample_ra_bundle())
```

## Next Build Steps

Done:

- ~~Replace heuristic Q-shared scorer with penalized least squares and backward induction.~~
- ~~Add real propensity modeling and held-out IPW policy value.~~

Next, in order:

1. **Proper uncertainty on the blip parameters.** Confidence bands are still a
   support-size heuristic. Bootstrap or sandwich standard errors on psi, then
   drive the equipoise threshold from them instead of a fixed Q-gap of 0.04.
2. **Cross-validation and a stability check.** One fixed split is a weak
   guarantee; k-fold plus a seed sweep would show whether the estimator ranking
   in `evaluate` is real or noise (currently the three are within their own
   standard error of each other, which the scorecard says out loud).
3. Train the GRU baseline on a real or synthetic RA corpus and compare against
   the handcrafted baseline. Only worthwhile once the cohort has richer
   longitudinal structure than two stages.
4. Replace the sample contraindication rules with a versioned clinical knowledge
   base, keyed to `shared/arms.py`.
5. Add guideline and drug-safety RAG ingestion, then assemble the 3-agent
   explanation system.
6. Resolve the `precisionrx_agent` / `treatmentrx` fork (see CLAUDE.md) before
   wrapping the pipeline in FastAPI, so only one decision path can be served.
