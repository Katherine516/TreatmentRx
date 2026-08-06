# TreatmentRx

A research-stage clinical decision-support agent for **sequential** treatment
decisions in rheumatoid arthritis. It estimates the value of each next-step
therapy from a patient's longitudinal history, reports how confident it is,
enforces safety in code, and explains itself from the model rather than around
it.

**Research scaffolding, not a medical device.** Nothing here is clinically
validated, and the synthetic cohort is a test fixture, not evidence.

```bash
PYTHONPATH=src python3 -m treatmentrx.cli demo
```

```bash
PYTHONPATH=src python3 -m treatmentrx.cli evaluate
```

```bash
PYTHONPATH=src python3 -m treatmentrx.cli inference
```

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

```python
from treatmentrx import TreatmentRxOrchestrator
from treatmentrx.demo_data import sample_ra_bundle

recommendation = TreatmentRxOrchestrator().run(sample_ra_bundle())
```

Pure Python 3.9, zero third-party dependencies.

## Architecture

Six layers, one direction. The order is load-bearing: safety runs before any
language is generated, and feedback runs last and cannot change the answer.

```text
FHIR bundle
  ↓  treatmentrx.data          Layer 1  → PatientState
  ↓  treatmentrx.estimation    Layer 2  → RegimeEstimate[]
  ↓  treatmentrx.decision      Layer 3  → Decision
  ↓  treatmentrx.safety        Layer 4  → SafeDecision
  ↓  treatmentrx.agent         Layer 5  → Recommendation
  ↓  treatmentrx.feedback      Layer 6  → FeedbackReceipt
```

| Layer | Does | Key modules |
| --- | --- | --- |
| **0 Simulation** | Multi-stage cohort with known blips, confounded assignment, informative dropout and irregular visits — plus a FHIR exporter so simulated patients re-enter through Layer 1 | `simulation/` |
| **1 Data** | FHIR ingestion, RA data contract, stage construction, irregular timing, switching/rescue capture, belief filtering over latent disease activity, competing risks, causal DAG identifiability, leakage guards | `data/` |
| **2 Estimation** | Three treatment-regime estimators over one arm menu, one scale, one training split | `estimation/` |
| **3 Decision** | Bayesian model averaging, four-type uncertainty, held-out calibration, competing-risk and belief adjustment, goal-conditioned thresholds, blip explainability, contrast testing | `decision/`, `estimation/inference.py` |
| **4 Safety** | Composite-action feasible set, hard clinical rules, delayed-toxicity detection | `safety/` |
| **5 Agent** | PHI-minimised context, episodic memory + guideline RAG, three explanation agents | `agent/` |
| **6 Feedback** | ITT/per-protocol/as-treated estimands, switching-aware OPE, validation ladder, override governance | `feedback/` |

## The estimation layer

All three estimators are fit once per process on the same seeded training split
and scored on the same held-out split.

| Method | What it is |
| --- | --- |
| Q-Shared + Penalized | Ridge-penalized Q-learning, blip parameters shared across stages, fit by backward induction on pseudo-outcomes |
| Stage-Specific Q-learning | Same machinery, an independent blip per stage |
| dWOLS-Shared | Doubly-robust weighted blip regression, each arm fit one-vs-reference |

They are kept different on purpose: Q-learning is an outcome-model method and
dWOLS is doubly robust, so they fail in different ways. Averaging them is only
informative because of that.

`RegimeEstimate.policy_value` is the estimator's **held-out IPW policy value** —
a model-level score, not a transform of its own Q-values — so model averaging
weights estimators by out-of-sample performance. Calibration is likewise measured
on held-out patients; a patient's own outcomes are never scored against
themselves.

### What the tests actually assert

`tests/test_estimators.py` and `tests/test_inference.py` check the statistics,
not the plumbing:

- every arm's blip function is recovered under confounded assignment;
- the adjusted estimate is closer to truth than the naive contrast;
- backward induction beats an otherwise-identical myopic fit on a generating
  process with a delayed hepatotoxicity cost, and stops prescribing the
  hepatotoxic arms at stage 1;
- every learned policy beats the clinician policy that produced the data, by
  both held-out IPW and oracle rollout;
- standard errors shrink with √n, and an arm compared against itself is never
  reported as distinguishable.

### The generating process

Three decision points per patient, six arms, and three mechanisms that make the
estimation problem real rather than decorative:

| Mechanism | What it breaks | What has to handle it |
| --- | --- | --- |
| Confounded assignment | Naive arm means | Propensity weighting (dWOLS) or outcome modelling (Q-learning) |
| Delayed hepatotoxicity | Myopic policies | Backward induction |
| Informative, arm-differential dropout | Complete-case analysis | Censored-row handling and IPCW |

Dropout depends on toxicity and on response, and burdensome infusion arms are
abandoned unless they are clearly working — so retention is *differentially*
response-dependent by arm. Trajectories end early, visit intervals shorten with
disease activity, and `simulation/fhir_export.py` renders any trajectory as an
ingestible bundle, so Layer 1's timing, switching and competing-risk machinery is
finally checked against patients whose truth is known by construction rather
than against one hand-written demo.

### Uncertainty is a standard error, not a heuristic

Confidence bands come from a cluster-robust (sandwich) covariance matrix,
clustered by patient because two stages from one trajectory are correlated. The
decision-relevant quantity is the **contrast** between the top arm and the
runner-up:

```text
Separation: rituximab over IL-6 inhibitor is +0.083
(SE 0.021, 95% CI [+0.041, +0.125]) — separable at this sample size.
```

Equipoise now needs two independent conditions to fail. The care goal sets how
large a difference is worth acting on; the interval decides whether the data can
resolve a difference that size at all. A gap that clears the clinical bar but
sits inside its own confidence interval is equipoise, not a recommendation.

### Where the sandwich is not enough

The sandwich treats the pseudo-outcomes as fixed data when they are themselves
estimated, so it is optimistic wherever backward induction is involved — which
with a *shared* blip is every stage, since one parameter vector is fit jointly
from every stage's rows.

The repair is an m-out-of-n bootstrap, which re-runs the entire procedure per
replicate. The ordinary bootstrap is inconsistent here: the `max` in the
pseudo-outcome is not smooth where two arms are tied, so the resample size
adapts to the measured proportion of patients sitting near such a tie.

```bash
PYTHONPATH=src python3 -m treatmentrx.cli inference
```

```text
Q-Shared + Penalized      n=280  m=191  non-regularity 0.20
Stage-Specific Q-learning n=280  m=88   non-regularity 0.61

rituximab vs IL-6 inhibitor, stage 0
  sandwich  SE 0.0062  CI [0.0239, 0.0481]
  bootstrap SE 0.0074  CI [0.0218, 0.0505]     ratio 1.20
```

The sandwich understates the interval by about 20% here. It stays the default —
it is exact at a stage-specific terminal block, it costs nothing, and most
decisions are terminal — but `training.enable_bootstrap_inference()` switches
the contrast over when an interval has to be defensible.

### Is the estimator ranking real?

```bash
PYTHONPATH=src python3 -m treatmentrx.cli stability
```

Refits across k folds and across cohort seeds, then says plainly whether the
leader's advantage survives the spread. Today it does not:

> Q-Shared + Penalized leads dWOLS-Shared by only 0.0002 against a 0.0138
> combined standard error (winning 25% of resamples). The estimators are not
> distinguishable at this sample size; treat the scorecard ordering as arbitrary
> and keep averaging them.

That is why the model-averaging weights come out near-uniform. Sharpening them
would manufacture a ranking the data does not support.

## What is real vs. still a placeholder

**Real:** the three estimators, the synthetic cohort and its known blips,
informative-dropout handling and IPCW, held-out IPW policy evaluation,
calibration, sandwich standard errors, m-out-of-n bootstrap intervals, contrast
tests, cross-validated stability, blip attributions.

**Deliberately simple, and labelled as such in-module:** `GRUBaselineEncoder` (a
deterministic summariser, not a trained GRU), `CausalDAGRegistry` (hand-listed
adjustment sets, no formal identifiability), `SemanticKnowledgeBase` (five
hard-coded passages, not a real RAG index), the E-value in the sensitivity
report, and the regime selector's BIC proxy.

The value of this codebase is that a reader can tell the difference.

## Next build steps

Done: penalized Q-learning with backward induction; held-out IPW policy value;
sandwich standard errors and contrast-driven equipoise; cross-validated
stability; m-out-of-n bootstrap for the non-regular stages; a multi-stage cohort
with informative dropout and irregular visits, ingestible through Layer 1.

Next, in order:

1. **Estimate the visit-intensity weights.** Dropout is now modelled from the
   data, but `IPCWHandler` still assigns visit weights heuristically. The cohort
   generates severity-driven visit spacing, so the ground truth to fit against
   exists — it just is not used yet.
2. **A misspecified-outcome-model arm of the simulation.** IPCW is currently a
   small correction because the outcome model is correctly specified. The case
   where it earns its keep is the one not yet simulated.
3. **Train the GRU baseline** and compare against the handcrafted encoder, now
   that trajectories have three stages and irregular timing to learn from.
4. **A versioned clinical knowledge base** keyed to `arms.py`, replacing the
   sample contraindication rules and the five hard-coded RAG passages.
5. **FastAPI service and clinician dashboard**, once 1–4 make the numbers worth
   serving.

## Research proposal

`docs/research_proposal.md` covers objectives, significance, architecture,
technical approach, and agent functions.
