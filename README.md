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
PYTHONPATH=src python3 -m treatmentrx.cli audit
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

### Do the intervals actually cover?

Everything above about uncertainty rests on one claim nothing was testing: that a
nominal 95% interval contains the truth 95% of the time. A standard error can
shrink correctly with sqrt(n), be reported on the right scale, and still miss.

```bash
PYTHONPATH=src python3 -m treatmentrx.cli coverage --bootstrap
```

| interval | coverage | reported SE / actual spread | bias |
| --- | --- | --- | --- |
| sandwich, stage-specific | 88% | 0.88 | −0.008 |
| sandwich, shared blip | 79% | 0.86 | **+0.019** |
| m-out-of-n bootstrap | **93%** | 1.27 | −0.003 |

Only the bootstrap reaches nominal. The sandwich reports about 88% of the
estimator's actual spread — an independent confirmation of the ~1.2 ratio the
bootstrap comparison found by a completely different route. The shared blip loses
a further nine points to *bias*, not width: it targets a stage-averaged estimand
by design, so missing the single-visit truth is the price of parameter sharing,
not a defect.

This study also set `DEFAULT_BLIP_RIDGE`. At 1.0 the penalty cost 0.014 of bias
on an effect of 0.088 and three points of coverage while buying 6% of variance;
at 0 the 78-parameter stage-specific fit destabilises at the sample sizes the
bootstrap resamples to. 0.25 is the best worst-case parameter error at n=88, 120
and 250 alike.

### The interval has to describe the decision

Layer 3 decides on the model-averaged Q-values, so the interval has to be for the
*averaged* contrast. It previously reported the widest of the three estimators'
intervals, which was incoherent twice over — the difference came from one model
while the decision came from the ensemble, and which model supplied it moved with
the data.

Measured end to end, that rule covered **78%** against a nominal 95%, worse than
any of its own components. Centring on the averaged contrast and bounding its
variance by the weighted sum of the component standard errors reaches nominal:
averaging removes the selection variability (empirical spread 0.029 → 0.013) and
the components' opposing biases partly cancel (+0.012 → +0.005).

The result errs wide rather than narrow, which is the safe direction — and it
must then be exempt from the sandwich-inflation guard, or the same correction is
charged twice.

That bound assumes the estimators are perfectly correlated. They are strongly
correlated but not perfectly, so it costs about a quarter of the interval width
for nothing. Refitting all three on the *same* resamples measures the covariance
instead of bounding it:

```python
from treatmentrx.estimation import training
training.enable_joint_inference()      # one refit per estimator per replicate
```

Measured over 40 regenerated cohorts, both reach nominal — but one does it
without the slack:

| Rule | Coverage | reported SE / actual spread | Mean width |
| --- | --- | --- | --- |
| Correlation bound | 97.5% | 1.13 | 0.077 |
| **Joint bootstrap** | **92.5%** | **1.05** | **0.056** |

A ratio of 1.05 is essentially calibrated. Across 120 simulated patients the mean
interval narrows from 0.083 to 0.068 and **six more of them get a recommendation**
instead of being told the arms cannot be separated — not because the bar moved,
but because the interval stopped assuming a correlation of exactly one.

Opt-in, because it costs a refit of each estimator per replicate. Without it the
decision layer falls back to the bound — visibly wide rather than silently
wrong.

### Which estimator, when policy value cannot decide?

`stability` reports that held-out policy value cannot separate the three. That is
honest but leaves the choice unjustified, so there is a second axis: bend the
treatment-free surface past what their linear basis can represent, leaving the
blips untouched, and see which survives.

```bash
PYTHONPATH=src python3 -m treatmentrx.cli misspecification
```

| Estimator | Nuisance model right | Worst case | Degradation |
| --- | --- | --- | --- |
| dWOLS-Shared | **0.170** | 0.619 | 3.64× |
| Stage-Specific Q-learning | 0.285 | 0.626 | 2.19× |
| Q-Shared + Penalized | 0.578 | 0.943 | **1.63×** |

They trade off in exactly the direction their designs predict — dWOLS is doubly
robust and most accurate when the assumptions hold; the shared-blip fit is the
most stable when they do not. No estimator wins on both axes, which is the
justification for averaging them rather than picking one. Until now that was an
assertion.

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

Performance, measured on the demo patient: a cold process pays 0.74s to fit the
three estimators; each subsequent recommendation costs ~2.2ms. The oracle rollout
benchmark is simulation-only and computed on request, so it stays off the path of
a process that just serves a patient.

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
