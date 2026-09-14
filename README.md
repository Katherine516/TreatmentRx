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
PYTHONPATH=src python3 -m treatmentrx.cli power
```

```bash
PYTHONPATH=src python3 -m treatmentrx.cli subgroups
```

```bash
PYTHONPATH=src python3 -m treatmentrx.cli transfer
```

```bash
PYTHONPATH=src python3 -m treatmentrx.cli serve
```

The service exposes `GET /capabilities`. It currently lists only
`rheumatoid_arthritis`; an unregistered diagnosis receives a typed 422 response
and is never scored by the RA model as a fallback.

It also exposes `GET /biomarkers`, `POST /trial/precision`, and
`POST /trial/power`. These are deliberately separate from `/recommend`:
randomized-trial precision adjustment estimates a marginal trial effect and can
never publish an individualized treatment recommendation. No frozen biomarker
artifact is registered by default.

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

```python
from treatmentrx import TreatmentRxOrchestrator
from treatmentrx.demo_data import sample_ra_bundle

recommendation = TreatmentRxOrchestrator().run(sample_ra_bundle())
```

Pure Python 3.9, zero third-party dependencies.

**What to feed it:** [`docs/INPUT_DATA.md`](docs/INPUT_DATA.md) — the FHIR subset
the adapter reads, the observation codes with their units and plausible ranges,
which six of them actually reach the estimators, what gets a record rejected, and
the separate and much larger contract a *training* cohort has to satisfy.

Three things a deployment supplies rather than inherits, each a seam with a
measured default: the **endpoint** (what a stage's outcome is), the **units**
(checked and converted, not assumed), and the **propensity** (fitted from the
data, never read off a generator).

## Architecture

Six layers, one direction. The order is load-bearing: safety runs before any
language is generated, and feedback runs last and cannot change the answer.

```text
FHIR bundle
  ↓  treatmentrx.diseases      resolve an explicit disease definition (fail closed)
  ↓  treatmentrx.scientific    pin dtr_research mode + a versioned estimand
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
| **2 Estimation** | Four treatment-regime estimators over one arm menu, one scale, one training split; two of them serve | `estimation/` |
| **3 Decision** | Bayesian model averaging, four-type uncertainty, held-out calibration, competing-risk and belief adjustment, goal-conditioned thresholds, blip explainability, contrast testing | `decision/`, `estimation/inference.py` |
| **4 Safety** | Composite-action feasible set, hard clinical rules, delayed-toxicity detection | `safety/` |
| **5 Agent** | PHI-minimised context, episodic memory + guideline RAG, three explanation agents | `agent/` |
| **6 Feedback** | ITT/per-protocol/as-treated estimands, switching-aware OPE, validation ladder, override governance | `feedback/` |

The six-layer sequence is reusable across diseases, but its clinical contents are
not. `diseases.py` binds one disease to its contract, treatment vocabulary,
endpoint, DAG, model family, safety policy and knowledge base. Only RA is
registered in this build.

## Scientific modes and biomarkers

Three typed modes prevent one analysis from borrowing another's claims:

- `dtr_research` is the only mode allowed to enter the patient recommendation
  workflow.
- `randomized_trial` compares an unadjusted continuous-outcome analysis with a
  frozen-prognostic-score-adjusted analysis.
- `biomarker_research` describes outcome prediction and carries no treatment
  authority.

Every estimator emitted by the serving path carries the same estimand
fingerprint; model averaging fails if fingerprints disagree. Frozen biomarker
artifacts fail closed on endpoint, horizon, reference treatment, disease, site,
platform, missing inputs, or post-decision feature timestamps. A prognostic
artifact is prohibited from claiming treatment-effect-modifier authority.

## The estimation layer

Four estimators are fit once per process on the same seeded training split and
scored on the same held-out split. **Two of them serve patients**
(`training.SERVING_ENSEMBLE`); the other two are the endpoints of the
shared-versus-stage-specific axis, kept as comparators for `stability`,
`misspecification` and `coverage`. Why only two serve is the subject of
"Only estimators that estimate the same quantity may be averaged" below.

| Method | What it is | Serves? |
| --- | --- | --- |
| **Q-Pooled** | A shared blip level plus per-stage deviations, only the deviations penalized. `pooling_ridge` interpolates between the two rows below | yes |
| **dWOLS-Shared** | Doubly-robust weighted blip regression, each arm fit one-vs-reference | yes |
| Q-Shared + Penalized | Ridge-penalized Q-learning, blip parameters shared across every stage, fit by backward induction on pseudo-outcomes | comparator |
| Stage-Specific Q-learning | Same machinery, an independent blip per stage | comparator |

The two serving estimators are kept different on purpose: Q-learning is an
outcome-model method and dWOLS is doubly robust, so they fail in different ways.
Averaging them is only *informative* because of that — and only *valid* because
they estimate the same quantity, which is the condition the shared-blip fit
fails.

`Q-Shared` and `Stage-Specific` were never two models so much as two settings of
one shrinkage parameter, which is what `Q-Pooled` makes explicit: `psi_{j,a} =
psibar_a + delta_{j,a}`, with only the per-stage deviations penalized, so
`pooling_ridge → ∞` recovers the shared fit and `→ 0` the stage-specific one. The
interior beats both ends — terminal parameter error 0.037 against 0.047 and
0.098, and lower total blip error than either at every curvature.

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

Because the winning arm is selected from six candidates, the serving decision
does not treat that post-selection top-two interval as an ordinary pointwise
95% interval. It applies a Bonferroni family-wise correction over all 15
unordered arm pairs; the same simultaneous threshold governs both the action
decision and candidate-set exclusions.

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

The study sweeps a six-patient grid spanning both effect modifiers and the
disease-activity range, at n=280 — the size the deployed models are fit on — and
contrasts each patient on their own true top-two arms, which is the quantity
Layer 3 reports. It used to run at a single patient, and that patient turned out
to be the best case for two of the three methods.

| interval | pooled | demo patient | worst patient | SE / spread | bias |
| --- | --- | --- | --- | --- | --- |
| sandwich, stage-specific | 91% | 92% | 85% | 0.89 | −0.002 |
| sandwich, shared blip | 28% | 74% | 0% | 0.38 | **+0.083** |
| **decision rule, correlation bound** | **95%** | — | 93% | 1.04 | 0.000 |
| decision rule, joint bootstrap (200 reps) | 93% | — | 85% | 0.94 | +0.001 |

The stage-specific sandwich is the honest one, and its remaining shortfall is
width: it reports about 90% of the estimator's actual spread, an independent
confirmation of the ~1.2 ratio the bootstrap comparison found by a completely
different route.

The shared blip's 29% is not a defect. Its estimate lands inside the span of the
true contrast across the decision points a shared psi pools over, for five of six
patients: it targets a *value-to-go* contrast, and a hepatotoxic arm's delayed ALT
cost is 0.165 per following stage — several times larger than most single-visit
contrasts in the grid. Scored against the single-visit truth, what you measure is
the estimand gap.

The decision rule is the open defect, and the reason the two rows below the fold
matter. Its intervals are wide enough everywhere (SE/spread 1.12–1.52 per patient)
and coverage still falls to 45%, because the miss is *centring*: it averages one
estimator targeting value-to-go with two targeting the single visit. Measuring the
estimators' covariance instead of bounding it — which looked like a free win at
the single patient — left the bias untouched and dropped coverage to 53%. The
conservative bound was the only thing holding coverage up, which is not a good
reason to keep it and an excellent reason not to remove it first.

### Is the bound too conservative? No, and here is the sweep that settles it

With the centring fixed, the correlation bound covered 98% at SE/spread 1.27 —
conservative, and the obvious question was whether measuring the covariance
instead would buy back interval width without losing coverage. The honest test is
to vary the one thing a percentile interval is most sensitive to:

| joint bootstrap | pooled | worst patient | SE/spread | width |
| --- | --- | --- | --- | --- |
| 25 replicates | 87% | 75% | 0.96 | 0.0607 |
| 50 replicates | 91% | 80% | 0.96 | 0.0638 |
| 100 replicates | 92% | 80% | 0.95 | 0.0653 |
| 200 replicates | 93% | 85% | 0.94 | 0.0662 |
| *correlation bound, before the covariance fix* | *98%* | *96%* | *1.27* | *0.0759* |
| **correlation bound, deployed** | **95%** | **93%** | **1.04** | **0.0663** |

Coverage climbs with the draw count **and the interval widens as it does** —
the signature of a percentile estimate stabilising, not of the method changing.
SE/spread sits at ~0.95 throughout, so the standard error was always honest; a
2.5th quantile from 25 draws is essentially the minimum, and that was the whole
shortfall.

At 200 replicates it reaches 93% pooled, within Monte Carlo error of nominal, but
**85% at the worst patient against the bound's 96%**, for only 1.15x the
narrowness. Trading 11 points of worst-patient coverage for 15% of width was
already the wrong direction for a clinical output.

**Keeping the dWOLS cross-arm covariance then closed the question outright.** The
bound's width fell from 0.0759 to 0.0663 against the bootstrap's 0.0662 — the two
intervals are now the same width to three decimals — while the bound covers 95%
pooled and 93% at its worst patient against the bootstrap's 93% and 85%. There is
no trade left to weigh: the bound is better on coverage and costs nothing on
width. It stays, on measurement rather than caution.

This study also set `DEFAULT_BLIP_RIDGE`. At 1.0 the penalty cost 0.014 of bias
on an effect of 0.088 and three points of coverage while buying 6% of variance;
at 0 the 78-parameter stage-specific fit destabilises at the sample sizes the
bootstrap resamples to. 0.25 is the best worst-case parameter error at n=88, 120
and 250 alike.

### Which stage is that 95% measured at?

```bash
PYTHONPATH=src python3 -m treatmentrx.cli coverage --stages
```

The patient grid exists because coverage at one covariate point is not coverage.
The stage index is a second axis with exactly the same property — it moves the
estimand and its standard error — and every study in `feedback/coverage.py`
pinned it at the terminal block. Nine call sites computing `n_stages - 1`. The
argument that justified sweeping the first axis was never applied to the second.

The terminal block is also the *most flattering* stage to measure at, and not by
accident: there is no future left, so a value-to-go blip and a single-visit blip
are the same number, and the two serving estimators target the same quantity
exactly. Swept, at n=280 over 40 refits of the six-patient grid:

| stage | coverage | worst | SE/spread | SE/within | bias spread | served? |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | **77.5%** | 37.5% | **0.64** | **1.13** | **0.0171** | no |
| 1 | 96.7% | 95.0% | 1.07 | 1.19 | 0.0055 | yes |
| terminal | 95.0% | 92.5% | 1.04 | 1.05 | 0.0024 | yes |

Stage 1 is at nominal. Stage 0 is not, and it is fitted, consumed by backward
induction, and **never used to score anyone**: `DataLayer.build_patient_state`
appends the pending visit, so `stage_index` is at least 1 for every patient the
pipeline sees. `coverage.SERVED_STAGE_INDICES` records that, a test checks the
pipeline still obeys it, and the row is reported rather than dropped, because what
keeps stage 0 out of reach is Layer 1's stage bookkeeping and not anything
statistical.

**Stage 0 fails on centring, not width — and the first version of this section
said the opposite.** `se_to_sd_ratio` is measured about each patient's *truth*,
so a bias that differs between patients lands in its denominator: 0.64 reads as
an interval a third too narrow. Measured about their own means the same estimates
give **1.13**, so that interval is slightly *wide*. The individual biases run
**−0.031 to +0.015** against a sampling spread near 0.013 and differ in sign, so
the pooled −0.009 cancels them away. `se_to_within_sd_ratio` and
`bias_dispersion` now separate the two, and a test asserts the width ratio stays
above 1 at every stage — because widening cannot repair a centre, and the two
defects want opposite fixes.

The mechanism is the one that motivated *Only estimators that estimate the same
quantity may be averaged*, recurring at a stage nobody swept. The ensemble
averages dWOLS's single-visit blip with Q-Pooled's value-to-go over the remaining
horizon; those coincide **exactly** at a terminal block and nowhere else, so the
bias is about half their gap — predicted −0.029 for the two seronegative patients
against −0.027 and −0.031 measured. The horizon rescaling shrinks the gap toward
the terminal block without closing it. Each member is nearly unbiased for its
*own* estimand at every stage, which is exactly the problem.

**Two scale mistakes in one study, in opposite directions.** Scored against the
*undivided* value-to-go the sweep reads 0% at stage 0 and 13% at stage 1 — pure
arithmetic, since `sandwich_contrast` divides by the remaining horizon and the
truth has to carry the same division. That one made the defect look
catastrophic. Then reading `se_to_sd_ratio` as a width ratio made it look like
the wrong *kind* of defect. Measure the denominator before believing the
quotient.

### The interval has to describe the decision

Layer 3 decides on the model-averaged Q-values, so the interval has to be for the
*averaged* contrast. It previously reported the widest of the three estimators'
intervals, which was incoherent twice over — the difference came from one model
while the decision came from the ensemble, and which model supplied it moved with
the data.

Measured end to end, that rule covered **78%** against a nominal 95%, worse than
any of its own components. Centring on the averaged contrast and bounding its
variance by the weighted sum of the component standard errors removed the
selection variability (empirical spread 0.029 → 0.013).

### Only estimators that estimate the same quantity may be averaged

Centring on the average was necessary and not sufficient. Swept across the patient
grid the averaged interval covered **74%**, and the miss was not width — its SE ran
1.12–1.52x the actual spread everywhere. It was the centre.

A shared blip is one ψ standing in for three stages of delayed effect, so its
terminal contrast is a stage-pooled compromise. Measured against the true
value-to-go contrast it runs −0.075 at stage 0, +0.009 at stage 1 and **+0.067 at
the terminal stage**, where 91% of the patients the pipeline is asked about sit
(a property of the fixture — `simulated_bundles` exports whole trajectories, so a
patient always presents at their last visit) and where the true contrasts range
0.008–0.088. Averaging it with two stage-resolved estimators put the ensemble
between two parameters. No weighting scheme repairs that; model averaging assumes
the members estimate the same thing.

Dropping it from the serving ensemble took coverage **74% → 97%** pooled and
**45% → 95%** at the worst patient, with the width essentially unchanged
(0.081 → 0.077), and cost nothing on the decision: true rollout value 2.1467 →
2.1473, oracle-arm agreement 0.888 → 0.900. Replacing the stage-specific member
with the partially pooled one took it the rest of the way, to **98% pooled and
96% at the worst patient**.

Every figure in the two paragraphs above prices a change against the interval of
the day, and the interval has since changed: keeping the dWOLS cross-arm
covariance removed 13% of the width, so the deployed ensemble now reads **95.0%
pooled, 93% at the worst patient, SE/spread 1.04**. The comparisons are what
justified the two swaps and are left as measured; the absolute figure to quote is
the one in the coverage table.

The visible consequence is more abstention — equipoise went from 41 to **85 of
120** audit patients, by way of 77 when the biased contrast was dropped, 99 once
the all-pairs correction went in, and back down as the dWOLS cross-arm covariance
removed width that was an error rather than a margin. The old contrast was biased
*away* from zero, so it manufactured separation. Patients the agent now declines
to separate have a true top-two gap of 0.026 against 0.081 for those it
recommends; the old split was 0.019 against 0.054. It recommends about half as
often and is right about it more often.

### Ignoring a covariance you can compute is not conservatism

dWOLS fits every arm one-vs-reference, so any two arms share every reference-arm
row and move together — measured over 60 refits the correlation between two arms'
blip estimates runs **+0.20 to +0.51**. The contrast standard error added their
variances as if independent, and ran **1.24×** the estimator's actual sampling
spread. With the cross term it runs **0.95×**.

`ArmFit.cross_covariance` is the usual M-estimator sandwich with a cross meat
term, `A_a⁻¹ (Σᵢ s_a,i s_b,i′) A_b⁻¹`, summed over the clusters present in *both*
fits. A patient who received only one of the two arms scores zero in the other
fit, so the whole cross term comes from the shared reference rows — which is
exactly the mechanism that correlates them. It carries the same small-cluster
correction `sandwich_covariance` applies, as the geometric mean
`√(scale_a · scale_b)`; without that it fails to reduce to the variance when the
two fits coincide, and `Var(x − x)` came out at a small positive residual instead
of zero. A test pins that identity, and it is what caught the omission.

The effect is system-wide, and all of it is recovered width rather than loosened
standards:

| | before | after |
| --- | --- | --- |
| decision-rule coverage | 98% | **95.0%** (nominal) |
| SE / actual spread | 1.27 | **1.04** |
| abstention | 78% | **67%** |
| transfer-site coverage | 97–100% | **95–97%** |

An interval that is too wide for a reason you can remove is not erring on the
safe side; it is declining to answer questions the data can answer. Here it fed
straight into the abstention rate, which is a clinical output.

### A per-decision value is not the regime's value

`evaluate_policy` walks stage-rows independently: it keeps the rows where the
policy agreed with the arm actually given, and weights by *that stage's*
propensity. For a dynamic regime that is a **per-decision** quantity. A patient
who deviated at stage 1 still contributes their stage-2 row — from a history the
regime would never have produced — and a single-stage probability is not the
cumulative product the sequential estimand requires.

`sequential_policy_value` computes the regime's own value. Reporting both is the
point; the denominators differ by a factor of three:

| | value | 95% interval | matched rows | ESS | max weight |
| --- | --- | --- | --- | --- | --- |
| per-decision (`ipw_policy_value`) | 0.7405 | — | 88 | **75.5** | 14.7 |
| sequential (the regime) | 0.8263 | 0.765 – 0.867 | 44 | **14.6** | 94.3 |

Three facts a reader needs. The gap (**0.086**) is larger than the whole claimed
gain over the behaviour policy (0.066). The sequential effective sample falls
**below `MIN_OPE_EFFECTIVE_SAMPLE`**, so by this repo's own standard the regime's
value is *not identified* at n=280 — `identified` is a field, the scorecard says
so in a note rather than leaving it to be inferred from a small number, and the
SILENT rung of the validation ladder now blocks on it. And only **3 of 120**
holdout trajectories follow the regime to the end (34, then 7, then 3).

That is not a defect in the estimator. It is what a three-stage regime costs in an
observational cohort this size, and it was invisible because nothing computed it.
`ipw_policy_value` still drives the model-averaging weights and `best_score()`,
which is defensible for a per-decision comparison against a per-decision
behaviour value — but it must never be described as the value of the regime.

#### Augmenting it: the doubly-robust version

Five contributing trajectories is what pure inverse weighting costs over three
decisions with six arms — and it is fixable, because the fitted Q-functions the
agent already serves from can supply the value wherever a trajectory leaves the
regime's path. `sequential_dr_value` is the standard doubly-robust backward
recursion (Murphy 2001; Bang & Robins 2005; Jiang & Li 2016):

```text
V_{T+1} = 0
V_t     = Q(x_t, d(x_t)) + 1{a_t = d(x_t)} / π_t · (y_t + V_{t+1} − Q(x_t, a_t))
```

Where the observed arm matches the regime, the residual is inverse-weighted in;
where it does not, the indicator is zero and the trajectory contributes the
model's own `Q(x_t, d(x_t))`. Every trajectory contributes, and the cumulative
propensity product never appears as an explicit weight. Against a known truth of
**2.2938**:

| estimator | value | SE | 95% interval | trajectories | ESS |
| --- | --- | --- | --- | --- | --- |
| **sequential DR (AIPW)** | 2.3278 | 0.0673 | (2.196, 2.460) | **120** | **120** |
| sequential IPW (Hajek) | 2.2864 | 0.2345 | (1.713, 2.542) | **5** | **3.9** |

Both cover it. The DR interval is **3.1× narrower**, its heaviest trajectory
carries **1.5–1.7%** of the estimate against the IPW estimate's **16.7%** (the
same quantity, measured the same way — uniform over 120 trajectories would be
0.83%), and both serving estimators' DR values land within **0.07 and 0.52
standard errors** of their own oracle.

**Which truth, and this is the trap.** The augmenting Q-model is IPCW-weighted,
so it targets the value-to-go *had the patient stayed in care* — 2.2938.
`oracle_rollout_value`, the figure printed beside every other policy value, is
what a patient accrues once dropout is simulated: **2.1469**. The 0.147 between
them is retention, not error, and scoring the DR estimate against the smaller one
would charge it for a gap it is not estimating.
`training.oracle_uncensored_value` exists so the comparison is made against the
right one.

**It does not retire the IPW gate.** The DR estimate says the regime's value *is*
identified — under the outcome model. The IPW estimate is the only thing here
that could falsify that model, and at an effective sample of 14.6 it has no power
to. A doubly-robust estimate whose inverse-weighted check cannot fail is a
model-based estimate wearing a robustness label. So the validation ladder's
blocker now reads "not identified *without leaning on the outcome model*" and
names the DR value alongside, instead of reporting only what is unknown.

#### How wrong would the outcome model have to be?

A caveat a reader cannot act on is half a finding. `cli evaluate` reports
`outcome_model_sensitivity`: scale every estimated blip by `1 + γ`, leave the
treatment-free surface alone, hold the evaluated regime **fixed**, and ask when
the claimed advantage over the behaviour policy stops excluding zero.

| γ | DR value | gain over behaviour | separated? |
| --- | --- | --- | --- |
| +0.00 | 2.2742 | +0.3686 | yes |
| +0.25 | 2.1951 | +0.2895 | yes |
| +0.40 | 2.1476 | +0.2420 | yes |
| **+0.434** | — | — | **tipping point** |
| +0.50 | 2.1159 | +0.2103 | no |
| +1.00 | 1.9575 | +0.0519 | no |

**The claim survives the model over-stating every treatment effect by 43%.** The
tipping point is bisected, not read off that grid, so adding a row for legibility
cannot move it.

**Where it lands is the point.** A 50% proportional blip error is the same
magnitude as the estimand shift in the transfer table's 1.5× row — the row where
calibration rises to 0.051 against 0.002–0.006 everywhere else *while policy
value gets better*. The misspecification that would overturn this claim is one
the deployment monitor already detects, and it detects it through calibration
rather than value. That is the estimand-shift finding arrived at from the other
end, and it is the argument for which monitor to watch.

Two caveats. The transfer row shifts the *truth* while the model stays put; γ
shifts the *model* while the truth stays put — both a 1.5× mismatch, but not the
same direction. And the benchmark (1.9056) is a simulation rollout on the
value-to-go scale; the per-decision behaviour value (0.674) is a different
quantity, and the 2.83 between them is a horizon, not an improvement.

**Every IPW number carries its weights.** `WeightDiagnostics` reports the maximum
weight, the mean, the share of mass in the heaviest row, and the share of rows
pinned to the propensity floor. A value of 0.74 at ESS 75 looks the same whether
the weights are flat or whether three rows carry a third of the mass, and the two
estimators differ exactly there: per-decision efficiency 0.86 with 3% of mass in
its heaviest row and positivity fine; sequential efficiency **0.33**, **16.7%** of
the mass in one row, 9% of rows at the floor, `positivity_ok` **false**.

### How much data buys how much confidence

```bash
PYTHONPATH=src python3 -m treatmentrx.cli power
```

The obvious next question — is abstaining on most patients a property of the
method or of the cohort? — has an answer. The same 240 patients, the ensemble
refit at each training size:

| train n | abstains | mean contrast SE | mean \|contrast\| | mean z |
| --- | --- | --- | --- | --- |
| 98 | 80% | 0.0261 | 0.045 | 1.73 |
| 196 | 86% | 0.0200 | 0.037 | 1.87 |
| **280 (deployed)** | **67%** | 0.0174 | 0.040 | 2.29 |
| 560 | 40% | 0.0124 | 0.044 | 3.54 |
| 1120 | 33% | 0.0092 | 0.045 | 4.86 |

The contrast itself is flat across the sweep — the effect is not changing, the
precision is. The standard error shrinks at **n^-0.43** against the n^-0.50 a
correctly specified estimator earns. With simultaneous all-pairs inference, the
current extrapolated estimate is about **1,490 training trajectories** to bring
abstention to 30%.

The n=196 row sitting *above* n=98 is not a reversal of the trend — it is the
same Monte Carlo noise the rest of the curve is fit through, on one draw per
cohort size, and the shrinkage exponent is estimated from the standard errors
rather than from the abstention rates for exactly that reason.

This study is also what caught a defect that had been invisible: `dwols.py` kept
its own module-level model cache, so `training.reset()` refit the Q-learning half
of the serving ensemble and left the dWOLS half at whatever cohort it first saw.
The two agreed on the default cohort, so nothing ever failed — but with half the
ensemble frozen the standard error appeared to shrink at n^-0.15, which reads
like a floor in the method rather than a stale object. One owner for every fit,
and the exponent is a regression test.

It also used to run on **60** scored patients, which was too small for a stable
headline abstention estimate. The reference evaluation now uses 240 patients and
must be regenerated whenever the decision rule or multiplicity correction
changes.

### Who does it abstain on?

```bash
PYTHONPATH=src python3 -m treatmentrx.cli subgroups
```

A pooled rate cannot see a subgroup, and this one hides a **58-point spread**.
Two things drive abstention and they mean opposite things: the arms are genuinely
close (*signal* — declining is correct), or the standard error is large
(*precision* — declining is a statement about the training data). Stratifying on
the non-intercept terms of `BLIP_BASIS`, the covariates the true effect actually
varies over, separates them:

| stratum | n | abstains | true gap | contrast SE | at pooled SE |
| --- | --- | --- | --- | --- | --- |
| anti-CCP negative | 90 | **97%** | 0.034 | 0.0179 | 97% |
| anti-CCP positive | 150 | 53% | 0.051 | 0.0169 | 74% |
| TNF naive | 158 | 76% | 0.038 | 0.0166 | 98% |
| prior TNF exposure | 82 | 57% | 0.056 | 0.0186 | 52% |
| das28 < 3.85 | 80 | 74% | 0.045 | 0.0167 | 78% |
| das28 3.85–5.89 | 80 | **39%** | 0.056 | 0.0155 | 78% |
| das28 ≥ 5.89 | 80 | **96%** | 0.032 | 0.0195 | 93% |

The last column holds each patient's own contrast and substitutes the pooled
standard error, so the gap between it and the abstention rate is the part that is
precision rather than signal.

**Abstention is earned in every stratum.** Within each cell, the patients
declined have closer arms than the patients recommended — `tests/test_subgroups.py`
asserts that cell by cell rather than pooled, because pooled can be true while a
subgroup has it backwards. The most-abstaining stratum on each axis is the one
with the smallest true gap.

**Precision is not evenly distributed.** In this reference run, low-activity and
prior-TNF strata carry about 2.5 and 2.4 percentage points of excess abstention
relative to pooled precision. Negative values in other strata mean their own
standard errors are smaller than the pooled substitution; they are not evidence
that uncertainty creates negative abstention.

**Seronegative patients are the case a deployment has to be told about.** 97%
abstention and *none* of it is precision. Rituximab's blip carries `+0.14 ×
anti_ccp`, so removing seropositivity removes the main thing separating the arms.
The agent is close to silent for 37% of this population and it is right to be.

This is a clinical-stratum analysis and deliberately **not** a fairness audit:
the synthetic cohort has no age, gender, steroid or comorbidity effect, so there
are no protected attributes to slice on. `ValidationLadder`'s `fairness_clean`
criterion remains unmet and unmeasured, and this does not claim to satisfy it.

The result errs wide rather than narrow, which is the safe direction — and it
must then be exempt from the sandwich-inflation guard, or the same correction is
charged twice.

That bound assumes the estimators are perfectly correlated. They are strongly
correlated but not perfectly, so in principle it costs interval width for
nothing. Refitting every member on the *same* resamples measures the covariance
instead of bounding it:

```python
from treatmentrx.estimation import training
training.enable_joint_inference()      # one refit per estimator per replicate
```

**It is off, and the sweep above is why.** An earlier reading here had the joint
bootstrap at 1.05 SE/spread and 0.056 width against the bound's 1.13 and 0.077 —
apparently calibrated and a third narrower, recovering six recommendations per
120 patients. Both halves of that were wrong for the same reason: the study was
running on the collapsed ensemble described under *Only estimators that estimate
the same quantity may be averaged*, so it was measuring dWOLS alone, whose
smaller parameter space gives a correspondingly narrower interval. Corrected, the
bootstrap is 1.15x narrower, not 1.43x — and since the dWOLS cross-arm covariance
has been kept the bound is the same width as the bootstrap while covering eight
points better at the worst patient.

Opt-in, because it costs a refit of each estimator per replicate, and because
nothing is left to buy. Without it the decision layer uses the bound, which is
now the better of the two on every axis measured.

### The assumption everything else rests on

```bash
PYTHONPATH=src python3 -m treatmentrx.cli misspecification --omitted-modifier
```

Every result above is conditional on the **blip basis** being right — on the true
effect modifiers being `anti_ccp` and `prior_tnf` and nothing else. `curvature`
tests a wrong *nuisance* surface, which is the failure double robustness exists
to survive. This tests a wrong *estimand*: the true effect also varies over
standardised CRP, which is in the treatment-free basis and deliberately not in
the blip basis, so an estimator can adjust for it as a confounder and still be
unable to represent it as an effect modifier.

| blip modifier | pooled coverage | worst patient | bias | max patient bias | SE / spread |
| --- | --- | --- | --- | --- | --- |
| 0 (correct) | 100% | 100% | +0.003 | 0.006 | 1.14 |
| 0.05 | 48% | 0% | −0.035 | 0.087 | 0.44 |
| 0.10 | 35% | 0% | −0.058 | 0.138 | 0.35 |
| 0.20 | 29% | 0% | −0.125 | 0.279 | 0.35 |

Coverage collapses and **SE/spread falls with it** — the interval does not widen
to absorb the damage, because a standard error computed under the wrong basis has
no way to see a bias in the estimand. There is no estimator-side defence and no
amount of data helps; what the ensemble converges to is the CRP-averaged
contrast, which is simply a different quantity from the patient's own.

This is the failure mode a real cohort will have, and it is the honest boundary
on everything else in this README. The defence is getting the basis right, which
is a clinical question, not a statistical one.

The study above compares against `true_blip` and so exists only in simulation.
The part that transfers is the specification test, which augments the blip block
with a named candidate and tests whether its coefficient is zero — a question an
analyst can ask of their own data. **It runs on every fit** (0.18s), and a flag
blocks the validation ladder, raises an uncertainty flag on every patient, and
puts a `CAVEAT` on the clinician card under the separation line it undercuts. It
deliberately does not change the recommendation: the fix is to add the covariate
and refit, and a diagnostic that silently alters behaviour is worse than a loud
one. `cli specification` shows the detail. Its error rates are measured: **0/30** false
positives on null cohorts, and 10/10 detection once the modifier reaches 0.05.

Getting there took two corrections that only measurement would have found. The
first version rejected on **33%** of null cohorts, because it corrected
multiplicity over arms while running covariates-times-arms tests and took the
sandwich at face value. And pointing it at ALT rejects on a cohort with no ALT
effect modification at all — ALT is *caused* by the previous arm (mean 62.2 after
a hepatotoxic one against 27.9 otherwise), so the test measures mediation rather
than modification. Candidates have to be pre-treatment, and `EXCLUDED_CANDIDATES`
records the one that is not.

### The near-flag, and why adding the term would be wrong

```bash
PYTHONPATH=src python3 -m treatmentrx.cli misspecification --extra-modifier
```

`das28_squared` tests at **max |z| = 3.367 against a 3.669 threshold** — 92% of
the way to firing, on the same covariate where `cli subgroups` finds 96%
abstention and the worst standard error in the cohort. That is not a coincidence,
and the mechanism is not noise: the estimators target the blip, which is linear in
`das28_std` by construction, but the quantity the *decision* uses is the
value-to-go contrast, and backward induction composes the blip with a `max` over
arms — curved even when the blip is not.

So the term was measured rather than argued about, at n=240 over three seeds:

| quantity | change |
| --- | --- |
| contrast interval width | **+13%** |
| pooled abstention | **+2.9 points** |
| top das28 tertile abstention | **−6.2 points**, on every seed |
| bottom das28 tertile abstention | **+13.7 points** |
| error on the four true parameters | −0.013 of 0.190, across twenty coefficients |

It does help the stratum the near-flag points at, consistently. It buys those six
points by paying fourteen in the bottom tertile and three pooled, and by widening
every interval in the system by 13% — while recovery of the parameters that
actually exist barely moves, so the gain is not better estimation. **The four-term
basis stays.**

This is the counterexample to the obvious misreading of `--omitted-modifier`: if
omitting a real modifier is unrecoverable, add every candidate. It is not free,
and this is the price.

### Which estimator, when policy value cannot decide?

`stability` reports that held-out policy value cannot separate them. That is
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

### What abstention actually costs

`cli audit` used to report the decision layer as flawless — oracle-arm rate 1.0,
max regret 0.0. It wasn't. `recommended_arm` is `None` unless the agent commits,
so scoring only those rows had quietly changed the question from *how good is the
ranking* to *when it commits, is it right* — whose answer is trivially yes,
because it commits only when the gap is large. The denominator had gone 120 → 43
without a word, and the guarding assertion (`oracle_arm_rate > 0.7`) waved the
perfect score straight through.

Two questions, two names, both carrying `patients`:

| | n | oracle-arm rate | mean regret | max regret |
| --- | --- | --- | --- | --- |
| `when_it_commits` | 35 | 1.000 | 0.0000 | 0.0000 |
| `if_forced_to_commit` | 120 | **0.9083** | **0.0011** | **0.0406** |

The second row is the interesting one, and it reframes the abstention story: the
agent declines on 71% of these patients, and its top-scored arm would have been
the oracle-optimal one **90.8%** of the time anyway. Its *ranking* is considerably
better than its own (deliberately conservative) intervals let it claim.

That does not argue for abstaining less — the intervals genuinely include zero.
It argues for saying what declining costs, which now depends on what the
clinician does next, over the 85 declined patients:

| the clinician takes | mean regret | max |
| --- | --- | --- |
| the model's own top arm | 0.0015 | 0.0406 |
| the worst arm in the candidate set | 0.0474 | 0.2142 |
| the worst arm on the whole menu | 0.2058 | 0.4093 |

Which is the retrospective case for the section above: handing back a bare
`equipoise` status, with nothing to choose from, was in regret terms roughly **4.3×
worse** than handing back the set. That ratio was 6× before the all-pairs
correction widened the set — a wider set is safer to be inside and less decisive
to choose within, and both halves of that show up here.

### What the agent says when it will not recommend

The most consequential measured fact about this agent is that it declines to name
one arm for **most** patients it sees — 67% at the training population, 57–89%
across the sites above, 97% for seronegative patients. That abstention is earned
(`cli subgroups` checks it stratum by stratum). But until recently it produced a
status and a paragraph, and the clinician, who still has to prescribe something,
got nothing to prescribe with.

`Decision.candidate_arms` is the leader plus every arm whose model-averaged
interval fails to exclude zero — the same `robustly_distinguishable` rule the
separation line reports, asked of every arm instead of only the runner-up:

```text
Cannot separate: IL-6 inhibitor, rituximab, methotrexate-optimization, JAK-inhibitor.
The data does not distinguish these from each other at this sample size; the other
2 arms on the menu were ruled out. This is not a recommendation — it is the
narrowest defensible set, and the choice within it should turn on tolerability,
route, comorbidity and patient preference rather than on these numbers.
```

| | |
| --- | --- |
| mean set among declined patients | **2.7 of 6 arms** |
| contains the truly optimal arm (40 refits × 6 patients) | **240/240**, mean size 1.97 |
| worst-case regret, whole menu → candidate set | **0.206 → 0.047 (−77%)** |
| same, replicated on the coverage grid | **−88%** |

**The level is simultaneous.** The leader and comparator are chosen from the very
estimates used for inference, so a pointwise 95% interval does not account for
that search. `inference.simultaneous_alpha` applies a Bonferroni family-wise
alpha over all 15 unordered pairs of six arms, which is what makes this an
all-pairs confidence set. That correction took abstention from 58% to 78%;
keeping the dWOLS cross-arm covariance then brought it back to 67%, by removing
width that was an error rather than a margin.

It has one definition on purpose. The coverage study briefly hard-coded 0.05
while the decision layer emitted at 0.05/15, and reported a mean set of 1.67 with
93% regret reduction for a rule nobody deployed. The numbers stayed entirely
plausible; they just described a different object.

```bash
PYTHONPATH=src python3 -m treatmentrx.cli coverage --candidate-set
```

The containment figure is replicated over *refits*, not read off the deployed fit
— on one fit every patient's set shares the same parameters. And 240/240 is not a
guarantee: at that draw count the miss rate is bounded near 1%. Read containment
carefully — it is a different property from the interval's nominal 95% and has a
naturally higher rate, because the set always holds the leader and the leader is
the true best arm about 91% of the time. Comparing the two directly was the error
in an earlier version of this paragraph, which also leaned on the interval
running 1.27× the actual spread; it runs 1.04 now.

**This is not a way to recommend more often.** The action bar is untouched, the
status is unchanged, and a test pins the abstention rate so a later change cannot
quietly collapse the set toward a single arm and call it an improvement. An arm
inside the set has not been recommended.

It also caught a contradiction on the card. `WHY_NOT_REASONS` is hard-coded
clinical prose hung on a model-derived Q-gap, and it was printed for the top two
runners-up regardless of whether the model could separate them — so the card said
*"cannot separate these four"* and then explained why two of them were wrong.
It now reports only arms outside the set.

### When a contraindication lands on an arm nobody recommended

`SafetyLayer` blocked whenever the top-scored arm was infeasible — reading that
field regardless of whether Layer 3 had actually recommended it. On an equipoise
decision it is only the argmax, and the layer has already said it cannot be told
apart from the rest of the candidate set. There was no recommendation for the
contraindication to strike down.

Measured over injected contraindications, **12 of 34 blocked cases were that**:
the patient was going to be told "these arms cannot be separated", one turned out
to be contraindicated, and the case escalated as though a recommendation had been
refused. Those now route to **review** carrying the survivors:

```text
MANUAL REVIEW — no arm was separated well enough to recommend, and
methotrexate-optimization, which scored highest, is not feasible for this
patient. Nothing has been substituted in its place.

Only rituximab remains in contention. The data could not separate it from 1
other arm that the safety layer then removed, so it stands by elimination
rather than by evidence — it has not been recommended.

Removed from this set by the safety layer: methotrexate-optimization
(methotrexate/leflunomide contraindicated in pregnancy).
```

**Everything else still blocks, and that is the point.** A *recommended* arm
being infeasible blocks; an empty surviving set blocks. `BLOCKED` is the only
status that stops, and the bug the no-substitution rule exists for — a naming
mismatch silently removing an arm — must still halt rather than produce a tidy
list of alternatives. `recommended_arm` stays `None` through the review path, and
a test asserts it end to end: a name appearing there means something was
promoted, whatever the status says.

The allergy rule has the same argmax defect and was deliberately left alone. A
recorded allergy is the strongest contraindication here and blocking is its
fail-safe direction; the arm is removed from the feasible set either way, so no
patient is exposed. Moving two safety paths in one change is how a regression
gets in. That boundary is pinned by a test.

`cli audit` reports the four branches under `contraindication_routing`.

### Does any of it survive data it was not fit on?

```bash
PYTHONPATH=src python3 -m treatmentrx.cli transfer
```

Every other number here is measured on the process the estimators were built
against, and their blip basis *is* that process's basis. So: fit at site A, score
at site B. The shift is confined to things that leave the estimand alone — case
mix, prescribing habits, retention — because a bent estimand is already measured
by `--omitted-modifier` and nothing survives it. What is open is whether a
*correctly specified* pipeline still works when the population and the practice
change, which is the shift a real deployment meets.

| site | abstains | coverage | SE/spread | ECE | IPW | local practice | rollout gain |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline (same process) | 68% | 97% | 1.11 | 0.004 | 0.734 | 0.639 | +0.386 |
| sicker, more seronegative | **89%** | 95% | 1.04 | 0.004 | 0.600 | 0.527 | +0.372 |
| milder, mostly seropositive | **57%** | 96% | 1.03 | 0.003 | 0.846 | 0.753 | +0.437 |
| TNF-first, rituximab-averse | 65% | 95% | 1.04 | 0.004 | 0.726 | 0.647 | +0.386 |
| heavy attrition | 75% | 97% | 1.10 | 0.005 | 0.748 | 0.669 | +0.469 |
| near-complete follow-up | 74% | 95% | 0.99 | 0.006 | 0.720 | 0.621 | +0.395 |
| combined: all three | 84% | 95% | 1.08 | 0.002 | 0.617 | 0.555 | +0.346 |
| **estimand shifted: blips 1.5×** | 70% | — | — | **0.051** | 0.824 | 0.695 | +0.538 |

**Three of the four headline claims transfer.** The policy beats local practice
at 7/7 estimand-preserving sites (+0.346 to +0.469) — including the site where
clinicians already prescribe TNF-first. Coverage holds at 95–97% — at nominal
now that the dWOLS cross-arm covariance is kept rather than dropped, where it
used to sit at 97–100% on an inflated interval — calibration holds at
0.002–0.006, and Layer 1 rejected **zero** records anywhere.

**Abstention does not.** It runs 57% to 89% against the 68% on the model card,
and the swing is driven by case mix in the direction that matters: the sicker,
more seronegative site is where the agent goes nearly silent. A deployment that
reads 68% as a property of the tool will be wrong by tens of points.

**The last row is the one to read twice.** Scaling every blip by 1.5 leaves the
*ordering* of arms intact, so the recommendation stays right and the policy value
stays good — the best rollout gain in the table — while every reported Q-value is
wrong by half. Calibration catches it at 13× the baseline error. **Policy value
cannot detect an estimand shift; calibration can.** A monitor watching value
alone is blind to exactly the failure that makes the clinician card meaningless.

Two things this is not. It is not live data — `live_data` stays False and the
validation ladder does not move. And the coverage column is replicated over
site-A *refits*, not read off the deployed fit: every patient's interval shares
one fit's parameters, so a single-fit patient fraction is a property of the draw
you took. Measured that way the baseline read 87% against the 95% `cli coverage`
reports for the same rule — the entire gap was the method of measurement.

## Serving it

```bash
PYTHONPATH=src python3 -m treatmentrx.cli serve
```

Three routes on `http.server` — no framework, so `dependencies = []` survives:

```text
GET  /health      is the process up, are the models fitted
GET  /model       the model card
POST /recommend   a FHIR bundle in, a Recommendation out
```

The service decides nothing. `tests/test_service.py` asserts the served answer
equals the orchestrator's for the same bundle, because a second decision path is
the failure this repo's notes open with. What it does add is the boundary: a
record the contract rejects comes back **422 with the failing fields**, not 500
and not a silent default; malformed JSON is 400; no traceback ever reaches the
caller, because a stack trace quotes field values and field values are PHI; the
access log records method, path and status and never a body. It binds to
loopback and warns if told not to — an unauthenticated decision-support endpoint
on `0.0.0.0` is a different class of mistake from a wrong standard error.

`GET /model` is the route that matters. A consumer holding one `equipoise`
response cannot see that the agent abstains on ~67% of patients by
design, so the card says it, alongside the measured interval coverage, the four
confounders nothing adjusts for, that six covariates reach the estimators at all,
that there is no held-back final test partition, and that the selection between
the two serving estimators is a tie-break rather than a measurement
(`ranking_resolved: false`). The validation block is *asked*
of `ValidationLadder`, not asserted by the card:

```json
"validation": {
  "rung": "silent",
  "gate_passed": false,
  "blockers": [
    "the regime's own value is not identified: sequential OPE effective sample
     size 15 is below 30; 3 trajectories follow the regime to the end",
    "no live data: the held-out estimate comes from the same generating process
     the model was fit on, so it cannot establish stability on live data"
  ]
}
```

The first blocker is new and it matters more than it looks. The gate used to read
`ope_effective_sample_size` — the *per-decision* one, 75.5, comfortably over the
threshold — while what advances a rung is a three-stage regime whose own value is
estimated at an effective sample of 14.6. So the only thing holding SILENT shut
was `live_data`, which is structural: flip it, as a first real cohort would, and
the gate would have opened on a regime whose value is not identified. Both
effective samples are now reported and both are gated, and a test strips
`live_data` to check the statistical criteria can still close the gate on their
own.

**There is no held-back final test.** `training.evaluation_partition()` builds
the partition this build actually has and the card reports it: 280 training
trajectories, 120 evaluation, nothing else. That evaluation split does four jobs
— it sets the model-averaging weights, selects the serving estimator through
`best_score()`, supplies the calibration the validation ladder gates on, and is
the held-out policy value quoted on the card. Each is defensible alone; together
they mean nothing untouched remains to check the headline numbers against. The
Monte Carlo studies (`coverage`, `power`, `misspecification`) do draw fresh
cohorts, so the tuned constants are not fitted to this split — but fresh seeds do
not give back a test set. `EvaluationPartitionContract` used to be exported,
unit-tested and never constructed, which meant the package advertised a locked
final test that did not exist; `has_final_test` now says so.

**A three-way split is not the remedy at this cohort size**, and `cli power`
prices it rather than leaving it to intuition:

| held back | evaluation ESS | final-test ESS | identifies per-decision | identifies the regime |
| --- | --- | --- | --- | --- |
| none (today) | 73.0 / 14.0 | — | — | — |
| 25% | 52.7 / 10.5 | 20.4 / 5.4 | no | no |
| 40% | 42.9 / 8.2 | 30.3 / 6.2 | **yes** | no |
| 50% | 33.8 / 7.4 | 39.4 / 7.5 | **yes** | no |

The two quantities disagree, which is why both are reported. A final test that
identifies the *per-decision* value exists at 40–50% held back — bought by taking
the evaluation split from 73 to 43, and the per-decision value is not what the
agent deploys. For the **regime's own** value no split works: 5.4 to 7.5 against
a threshold of 30, so the confirmation would itself be unidentified. Keeping
today's evaluation precision *and* adding an identified final test needs roughly
**565** trajectories for the per-decision value and **1,258** for the regime's —
against 400 today, and alongside `cli power`'s 1,490 for 30% abstention. Two
different questions, the same answer about this cohort.

What it is not is production infrastructure: single-process, no TLS, no auth.
Those gaps are listed in `LIMITATIONS` and returned on `/health` rather than left
for a reader to infer.

## What is real vs. still a placeholder

**Real:** the four estimators, the synthetic cohort and its known blips,
informative-dropout handling and IPCW, held-out IPW policy evaluation — both
per-decision and sequential, each with a trajectory-clustered bootstrap interval
and its weight diagnostics — calibration, cluster-robust sandwich standard errors
including the dWOLS cross-arm covariance, m-out-of-n bootstrap intervals,
contrast tests, cross-validated stability, blip attributions, the
backward-induction oracle used for regret, the safety sweep, and the
specification test with measured false-positive rates.

**Deliberately simple, and labelled as such in-module:**
`HandcraftedFeatureEncoder` (nine clinical features normalised and tiled — not a
learned representation, and it does not claim to be),
`SemanticKnowledgeBase` (five hard-coded passages, not a real RAG index), `WHY_NOT_REASONS` (hard-coded clinical prose on a model-derived Q-gap),
and the E-value in the sensitivity report.

**Deleted rather than improved:** `SwitchingAwareOPE` used to publish an
`iptw_policy_value` — one patient's observed outcomes reweighted by `adherence ×
0.5^switched × 0.7^rescue`. Measuring the cohort ended it. Over 728 stage-rows,
`adherence` was identically 1.000 and rescue fired **zero** times, so two of the
three factors were dead and the whole weight collapsed to *double it if they
switched* — one invented constant, moving the published number by up to **0.112**
against outcomes near 0.64. And the estimand did not exist: IPTW recovers a
population policy value by correcting confounded assignment, so on a single
patient's three or four outcomes there is no counterfactual to recover. Fitting
the deviation model would have fixed the constant and left that untouched. It now
reports what it was standing in front of — the observed mean, the visit count, and
a plain tally of deviations — labelled descriptive, with no interval.

The value of this codebase is that a reader can tell the difference.

## Next build steps

Done: penalized Q-learning with backward induction; held-out IPW policy value;
sandwich standard errors and contrast-driven equipoise; cross-validated
stability; m-out-of-n bootstrap for the non-regular stages; a multi-stage cohort
with informative dropout and irregular visits, ingestible through Layer 1.

Also done: the visit-intensity model is fitted from the data (and measurably
does not help on this generating process, so it is off by default with the
numbers recorded beside the flag); the misspecified-nuisance arm of the
simulation; the joint bootstrap that measures the estimators' covariance instead
of bounding it; held-out policy values with bootstrap intervals, so an argmax
over three indistinguishable numbers is no longer mistaken for a ranking; a
backward-induction oracle to measure regret against; a coverage study that
sweeps a patient grid rather than one reference point; `Q-Pooled`, which turns
the shared-versus-stage-specific choice into a shrinkage parameter and beats both
endpoints; a power curve that prices abstention against cohort size; unit
validation with conversion; a fitted evaluation propensity replacing the
generator's; an explicit endpoint seam with EULAR response alongside the
free-text mapping; a specification test with measured false-positive rates that
runs on every fit; and an HTTP surface with a model card, on the standard
library, so `dependencies = []` still holds.

Performance, measured on the demo patient: a cold process pays **2.9s** to fit
the four estimators, fit the evaluation propensity, and bootstrap the held-out
values; each subsequent recommendation costs **~2ms**. The propensity fit is
about half of that cold cost and is what removes the generator from the
evaluation path. The oracle rollout benchmark is simulation-only
and computed on request, so it stays off the path of a process that just serves
a patient.

Next, in order:

1. **A versioned clinical knowledge base** keyed to `arms.py`, replacing the
   sample contraindication rules and the five hard-coded RAG passages. The
   composite action set is currently a literal in `estimation/actions.py`, and
   its breadth is load-bearing for whether a contraindication blocks a patient.
   `WHY_NOT_REASONS` belongs there too — it is the one place Layer 5 asserts
   something the model did not produce.
2. **A learned state representation, if one is ever wanted.** The previous entry
   here offered three options for `GRUBaselineEncoder` — train it, shrink it, or
   drop the tail — and the tail is dropped. Once the out-of-distribution term
   that read `vector[:32]` was removed as unfirable, nothing read the vector at
   all, and its only surviving use was its own name and length in an audit line:
   177 µs per request, 29% of Layer 1, for a label. `build_patient_state` now
   costs 561 µs against 723 µs. Training a real encoder is still a coherent
   project; keeping an untrained one warm against that day was not.
3. **A clinician-facing view of the card.** The numbers are served
   (`cli serve`); what is not built is a reading surface for them. The
   interesting question there is not layout — it is whether an interface can
   make `equipoise` on half of patients read as the measured statement it is
   rather than as the tool failing.

Not on this list, deliberately: **more estimators**, and **a bigger cohort to
make the abstention rate look better**. `cli stability` cannot separate the ones
already here, and `cli power` shows the abstention rate is a statement about
sample size, not about the method.

## Research proposal

`docs/research_proposal.md` covers objectives, significance, architecture,
technical approach, and agent functions.
