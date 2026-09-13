# What data TreatmentRx needs

Two different questions, and they have very different answers:

* **[Inference](#1-inference-one-patient)** — what one patient's record must
  contain for the agent to produce a recommendation. Small, and fully specified
  by the FHIR adapter and the data contract.
* **[Training](#2-training-a-cohort)** — what a cohort must contain for the
  estimators to be *fit* on real data instead of the simulator. Considerably
  more, and three of the requirements are not satisfiable by a claims extract.

Everything below is what the code actually reads today, not an aspiration.
Section 3 lists what is missing.

---

## 1. Inference: one patient

### 1.1 Transport

A FHIR R4-style `Bundle` dictionary, passed to
`TreatmentRxOrchestrator.run(bundle)`. `data/fhir.py` reads a deliberate subset
and ignores everything else, so extra resources are harmless.

**Time is integer days from that patient's baseline, never a date.** The adapter
reads `effectiveDay`, `authoredOnDay`, `startDay`, `stopDay`, `day`, and
`period.startDay`. This is a prototype simplification: a real adapter has to
convert real timestamps to a per-patient day offset before this layer sees them,
and that conversion is where timezone and partial-date bugs will live.
`Patient.birthDate` is the one real date read, and nothing numeric uses it.

### 1.2 Resources read

| Resource | Cardinality | Fields read |
| --- | --- | --- |
| `Patient` | exactly 1, required | `id`, `gender`, `birthDate` |
| `Condition` | ≥ 1, required | `code.text` or `code.coding[0].display` |
| `MedicationRequest` / `MedicationStatement` | ≥ 1, required | `medicationCodeableConcept.text`, `authoredOnDay`\|`startDay`, `dose`, `stopDay`, `response`, `discontinuationReason` |
| `MedicationDispense` / `MedicationAdministration` | 0+ | `medicationCodeableConcept.text`, `whenHandedOverDay`\|`effectiveDay`\|`day`, `daysSupply` |
| `Observation` | 0+ | `code.text`, one of `valueQuantity.value` / `valueString` / `valueBoolean`, `valueQuantity.unit`, `effectiveDay`\|`day` |
| `Encounter` | 0+ (≥ 2 to avoid a warning) | `period.startDay` \| `day` |
| `AllergyIntolerance` | 0+ | `code.text` |

`Patient.id` never leaves Layer 1 — everything downstream sees
`sha256(id)[:16]`.

The adapter enforces exactly one `Patient`. If a supported resource carries a
`subject` or `patient` reference, that reference must identify the same Patient;
a mixed-patient Bundle is rejected rather than merged.

### 1.3 Line events versus concomitant medications

Not every `MedicationRequest` is a decision point. A medication that maps to no
arm *and* matches `arms.CONCOMITANT_TOKENS` — a steroid bridge, typically — is
recorded alongside the line rather than as one: it does not create a stage, and
it sets `rescue_therapy` on whichever stage window it falls inside.

The rule is conservative. A medication that maps to no arm and matches no
concomitant token stays in the line sequence and surfaces as `manual-review`, so
a biologic newer than `ARM_SYNONYMS` is flagged for a human rather than dropped.

### 1.4 The decision point is a medication event

This is the least obvious requirement. **Stages are built from treatment events,
one per `MedicationRequest`, and the visit you want a recommendation for must be
present as its own event.** The demo bundle ends with:

```json
{"resourceType": "MedicationRequest",
 "medicationCodeableConcept": {"text": "current decision point"},
 "authoredOnDay": 365,
 "response": "response unknown"}
```

Without that trailing open event there is no stage to decide at, and the agent
rejects the record rather than answering for the *previous* visit. The marker
must be open (no `stopDay`) and cannot carry a known response. `"current decision point"` maps to the
`continue-current` reference arm through `arms.ARM_SYNONYMS`.

### 1.5 Observation codes

Codes are normalised by lowercasing and replacing `-` and space with `_`, so
`anti-CCP`, `Anti CCP` and `anti_ccp` are the same field. Only these are read:

| Family | Accepted codes | Type | Plausible range |
| --- | --- | --- | --- |
| disease activity | `das28`, `cdai`, `sdai`, `tender_joint_count`, `swollen_joint_count`, `haq_di` | number | `das28` 0–10, `haq_di` 0–3 |
| inflammation | `crp`, `esr` | number | `crp` 0–500, `esr` 0–200 |
| serostatus | `anti_ccp`, `anti_ccp_positive`, `rheumatoid_factor`, `rf_positive` | boolean or number | — |
| safety labs | `egfr`, `alt`, `ast`, `pregnant`, `pregnancy` | number; `pregnant` boolean | `egfr` 0–200, `alt`/`ast` 0–5000 |

**Units are checked and converted** (`data/units.py`), with three outcomes:

| Reported unit | What happens |
| --- | --- |
| the canonical one, or a known equivalent (`mg/dL` for CRP) | converted, and the conversion recorded as an `info` issue |
| a known-incompatible one (`U/L` for CRP) | `DataContractError` — the number is not the quantity it claims to be |
| unrecognised, or absent | assumed canonical, modelled as given, flagged as a `warning` |

The conversion runs *before* the plausible-range check, because a range check on
a value in the wrong unit checks the wrong number: 60 mg/dL is 600 mg/L, outside
the CRP range, and read as mg/L it would have passed.

**A value outside its plausible range is an error, not an outlier.** It raises
`DataContractError` out of `build_patient_state`. A DAS28 of −5 is a corrupt
record, and quietly modelling it moves the recommendation with nothing to show.

**A measured zero is a value, not a missing one.** An eGFR of 0 is the most
extreme patient the range admits and is handled as such throughout.

### 1.6 What actually reaches the model

Of everything ingested, the fitted estimators condition on exactly six numbers,
taken from the **latest** stage:

```text
blip basis h(X)            intercept, das28_std, anti_ccp, prior_tnf
treatment-free basis f(X)  intercept, das28_std, crp_std, anti_ccp, prior_tnf, alt_excess
```

where `das28_std = (das28 − 5.5) / 1.5`, `crp_std = (crp − 30) / 25`, and
`alt_excess = max(alt − 40, 0) / 25`. `prior_tnf` is derived: 1.0 if any
*completed* stage used a TNF inhibitor (the open stage is excluded, since at the
decision point the patient has not yet been exposed to what is being decided).

`egfr` and `alt` additionally drive the Layer 4 safety filter. `haq_di`, `esr`,
`cdai`, `sdai` and the joint counts are ingested, contribute to the belief
filter and the data-contract family check, and **are not covariates in any
estimator**. Do not read a recommendation as having accounted for them.

Missing values fall back to deliberately mid-range defaults rather than extreme
ones — `das28` 5.0, `crp` 15.0, `anti_ccp` 0, `prior_tnf` 0, `egfr` 90,
`alt` 25 — and the data contract is what flags the absence.

### 1.7 Medication names

Free text, mapped to the six canonical arms by substring match in order
(`arms.ARM_SYNONYMS`); the first match wins, so combination therapy maps to the
advanced agent rather than its csDMARD anchor.

| Arm | Tokens |
| --- | --- |
| `continue-current` | current decision, continue |
| `TNF-inhibitor` | tnf, adalimumab, etanercept, infliximab, certolizumab, golimumab |
| `IL-6 inhibitor` | il-6, il6, tocilizumab, sarilumab |
| `JAK-inhibitor` | jak, tofacitinib, baricitinib, upadacitinib |
| `rituximab` | rituximab, abatacept |
| `methotrexate-optimization` | methotrexate, mtx, hydroxychloroquine, sulfasalazine, leflunomide |

Anything unmatched becomes `manual-review`, which is deliberately excluded from
the scoreable menu. Note that `abatacept` is mapped onto the rituximab arm: they
are pharmacologically distinct and this is a modelling shortcut, not a clinical
claim.

### 1.8 Rejection rules

`DataLayer.build_patient_state` raises rather than degrading:

| Severity | Condition | Effect |
| --- | --- | --- |
| **error** | `Condition` does not contain "rheumatoid" | `DataContractError` |
| **error** | zero medication events | `DataContractError` |
| **error** | no completed treatment line followed by an explicit open current-decision event | `DataContractError` |
| **error** | multiple Patient resources or a resource referencing another patient | rejected before the data contract |
| **error** | medication `start_day`s not chronological | `DataContractError` |
| **error** | any observation outside `PLAUSIBLE_RANGES` | `DataContractError` |
| **error** | a feature traceable only to a post-decision observation | `LeakageError` |
| warning | fewer than 2 encounters | limits visit-intensity correction |
| warning | a required variable family absent | recorded in the contract report |
| warning | a ranged field recorded as a non-numeric string | estimators use a default, and you are told |
| warning | no medication maps to a configured arm | recorded in the contract report |

### 1.9 Minimum viable record

```text
1 Patient           id
1 Condition         text containing "rheumatoid"
2 MedicationRequest the prior line, plus the open decision point
2 Encounter         to avoid the visit-intensity warning
1 Observation       das28 at or before the decision day
```

Everything else improves the estimate or the safety check. With only that, every
other covariate takes its default and the recommendation is close to a
population average.

---

## 2. Training: a cohort

Today the estimators are fit on `simulation/ra_cohort.py`. To fit them on real
data you must supply `list[CohortTrajectory]`, one per patient:

```python
CohortTrajectory(patient_index, stages, censored, censoring_reason)
CohortStage(stage, day, features, arm, propensity, outcome,
            uncensored_probability, interval_days, event)
```

### 2.1 Per decision point

| Field | Requirement |
| --- | --- |
| `features` | the six model covariates, already normalised — `das28`, `crp`, `anti_ccp`, `prior_tnf`, `egfr`, `alt` |
| `arm` | one of the six canonical arms, as *given* |
| `outcome` | the reward on a 0–1 scale, **at this visit** |
| `propensity` | P(this arm \| covariates) under the clinician policy that generated the data |
| `day`, `interval_days` | integer days; irregular spacing is expected and modelled |
| `event` | terminating event, or `"ongoing"` |
| `uncensored_probability` | simulation-only ground truth; leave at 1.0 on real data — `estimation/censoring.py` estimates it |

### 2.2 Structural requirements

* **Multiple decision points per patient.** Backward induction is the reason
  this system exists; a single-visit cohort reduces it to a regression. The
  cohort assumes 3 stages, and `n_stages` is read from the longest trajectory.
* **Positivity.** Every arm needs non-trivial probability for every covariate
  profile, or the inverse-probability weights diverge. Propensities are clipped
  at 0.02/0.98, which bounds the damage but does not identify the effect.
* **Variation in the arm actually given.** Blips are identified against
  `continue-current`; an arm nobody received has no blip.
* **Enough patients.** The fit is at n=280 trajectories, and the coverage study
  is run at that size. The stage-specific parameterisation carries 78 blip
  parameters and the pooled one 98, so n in the low hundreds is a floor, not a
  target.
* **Observed dropout.** Informative censoring is modelled and corrected; you
  need the trajectories that *ended*, not just the completers. Dropping them
  before ingestion reintroduces exactly the bias `use_ipcw` removes.

### 2.3 Propensity is fitted, not taken

`CohortStage.propensity` is what the *simulator* drew from — oracle knowledge no
deployment has. `estimation/propensity.py` fits the same quantity from the data
an analyst would hold: a multinomial logit over the blip basis, fit on the
training split by IRLS and applied to the holdout. **The deployed scores use the
fitted model**; the oracle column exists only to size the difference.

`cli evaluate` reports both. Swapping the generator's propensity for the fitted
one moves each estimator's held-out value by +0.006 to +0.011 — under a sixth of
its interval width, in the same direction for every estimator, so the ordering is
unchanged. Held-out calibration against the true propensities is 0.021 mean
absolute error.

That establishes the weaker and necessary claim: the evaluation does not *require*
the generating process. It is **not** a claim about unmeasured confounding, which
is where a propensity model actually fails and which a simulation whose
assignment depends only on recorded covariates cannot test. On real data the
assignment depends on things the record does not hold, and no amount of fitting
recovers those.

---

## 3. What is missing today

Ordered by how much it would change a recommendation. Units, the endpoint seam
and the fitted propensity used to head this list; they are §1.5, §3.1 and §2.3
above now.

1. **Choose an endpoint deliberately.** `data/endpoints.py` offers two:
   `ResponseTextEndpoint` (the keyword mapping, a placeholder) and
   `EULARResponseEndpoint` (the 1996 criteria, computed from the DAS28 change
   across the stage and the level attained). Pass one to `DataLayer`.

   The default is the *text* one, which is the surprising choice and is
   deliberate: `simulation/ra_cohort` draws a synthetic 0–1 response and then
   moves DAS28 by `3 * (outcome - 0.5)`, so an outcome of 0.55 improves DAS28 by
   0.15 — which EULAR correctly calls no response. Measured over 113 simulated
   stages the two agree on 24%, and EULAR sits further from the generator's own
   outcome (mean absolute difference 0.43) than the text mapping does (0.19).
   The scales are different by construction, and reconciling them would mean
   re-tuning the simulation to flatter an endpoint. **On real data, set
   `EULARResponseEndpoint` or your own.**

2. **Supply the dispensing records.** `MedicationDispense` and
   `MedicationAdministration` are now read (`whenHandedOverDay` / `effectiveDay`,
   `daysSupply`), and they are what make `realized` differ from `assigned` and
   adherence a measured proportion of days covered rather than a default of 1.0.
   Without them the ITT / per-protocol / as-treated split has no input and
   adherence is an assumption. A cross-arm substitution is flagged as a switch; a
   within-class one (etanercept against an adalimumab order) is not, because the
   arm is what the model reasons about.
3. **Four confounders are unadjusted.** `data/dag.py` derives the adjustment
   set from the edge list by the backdoor criterion and splits it by what the
   estimator bases carry: `age`, `gender`, `steroid_use` and
   `comorbidity_burden` end up in `unmodelled_confounders`, reported as a
   warning on every patient. The model's set does not satisfy the backdoor
   criterion and will not until those covariates are ingested and put in a
   basis.
4. **Comorbidity and CV risk.** The JAK boxed warning (MACE, VTE, malignancy)
   is in the knowledge base but nothing in the feasible set reads a
   cardiovascular history, because no such field is ingested.
5. **Concomitant medications are recognised by token, not by code.**
   `arms.CONCOMITANT_TOKENS` matches prednisone, methylprednisolone,
   dexamethasone and the rest by substring. A steroid recorded under a name the
   list does not carry becomes a treatment-line event and a phantom decision
   point. A real deployment maps these from RxNorm or ATC rather than free text.
