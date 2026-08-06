# TreatmentRx Agent Research Proposal

## 1. Research Objectives

TreatmentRx is a research-stage clinical decision-support agent designed to support personalized, sequential treatment recommendations for chronic diseases. The first target disease is rheumatoid arthritis (RA), where patients often move through multiple treatment stages, including conventional synthetic DMARDs, biologic therapies, JAK inhibitors, treatment switching, and specialist review.

The main research objective is to develop a causal, calibrated, and explainable treatment recommendation agent that can use longitudinal patient data to support the next clinical treatment decision while maintaining strict safety controls.

Specific objectives are:

- Build a longitudinal RA treatment-decision pipeline that can ingest patient history, treatment exposure, disease activity, lab values, safety variables, and outcomes.
- Segment patient history into clinically meaningful treatment stages.
- Use a disease-specific causal model to identify valid adjustment variables and prevent inappropriate causal claims.
- Estimate treatment value across candidate therapies using sequential treatment-regime methods.
- Combine multiple estimators through Bayesian model averaging rather than relying on one model.
- Quantify uncertainty, calibration, model disagreement, and out-of-distribution risk.
- Generate clinician-facing and patient-facing explanations that are grounded in model outputs, safety rules, and retrieved clinical evidence.
- Keep LLM agents separate from the statistical recommendation layer so that language models explain recommendations but do not independently make medical decisions.

## 2. Research Significance

Clinical treatment decisions for chronic diseases are rarely one-time choices. They are sequential, context-dependent, and shaped by prior treatments, disease response, adverse effects, patient preference, and changing clinical risk. Many existing AI systems simplify this problem into single-step prediction, which does not fully reflect real clinical decision-making.

TreatmentRx is significant because it treats treatment recommendation as a multi-stage decision problem. Instead of predicting only whether a patient will respond to one therapy, the system is designed to estimate the value of different next-step treatment options based on the patient's longitudinal history.

The project also addresses several important limitations in current clinical AI systems:

- It separates causal estimation from language-model explanation.
- It uses an explicit causal graph to reduce the risk of misleading associations.
- It reports uncertainty and model disagreement instead of hiding uncertainty behind a single recommendation.
- It includes safety gates that can block unsafe recommendations before any LLM output is generated.
- It supports auditable recommendation generation through data contracts, model versioning, DAG versioning, and provenance logging.

If successful, TreatmentRx can provide a reusable framework for clinical decision-support agents in other sequential treatment domains, such as depression, inflammatory bowel disease, oncology supportive care, and cardiovascular risk management.

## 3. Agent Architecture

TreatmentRx is organized as a layered clinical AI system. Each layer has a specific responsibility and can be tested independently.

### 3.1 Data Layer

The data layer ingests patient records in a FHIR-like structure. It extracts demographics, diagnosis, medication history, treatment start and stop dates, disease activity measurements, lab values, encounters, allergies, safety variables, and outcomes.

For the RA prototype, the data layer validates whether the patient record contains enough information for treatment-regime estimation. Required information includes treatment history, disease activity measures, inflammatory markers, serostatus, and key safety labs.

### 3.2 Stage Construction Layer

The stage construction layer converts longitudinal patient history into treatment stages. Each stage represents a decision point, such as starting methotrexate, switching to a TNF inhibitor, or considering a new biologic after inadequate response.

This layer also supports visit-intensity and censoring adjustments so that irregular visit schedules and incomplete follow-up can be handled more appropriately.

### 3.3 Patient State Encoder

The patient state encoder converts clinical history into a structured patient state representation. The current design uses:

- a transparent handcrafted feature baseline;
- a GRU-compatible sequence encoder interface;
- a future Transformer encoder for larger EHR datasets.

The encoder produces a patient state vector, `z_t`, which is used by the treatment-regime estimators.

### 3.4 Causal DAG Layer

The causal DAG layer stores a disease-specific causal graph. For RA, the graph includes baseline disease activity, inflammatory markers, serostatus, prior biologic exposure, steroid use, comorbidities, treatment exposure, treatment response, and safety variables.

The DAG layer identifies adjustment variables and generates deterministic causal path text. If the treatment effect is not identifiable from the available data, the system should block causal recommendation and route the case to clinical review.

### 3.5 Treatment Estimation Layer

The treatment estimation layer compares candidate treatment actions. The planned estimators include:

- Q-Shared sequential treatment-regime estimation;
- dWOLS-Shared doubly robust estimation;
- stage-specific Q-learning;
- Bayesian model averaging across estimators.

The goal is not only to recommend the highest-value treatment option, but also to show how much the candidate models agree or disagree.

### 3.6 Uncertainty and Calibration Layer

The uncertainty layer reports four types of uncertainty:

- aleatoric uncertainty from natural outcome variability;
- epistemic uncertainty from limited data;
- model uncertainty from disagreement across estimators;
- out-of-distribution uncertainty when a patient is far from the training population.

The calibration layer evaluates whether predicted treatment values are reliable. Poor calibration should trigger recalibration or prevent deployment of the model.

### 3.7 Safety Layer

The safety layer runs before the LLM agent layer. It checks contraindications, allergies, organ function risks, pregnancy flags, identifiability failures, equipoise, and out-of-distribution warnings.

Hard safety blocks are enforced in application code and cannot be overridden by an LLM.

### 3.8 Multi-Agent Explanation Layer

The LLM layer is used for explanation and communication, not independent medical decision-making. It contains three agents:

- Safety Agent: formats safety findings and uncertainty warnings.
- Guideline Agent: summarizes retrieved guideline and drug-safety evidence.
- Synthesis Agent: produces a clinician recommendation card and a patient-facing summary.

The agents must follow strict constraints: they cannot choose a new treatment, invent citations, modify the causal graph, bypass safety blocks, or make causal claims beyond the system-generated causal path text.

## 4. Technical Approach

The TreatmentRx prototype will use a modular Python-based architecture.

Core technologies include:

- Python for the clinical pipeline and orchestration.
- FHIR-like data structures for patient record ingestion.
- PostgreSQL or TimescaleDB for longitudinal patient data in later deployment.
- Qdrant or another vector database for guideline and drug-safety retrieval.
- PyTorch for future GRU and Transformer encoder training.
- NumPy, SciPy, and scikit-learn-style estimators for baseline statistical modeling.
- NetworkX or a similar graph library for causal DAG storage and traversal.
- MLflow and DVC-style artifact tracking for model, DAG, calibration, and dataset versioning.
- Expected calibration error and reliability summaries for calibration monitoring.
- Retrieval-augmented generation for guideline-grounded explanations.
- Application-code safety gates for contraindication and identifiability blocking.

The prototype currently focuses on a working research scaffold rather than production deployment. It includes RA data validation, stage segmentation, baseline encoders, causal DAG scaffolding, Bayesian model averaging, uncertainty reporting, calibration reporting, and deterministic explanation generation.

## 5. Agent Functions

The TreatmentRx agent is intended to support clinicians by producing structured treatment-decision assistance. Its main functions are:

- Ingest a longitudinal patient record.
- Validate whether the patient record meets the RA data contract.
- Build treatment stages from medication and encounter history.
- Extract clinical features such as disease activity, inflammatory markers, serostatus, renal function, liver function, and prior treatment exposure.
- Generate a patient state representation for modeling.
- Check the RA causal DAG and identify adjustment requirements.
- Estimate candidate treatment values.
- Combine estimator outputs using Bayesian model averaging.
- Report treatment scores, confidence bands, uncertainty values, and model disagreement.
- Run safety checks before producing an explanation.
- Return a clinician-facing explanation with treatment comparison, safety status, uncertainty, and rationale.
- Return a patient-facing summary written in plain language.
- Log provenance information for audit, including model version, DAG version, safety findings, and recommendation output.

The agent is not intended to replace the clinician. It is designed to support decision-making, highlight uncertainty, surface safety concerns, and make the reasoning behind a recommendation easier to inspect.

## 6. Summary

TreatmentRx proposes a clinically grounded architecture for treatment recommendation agents. The system combines sequential treatment-regime modeling, causal DAGs, Bayesian model averaging, uncertainty quantification, calibration, safety gates, and controlled LLM-based explanation.

The central design principle is separation of responsibilities. Statistical models estimate treatment value; causal graphs constrain causal interpretation; safety gates enforce hard clinical rules; retrieval systems provide guideline evidence; and LLM agents communicate the result in a structured, readable form.

The first research target is rheumatoid arthritis. The same architecture can later be adapted to other chronic diseases that require sequential treatment decisions. This project aims to create a safer, more auditable, and more clinically realistic foundation for medical AI agents.
