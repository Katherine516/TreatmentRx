from __future__ import annotations

from dataclasses import dataclass

from treatmentrx.domain import DAGValidationResult, PatientRecord


@dataclass(frozen=True)
class CausalDAG:
    name: str
    version: str
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    adjustment_set: tuple[str, ...]
    colliders: tuple[str, ...]
    safety_nodes: tuple[str, ...]


class CausalDAGRegistry:
    """Small deterministic registry for disease DAGs.

    The production version should persist YAML DAGs and run a formal
    identifiability package. This module gives the pipeline the same contract
    without adding heavy dependencies to the prototype.
    """

    def __init__(self) -> None:
        self._dags = {"rheumatoid arthritis": self._ra_v1()}

    def match(self, patient: PatientRecord) -> CausalDAG:
        disease = patient.disease.lower()
        for key, dag in self._dags.items():
            if key in disease:
                return dag
        raise ValueError(f"No causal DAG registered for disease: {patient.disease}")

    def validate(self, patient: PatientRecord, treatment: str, outcome: str = "RA response") -> DAGValidationResult:
        dag = self.match(patient)
        observed_features = {
            observation.code.lower().replace("-", "_").replace(" ", "_")
            for observation in patient.observations
        }
        missing_adjusters = [
            node for node in dag.adjustment_set
            if not self._has_adjuster(node, observed_features, patient)
        ]
        identified = len(missing_adjusters) == 0
        blocked_reason = None
        if missing_adjusters:
            blocked_reason = "Missing required adjustment variables: " + ", ".join(missing_adjusters)

        return DAGValidationResult(
            dag_name=dag.name,
            version=dag.version,
            identified=identified,
            adjustment_set=list(dag.adjustment_set),
            causal_path_text=self.causal_path_text(dag, treatment=treatment, outcome=outcome),
            blocked_reason=blocked_reason,
        )

    def _has_adjuster(self, node: str, observed_features: set[str], patient: PatientRecord) -> bool:
        if node == "age":
            return bool(patient.demographics.get("age") or patient.demographics.get("birthDate"))
        if node == "gender":
            return bool(patient.demographics.get("gender"))
        if node == "baseline_disease_activity":
            return bool(observed_features & {"das28", "cdai", "sdai", "haq_di"})
        if node == "prior_biologic_exposure":
            names = " ".join(medication.name.lower() for medication in patient.medications)
            return any(token in names for token in ("tnf", "adalimumab", "etanercept", "tocilizumab", "jak"))
        if node == "steroid_use":
            return bool(patient.medications)
        return node in observed_features or node in patient.demographics

    def causal_path_text(self, dag: CausalDAG, treatment: str, outcome: str) -> str:
        return (
            f"In {dag.name} {dag.version}, {treatment} is evaluated as a treatment exposure affecting "
            f"{outcome} through post-treatment disease activity and inflammatory marker response. "
            "The minimum adjustment set controls baseline disease activity, inflammation, serostatus, "
            "prior advanced-therapy exposure, steroid use, and core demographics. Visit frequency and "
            "treatment switching are treated as colliders or downstream process variables and are not "
            "used as adjustment covariates."
        )

    def _ra_v1(self) -> CausalDAG:
        nodes = (
            "age",
            "gender",
            "disease_duration",
            "baseline_disease_activity",
            "crp",
            "esr",
            "anti_ccp",
            "rheumatoid_factor",
            "prior_biologic_exposure",
            "steroid_use",
            "comorbidity_burden",
            "treatment",
            "post_treatment_disease_activity",
            "ra_response",
            "visit_frequency",
            "treatment_switching",
            "egfr",
            "alt",
            "ast",
            "pregnant",
            "serious_infection_history",
            "heart_failure",
        )
        edges = (
            ("age", "treatment"),
            ("age", "ra_response"),
            ("gender", "treatment"),
            ("gender", "ra_response"),
            ("baseline_disease_activity", "treatment"),
            ("baseline_disease_activity", "ra_response"),
            ("crp", "treatment"),
            ("crp", "ra_response"),
            ("anti_ccp", "treatment"),
            ("anti_ccp", "ra_response"),
            ("prior_biologic_exposure", "treatment"),
            ("prior_biologic_exposure", "ra_response"),
            ("steroid_use", "treatment"),
            ("steroid_use", "ra_response"),
            ("comorbidity_burden", "treatment"),
            ("comorbidity_burden", "ra_response"),
            ("treatment", "post_treatment_disease_activity"),
            ("post_treatment_disease_activity", "ra_response"),
            ("baseline_disease_activity", "visit_frequency"),
            ("treatment", "treatment_switching"),
            ("ra_response", "treatment_switching"),
        )
        return CausalDAG(
            name="Rheumatoid Arthritis DAG",
            version="v1.0",
            nodes=nodes,
            edges=edges,
            adjustment_set=(
                "age",
                "gender",
                "baseline_disease_activity",
                "crp",
                "anti_ccp",
                "prior_biologic_exposure",
                "steroid_use",
            ),
            colliders=("visit_frequency", "treatment_switching"),
            safety_nodes=("egfr", "alt", "ast", "pregnant", "serious_infection_history", "heart_failure"),
        )
