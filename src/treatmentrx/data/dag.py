"""Whether the effect is identified — checked against the model, not the record.

The previous version asked whether the *bundle mentioned* each adjuster. A
`birthDate` counted as adjusting for age; any medication at all counted as
adjusting for steroid use. Both are false: what matters is whether the estimator
conditions on the variable, and no basis in `estimation/basis.py` carries age,
gender, steroid use or comorbidity burden. So the certificate said "identified"
for essentially every patient while the model adjusted for none of them.

Two lists now, and the distinction is the point:

* `adjustment_set` — confounders the model *does* condition on. `identified`
  requires every one of them to be present in the record, and it can fail: a
  patient with no disease-activity measurement is running the estimators on a
  default, and the effect genuinely is not identified for them.
* `unmodelled_confounders` — nodes the DAG believes affect both treatment and
  outcome that no basis carries. This is a standing limitation of the model
  rather than a property of any record, so it is reported once, as a warning, for
  every patient. On the synthetic cohort it costs nothing because the generating
  process has no age, gender, steroid or comorbidity effect; on real data it is
  exactly the residual confounding nobody can rule out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from treatmentrx.domain import DAGValidationResult, PatientRecord

# DAG node -> the basis term the estimators actually condition on. Anything not
# in here is unmodelled by construction, whatever the record contains.
MODELLED_BY = {
    "baseline_disease_activity": "das28_std",
    "crp": "crp_std",
    "anti_ccp": "anti_ccp",
    "prior_biologic_exposure": "prior_tnf",
}


@dataclass(frozen=True)
class CausalDAG:
    name: str
    version: str
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    adjustment_set: tuple[str, ...]
    colliders: tuple[str, ...]
    safety_nodes: tuple[str, ...]
    # Confounders the DAG asserts and the estimators cannot condition on.
    unmodelled_confounders: tuple[str, ...] = field(default=())


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
            blocked_reason = (
                "Missing required adjustment variables: " + ", ".join(missing_adjusters)
            )

        return DAGValidationResult(
            dag_name=dag.name,
            version=dag.version,
            identified=identified,
            adjustment_set=list(dag.adjustment_set),
            causal_path_text=self.causal_path_text(dag, treatment=treatment, outcome=outcome),
            blocked_reason=blocked_reason,
        )

    def unmodelled_confounders(self, patient: PatientRecord) -> list[str]:
        """Confounders this DAG asserts that no estimator basis carries.

        Constant per DAG, not per patient — it is a property of the model. Read
        it as the list of things the reported effect has *not* been adjusted for.
        """
        return list(self.match(patient).unmodelled_confounders)

    def _has_adjuster(self, node: str, observed_features: set[str], patient: PatientRecord) -> bool:
        """Is this adjuster actually available for this patient?

        Only called for nodes in `adjustment_set`, which by construction are the
        ones the model conditions on. A node the model cannot use is never asked
        about here — it is in `unmodelled_confounders` instead.
        """
        if node == "baseline_disease_activity":
            return bool(observed_features & {"das28", "cdai", "sdai", "haq_di"})
        if node == "prior_biologic_exposure":
            # The adjuster is *known* whenever a medication history exists —
            # "no prior biologic" is a value of this variable, not a missing one.
            # Requiring a biologic to appear blocked every csDMARD-only patient
            # for having been treated conservatively.
            return bool(patient.medications)
        return node in observed_features or node in patient.demographics

    def causal_path_text(self, dag: CausalDAG, treatment: str, outcome: str) -> str:
        return (
            f"In {dag.name} {dag.version}, {treatment} is evaluated as a treatment exposure affecting "
            f"{outcome} through post-treatment disease activity and inflammatory marker response. "
            "The adjustment set the estimators actually condition on is baseline disease "
            "activity, inflammation, serostatus and prior advanced-therapy exposure. "
            f"{', '.join(dag.unmodelled_confounders)} are confounders in this DAG that no "
            "estimator basis carries, so the effect is not adjusted for them. Visit "
            "frequency and treatment switching are colliders or downstream process "
            "variables and are not used as adjustment covariates."
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
        skeleton = CausalDAG(
            name="Rheumatoid Arthritis DAG",
            version="v1.0",
            nodes=nodes,
            edges=edges,
            # Both sets are *derived* below, from the edge list and from what
            # `estimation/basis.py` actually carries. Hand-listing them is how
            # the certificate came to claim adjustments nothing performed.
            adjustment_set=(),
            unmodelled_confounders=(),
            colliders=("visit_frequency", "treatment_switching"),
            safety_nodes=("egfr", "alt", "ast", "pregnant", "serious_infection_history", "heart_failure"),
        )
        return _split_by_what_the_model_carries(skeleton)


# --------------------------------------------------------------------------
# Backdoor criterion — the adjustment set derived rather than declared
# --------------------------------------------------------------------------


def _parents(edges: tuple[tuple[str, str], ...], node: str) -> set[str]:
    return {source for source, target in edges if target == node}


def _descendants(edges: tuple[tuple[str, str], ...], node: str) -> set[str]:
    """Everything reachable from `node` following edge direction."""
    seen: set[str] = set()
    frontier = [node]
    while frontier:
        current = frontier.pop()
        for source, target in edges:
            if source == current and target not in seen:
                seen.add(target)
                frontier.append(target)
    return seen


def backdoor_paths(
    edges: tuple[tuple[str, str], ...], treatment: str, outcome: str
) -> list[list[str]]:
    """Every path from treatment to outcome that starts with an arrow *into* the
    treatment.

    Enumerated by walking the undirected skeleton and keeping the paths whose
    first step is a parent of the treatment. The DAG here has 22 nodes and 21
    edges, so exhaustive enumeration is cheap and exact; a larger graph would
    want a proper algorithm rather than this.
    """
    neighbours: dict[str, set[str]] = {}
    for source, target in edges:
        neighbours.setdefault(source, set()).add(target)
        neighbours.setdefault(target, set()).add(source)

    treatment_parents = _parents(edges, treatment)
    paths: list[list[str]] = []

    def walk(node: str, path: list[str]) -> None:
        if node == outcome:
            paths.append(list(path))
            return
        for neighbour in sorted(neighbours.get(node, ())):
            if neighbour in path:
                continue
            walk(neighbour, path + [neighbour])

    for parent in sorted(treatment_parents):
        walk(parent, [treatment, parent])
    return paths


def _is_collider(edges, path: list[str], index: int) -> bool:
    """Does the path meet head-to-head at `path[index]`?"""
    before, node, after = path[index - 1], path[index], path[index + 1]
    return (before, node) in edges and (after, node) in edges


def path_is_blocked(edges, path: list[str], adjusted: set[str]) -> bool:
    """d-separation along one path, given the adjustment set.

    A non-collider blocks when it is adjusted for; a collider blocks unless it —
    or one of its descendants — is adjusted for. Conditioning on a collider is
    how an adjustment set makes things worse rather than better, which is why
    `visit_frequency` and `treatment_switching` must stay out of it.
    """
    for index in range(1, len(path) - 1):
        node = path[index]
        if _is_collider(edges, path, index):
            opened = node in adjusted or bool(
                _descendants(edges, node) & adjusted
            )
            if not opened:
                return True
        elif node in adjusted:
            return True
    return False


def satisfies_backdoor(
    dag: CausalDAG, adjusted: set[str], treatment: str = "treatment", outcome: str = "ra_response"
) -> tuple[bool, list[list[str]]]:
    """Does `adjusted` satisfy the backdoor criterion, and what is left open?

    Two conditions: no adjusted node may be a descendant of the treatment, and
    every backdoor path must be blocked. Returns the verdict and the paths that
    are still open, because "not identified" is only actionable if it says which
    path is the problem.
    """
    descendants = _descendants(dag.edges, treatment)
    if adjusted & descendants:
        return False, [["adjusts a descendant of treatment: " + ", ".join(sorted(adjusted & descendants))]]
    open_paths = [
        path
        for path in backdoor_paths(dag.edges, treatment, outcome)
        if not path_is_blocked(dag.edges, path, adjusted)
    ]
    return not open_paths, open_paths


def minimal_backdoor_set(
    dag: CausalDAG, treatment: str = "treatment", outcome: str = "ra_response"
) -> set[str]:
    """The smallest set of non-descendants that blocks every backdoor path.

    Greedy over the nodes that actually appear on a backdoor path, which for this
    graph is exact: every backdoor path here is a single confounder with arrows
    into both treatment and outcome, so blocking requires that confounder and
    nothing substitutes for it.
    """
    descendants = _descendants(dag.edges, treatment)
    candidates = {
        node
        for path in backdoor_paths(dag.edges, treatment, outcome)
        for node in path[1:-1]
        if node not in descendants and node not in {treatment, outcome}
    }
    required: set[str] = set()
    for node in sorted(candidates):
        without = candidates - {node} if not required else required
        blocked, _ = satisfies_backdoor(dag, required | {node}, treatment, outcome)
        remaining, _ = satisfies_backdoor(dag, required, treatment, outcome)
        if not remaining:
            required.add(node)
        if blocked and satisfies_backdoor(dag, required, treatment, outcome)[0]:
            break
    # Verify and repair: anything still open gets added.
    identified, open_paths = satisfies_backdoor(dag, required, treatment, outcome)
    while not identified and open_paths:
        for path in open_paths:
            for node in path[1:-1]:
                if node not in descendants:
                    required.add(node)
                    break
        identified, open_paths = satisfies_backdoor(dag, required, treatment, outcome)
    return required


def _split_by_what_the_model_carries(dag: CausalDAG) -> CausalDAG:
    """Derive the adjustment set from the graph, then split it by the basis.

    Two steps, and keeping them separate is the point:

    1. **What the graph requires** — `minimal_backdoor_set` over the edge list.
       This is a property of the causal assumptions and owes nothing to what the
       estimators happen to implement.
    2. **What the model can do about it** — split by `MODELLED_BY`. Adjusters
       that map to a term in `TREATMENT_FREE_BASIS` become `adjustment_set`;
       the rest become `unmodelled_confounders` and are reported as a standing
       warning on every patient.

    Deriving rather than declaring means the two cannot drift: adding a node to
    the graph puts it in one list or the other automatically, and adding a
    covariate to the basis moves it from the second to the first.
    """
    required = minimal_backdoor_set(dag)
    modelled = tuple(sorted(node for node in required if node in MODELLED_BY))
    unmodelled = tuple(sorted(node for node in required if node not in MODELLED_BY))
    return CausalDAG(
        name=dag.name,
        version=dag.version,
        nodes=dag.nodes,
        edges=dag.edges,
        adjustment_set=modelled,
        colliders=dag.colliders,
        safety_nodes=dag.safety_nodes,
        unmodelled_confounders=unmodelled,
    )
