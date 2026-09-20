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
  default, and the effect genuinely is not identified for them. "Present" means
  present *as the estimators read it* — `OBSERVED_AS` names the record key each
  term is built from, because a hand-written list of observation codes is how
  this check came to accept a HAQ-DI in place of a DAS28 and certify a defaulted
  covariate as identified.
* `unmodelled_confounders` — nodes the DAG believes affect both treatment and
  outcome that no basis carries. This is a standing limitation of the model
  rather than a property of any record, so it is reported once, as a warning, for
  every patient. On the synthetic cohort it costs nothing because the generating
  process has no age, gender, steroid or comorbidity effect; on real data it is
  exactly the residual confounding nobody can rule out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from treatmentrx.domain import DAGValidationResult, PatientRecord, StageRecord

# DAG node -> the basis term the estimators actually condition on. Anything not
# in here is unmodelled by construction, whatever the record contains.
MODELLED_BY = {
    "baseline_disease_activity": "das28_std",
    "crp": "crp_std",
    "anti_ccp": "anti_ccp",
    "prior_biologic_exposure": "prior_tnf",
}

# DAG node -> the record keys that basis term is built from, and whether the
# covariate is read as a number.
#
# `MODELLED_BY` says which term the estimators carry; this says which
# observation the term is *built from*, which is the thing a record either has
# or does not. They were the same question asked in two vocabularies, and the
# second one was a hand-written list of observation codes that had drifted:
# `baseline_disease_activity` accepted `cdai`, `sdai` and `haq_di`, none of
# which produces a `das28`. A patient with a HAQ-DI and no DAS28 was certified
# identified while `estimation.features` handed the estimators
# `FEATURE_DEFAULTS["das28"]` — the exact case this check exists to catch.
#
# `prior_biologic_exposure` is deliberately absent: it is read from the
# medication history rather than from an observation, and "no prior biologic" is
# a value of that variable rather than a missing one. The contract already makes
# a record with no treatment history an error, so the adjuster is present by the
# time this runs.
#
# The numeric flag mirrors `estimation.features.numeric_feature`, which falls
# back to the default for a non-numeric or boolean value. Serostatus is read for
# truth rather than magnitude, so a recorded negative is an observation and not
# an absence — which is the distinction invariant 28 is about, one covariate
# over.
OBSERVED_AS = {
    "baseline_disease_activity": (("das28",), True),
    "crp": (("crp",), True),
    "anti_ccp": (("anti_ccp", "anti_ccp_positive"), False),
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

    def validate(
        self,
        patient: PatientRecord,
        stages: list[StageRecord],
        outcome: str = "RA response",
    ) -> DAGValidationResult:
        """Is the effect identified for the patient *as the model will read them*?

        `stages` rather than `patient.observations`, because the estimators read
        the decision point's carried-forward feature map and nothing else. The
        treatment is taken from the same place for the same reason: two
        arguments describing one stage are two things that can disagree.
        """
        dag = self.match(patient)
        latest = stages[-1]
        treatment = latest.treatment
        missing_adjusters = [
            node for node in dag.adjustment_set
            if not self._has_adjuster(node, latest, patient)
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

    def _has_adjuster(self, node: str, stage: StageRecord, patient: PatientRecord) -> bool:
        """Does the record carry the value the estimators will actually read?

        Only called for nodes in `adjustment_set`, which by construction are the
        ones the model conditions on. A node the model cannot use is never asked
        about here — it is in `unmodelled_confounders` instead.

        The question is deliberately narrow: not "does the bundle mention
        something in this family" but "will `estimation.features` find a usable
        value under the key this term is built from, or fall back to a default".
        The family question belongs to the data contract and is a warning there;
        answering it here is what certified a defaulted DAS28 as identified.
        """
        if node == "prior_biologic_exposure":
            # Read from the medication history rather than an observation, and
            # "no prior biologic" is a value of this variable, not a missing one.
            # Requiring a biologic to appear blocked every csDMARD-only patient
            # for having been treated conservatively.
            return bool(patient.medications)
        keys, numeric = OBSERVED_AS[node]
        for key in keys:
            if key not in stage.features:
                continue
            if not numeric:
                return True
            value = stage.features[key]
            # Same test `numeric_feature` applies before falling back: a DAS28
            # recorded as the string "high" is not a disease-activity measurement
            # the estimators can use, however present it looks.
            if not isinstance(value, bool) and isinstance(value, (int, float)):
                return True
        return False

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
    """An *irreducible* set of non-descendants that blocks every backdoor path.

    Minimal in the sense the causal literature uses: no proper subset of the
    result also satisfies the criterion. That is not the same as *minimum*
    cardinality, which is NP-hard in general; where the two differ this returns
    an irreducible set and says so rather than claiming the smaller guarantee.

    **The previous version was neither.** It walked the candidates in sorted
    order and added a node whenever the set so far did not yet block, without
    ever testing whether *that* node helped — so a redundant node got in simply
    by sorting first. On `T <- Z -> Y` plus `T <- Z <- W -> Y`, where `{Z}`
    blocks both paths on its own, it returned `{W, Z}`. It also carried a
    `without = ...` local that was computed and never read, which is usually what
    a half-finished condition leaves behind.

    It happened to be right where it is actually used: on the RA graph every
    backdoor path is a single confounder with arrows into both treatment and
    outcome, so nothing substitutes for anything and the greedy order cannot
    matter. The deployed adjustment set never changed. But a function whose name
    promises minimality has to earn it on graphs other than the one it ships
    with, because `_split_by_what_the_model_carries` derives
    `unmodelled_confounders` from this — and a spurious entry there is a
    confounder the model card claims it failed to adjust for when it never
    needed to.

    Two passes. Cover: add nodes until every backdoor path is blocked. Prune:
    try removing each one; drop it if the set still blocks. Both iterate in
    sorted order, so the result is deterministic and reproducible for audit.
    """
    descendants = _descendants(dag.edges, treatment)
    candidates = sorted(
        {
            node
            for path in backdoor_paths(dag.edges, treatment, outcome)
            for node in path[1:-1]
            if node not in descendants and node not in {treatment, outcome}
        }
    )

    required: set[str] = set()
    for node in candidates:
        if satisfies_backdoor(dag, required, treatment, outcome)[0]:
            break
        required.add(node)

    # Prune. A node earns its place only if removing it re-opens a path — which
    # is exactly the definition of irreducible, and is the step the previous
    # version was missing.
    for node in sorted(required):
        trimmed = required - {node}
        if satisfies_backdoor(dag, trimmed, treatment, outcome)[0]:
            required = trimmed
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
