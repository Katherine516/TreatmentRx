"""Disease isolation: one explicit definition owns each clinical workflow.

The six-layer sequence is reusable; its clinical contents are not.  A disease
definition binds the data contract, treatment vocabulary, endpoint, DAG, model
family, safety policy, explanation knowledge and feedback implementation into one
unit.  The registry fails closed when no such unit exists, so an unsupported
diagnosis can never fall through to the rheumatoid-arthritis model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from treatmentrx.arms import TREATMENT_ARMS
from treatmentrx.biomarkers import BiomarkerRegistry
from treatmentrx.scientific import EstimandContract, ScientificMode, ra_dtr_estimand


class UnsupportedDiseaseError(ValueError):
    """No registered workflow is allowed to score this diagnosis."""

    def __init__(self, disease: str, supported: list[str]) -> None:
        self.disease = disease
        self.supported = supported
        super().__init__(
            f"No disease workflow is registered for {disease!r}; supported: "
            + ", ".join(supported)
        )


@dataclass
class DiseaseWorkflow:
    """The concrete six layers and model owner for one disease."""

    data: Any
    estimation: Any
    decision: Any
    safety: Any
    agent: Any
    feedback: Any
    training: Any
    biomarkers: BiomarkerRegistry = field(default_factory=BiomarkerRegistry)


@dataclass(frozen=True)
class DiseaseDefinition:
    disease_id: str
    display_name: str
    diagnosis_terms: tuple[str, ...]
    treatment_arms: tuple[str, ...]
    endpoint: str
    dag: str
    model_family: str
    safety_policy: str
    knowledge_base: str
    workflow_factory: Callable[[], DiseaseWorkflow] = field(repr=False, compare=False)
    operating_modes: tuple[ScientificMode, ...] = (ScientificMode.DTR_RESEARCH,)
    estimand_contracts: tuple[EstimandContract, ...] = ()

    def matches(self, diagnosis: str) -> bool:
        normalised = diagnosis.lower()
        return any(term.lower() in normalised for term in self.diagnosis_terms)

    def capability(self) -> dict[str, object]:
        return {
            "disease_id": self.disease_id,
            "display_name": self.display_name,
            "diagnosis_terms": list(self.diagnosis_terms),
            "treatment_arms": list(self.treatment_arms),
            "endpoint": self.endpoint,
            "dag": self.dag,
            "model_family": self.model_family,
            "safety_policy": self.safety_policy,
            "knowledge_base": self.knowledge_base,
            "operating_modes": [mode.value for mode in self.operating_modes],
            "estimand_contracts": [
                estimand.as_dict() for estimand in self.estimand_contracts
            ],
        }

    def estimand_for(self, mode: ScientificMode) -> EstimandContract:
        matches = [
            estimand for estimand in self.estimand_contracts if estimand.mode is mode
        ]
        if len(matches) != 1:
            raise ValueError(
                f"disease {self.disease_id!r} requires exactly one estimand for "
                f"mode {mode.value!r}; found {len(matches)}"
            )
        return matches[0]


class DiseaseRegistry:
    """Deterministic diagnosis-to-workflow routing, with no default fallback."""

    def __init__(self, definitions: tuple[DiseaseDefinition, ...] | None = None) -> None:
        definitions = definitions or (rheumatoid_arthritis_definition(),)
        ids = [definition.disease_id for definition in definitions]
        if len(ids) != len(set(ids)):
            raise ValueError("Disease definitions must have unique disease_id values")
        self._definitions = {definition.disease_id: definition for definition in definitions}

    def resolve(self, diagnosis: str) -> DiseaseDefinition:
        return self.resolve_diagnoses([diagnosis])

    def resolve_diagnoses(self, diagnoses: list[str]) -> DiseaseDefinition:
        """Resolve across a problem list rather than trusting its first entry."""
        matches = [
            definition for definition in self._definitions.values()
            if any(definition.matches(diagnosis) for diagnosis in diagnoses)
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Diagnoses {diagnoses!r} ambiguously match "
                + ", ".join(definition.disease_id for definition in matches)
            )
        rendered = "; ".join(diagnoses) or "Unknown disease"
        raise UnsupportedDiseaseError(rendered, self.supported_ids())

    def get(self, disease_id: str) -> DiseaseDefinition:
        try:
            return self._definitions[disease_id]
        except KeyError:
            raise UnsupportedDiseaseError(disease_id, self.supported_ids())

    def supported_ids(self) -> list[str]:
        return sorted(self._definitions)

    def capabilities(self) -> list[dict[str, object]]:
        return [
            self._definitions[disease_id].capability()
            for disease_id in self.supported_ids()
        ]


def _ra_workflow() -> DiseaseWorkflow:
    # Local imports keep the registry from creating a package-wide import cycle.
    from treatmentrx.agent import AgentLayer
    from treatmentrx.data import DataLayer
    from treatmentrx.decision import DecisionLayer
    from treatmentrx.estimation import EstimationLayer, training
    from treatmentrx.feedback import FeedbackLayer
    from treatmentrx.safety import SafetyLayer

    return DiseaseWorkflow(
        data=DataLayer(),
        estimation=EstimationLayer(),
        decision=DecisionLayer(),
        safety=SafetyLayer(),
        agent=AgentLayer(),
        feedback=FeedbackLayer(),
        training=training,
    )


def rheumatoid_arthritis_definition() -> DiseaseDefinition:
    return DiseaseDefinition(
        disease_id="rheumatoid_arthritis",
        display_name="Rheumatoid Arthritis",
        diagnosis_terms=("rheumatoid",),
        treatment_arms=TREATMENT_ARMS,
        endpoint="response_text (synthetic-cohort default)",
        dag="RA_v1.0",
        model_family="sequential_ra_regime",
        safety_policy="ra_safety_v1",
        knowledge_base="ra-kb-2026Q1",
        workflow_factory=_ra_workflow,
        operating_modes=(ScientificMode.DTR_RESEARCH,),
        estimand_contracts=(ra_dtr_estimand(TREATMENT_ARMS),),
    )


__all__ = [
    "DiseaseDefinition",
    "DiseaseRegistry",
    "DiseaseWorkflow",
    "UnsupportedDiseaseError",
    "rheumatoid_arthritis_definition",
]
