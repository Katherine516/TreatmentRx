"""v5.1 Memory Architecture — three tiers with a hard influence boundary.

Tier 1 (working): the ContextBundle for one inference — not persisted here.
Tier 2 (episodic): per-patient, *structured* (not embedded), deterministically
    retrieved by patient_id. The agent's longitudinal awareness.
Tier 3 (semantic): the shared medical knowledge base, the only place vector /
    keyword retrieval belongs. Never patient-specific, read-only at inference.

The non-negotiable rule: memory must never influence the statistical
recommendation. It influences only narrative, framing, and retrieval. Q-values
come exclusively from the current-state estimators. `apply_memory` enforces this
with an assertion, not a convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class MemoryInfluenceError(AssertionError):
    """Raised if memory attempts to move a statistical quantity."""


@dataclass(frozen=True)
class EpisodicItem:
    stage: int
    recommended_action: str
    clinician_action: str | None
    override_reason: str | None
    outcome_summary: str | None
    preference: str | None
    ttl_days: int = 3650


@dataclass
class EpisodicMemory:
    """Structured, patient-scoped memory. No fuzzy matching; deterministic recall."""

    _store: dict[str, list[EpisodicItem]] = field(default_factory=dict)

    def record(self, patient_hash: str, item: EpisodicItem) -> None:
        self._store.setdefault(patient_hash, []).append(item)

    def recall(self, patient_hash: str) -> list[EpisodicItem]:
        # Deterministic: same patient, same retrieval — reproducible for audit.
        return list(self._store.get(patient_hash, []))

    def preferences(self, patient_hash: str) -> list[str]:
        return [item.preference for item in self.recall(patient_hash) if item.preference]

    def recent_summary(self, patient_hash: str, limit: int = 3) -> str:
        items = self.recall(patient_hash)[-limit:]
        if not items:
            return "No prior recorded encounters for this patient."
        return "; ".join(
            f"stage {it.stage}: recommended {it.recommended_action}"
            + (f", clinician {it.clinician_action}" if it.clinician_action else "")
            + (f" ({it.outcome_summary})" if it.outcome_summary else "")
            for it in items
        )

    def reset(self, patient_hash: str) -> None:
        """Per-patient reset — required for trust and data-rights compliance."""
        self._store.pop(patient_hash, None)


# Minimal RA knowledge base. In production this is a versioned Qdrant collection;
# here it is a read-only keyword index so the seam and contract exist.
_KNOWLEDGE_BASE = [
    ("tnf inadequate response", "After inadequate TNF response, guidelines support switching mechanism of action (IL-6, JAK, or rituximab/abatacept)."),
    ("il-6 tocilizumab", "IL-6 inhibition (tocilizumab) is effective as monotherapy or with MTX; monitor lipids, ALT, neutrophils."),
    ("jak safety", "JAK inhibitors carry boxed warnings (MACE, VTE, malignancy); avoid in high CV risk and pregnancy."),
    ("methotrexate", "Methotrexate is the anchor csDMARD; monitor LFTs and renal function, contraindicated in pregnancy."),
    ("seropositive rituximab", "Anti-CCP / RF seropositive patients tend to respond better to rituximab."),
]


@dataclass
class SemanticKnowledgeBase:
    """Tier 3 — shared, read-only RAG over the medical knowledge base."""

    version: str = "ra-kb-2026Q1"

    def retrieve(self, query: str, k: int = 2) -> list[str]:
        tokens = set(query.lower().split())
        scored = []
        for key, passage in _KNOWLEDGE_BASE:
            overlap = len(tokens & set(key.split()))
            if overlap:
                scored.append((overlap, passage))
        scored.sort(key=lambda kv: kv[0], reverse=True)
        return [passage for _, passage in scored[:k]]


# Statistical keys memory is forbidden from touching.
_FORBIDDEN_KEYS = {"q_values", "policy_value", "confidence_band", "recommended_action", "safety_status"}


def apply_memory(bundle: dict[str, Any], memory: EpisodicMemory, kb: SemanticKnowledgeBase) -> dict[str, Any]:
    """Apply episodic + semantic memory to a context bundle — narrative only.

    ALLOWED: framing_hints, prior_context, rag evidence, provenance.
    FORBIDDEN: anything under statistical_output.
    """
    patient_hash = bundle["patient_context"]["patient_id"]
    before = dict(bundle["statistical_output"])  # snapshot to prove non-influence

    preferences = memory.preferences(patient_hash)
    rag_query = " ".join(
        [bundle["statistical_output"]["recommended_action"], bundle["patient_context"].get("history_summary", "")]
    )
    evidence = kb.retrieve(rag_query)

    bundle["memory"] = {
        "framing_hints": preferences,
        "prior_context": memory.recent_summary(patient_hash),
        "evidence": evidence,
        "kb_version": kb.version,
        "provenance": {
            "episodic_items": len(memory.recall(patient_hash)),
            "evidence_passages": len(evidence),
            "influence": "narrative_and_retrieval_only",
        },
    }

    # Enforce the boundary, do not trust it.
    after = bundle["statistical_output"]
    for key in _FORBIDDEN_KEYS:
        if before.get(key) != after.get(key):
            raise MemoryInfluenceError(f"memory mutated statistical_output['{key}']")
    return bundle
