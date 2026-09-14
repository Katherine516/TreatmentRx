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

import copy
import re
from dataclasses import dataclass, field
from typing import Any


class MemoryInfluenceError(AssertionError):
    """Raised if memory attempts to move a statistical quantity."""


@dataclass(frozen=True)
class EpisodicItem:
    stage: int
    recommended_arm: str
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
            f"stage {it.stage}: recommended {it.recommended_arm}"
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


#: Words, not whitespace-delimited blobs. The arm vocabulary is hyphenated
#: (`TNF-inhibitor`, `JAK-inhibitor`, `methotrexate-optimization`) and the
#: knowledge-base keys are not, so splitting on whitespace produced a single
#: token that matched nothing: **four of six arms retrieved zero passages from
#: their own name**. This is invariant 26's failure with the sign reversed —
#: there a short token matched too much, here a compound token matched nothing.
_WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


@dataclass
class SemanticKnowledgeBase:
    """Tier 3 — shared, read-only keyword index over the medical knowledge base.

    Still five hard-coded passages and still not a vector store; what changed is
    that the keyword index now matches keywords.

    **Two defects, and the second only became visible once the first was fixed.**
    Tokenising on whitespace meant `TNF-inhibitor` never matched the key
    `tnf inadequate response`, so `continue-current`,
    `methotrexate-optimization`, `TNF-inhibitor` and `JAK-inhibitor` — four of
    six arms — retrieved nothing for their own name. Only `IL-6 inhibitor` (it
    contains a space) and `rituximab` (no hyphen) worked.

    Then the ranking. The query is the recommended arm *plus* the history
    summary, every match scored one point, and `sort` is stable — so ties fell
    back to the order passages happen to appear in `_KNOWLEDGE_BASE`. The demo
    patient is recommended **rituximab**, the knowledge base contains a
    rituximab passage, and the two retrieved were about TNF response and
    methotrexate: the arm's own evidence lost to history tokens on insertion
    order. A card that labels this "Evidence:" under a recommendation was citing
    passages about something else.

    `subject` is what the passage is supposed to be evidence *for*. A match on it
    outranks a match on the surrounding history, and the remaining ties break by
    knowledge-base order, which is arbitrary but deterministic and auditable.
    """

    version: str = "ra-kb-2026Q1"

    #: How much more a subject match is worth than a history match. Only the
    #: ordering matters, not the value: it has to exceed the largest possible
    #: history overlap, which is bounded by the longest key.
    subject_weight: int = 10

    def retrieve(self, query: str, k: int = 2, subject: str = "") -> list[str]:
        query_words = _words(query)
        subject_words = _words(subject)
        scored = []
        for index, (key, passage) in enumerate(_KNOWLEDGE_BASE):
            key_words = _words(key)
            on_subject = len(subject_words & key_words)
            on_query = len(query_words & key_words)
            if not on_subject and not on_query:
                continue
            # Negative index so the sort stays descending overall and ties keep
            # knowledge-base order rather than inheriting it by accident.
            scored.append((self.subject_weight * on_subject + on_query, -index, passage))
        scored.sort(reverse=True)
        return [passage for _, _, passage in scored[:k]]


# Statistical keys memory is forbidden from touching.
_FORBIDDEN_KEYS = {"q_values", "policy_value", "confidence_band", "recommended_arm", "safety_status"}


def apply_memory(bundle: dict[str, Any], memory: EpisodicMemory, kb: SemanticKnowledgeBase) -> dict[str, Any]:
    """Apply episodic + semantic memory to a context bundle — narrative only.

    ALLOWED: framing_hints, prior_context, rag evidence, provenance.
    FORBIDDEN: anything under statistical_output.
    """
    patient_hash = bundle["patient_context"]["patient_id"]
    # Deep, not shallow. `q_values` and `confidence_band` are containers: a
    # shallow snapshot holds the *same* objects the bundle does, so a component
    # that mutates one in place — `q_values["TNF-inhibitor"] = 0.99` — passes the
    # comparison below unchanged. The guard is the strongest claim this layer
    # makes and it only held because `apply_memory` happens to replace containers
    # rather than edit them; that stops being true the moment a real memory
    # component gets a handle on the bundle.
    before = copy.deepcopy(bundle["statistical_output"])

    preferences = memory.preferences(patient_hash)
    # The arm is the *subject* — what the evidence is supposed to be about — and
    # the history is context around it. Passing them as one string made them
    # compete, and the history usually won: see `SemanticKnowledgeBase`.
    recommended = bundle["statistical_output"]["recommended_arm"]
    rag_query = " ".join([recommended, bundle["patient_context"].get("history_summary", "")])
    evidence = kb.retrieve(rag_query, subject=recommended)

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
