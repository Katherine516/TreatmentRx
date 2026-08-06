"""The canonical RA treatment-arm vocabulary.

Every layer must name arms identically. When the data contract and the
estimators disagreed on a name, the v5 safety layer classified the estimators'
top arm as "not in the feasible set" and silently substituted the runner-up —
a recommendation downgrade with no clinical cause, reported to the clinician as
a safety action. One vocabulary, defined here, is what prevents that.

`MANUAL_REVIEW` is not a treatment: it is the escape value for a medication that
maps to no configured arm, and it is deliberately excluded from `TREATMENT_ARMS`
so nothing can ever score or recommend it.
"""

from __future__ import annotations

REFERENCE_ARM = "continue-current"
MANUAL_REVIEW = "manual-review"

TREATMENT_ARMS: tuple[str, ...] = (
    "continue-current",
    "methotrexate-optimization",
    "TNF-inhibitor",
    "IL-6 inhibitor",
    "JAK-inhibitor",
    "rituximab",
)

# Free-text medication tokens that map onto each arm, in match order. Ordering
# matters: a combination like "tocilizumab + methotrexate" should map to the
# advanced therapy that defines the line, not to its csDMARD anchor.
ARM_SYNONYMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("continue-current", ("current decision", "continue")),
    ("TNF-inhibitor", ("tnf", "adalimumab", "etanercept", "infliximab", "certolizumab", "golimumab")),
    ("IL-6 inhibitor", ("il-6", "il6", "tocilizumab", "sarilumab")),
    ("JAK-inhibitor", ("jak", "tofacitinib", "baricitinib", "upadacitinib")),
    ("rituximab", ("rituximab", "abatacept")),
    ("methotrexate-optimization", ("methotrexate", "mtx", "hydroxychloroquine", "sulfasalazine", "leflunomide")),
)


def normalize_arm(medication_name: str) -> str:
    """Map a free-text medication to a canonical arm, or to `MANUAL_REVIEW`."""
    name = medication_name.lower()
    for arm, tokens in ARM_SYNONYMS:
        if any(token in name for token in tokens):
            return arm
    return MANUAL_REVIEW
