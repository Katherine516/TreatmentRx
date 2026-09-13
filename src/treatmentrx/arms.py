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
CURRENT_DECISION_TOKEN = "current decision"

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


# Medications given *alongside* a DMARD line rather than as one. A steroid
# bridge is the standard RA example: it is prescribed while a slow-acting agent
# takes effect, and it is not a change of treatment line.
#
# This distinction was missing and it was not cosmetic. Every `MedicationRequest`
# became a stage, so a prednisone taper recorded mid-line produced a **phantom
# decision point** — the agent read it as the clinician switching to an arm that
# maps to `manual-review`, renumbered every later stage, and truncated the real
# DMARD line's window to end at the steroid's start day.
CONCOMITANT_TOKENS: tuple[str, ...] = (
    "prednis",       # prednisone, prednisolone
    "methylpred",
    "dexamethasone",
    "hydrocortisone",
    "steroid",
    "glucocorticoid",
    "depo-medrol",
    "kenalog",
    "triamcinolone",
)

# Concomitant medications that specifically indicate rescue therapy — a burst
# given because the line is failing, which is what `SwitchingRecord.rescue_therapy`
# is meant to capture.
RESCUE_TOKENS: tuple[str, ...] = CONCOMITANT_TOKENS + ("rescue", "bridge", "burst")


def normalize_arm(medication_name: str) -> str:
    """Map a free-text medication to a canonical arm, or to `MANUAL_REVIEW`."""
    name = medication_name.lower()
    for arm, tokens in ARM_SYNONYMS:
        if any(token in name for token in tokens):
            return arm
    return MANUAL_REVIEW


def is_concomitant(medication_name: str) -> bool:
    """Is this given alongside a line rather than as one?

    Deliberately conservative: a medication only counts as concomitant if it
    maps to no arm *and* matches a known token. An unrecognised DMARD — a
    biologic newer than `ARM_SYNONYMS` — stays in the line sequence and surfaces
    as `manual-review`, which is the existing safe behaviour and the one a
    clinician needs to see.
    """
    if normalize_arm(medication_name) != MANUAL_REVIEW:
        return False
    name = medication_name.lower()
    return any(token in name for token in CONCOMITANT_TOKENS)


def is_rescue(medication_name: str) -> bool:
    """Does this concomitant medication read as a rescue or bridge?"""
    name = medication_name.lower()
    return any(token in name for token in RESCUE_TOKENS)


def is_current_decision(medication_name: str) -> bool:
    """Whether this event explicitly marks the visit being decided now.

    A generic ``continue`` order is a treatment choice, not necessarily a request
    for a new recommendation.  The inference contract therefore requires the
    explicit marker documented in ``docs/INPUT_DATA.md``.
    """
    return CURRENT_DECISION_TOKEN in medication_name.lower()
