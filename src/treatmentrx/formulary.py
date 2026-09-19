"""The molecules the agent may offer, and the hazards they carry.

`arms.py` is the canonical *arm* vocabulary and invariant 3 says the data
contract, the estimators, the composite action space and the feasible-set filter
must all agree on it. They do. What they did not agree on is the layer below:
**which molecules belong to an arm, and which hazards they carry** was declared
in four places.

    arms.ARM_SYNONYMS              what a medication history might say
    estimation.actions.ARM_CANDIDATES   what the agent may propose
    safety.feasible_set.{JAK_DRUGS, HEPATOTOXIC_DRUGS}   composite-level hazards
    safety.rules.HEPATOTOXIC_TOKENS     arm-level hazards

Measured, those four agree on hazard *classification* for every arm — both
safety modules class `methotrexate-optimization` and `JAK-inhibitor` as
ALT-gated and the other three as not — but the agreement is a coincidence of
curation, not a guarantee, and their token lists are not subsets of one another
(`mtx` only in one, the three JAK molecules only in the other).

And on *membership* they had already drifted. `arms.py` recognises three JAK
molecules and `feasible_set.py` hazard-classes the same three; `actions.py`
offered **one**. Nothing caught it, because nothing compared them.

**Three vocabularies, deliberately different.** Collapsing them would be wrong:

* *Recognition* is the broadest — a history may name a drug the agent would
  never propose, and `normalize_arm` must still place it. `ARM_SYNONYMS` owns
  this and keeps class tokens (`tnf`, `jak`) that are not molecules at all.
* *Offer* is a curated subset. Which molecules a service is willing to propose
  is a formulary decision, not a modelling one.
* *Hazard* has to cover everything offerable and should cover everything
  recognisable, and its match tokens are spellings rather than molecules —
  `mtx` is needed because a composite carries `combination="MTX"`.

This module owns the last two and states their relationship to the first, so
`tests/test_formulary.py` can check what was previously left to four literals
staying in sync by hand.

**The curation here is illustrative and every entry says so.** Nothing in this
file is sourced from a guideline or a formulary; `provenance` carries that per
molecule rather than in a comment at the top, because a reader inspecting one
entry should not have to find the header to learn it is a placeholder. Replacing
it is a clinical task, and `cli audit` reports what the current breadth costs so
that task starts from a number.
"""

from __future__ import annotations

from dataclasses import dataclass

from treatmentrx.arms import REFERENCE_ARM, TREATMENT_ARMS, normalize_arm
from treatmentrx.domain import CompositeAction

#: Bumped when the curation changes, not when this module is refactored. It is
#: reported on the model card so a served recommendation names the formulary it
#: was drawn from.
VERSION = "ra-illustrative-2026Q1"

#: Hazard classes. These are the *reasons* a molecule can be withdrawn, and the
#: safety layer derives its match tokens from them rather than keeping its own
#: list. A class with no member here is not a class the filter can apply.
HAZARD_JAK = "jak-class"
HAZARD_HEPATOTOXIC = "hepatotoxic"
HAZARD_TERATOGENIC = "teratogenic"

_PLACEHOLDER = "illustrative curation, not a clinical source"


@dataclass(frozen=True)
class Molecule:
    """One drug, the arm it belongs to, and what it is dangerous for.

    `offerable` is the distinction that makes this file worth having. A molecule
    can be known to the hazard classes and absent from the menu — `leflunomide`
    is exactly that: recognised by `arms.py`, hepatotoxic, and never proposed.
    Before this it was a bare string in a tuple with nothing saying which of
    those it was.
    """

    name: str
    arm: str
    hazards: frozenset
    provenance: str
    #: Spellings the safety filter must match, beyond `name`. `mtx` exists
    #: because composites carry `combination="MTX"` and the filter substring
    #: matches that field.
    aliases: tuple = ()
    #: Whether the agent may propose it. False means "known, never offered".
    offerable: bool = True
    #: The regimens offered for this molecule. Empty for a non-offerable one.
    regimens: tuple = ()

    @property
    def match_tokens(self) -> tuple:
        return (self.name,) + tuple(self.aliases)


MOLECULES: tuple = (
    Molecule(
        name="methotrexate",
        arm="methotrexate-optimization",
        hazards=frozenset({HAZARD_HEPATOTOXIC, HAZARD_TERATOGENIC}),
        aliases=("mtx",),
        provenance=_PLACEHOLDER,
        regimens=(
            CompositeAction(drug="methotrexate", dose="25mg", route="PO", timing="weekly"),
            CompositeAction(drug="methotrexate", dose="25mg", route="SC", timing="weekly"),
        ),
    ),
    Molecule(
        name="leflunomide",
        arm="methotrexate-optimization",
        hazards=frozenset({HAZARD_HEPATOTOXIC, HAZARD_TERATOGENIC}),
        provenance=f"{_PLACEHOLDER}; hazard-classed but never proposed",
        offerable=False,
    ),
    Molecule(
        name="adalimumab",
        arm="TNF-inhibitor",
        hazards=frozenset(),
        provenance=_PLACEHOLDER,
        regimens=(
            CompositeAction(drug="adalimumab", dose="40mg", route="SC", timing="q2wk", combination="MTX"),
            CompositeAction(drug="adalimumab", dose="40mg", route="SC", timing="q2wk"),
        ),
    ),
    Molecule(
        name="etanercept",
        arm="TNF-inhibitor",
        hazards=frozenset(),
        provenance=_PLACEHOLDER,
        regimens=(
            CompositeAction(drug="etanercept", dose="50mg", route="SC", timing="weekly", combination="MTX"),
            CompositeAction(drug="etanercept", dose="50mg", route="SC", timing="weekly"),
        ),
    ),
    Molecule(
        name="tocilizumab",
        arm="IL-6 inhibitor",
        hazards=frozenset(),
        provenance=_PLACEHOLDER,
        regimens=(
            CompositeAction(drug="tocilizumab", dose="8mg/kg", route="IV", timing="q4wk", combination="MTX"),
            CompositeAction(drug="tocilizumab", dose="162mg", route="SC", timing="weekly"),
        ),
    ),
    Molecule(
        name="sarilumab",
        arm="IL-6 inhibitor",
        hazards=frozenset(),
        provenance=_PLACEHOLDER,
        regimens=(
            CompositeAction(drug="sarilumab", dose="200mg", route="SC", timing="q2wk"),
        ),
    ),
    Molecule(
        name="upadacitinib",
        arm="JAK-inhibitor",
        # Hepatotoxic as well as JAK-class: `safety/rules.py` already classed the
        # JAK arm that way for its delayed-toxicity warning, and the composite
        # filter reaches the same ALT ceiling through its JAK branch. Declaring
        # it once here is what lets the two agree by construction.
        hazards=frozenset({HAZARD_JAK, HAZARD_HEPATOTOXIC, HAZARD_TERATOGENIC}),
        provenance=_PLACEHOLDER,
        regimens=(
            CompositeAction(drug="upadacitinib", dose="15mg", route="PO", timing="daily"),
        ),
    ),
    Molecule(
        name="tofacitinib",
        arm="JAK-inhibitor",
        hazards=frozenset({HAZARD_JAK, HAZARD_HEPATOTOXIC, HAZARD_TERATOGENIC}),
        provenance=_PLACEHOLDER,
        regimens=(
            CompositeAction(drug="tofacitinib", dose="5mg", route="PO", timing="BD"),
        ),
    ),
    Molecule(
        name="baricitinib",
        arm="JAK-inhibitor",
        hazards=frozenset({HAZARD_JAK, HAZARD_HEPATOTOXIC, HAZARD_TERATOGENIC}),
        provenance=_PLACEHOLDER,
        regimens=(
            CompositeAction(drug="baricitinib", dose="4mg", route="PO", timing="daily"),
        ),
    ),
    Molecule(
        name="rituximab",
        arm="rituximab",
        hazards=frozenset(),
        provenance=_PLACEHOLDER,
        regimens=(
            CompositeAction(drug="rituximab", dose="1000mg", route="IV", timing="x2 q2wk", combination="MTX"),
            CompositeAction(drug="rituximab", dose="1000mg", route="IV", timing="x2 q2wk"),
        ),
    ),
)


def molecules_for(arm: str) -> tuple:
    """Every molecule declared for an arm, offerable or not."""
    return tuple(molecule for molecule in MOLECULES if molecule.arm == arm)


def offerable_for(arm: str) -> tuple:
    """The molecules the agent may propose for an arm."""
    return tuple(molecule for molecule in molecules_for(arm) if molecule.offerable)


def hazard_tokens(hazard: str) -> tuple:
    """Match tokens for a hazard class, spanning non-offerable molecules too.

    The filter has to recognise a hazard wherever it appears — in a composite's
    drug, or in the background it is combined with — so a molecule the agent
    never proposes still contributes its spellings.
    """
    tokens: list = []
    for molecule in MOLECULES:
        if hazard in molecule.hazards:
            tokens.extend(molecule.match_tokens)
    return tuple(dict.fromkeys(tokens))


def is_molecule_named(arm: str) -> bool:
    """Is the arm named after one of its own molecules?

    This decides whether a within-class alternative is even a coherent idea.
    `rituximab` and `methotrexate-optimization` are named for the drug they are;
    swapping the molecule would make them a different arm, so they are
    single-molecule by definition and a menu cannot widen them. `TNF-inhibitor`,
    `IL-6 inhibitor` and `JAK-inhibitor` name a *class*, and a class has members.

    Derived from the arm name rather than declared, so adding an arm cannot
    forget to say which kind it is. `arms.ARM_SYNONYMS` recognises
    `hydroxychloroquine` under the methotrexate arm and `abatacept` under
    rituximab — for reading a history, where "some csDMARD" and "some non-TNF
    advanced therapy" is the right granularity. Neither is a substitute *within*
    the arm's meaning, which is why the recognition vocabulary stays separate
    from this one.
    """
    return any(molecule.name in arm.lower() for molecule in molecules_for(arm))


def arms_with_hazard(hazard: str) -> tuple:
    """Arms carrying at least one molecule with this hazard.

    `safety/rules.py` asks an arm-level question — "is this arm hepatotoxic" —
    and answered it by substring-matching a list of *molecule* spellings against
    an arm name. Four of its six tokens could never match, because no arm name
    contains `tofacitinib`, `baricitinib`, `upadacitinib` or `leflunomide`; only
    `methotrexate` and `jak` did any work. An arm-level question wants arm-level
    membership, and that is derivable from the molecules rather than restated.
    """
    return tuple(
        arm
        for arm in TREATMENT_ARMS
        if any(hazard in molecule.hazards for molecule in molecules_for(arm))
    )


def composites_for(arm: str) -> list:
    """The curated regimens for an arm, in declaration order.

    `continue-current` is the reference arm and carries no molecule: staying on
    the current line is a decision, not a prescription, so it is built here
    rather than declared as a drug.
    """
    if arm == REFERENCE_ARM:
        return [CompositeAction(drug=REFERENCE_ARM, stop_continue="continue")]
    regimens: list = []
    for molecule in offerable_for(arm):
        regimens.extend(molecule.regimens)
    return regimens


def arm_candidates() -> dict:
    """The whole menu, keyed by arm — what `ARM_CANDIDATES` used to be."""
    return {arm: composites_for(arm) for arm in TREATMENT_ARMS}


def breadth() -> dict:
    """Per arm: what is recognised, what is offered, and what a single allergy costs.

    The narrowness of the offer list is not a modelling property and it is not
    inherent — it is a curation choice, and this is the number that prices it. An
    arm offered as one molecule is an arm a single drug allergy removes outright,
    while an arm offered as two survives on the other. Reported rather than
    tuned: widening the menu means writing regimens for molecules this file
    declares but does not propose, which is a clinical task.
    """
    summary: dict = {}
    for arm in TREATMENT_ARMS:
        if arm == REFERENCE_ARM:
            continue
        declared = molecules_for(arm)
        offered = offerable_for(arm)
        molecule_named = is_molecule_named(arm)
        summary[arm] = {
            "declared_molecules": [molecule.name for molecule in declared],
            "offerable_molecules": [molecule.name for molecule in offered],
            "composites": len(composites_for(arm)),
            # One molecule means one allergy takes the arm; two means it survives.
            "survives_a_single_molecule_allergy": len(offered) > 1,
            # Whether widening is even coherent. Reporting "1 of 5 arms survive"
            # without this reads as three-fifths of a gap, when two of those arms
            # are named for their only molecule and cannot be widened at all.
            "named_after_its_molecule": molecule_named,
            "widening_is_possible": not molecule_named,
        }
    return summary


def unrecognised_offerables() -> tuple:
    """Offerable molecules `arms.normalize_arm` would not place on their own arm.

    The agent must be able to read back a line it proposed. A molecule it offers
    but cannot recognise would arrive in the next visit's history as
    `manual-review`, and the trajectory would record a decision the agent never
    made.
    """
    return tuple(
        molecule.name
        for molecule in MOLECULES
        if molecule.offerable and normalize_arm(molecule.name) != molecule.arm
    )


__all__ = [
    "HAZARD_HEPATOTOXIC",
    "HAZARD_JAK",
    "HAZARD_TERATOGENIC",
    "MOLECULES",
    "VERSION",
    "Molecule",
    "arm_candidates",
    "arms_with_hazard",
    "breadth",
    "composites_for",
    "hazard_tokens",
    "is_molecule_named",
    "molecules_for",
    "offerable_for",
    "unrecognised_offerables",
]
