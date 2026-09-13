"""Units for the observations the estimators condition on.

The adapter used to read `valueQuantity.unit`, store it, and never look at it
again. Every range check, every plausible-value gate and every basis term then
assumed a canonical unit that nothing enforced — so a CRP reported in mg/dL, a
perfectly ordinary convention, entered `crp_std = (crp - 30) / 25` an order of
magnitude too small and moved the recommendation with nothing to show for it.
That is the same failure mode as invariant 13's implausible values, one level
further out: the number looks fine, and only the label says otherwise.

Three outcomes, in decreasing order of confidence:

* **Known equivalent** — converted to the canonical unit and recorded, so the
  conversion appears in the contract report rather than happening invisibly.
* **Known incompatible** — a mass where a concentration belongs. That is a
  corrupt record and the contract rejects it.
* **Unknown or absent** — assumed canonical and flagged. Assuming is the only
  thing that can be done with an unlabelled number, and saying so is the
  difference between a documented assumption and a silent one.

The table is deliberately small. It covers the conventions that actually appear
in RA data and the confusions that actually happen; it is not a units library.
"""

from __future__ import annotations

from dataclasses import dataclass

# Canonical unit per observation code, and the factors that reach it. A factor
# multiplies the reported value: CRP in mg/dL is ten times the same
# concentration in mg/L, so 2.8 mg/dL becomes 28 mg/L.
#
# Keys are normalised the same way observation codes are — lowercased, with `/`
# and spaces and punctuation stripped — so "mg/L", "mg/l" and "MG / L" agree.
CANONICAL_UNITS: dict[str, str] = {
    "crp": "mg/L",
    "esr": "mm/hr",
    "egfr": "mL/min/1.73m2",
    "alt": "U/L",
    "ast": "U/L",
    "das28": "score",
    "haq_di": "score",
}

CONVERSIONS: dict[str, dict[str, float]] = {
    "crp": {
        "mgl": 1.0,
        "milligramsperliter": 1.0,
        "mgdl": 10.0,      # the one that actually bites
        "milligramsperdeciliter": 10.0,
        "gl": 1000.0,
        "nmoll": 0.0001051,  # rarely reported, included because it is unambiguous
    },
    "esr": {"mmhr": 1.0, "mmh": 1.0, "mm1hr": 1.0, "millimetersperhour": 1.0},
    "egfr": {
        "mlmin173m2": 1.0,
        "mlmin1732": 1.0,
        "mlmin": 1.0,  # almost always already indexed to BSA despite the label
        "millilitersperminute": 1.0,
    },
    "alt": {"ul": 1.0, "iul": 1.0, "unitsperliter": 1.0, "ukatl": 60.0},
    "ast": {"ul": 1.0, "iul": 1.0, "unitsperliter": 1.0, "ukatl": 60.0},
    # Scores are dimensionless. The empty string is how an omitted unit arrives.
    "das28": {"score": 1.0, "": 1.0, "1": 1.0, "unitless": 1.0, "points": 1.0},
    "haq_di": {"score": 1.0, "": 1.0, "1": 1.0, "unitless": 1.0, "points": 1.0},
}

# Units that are recognisable but cannot be what this code means. Listing them
# is what turns "I do not know this unit" into "this record is wrong": an ALT in
# mg/dL is not an enzyme activity, and converting it is not possible.
INCOMPATIBLE: dict[str, tuple[str, ...]] = {
    "crp": ("ul", "iul", "mmhr", "score"),
    "esr": ("mgl", "mgdl", "ul", "score"),
    "egfr": ("mgl", "mgdl", "ul", "score"),
    "alt": ("mgl", "mgdl", "mmhr", "score"),
    "ast": ("mgl", "mgdl", "mmhr", "score"),
    "das28": ("mgl", "mgdl", "ul", "mmhr"),
    "haq_di": ("mgl", "mgdl", "ul", "mmhr"),
}


@dataclass(frozen=True)
class UnitCheck:
    """What happened to one observation's unit."""

    code: str
    reported: str | None
    canonical: str
    factor: float
    status: str  # "canonical" | "converted" | "incompatible" | "unrecognised" | "absent"

    @property
    def message(self) -> str:
        if self.status == "converted":
            return (
                f"{self.code} reported in {self.reported!r}; converted to "
                f"{self.canonical} (x{self.factor:g})."
            )
        if self.status == "incompatible":
            return (
                f"{self.code} reported in {self.reported!r}, which is not a "
                f"{self.canonical} quantity — the record is not usable as written."
            )
        if self.status == "unrecognised":
            return (
                f"{self.code} reported in {self.reported!r}, which is not a unit "
                f"this build recognises; assuming {self.canonical} and modelling "
                f"the value as given."
            )
        if self.status == "absent":
            return (
                f"{self.code} carries no unit; assuming {self.canonical}. An "
                f"unlabelled number cannot be checked."
            )
        return f"{self.code} is in {self.canonical}."


def normalise_unit(unit: str | None) -> str:
    """Strip a unit string to a comparison key.

    `mg/L`, `mg / l` and `MG.L` are the same unit written three ways, and a
    lookup table that distinguishes them is a lookup table that misses.
    """
    if unit is None:
        return ""
    return "".join(character for character in unit.lower() if character.isalnum())


def check(code: str, unit: str | None) -> UnitCheck | None:
    """Classify a reported unit for `code`, or None if the code is unitless here."""
    canonical = CANONICAL_UNITS.get(code)
    if canonical is None:
        return None
    key = normalise_unit(unit)
    factors = CONVERSIONS.get(code, {})

    if unit is None or key == "":
        # A score with no unit is canonical; a concentration with no unit is an
        # assumption, and the two should not read the same in the report.
        status = "canonical" if key in factors else "absent"
        return UnitCheck(code, unit, canonical, 1.0, status)
    if key in INCOMPATIBLE.get(code, ()):
        return UnitCheck(code, unit, canonical, 1.0, "incompatible")
    factor = factors.get(key)
    if factor is None:
        return UnitCheck(code, unit, canonical, 1.0, "unrecognised")
    return UnitCheck(
        code, unit, canonical, factor, "canonical" if factor == 1.0 else "converted"
    )


def convert(code: str, value, unit: str | None):
    """The value in canonical units, plus what was done to get there.

    Non-numeric and boolean values pass through untouched — the contract has its
    own complaint about those, and multiplying a string is not an improvement.
    """
    result = check(code, unit)
    if result is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return value, result
    if result.status == "converted":
        return float(value) * result.factor, result
    return value, result


__all__ = [
    "CANONICAL_UNITS",
    "CONVERSIONS",
    "INCOMPATIBLE",
    "UnitCheck",
    "check",
    "convert",
    "normalise_unit",
]
