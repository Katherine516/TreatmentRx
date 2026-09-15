"""Fail-closed contracts for externally trained biomarker predictors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class BiomarkerRole(str, Enum):
    PROGNOSTIC_ADJUSTMENT = "prognostic_adjustment"
    PREDICTIVE_MODIFIER = "predictive_modifier"


class BiomarkerError(ValueError):
    pass


class BiomarkerDomainError(BiomarkerError):
    pass


class BiomarkerInputError(BiomarkerError):
    pass


class BiomarkerLeakageError(BiomarkerError):
    pass


@dataclass(frozen=True)
class BiomarkerDomain:
    """Where a frozen artifact was validated — categorically *and* numerically.

    The categorical half (disease, modality, endpoint, horizon, site, platform)
    decides whether this is the right artifact at all, and a mismatch there
    **raises**: scoring a lung-CT model on a rheumatology record is not a
    borderline case.

    `validated_ranges` is the other half and it was missing. A frozen linear
    score applied to a CRP of 400 when it was fit on 0-50 is extrapolating, and
    nothing noticed — `PrognosticScore.within_validated_domain` was assigned
    `True` at its only construction site and could not take any other value, so
    a consumer branching on it was writing dead code.

    Ranges are optional per feature: an artifact that declares none for a
    covariate is saying it does not know, which is different from saying the
    covariate is unbounded, and `within_validated_domain` stays True for the
    ones it cannot judge. That mirrors the two-tier split the main model already
    uses — `data/contract.PLAUSIBLE_RANGES` raises on impossible values while
    `safety/rules._out_of_support` warns on extreme-but-possible ones.
    """

    disease_id: str
    modality: str
    endpoint: str
    horizon_days: int
    reference_treatment: str
    eligible_sites: tuple[str, ...]
    eligible_platforms: tuple[str, ...]
    #: feature name -> (low, high) the artifact was validated over. Declaring
    #: none leaves the numeric half unjudged rather than asserted.
    validated_ranges: tuple[tuple[str, tuple[float, float]], ...] = ()

    def __post_init__(self) -> None:
        string_fields = (
            self.disease_id,
            self.modality,
            self.endpoint,
            self.reference_treatment,
        )
        if any(not value.strip() for value in string_fields):
            raise BiomarkerDomainError("biomarker domain fields may not be blank")
        if self.horizon_days <= 0:
            raise BiomarkerDomainError("biomarker horizon_days must be positive")
        if not self.eligible_sites or not self.eligible_platforms:
            raise BiomarkerDomainError(
                "eligible_sites and eligible_platforms must be explicit"
            )
        names = [name for name, _ in self.validated_ranges]
        if len(names) != len(set(names)):
            raise BiomarkerDomainError("validated_ranges names must be unique")
        for name, bounds in self.validated_ranges:
            if not name.strip():
                raise BiomarkerDomainError("validated_ranges names may not be blank")
            low, high = bounds
            if not (low < high):
                raise BiomarkerDomainError(
                    f"validated range for {name!r} must have low < high"
                )

    def out_of_range(self, values: dict[str, float]) -> tuple[str, ...]:
        """Features whose value falls outside the artifact's validated range.

        Silent about features with no declared range — an absent range is a
        missing measurement, not a guarantee.
        """
        outside = []
        for name, (low, high) in self.validated_ranges:
            value = values.get(name)
            if value is None:
                continue
            if value < low or value > high:
                outside.append(name)
        return tuple(sorted(outside))

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["eligible_sites"] = list(self.eligible_sites)
        payload["eligible_platforms"] = list(self.eligible_platforms)
        payload["validated_ranges"] = {
            name: list(bounds) for name, bounds in self.validated_ranges
        }
        return payload


@dataclass(frozen=True)
class BiomarkerRequest:
    disease_id: str
    modality: str
    endpoint: str
    horizon_days: int
    reference_treatment: str
    site: str
    platform: str
    decision_day: int


@dataclass(frozen=True)
class PrognosticScore:
    value: float
    artifact_id: str
    artifact_version: str
    role: BiomarkerRole
    endpoint: str
    horizon_days: int
    reference_treatment: str
    #: True only when every feature carrying a declared range fell inside it.
    #: Categorical domain mismatches raise instead of landing here, so this is
    #: the *numeric* half of the question and nothing else. It used to be
    #: hard-coded True at the only place a score is built.
    within_validated_domain: bool
    feature_days: dict[str, int]
    #: Named so a consumer can say which feature left the range, rather than
    #: being handed a bare False.
    out_of_range_features: tuple[str, ...] = ()

    @property
    def may_modify_treatment_effect(self) -> bool:
        return self.role is BiomarkerRole.PREDICTIVE_MODIFIER

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["role"] = self.role.value
        payload["may_modify_treatment_effect"] = self.may_modify_treatment_effect
        payload["out_of_range_features"] = list(self.out_of_range_features)
        return payload


__all__ = [
    "BiomarkerDomain",
    "BiomarkerDomainError",
    "BiomarkerError",
    "BiomarkerInputError",
    "BiomarkerLeakageError",
    "BiomarkerRequest",
    "BiomarkerRole",
    "PrognosticScore",
]
