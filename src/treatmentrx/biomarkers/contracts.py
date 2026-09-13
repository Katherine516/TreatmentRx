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
    disease_id: str
    modality: str
    endpoint: str
    horizon_days: int
    reference_treatment: str
    eligible_sites: tuple[str, ...]
    eligible_platforms: tuple[str, ...]

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

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["eligible_sites"] = list(self.eligible_sites)
        payload["eligible_platforms"] = list(self.eligible_platforms)
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
    within_validated_domain: bool
    feature_days: dict[str, int]

    @property
    def may_modify_treatment_effect(self) -> bool:
        return self.role is BiomarkerRole.PREDICTIVE_MODIFIER

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["role"] = self.role.value
        payload["may_modify_treatment_effect"] = self.may_modify_treatment_effect
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
