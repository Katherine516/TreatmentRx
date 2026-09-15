"""Immutable prognostic models trained outside the scored patient record."""

from __future__ import annotations

import math
from dataclasses import dataclass

from treatmentrx.biomarkers.contracts import (
    BiomarkerDomain,
    BiomarkerDomainError,
    BiomarkerInputError,
    BiomarkerLeakageError,
    BiomarkerRequest,
    BiomarkerRole,
    PrognosticScore,
)


def _normalise(value: str) -> str:
    return " ".join(value.strip().lower().split())


@dataclass(frozen=True)
class FrozenPrognosticModel:
    """A locked linear score with no training or treatment-selection method.

    The deliberately small implementation is an artifact interface, not a
    claim that biomarkers should be linear.  A complex external model can be
    wrapped behind the same contract after its preprocessing and validation are
    frozen.
    """

    artifact_id: str
    version: str
    training_population: str
    domain: BiomarkerDomain
    intercept: float
    coefficients: tuple[tuple[str, float], ...]
    calibration_intercept: float = 0.0
    calibration_slope: float = 1.0
    role: BiomarkerRole = BiomarkerRole.PROGNOSTIC_ADJUSTMENT

    def __post_init__(self) -> None:
        if self.role is not BiomarkerRole.PROGNOSTIC_ADJUSTMENT:
            raise BiomarkerDomainError(
                "FrozenPrognosticModel only authorizes prognostic adjustment; "
                "predictive modifiers require a separately validated artifact"
            )
        if not self.artifact_id.strip() or not self.version.strip():
            raise BiomarkerInputError("artifact_id and version may not be blank")
        if not self.training_population.strip():
            raise BiomarkerInputError("training_population may not be blank")
        names = [name for name, _ in self.coefficients]
        if not names or any(not name.strip() for name in names):
            raise BiomarkerInputError("at least one named coefficient is required")
        if len(names) != len(set(names)):
            raise BiomarkerInputError("biomarker coefficient names must be unique")
        numeric = [
            self.intercept,
            self.calibration_intercept,
            self.calibration_slope,
            *(coefficient for _, coefficient in self.coefficients),
        ]
        if not all(math.isfinite(float(value)) for value in numeric):
            raise BiomarkerInputError("biomarker parameters must be finite")
        if self.calibration_slope <= 0.0:
            raise BiomarkerInputError("calibration_slope must be positive")

    def score(
        self,
        features: dict[str, float],
        feature_days: dict[str, int],
        request: BiomarkerRequest,
    ) -> PrognosticScore:
        self._validate_domain(request)
        required = [name for name, _ in self.coefficients]
        missing = sorted(set(required) - set(features))
        if missing:
            raise BiomarkerInputError(
                "missing required biomarker features: " + ", ".join(missing)
            )
        missing_days = sorted(set(required) - set(feature_days))
        if missing_days:
            raise BiomarkerInputError(
                "missing feature timestamps: " + ", ".join(missing_days)
            )
        leaked = sorted(
            name for name in required if feature_days[name] > request.decision_day
        )
        if leaked:
            raise BiomarkerLeakageError(
                "post-decision biomarker features are prohibited: "
                + ", ".join(leaked)
            )
        values: dict[str, float] = {}
        for name in required:
            value = features[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise BiomarkerInputError(f"biomarker feature {name!r} is not numeric")
            value = float(value)
            if not math.isfinite(value):
                raise BiomarkerInputError(f"biomarker feature {name!r} is not finite")
            values[name] = value

        raw = self.intercept + sum(
            coefficient * values[name] for name, coefficient in self.coefficients
        )
        calibrated = self.calibration_intercept + self.calibration_slope * raw

        # The numeric half of "is this artifact valid here". The categorical
        # half already raised above, so this cannot be a restatement of it:
        # `within_validated_domain` was previously hard-coded True at this one
        # construction site and could not take any other value, which made every
        # consumer branch on it dead code. A frozen linear score applied outside
        # the range it was fit on is extrapolating, and that is the classic way
        # an external biomarker fails.
        outside = self.domain.out_of_range(values)
        return PrognosticScore(
            value=calibrated,
            artifact_id=self.artifact_id,
            artifact_version=self.version,
            role=self.role,
            endpoint=self.domain.endpoint,
            horizon_days=self.domain.horizon_days,
            reference_treatment=self.domain.reference_treatment,
            within_validated_domain=not outside,
            feature_days={name: feature_days[name] for name in required},
            out_of_range_features=outside,
        )

    def capability(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "version": self.version,
            "training_population": self.training_population,
            "role": self.role.value,
            "may_modify_treatment_effect": False,
            "domain": self.domain.as_dict(),
            "required_features": [name for name, _ in self.coefficients],
            # An artifact that declares no ranges is saying it does not know
            # where it was validated, which a consumer should be able to see
            # without inspecting the domain.
            "declares_validated_ranges": bool(self.domain.validated_ranges),
        }

    def _validate_domain(self, request: BiomarkerRequest) -> None:
        """Categorical domain membership — a mismatch here raises.

        Scoring a lung-CT artifact against a rheumatology record is not a
        borderline case, so this is a gate rather than a flag. The *numeric*
        question — are the feature values inside the range the artifact was fit
        on — is answered in `score` and reported through
        `PrognosticScore.within_validated_domain`, because an extreme but
        admissible patient is a caveat rather than a wrong artifact. Same
        two-tier split the main model uses between `data/contract` and
        `safety/rules._out_of_support`.
        """
        expected = self.domain
        mismatches = []
        for name in (
            "disease_id",
            "modality",
            "endpoint",
            "reference_treatment",
        ):
            if _normalise(str(getattr(request, name))) != _normalise(
                str(getattr(expected, name))
            ):
                mismatches.append(name)
        if request.horizon_days != expected.horizon_days:
            mismatches.append("horizon_days")
        if _normalise(request.site) not in {
            _normalise(site) for site in expected.eligible_sites
        }:
            mismatches.append("site")
        if _normalise(request.platform) not in {
            _normalise(platform) for platform in expected.eligible_platforms
        }:
            mismatches.append("platform")
        if mismatches:
            raise BiomarkerDomainError(
                "biomarker artifact is outside its validated domain: "
                + ", ".join(sorted(set(mismatches)))
            )


class BiomarkerRegistry:
    """Registered frozen artifacts; absence is unsupported, never a fallback."""

    def __init__(
        self, artifacts: tuple[FrozenPrognosticModel, ...] = ()
    ) -> None:
        ids = [artifact.artifact_id for artifact in artifacts]
        if len(ids) != len(set(ids)):
            raise BiomarkerInputError("biomarker artifact_id values must be unique")
        self._artifacts = {artifact.artifact_id: artifact for artifact in artifacts}

    def get(self, artifact_id: str) -> FrozenPrognosticModel:
        try:
            return self._artifacts[artifact_id]
        except KeyError:
            raise BiomarkerDomainError(
                f"no frozen biomarker artifact is registered for {artifact_id!r}"
            )

    def capabilities(self) -> list[dict[str, object]]:
        return [
            self._artifacts[key].capability() for key in sorted(self._artifacts)
        ]


__all__ = ["BiomarkerRegistry", "FrozenPrognosticModel"]
