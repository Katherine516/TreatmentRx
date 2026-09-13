"""Frozen prognostic scores fail closed and never gain treatment authority."""

import unittest

from treatmentrx.biomarkers import (
    BiomarkerDomain,
    BiomarkerDomainError,
    BiomarkerInputError,
    BiomarkerLeakageError,
    BiomarkerRegistry,
    BiomarkerRequest,
    BiomarkerRole,
    FrozenPrognosticModel,
)


def model():
    return FrozenPrognosticModel(
        artifact_id="spatial-ra-prognosis",
        version="1.0",
        training_population="external RA cohort",
        domain=BiomarkerDomain(
            disease_id="rheumatoid_arthritis",
            modality="spatial_transcriptomics",
            endpoint="90-day response",
            horizon_days=90,
            reference_treatment="continue-current",
            eligible_sites=("site-a",),
            eligible_platforms=("visium-v2",),
        ),
        intercept=0.1,
        coefficients=(("immune_score", 0.4), ("stromal_score", -0.2)),
        calibration_intercept=0.05,
        calibration_slope=0.9,
    )


def request(**changes):
    values = {
        "disease_id": "rheumatoid_arthritis",
        "modality": "spatial_transcriptomics",
        "endpoint": "90-day response",
        "horizon_days": 90,
        "reference_treatment": "continue-current",
        "site": "site-a",
        "platform": "visium-v2",
        "decision_day": 100,
    }
    values.update(changes)
    return BiomarkerRequest(**values)


class FrozenBiomarkerTests(unittest.TestCase):
    def test_valid_score_is_prognostic_only(self):
        score = model().score(
            {"immune_score": 2.0, "stromal_score": 1.0},
            {"immune_score": 90, "stromal_score": 95},
            request(),
        )
        self.assertAlmostEqual(score.value, 0.68)
        self.assertEqual(score.role, BiomarkerRole.PROGNOSTIC_ADJUSTMENT)
        self.assertFalse(score.may_modify_treatment_effect)
        self.assertTrue(score.within_validated_domain)

    def test_post_decision_feature_is_a_hard_failure(self):
        with self.assertRaises(BiomarkerLeakageError):
            model().score(
                {"immune_score": 2.0, "stromal_score": 1.0},
                {"immune_score": 101, "stromal_score": 95},
                request(),
            )

    def test_unvalidated_site_is_not_silently_accepted(self):
        with self.assertRaisesRegex(BiomarkerDomainError, "site"):
            model().score(
                {"immune_score": 2.0, "stromal_score": 1.0},
                {"immune_score": 90, "stromal_score": 95},
                request(site="site-b"),
            )

    def test_missing_feature_is_not_imputed(self):
        with self.assertRaises(BiomarkerInputError):
            model().score(
                {"immune_score": 2.0}, {"immune_score": 90}, request()
            )

    def test_prognostic_artifact_cannot_claim_predictive_authority(self):
        base = model()
        with self.assertRaisesRegex(BiomarkerDomainError, "predictive"):
            FrozenPrognosticModel(
                artifact_id=base.artifact_id,
                version=base.version,
                training_population=base.training_population,
                domain=base.domain,
                intercept=base.intercept,
                coefficients=base.coefficients,
                role=BiomarkerRole.PREDICTIVE_MODIFIER,
            )

    def test_registry_has_no_fallback(self):
        registry = BiomarkerRegistry((model(),))
        self.assertEqual(registry.get("spatial-ra-prognosis").version, "1.0")
        with self.assertRaises(BiomarkerDomainError):
            registry.get("unknown")


if __name__ == "__main__":
    unittest.main()
