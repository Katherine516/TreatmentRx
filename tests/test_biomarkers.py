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



class ValidatedRangeTests(unittest.TestCase):
    """`within_validated_domain` was hard-coded True and could not be anything else.

    Categorical domain mismatches *raise*, so the field was a restatement of
    "you got a score at all" and any consumer branching on it wrote dead code.
    The numeric half — is the patient inside the range the artifact was fit on —
    was missing, and that is the classic way an external biomarker fails.
    """

    @staticmethod
    def _domain(ranges=()):
        return BiomarkerDomain(
            "rheumatoid_arthritis", "serum", "stage response", 90,
            "continue-current", ("site-a",), ("platform-x",),
            validated_ranges=ranges,
        )

    @staticmethod
    def _request():
        return BiomarkerRequest(
            "rheumatoid_arthritis", "serum", "stage response", 90,
            "continue-current", "site-a", "platform-x", 30,
        )

    def _model(self, ranges=()):
        return FrozenPrognosticModel(
            "ra-crp-v1", "1", "derivation cohort", self._domain(ranges),
            0.1, (("crp", 0.01),),
        )

    def test_an_in_range_patient_is_inside_the_domain(self):
        score = self._model((("crp", (0.0, 50.0)),)).score(
            {"crp": 20.0}, {"crp": 10}, self._request()
        )
        self.assertTrue(score.within_validated_domain)
        self.assertEqual(score.out_of_range_features, ())

    def test_an_out_of_range_patient_is_flagged_and_named(self):
        """A bare False would not tell a reader which feature left the range."""
        score = self._model((("crp", (0.0, 50.0)),)).score(
            {"crp": 400.0}, {"crp": 10}, self._request()
        )
        self.assertFalse(score.within_validated_domain)
        self.assertEqual(score.out_of_range_features, ("crp",))

    def test_the_field_can_take_both_values(self):
        """The property the old hard-coded True could not have."""
        model = self._model((("crp", (0.0, 50.0)),))
        inside = model.score({"crp": 20.0}, {"crp": 10}, self._request())
        outside = model.score({"crp": 400.0}, {"crp": 10}, self._request())
        self.assertNotEqual(
            inside.within_validated_domain, outside.within_validated_domain
        )

    def test_an_undeclared_range_is_unjudged_rather_than_asserted(self):
        """No declared range means the artifact does not know, not that it is safe."""
        score = self._model().score({"crp": 400.0}, {"crp": 10}, self._request())
        self.assertTrue(score.within_validated_domain)
        self.assertEqual(score.out_of_range_features, ())
        self.assertFalse(self._model().capability()["declares_validated_ranges"])

    def test_a_categorical_mismatch_still_raises(self):
        """Wrong artifact entirely is a gate, not a caveat — the two-tier split."""
        wrong = BiomarkerRequest(
            "rheumatoid_arthritis", "imaging", "stage response", 90,
            "continue-current", "site-a", "platform-x", 30,
        )
        with self.assertRaises(BiomarkerDomainError):
            self._model((("crp", (0.0, 50.0)),)).score({"crp": 20.0}, {"crp": 10}, wrong)

    def test_an_inverted_range_is_refused(self):
        with self.assertRaises(BiomarkerDomainError):
            self._domain((("crp", (50.0, 0.0)),))

if __name__ == "__main__":
    unittest.main()
