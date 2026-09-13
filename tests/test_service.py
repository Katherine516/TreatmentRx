"""The HTTP surface: routing, status codes, and what must never leak.

The service adds no clinical logic — the orchestrator already decided. What it
can get wrong is everything around that: a malformed record returning 500 instead
of a reason, a traceback carrying field values to a caller, a default bind
address that puts an unauthenticated decision-support endpoint on the network.
"""

import contextlib
import copy
import io
import json
import threading
import time
import unittest
import urllib.error
import urllib.request

from treatmentrx.demo_data import sample_ra_bundle
from treatmentrx.service import (
    DEFAULT_HOST,
    DISCLAIMER,
    MAX_BODY_BYTES,
    RecommendationService,
    build_server,
    exposure_warning,
)


class ServiceRouteTests(unittest.TestCase):
    """Driven over a real socket — the point is the transport, not the agent."""

    @classmethod
    def setUpClass(cls):
        cls.server = build_server(port=0)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        # The access log is a real feature and stays on; it just belongs
        # somewhere other than the middle of the suite's output, where fourteen
        # request lines make an actual failure harder to see. unittest records
        # failures and prints them after tearDownClass, so nothing is lost.
        cls._stderr = contextlib.redirect_stderr(io.StringIO())
        cls._stderr.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls._stderr.__exit__(None, None, None)

    def call(self, method, path, body=None, raw=None):
        data = raw if raw is not None else (
            json.dumps(body).encode() if body is not None else None
        )
        request = urllib.request.Request(
            self.base + path, data=data, method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, json.load(error)

    def test_health_reports_whether_the_models_are_fitted(self):
        status, payload = self.call("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["models_fitted"], "build_server should warm the fit")
        self.assertEqual(payload["supported_diseases"], ["rheumatoid_arthritis"])

    def test_capabilities_are_explicit_and_have_no_fallback(self):
        status, payload = self.call("GET", "/capabilities")
        self.assertEqual(status, 200)
        self.assertFalse(payload["fallback_to_other_disease_model"])
        self.assertEqual(
            [item["disease_id"] for item in payload["supported_diseases"]],
            ["rheumatoid_arthritis"],
        )
        self.assertIn(
            "dtr_research",
            payload["supported_diseases"][0]["operating_modes"],
        )
        self.assertFalse(
            payload["analysis_services"]["randomized_trial_precision"]
            ["can_recommend_individual_treatment"]
        )

    def test_biomarker_registry_is_explicitly_empty(self):
        status, payload = self.call("GET", "/biomarkers")
        self.assertEqual(status, 200)
        self.assertEqual(payload["artifacts"], [])
        self.assertFalse(payload["fallback_to_unvalidated_artifact"])

    def test_randomized_trial_precision_is_a_separate_non_recommending_route(self):
        records = []
        for index in range(40):
            treatment = index % 2
            score = (index - 20) / 8
            records.append(
                {
                    "outcome": 0.3 * treatment + 0.7 * score + (index % 3) / 10,
                    "treatment": treatment,
                    "prognostic_score": score,
                }
            )
        status, payload = self.call(
            "POST",
            "/trial/precision",
            {
                "records": records,
                "outcome": "365-day response",
                "horizon_days": 365,
                "planning_effect": 0.3,
                "prognostic_artifact": {
                    "artifact_id": "external-score",
                    "artifact_version": "1.0",
                    "endpoint": "365-day response",
                    "horizon_days": 365,
                    "reference_treatment": "control",
                    "independence_strategy": "external_frozen",
                    "externally_validated": True,
                },
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["operating_mode"], "randomized_trial")
        self.assertFalse(payload["can_recommend_individual_treatment"])
        self.assertIn("prognostic_adjusted", payload)

    def test_invalid_trial_request_is_422_not_a_patient_error(self):
        status, payload = self.call(
            "POST", "/trial/precision", {"records": []}
        )
        self.assertEqual(status, 422)
        self.assertFalse(payload["can_recommend_individual_treatment"])

    def test_trial_power_route_reports_monte_carlo_error(self):
        status, payload = self.call(
            "POST",
            "/trial/power",
            {
                "sample_size": 40,
                "treatment_effect": 0.0,
                "prognostic_coefficient": 1.0,
                "error_sd": 1.0,
                "replicates": 20,
                "seed": 3,
            },
        )
        self.assertEqual(status, 200)
        self.assertIn("monte_carlo_standard_error", payload)
        self.assertFalse(payload["can_recommend_individual_treatment"])

    def test_a_recommendation_round_trips(self):
        status, payload = self.call("POST", "/recommend", sample_ra_bundle())
        self.assertEqual(status, 200)
        self.assertIn(payload["status"], {"recommend", "equipoise", "review", "blocked"})
        self.assertIn("clinician_card", payload)
        self.assertIn("audit_event", payload)

    def test_an_unsupported_disease_is_422_with_the_supported_menu(self):
        """Not 400 and not 500: the request was well-formed and the content is
        not, which is a different thing for a caller to act on."""
        bundle = {
            "resourceType": "Bundle",
            "entry": [
                {"resource": {"resourceType": "Patient", "id": "x"}},
                {"resource": {"resourceType": "Condition", "code": {"text": "Asthma"}}},
            ],
        }
        status, payload = self.call("POST", "/recommend", bundle)
        self.assertEqual(status, 422)
        self.assertEqual(payload["error"], "unsupported disease")
        self.assertEqual(payload["disease"], "Asthma")
        self.assertEqual(payload["supported_diseases"], ["rheumatoid_arthritis"])

    def test_an_implausible_value_is_reported_not_modelled(self):
        bundle = copy.deepcopy(sample_ra_bundle())
        bundle["entry"].append(
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"text": "DAS28"},
                    "valueQuantity": {"value": -5, "unit": "score"},
                    "effectiveDay": 365,
                }
            }
        )
        status, payload = self.call("POST", "/recommend", bundle)
        self.assertEqual(status, 422)
        self.assertTrue(any("das28" in issue["field"] for issue in payload["issues"]))

    def test_malformed_json_is_400(self):
        status, payload = self.call("POST", "/recommend", raw=b"{not json")
        self.assertEqual(status, 400)
        self.assertIn("JSON", payload["error"])

    def test_a_non_object_body_is_rejected(self):
        status, payload = self.call("POST", "/recommend", body=[1, 2, 3])
        self.assertEqual(status, 400)
        self.assertIn("Bundle", payload["error"])

    def test_an_empty_body_is_400(self):
        status, payload = self.call("POST", "/recommend", raw=b"")
        self.assertEqual(status, 400)

    def test_unknown_routes_and_methods_are_404(self):
        self.assertEqual(self.call("GET", "/nope")[0], 404)
        self.assertEqual(self.call("POST", "/model", {})[0], 404)

    def test_every_response_carries_the_disclaimer(self):
        for method, path, body in (
            ("GET", "/health", None),
            ("GET", "/capabilities", None),
            ("GET", "/model", None),
            ("GET", "/biomarkers", None),
            ("GET", "/", None),
            ("POST", "/recommend", sample_ra_bundle()),
        ):
            with self.subTest(path=path):
                _, payload = self.call(method, path, body)
                self.assertEqual(payload["disclaimer"], DISCLAIMER)


class ModelCardTests(unittest.TestCase):
    """The card exists so a consumer can interpret a single recommendation.

    Without it, a caller sees `equipoise` and reads a failure rather than a
    measured result about how much data the model has.
    """

    @classmethod
    def setUpClass(cls):
        cls.card = RecommendationService().model_card()

    def test_it_says_the_cohort_is_synthetic(self):
        self.assertIn("synthetic", self.card["training"]["cohort"])

    def test_it_pins_the_disease(self):
        self.assertEqual(self.card["disease_id"], "rheumatoid_arthritis")

    def test_it_names_the_serving_ensemble(self):
        from treatmentrx.estimation import training

        self.assertEqual(
            set(self.card["serving_ensemble"]), set(training.SERVING_ENSEMBLE)
        )
        self.assertEqual(set(self.card["held_out"]), set(training.SERVING_ENSEMBLE))

    def test_it_states_the_abstention_rate_as_a_limitation(self):
        abstention = self.card["known_limitations"]["abstention"]
        self.assertIn("equipoise", abstention["what_it_means"].lower())
        self.assertGreater(abstention["pooled_rate"], 0.5)

    def test_it_says_the_abstention_rate_does_not_transfer(self):
        """The strongest finding from `cli transfer`, and the one a consumer at a
        different site most needs. The baseline rate is a property of the
        training population, not a universal method constant."""
        abstention = self.card["known_limitations"]["abstention"]
        self.assertIn("varies_by_population", abstention)
        self.assertIn("transfer", self.card["known_limitations"])

    def test_it_says_the_abstention_rate_is_not_uniform(self):
        """The pooled rate is what a consumer quotes, and it hides a large
        spread. A clinician seeing equipoise for a seronegative patient is
        seeing the common case, not an unlucky one."""
        abstention = self.card["known_limitations"]["abstention"]
        self.assertIn("not_uniform", abstention)
        self.assertIn("subgroups", abstention["not_uniform"])

    def test_it_names_the_unadjusted_confounders(self):
        unadjusted = self.card["known_limitations"]["unadjusted_confounders"]
        self.assertIn("age", unadjusted["nodes"])
        self.assertIn("steroid_use", unadjusted["nodes"])

    def test_it_says_the_unadjusted_confounders_cost_nothing_here(self):
        """A bare list reads as measured residual confounding, and it is not.

        The generating process has no age, gender, steroid or comorbidity
        effect, so on this cohort the omission costs zero. It is a property of
        the basis that would bite on real data, and the card has to say which of
        the two it is or a reader will price it wrong.
        """
        unadjusted = self.card["known_limitations"]["unadjusted_confounders"]
        self.assertIn("cost_on_this_cohort", unadjusted)
        self.assertIn("on_real_data", unadjusted)
        self.assertIn("none", unadjusted["cost_on_this_cohort"])

    def test_it_reports_whether_the_ranking_is_resolved(self):
        """An estimator chosen by tie-break is not an estimator chosen by data."""
        self.assertIn("ranking_resolved", self.card["selection"])

    def test_it_reports_the_blip_basis_check(self):
        self.assertIn("flagged_modifiers", self.card["blip_basis"])
        self.assertTrue(self.card["blip_basis"]["candidates_tested"])

    def test_the_validation_gate_is_asked_not_asserted(self):
        """The card must not hold its own opinion about the gate.

        A hard-coded `gate_passed: false` looks identical to a correct one and
        would keep reading false after the ladder started saying otherwise. This
        pins the card to the ladder's own verdict and wording.
        """
        from treatmentrx.domain import ValidationRung
        from treatmentrx.estimation import training
        from treatmentrx.feedback.validation_ladder import ValidationLadder

        status = ValidationLadder().assess(
            ValidationRung.SILENT, training.deployment_readiness()
        )
        self.assertEqual(self.card["validation"]["gate_passed"], status.gate_passed)
        self.assertEqual(self.card["validation"]["blockers"], list(status.blockers))
        self.assertFalse(self.card["validation"]["gate_passed"], "no live data in this build")
        self.assertFalse(self.card["validation"]["retraining_allowed"])


class ServiceSafetyTests(unittest.TestCase):
    def test_the_default_bind_is_loopback(self):
        """An unauthenticated decision-support endpoint must not reach the
        network by default."""
        self.assertEqual(DEFAULT_HOST, "127.0.0.1")
        self.assertIsNone(exposure_warning(DEFAULT_HOST))

    def test_a_routable_bind_is_warned_about(self):
        warning = exposure_warning("0.0.0.0")
        self.assertIsNotNone(warning)
        self.assertIn("no authentication", warning)
        self.assertIn(DISCLAIMER, warning)

    def test_the_access_log_drops_the_query_string(self):
        """A query string is somewhere an identifier ends up.

        The default `BaseHTTPRequestHandler` logs the whole request line. Here a
        caller passing `?patient=...` would have written it to the log of a
        service whose entire point is that identifiers stop at Layer 1.
        """
        server = build_server(port=0, warm=False)
        base = f"http://127.0.0.1:{server.server_address[1]}"
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        captured = io.StringIO()
        try:
            with contextlib.redirect_stderr(captured):
                with contextlib.suppress(urllib.error.HTTPError):
                    urllib.request.urlopen(base + "/health?patient=MRN-90210-SECRET")
                time.sleep(0.2)  # the handler logs on its own thread
        finally:
            server.shutdown()
            server.server_close()
        log = captured.getvalue()
        self.assertIn("/health", log, "the request should still be logged")
        self.assertNotIn("MRN-90210-SECRET", log)
        self.assertNotIn("patient=", log)

    def test_the_body_limit_is_finite(self):
        """An unbounded read on an open port is a denial of service with no code."""
        self.assertGreater(MAX_BODY_BYTES, 0)
        self.assertLessEqual(MAX_BODY_BYTES, 8 * 1024 * 1024)

    def test_an_oversized_body_is_refused_before_it_is_read(self):
        server = build_server(port=0, warm=False)
        base = f"http://127.0.0.1:{server.server_address[1]}"
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                request = urllib.request.Request(
                    base + "/recommend", data=b"x" * 64, method="POST"
                )
                request.add_header("Content-Length", str(MAX_BODY_BYTES + 1))
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    urllib.request.urlopen(request)
            self.assertEqual(raised.exception.code, 413)
        finally:
            server.shutdown()
            server.server_close()

    def test_the_service_adds_no_clinical_logic(self):
        """Same rule as the orchestrator: transport only.

        The service's answer for a patient has to be the orchestrator's answer,
        or there is a second decision path — the failure mode this repo's notes
        open with.
        """
        from treatmentrx.orchestrator import TreatmentRxOrchestrator

        direct = TreatmentRxOrchestrator().run(sample_ra_bundle())
        served = RecommendationService().recommend(sample_ra_bundle())
        self.assertEqual(served["recommended_arm"], direct.recommended_arm)
        self.assertEqual(served["status"], direct.status.value)
        self.assertEqual(served["q_values"], direct.q_values)


if __name__ == "__main__":
    unittest.main()
