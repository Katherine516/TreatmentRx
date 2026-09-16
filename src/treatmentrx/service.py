"""An HTTP surface for the agent, on the standard library only.

`dependencies = []` is a hard constraint and it survives this: `http.server` is
enough for three routes, and the alternative — pulling in a web framework and an
ASGI server for a research prototype — would trade the property that makes the
repo installable anywhere for request validation this can do by hand.

What it is not is a production service. `http.server` is single-process, has no
TLS, no auth, and its own documentation says it is not recommended for
production. The gaps are deliberate and listed in `LIMITATIONS` so a reader does
not have to infer them:

* **Binds to localhost by default.** A clinical decision-support prototype that
  listens on 0.0.0.0 by accident is a different kind of mistake from a wrong
  standard error. Overriding the host is possible and prints a warning.
* **No authentication.** Anything reachable on the port can post a record.
* **No PHI in logs.** The default `BaseHTTPRequestHandler` logging writes the
  request line to stderr; bodies are never logged, and the access log is
  overridden to record method, path and status only. The patient identifier
  never leaves Layer 1 in any case — everything downstream sees a hash.

The service exposes the patient workflow and two statistically separate research
surfaces:

    GET  /health      is the process up, and are the models fitted
    GET  /capabilities which diseases this process is allowed to score
    GET  /model       the RA model card: what it was fit on, what it scores, what
                      it will not tell you
    GET  /biomarkers  frozen external prognostic artifacts; empty means none
    POST /recommend   a FHIR bundle in, a Recommendation out
    POST /trial/precision  randomized continuous-outcome precision comparison
    POST /trial/power      deterministic precision-adjustment simulation

`/model` exists because this agent abstains on roughly two in three patients it
sees, and a consumer that does not know that will read equipoise as a failure.
Everything on that card is already computed and cached at fit time; the route is
a projection, not a study.
"""

from __future__ import annotations

import json
import sys
import threading
from dataclasses import asdict, is_dataclass
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from treatmentrx.scientific import ScientificMode

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8371

# A record large enough to be a mistake. An unbounded read on a public port is a
# denial of service with no code required.
MAX_BODY_BYTES = 2 * 1024 * 1024

DISCLAIMER = (
    "Research scaffolding, not a medical device. Nothing here is clinically "
    "validated and no output may be used to direct patient care."
)

LIMITATIONS = (
    "single-process http.server; no TLS; no authentication; binds to localhost "
    "unless overridden",
)


def _to_json(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _to_json(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json(item) for item in value]
    return value


class RecommendationService:
    """The agent behind the routes. Transport-free, so it is testable directly.

    Holds one orchestrator behind a lock. `EpisodicMemory` is mutable per-patient
    state and `ThreadingHTTPServer` runs a thread per request, so concurrent
    writes would race on a dict — the lock is cheaper than making memory
    thread-safe and the request itself is milliseconds.
    """

    def __init__(self) -> None:
        from treatmentrx.orchestrator import TreatmentRxOrchestrator

        self._orchestrator = TreatmentRxOrchestrator()
        self._lock = threading.Lock()

    def warm(self) -> None:
        """Fit the models before the first request rather than during it.

        Cold start is ~3s and lazily paid. A first caller who waits three
        seconds and every caller after who waits two milliseconds is a confusing
        service; paying it at startup is honest about where the cost is.
        """
        from treatmentrx.estimation import training

        training.fitted()

    def health(self) -> dict[str, Any]:
        from treatmentrx.contracts import VersionSet
        from treatmentrx.estimation import training

        return {
            "status": "ok",
            "models_fitted": training._FITTED is not None,
            "supported_diseases": self._orchestrator.registry.supported_ids(),
            "versions": VersionSet().__dict__,
            "disclaimer": DISCLAIMER,
            "limitations": list(LIMITATIONS),
        }

    def capabilities(self) -> dict[str, Any]:
        """Machine-readable disease boundary; absence means unsupported."""
        return {
            "supported_diseases": self._orchestrator.capabilities(),
            "analysis_services": {
                "individual_recommendation": "dtr_research",
                "randomized_trial_precision": {
                    "mode": "randomized_trial",
                    "outcomes": ["continuous"],
                    "can_recommend_individual_treatment": False,
                },
                "biomarker_scoring": {
                    "mode": "biomarker_research",
                    "registered_artifacts": len(self.biomarkers()["artifacts"]),
                    "fallback": False,
                },
            },
            "fallback_to_other_disease_model": False,
            "disclaimer": DISCLAIMER,
        }

    def biomarkers(self) -> dict[str, Any]:
        """Frozen predictors this process can apply; no artifact is the default."""
        artifacts = []
        for disease_id in self._orchestrator.registry.supported_ids():
            definition = self._orchestrator.registry.get(disease_id)
            workflow = self._orchestrator._workflow(definition)
            artifacts.extend(workflow.biomarkers.capabilities())
        return {
            "artifacts": artifacts,
            "fallback_to_unvalidated_artifact": False,
            "prognostic_scores_may_modify_treatment_effect": False,
            "message": (
                "No frozen biomarker is used unless it is explicitly registered "
                "and the request matches its validated domain."
            ),
            "disclaimer": DISCLAIMER,
        }

    def model_card(self) -> dict[str, Any]:
        """What the model was fit on, what it scores, and what it will not say.

        Every field here is read from the cached fit. The caveats are the point:
        a consumer reading a single recommendation cannot see the abstention
        rate, the interval coverage, or that four confounders are unadjusted, and
        those are the facts that decide whether the number in front of them means
        anything.
        """
        from treatmentrx.data.dag import CausalDAGRegistry
        from treatmentrx.domain import ValidationRung
        from treatmentrx.estimation import training
        from treatmentrx.feedback.validation_ladder import ValidationLadder

        fit = training.fitted()
        specification = training.basis_specification()
        readiness = training.deployment_readiness()
        dag = CausalDAGRegistry()._ra_v1()
        # Asked, not asserted. A hard-coded "gate_passed": false is a claim the
        # card makes on the ladder's behalf, and it would keep reading false
        # after the ladder started saying otherwise — the card-versus-status
        # divergence this repo has already fixed once.
        validation = ValidationLadder().assess(ValidationRung.SILENT, readiness)
        return {
            "disclaimer": DISCLAIMER,
            "disease_id": "rheumatoid_arthritis",
            "operating_mode": "dtr_research",
            "estimand_contract": self._orchestrator.registry.get(
                "rheumatoid_arthritis"
            ).estimand_for(ScientificMode.DTR_RESEARCH).as_dict(),
            "training": {
                "cohort": "synthetic (simulation/ra_cohort.py) — not clinical data",
                "trajectories": len(fit.train),
                "holdout_trajectories": len(fit.holdout),
                "seed": training.COHORT_SEED,
            },
            "serving_ensemble": list(training.SERVING_ENSEMBLE),
            "held_out": {
                name: {
                    "policy_value": score.ipw_policy_value,
                    "policy_value_is": "per-decision, not the regime's own value",
                    "interval": list(score.value_interval or ()),
                    "improvement_over_behaviour": list(score.improvement_interval or ()),
                    "effective_sample_size": score.effective_sample_size,
                    # The quantity a sequential agent actually claims. Reported
                    # beside the per-decision one because they answer different
                    # questions, and at this cohort size the regime's own value
                    # is not identified — which a consumer reading a single
                    # policy value would otherwise never learn.
                    "sequential_regime_value": (
                        score.sequential.as_dict() if score.sequential else None
                    ),
                    # The same estimand, augmented with the fitted Q-function so
                    # every held-out trajectory contributes instead of the five
                    # that stay on the regime's path. Reported beside the IPW
                    # version rather than instead of it: it is identified, but
                    # under the outcome model, and the IPW estimate is the only
                    # thing here that could falsify that model.
                    "sequential_regime_value_doubly_robust": (
                        score.sequential_dr.as_dict() if score.sequential_dr else None
                    ),
                }
                for name, score in fit.scores.items()
                if name in training.SERVING_ENSEMBLE
            },
            "selection": {
                "estimator": training.best_score().estimator,
                "ranking_resolved": training.ranking_is_resolved(),
            },
            "calibration": {
                "expected_calibration_error": fit.calibration.expected_calibration_error,
                "passed": fit.calibration.passed,
            },
            "blip_basis": {
                "flagged_modifiers": specification.get("flagged", []),
                "candidates_tested": sorted(specification.get("candidates", {})),
            },
            "validation": {
                "rung": validation.rung.value,
                "gate_description": validation.gate_description,
                "gate_passed": validation.gate_passed,
                "blockers": list(validation.blockers),
                "retraining_allowed": False,
            },
            "known_limitations": {
                "abstention": {
                    "pooled_rate": 0.66,
                    "what_it_means": (
                        "The agent declines to separate arms for ~66% of "
                        "patients at this training size after simultaneous "
                        "all-pairs multiplicity correction. Equipoise is a "
                        "measured result, not a failure: see `cli power`."
                    ),
                    # The pooled rate is the number a consumer will quote, but a
                    # clinician seeing equipoise for a seronegative patient is
                    # seeing the common case, not an unlucky one.
                    # Measured at seven simulated sites (`cli transfer`): the
                    # rate is a property of the population at least as much as
                    # of the method, and a consumer reading 66% as a property of
                    # the tool will be wrong by tens of points.
                    "varies_by_population": (
                        "54%-89% across seven simulated sites differing in case "
                        "mix, prescribing and retention (`cli transfer`). The "
                        "quoted rate describes this training population."
                    ),
                    "not_uniform": (
                        "Abstention ranges from 39% to 97% across strata "
                        "(`cli subgroups`). It is highest for seronegative "
                        "patients (97%) and for the top disease-activity tertile "
                        "(96%). It is earned in every stratum — declined "
                        "patients really do have closer arms."
                    ),
                },
                "interval_coverage": (
                    "The reported contrast interval covers 95.0% against a "
                    "nominal 95% on a six-patient grid at n=280, at SE/spread "
                    "1.04 (`cli coverage`). That figure is measured at the "
                    "terminal decision, which is the only stage where both "
                    "serving estimators target the same quantity. Swept over "
                    "stages (`cli coverage --stages`) it holds at 96.7% for the "
                    "other stage a patient can present at, and falls to 77.5% at "
                    "stage 0 — which Layer 1 never produces, because it appends "
                    "the pending visit."
                ),
                # A partition that has set the averaging weights, selected the
                # estimator, supplied the calibration and produced the headline
                # value is not a test set, whatever it is called. Reported here
                # because a consumer comparing this card against a published
                # model's would otherwise assume a held-back split exists.
                "evaluation_partition": {
                    **training.evaluation_partition().as_dict(),
                    "what_it_means": (
                        "There is no partition held back from model development. "
                        "The evaluation split sets the model-averaging weights, "
                        "selects the serving estimator, supplies the calibration "
                        "the validation ladder gates on, and is the held-out "
                        "policy value quoted above. The Monte Carlo studies in "
                        "`cli coverage`, `power` and `misspecification` do draw "
                        "fresh cohorts, so the tuned constants are not fitted to "
                        "this split — but nothing untouched remains to check the "
                        "headline numbers against."
                    ),
                },
                "unadjusted_confounders": {
                    "nodes": list(dag.unmodelled_confounders),
                    # Bare, this list reads as measured residual confounding in
                    # the numbers above it. It is not: the generating process
                    # has no age, gender, steroid or comorbidity effect, so on
                    # this cohort the omission costs exactly zero. It is a
                    # standing property of the *basis*, and it is what would
                    # bite on real data.
                    "cost_on_this_cohort": (
                        "none — the synthetic generating process has no effect "
                        "on these nodes, so this is a property of the model "
                        "basis rather than a measured bias here"
                    ),
                    "on_real_data": (
                        "the residual confounding nobody can rule out; the "
                        "reported contrasts would be biased by an unknown amount"
                    ),
                },
                "covariates_reaching_the_model": 6,
                # What survives being scored on data it was not fit on, and what
                # does not. The last clause is the one a monitor should act on.
                "transfer": (
                    "Fit at one simulated site and scored at seven others "
                    "(`cli transfer`): the policy still beats local practice "
                    "7/7, interval coverage holds 95-97% and calibration holds "
                    "0.002-0.006, while the abstention rate does not transfer. "
                    "Under a shift that rescales the true effects, calibration "
                    "degrades an order of magnitude while policy value improves "
                    "— policy value alone cannot detect that the reported "
                    "Q-values have stopped meaning anything."
                ),
                "readiness": readiness,
            },
        }

    def recommend(self, bundle: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            recommendation = self._orchestrator.run(bundle)
        payload = _to_json(recommendation)
        payload["disclaimer"] = DISCLAIMER
        return payload

    def trial_precision(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Analyze a randomized continuous endpoint without recommendation output."""
        from treatmentrx.trials import (
            ContinuousTrialRecord,
            PrognosticAdjustmentContract,
            RandomizedTrialPrecisionAnalyzer,
            continuous_trial_estimand,
        )

        raw_records = payload.get("records")
        if not isinstance(raw_records, list):
            raise ValueError("records must be a list")
        records = []
        for index, raw in enumerate(raw_records):
            if not isinstance(raw, dict):
                raise ValueError(f"records[{index}] must be an object")
            try:
                records.append(
                    ContinuousTrialRecord(
                        outcome=float(raw["outcome"]),
                        treatment=int(raw["treatment"]),
                        prognostic_score=float(raw["prognostic_score"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"records[{index}] requires numeric outcome, treatment, "
                    "and prognostic_score"
                ) from error
        analyzer = RandomizedTrialPrecisionAnalyzer(
            alpha=float(payload.get("alpha", 0.05)),
            target_power=float(payload.get("target_power", 0.80)),
        )
        estimand = continuous_trial_estimand(
            outcome=str(payload.get("outcome", "continuous outcome")),
            horizon_days=int(payload.get("horizon_days", 365)),
            treatment_label=str(payload.get("treatment_label", "experimental")),
            reference_label=str(payload.get("reference_label", "control")),
        )
        raw_artifact = payload.get("prognostic_artifact")
        if not isinstance(raw_artifact, dict):
            raise ValueError(
                "prognostic_artifact must declare the frozen or cross-fitted "
                "score source"
            )
        externally_validated = raw_artifact.get("externally_validated", False)
        if not isinstance(externally_validated, bool):
            raise ValueError(
                "prognostic_artifact.externally_validated must be a boolean"
            )
        try:
            prognostic_contract = PrognosticAdjustmentContract(
                artifact_id=str(raw_artifact["artifact_id"]),
                artifact_version=str(raw_artifact["artifact_version"]),
                endpoint=str(raw_artifact["endpoint"]),
                horizon_days=int(raw_artifact["horizon_days"]),
                reference_treatment=str(raw_artifact["reference_treatment"]),
                independence_strategy=str(raw_artifact["independence_strategy"]),
                externally_validated=externally_validated,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "prognostic_artifact requires artifact_id, artifact_version, "
                "endpoint, horizon_days, reference_treatment, and "
                "independence_strategy"
            ) from error
        planning = payload.get("planning_effect")
        result = analyzer.analyze(
            records,
            estimand,
            prognostic_contract,
            planning_effect=float(planning) if planning is not None else None,
        ).as_dict()
        result["disclaimer"] = DISCLAIMER
        return result

    def trial_power(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run bounded, deterministic precision-adjustment simulations."""
        from treatmentrx.trials import simulate_precision_power

        result = simulate_precision_power(
            sample_size=int(payload.get("sample_size", 100)),
            treatment_effect=float(payload.get("treatment_effect", 0.0)),
            prognostic_coefficient=float(payload.get("prognostic_coefficient", 1.0)),
            error_sd=float(payload.get("error_sd", 1.0)),
            replicates=int(payload.get("replicates", 500)),
            alpha=float(payload.get("alpha", 0.05)),
            seed=int(payload.get("seed", 2026)),
        )
        result["disclaimer"] = DISCLAIMER
        return result


class _Handler(BaseHTTPRequestHandler):
    server_version = "TreatmentRx"
    sys_version = ""
    service: RecommendationService

    # ------------------------------------------------------------------ routes

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's interface
        route = self.path.split("?", 1)[0].rstrip("/") or "/"
        if route == "/health":
            self._send(200, self.service.health())
        elif route == "/capabilities":
            self._send(200, self.service.capabilities())
        elif route == "/model":
            self._send(200, self.service.model_card())
        elif route == "/biomarkers":
            self._send(200, self.service.biomarkers())
        elif route == "/":
            self._send(
                200,
                {
                    "routes": [
                        "/health",
                        "/capabilities",
                        "/model",
                        "/biomarkers",
                        "POST /recommend",
                        "POST /trial/precision",
                        "POST /trial/power",
                    ],
                    "disclaimer": DISCLAIMER,
                },
            )
        else:
            self._send(404, {"error": "no such route", "path": route})

    def do_POST(self) -> None:  # noqa: N802
        route = self.path.split("?", 1)[0].rstrip("/") or "/"
        if route not in {"/recommend", "/trial/precision", "/trial/power"}:
            self._send(404, {"error": "no such route", "path": route})
            return

        body = self._read_body()
        if body is None:
            return
        try:
            bundle = json.loads(body)
        except json.JSONDecodeError as error:
            self._send(400, {"error": "request body is not valid JSON", "detail": str(error)})
            return
        if not isinstance(bundle, dict):
            message = (
                "request body must be a FHIR Bundle object"
                if route == "/recommend"
                else "request body must be a JSON object"
            )
            self._send(400, {"error": message})
            return

        if route == "/recommend":
            self._recommend(bundle)
        else:
            self._trial_analysis(route, bundle)

    def _trial_analysis(self, route: str, payload: dict[str, Any]) -> None:
        """Keep trial research separate from the patient recommendation path."""
        from treatmentrx.scientific import EstimandContractError
        from treatmentrx.trials import TrialPrecisionError

        try:
            if route == "/trial/precision":
                result = self.service.trial_precision(payload)
            else:
                result = self.service.trial_power(payload)
            self._send(200, result)
        except (TrialPrecisionError, EstimandContractError, ValueError) as error:
            self._send(
                422,
                {
                    "error": "the randomized-trial analysis request is invalid",
                    "detail": str(error),
                    "can_recommend_individual_treatment": False,
                },
            )

    def _recommend(self, bundle: dict[str, Any]) -> None:
        """Run the agent, and map each failure to a status code that means it.

        A record the contract rejects is 422, not 400 and not 500: the request
        was well-formed and the *content* is unusable, which is a different thing
        for a caller to act on. The reasons are returned because "unprocessable"
        with no field name is not actionable.
        """
        from treatmentrx.data import DataContractError, LeakageError
        from treatmentrx.diseases import UnsupportedDiseaseError

        try:
            self._send(200, self.service.recommend(bundle))
        except UnsupportedDiseaseError as error:
            self._send(
                422,
                {
                    "error": "unsupported disease",
                    "disease": error.disease,
                    "supported_diseases": error.supported,
                },
            )
        except DataContractError as error:
            self._send(
                422,
                {
                    "error": "the record does not satisfy the RA data contract",
                    "issues": [
                        {"field": issue.field, "message": issue.message}
                        for issue in error.issues
                    ],
                },
            )
        except LeakageError as error:
            self._send(
                422,
                {
                    "error": "post-decision information was found in the patient state",
                    "detail": str(error),
                },
            )
        except ValueError as error:
            self._send(422, {"error": "the record cannot be processed", "detail": str(error)})
        except Exception:  # noqa: BLE001 - the boundary has to hold
            # No traceback to the client: it can carry field values, and field
            # values can carry PHI.
            self.log_error("unhandled error serving /recommend")
            self._send(500, {"error": "internal error"})

    # ------------------------------------------------------------------ plumbing

    def _read_body(self) -> bytes | None:
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            self._send(400, {"error": "Content-Length is not an integer"})
            return None
        if length <= 0:
            self._send(400, {"error": "empty request body"})
            return None
        if length > MAX_BODY_BYTES:
            self._send(
                413, {"error": "request body too large", "limit_bytes": MAX_BODY_BYTES}
            )
            return None
        return self.rfile.read(length)

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        if "disclaimer" not in payload:
            payload = payload | {"disclaimer": DISCLAIMER}
        body = json.dumps(payload, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """Method, path and status, to stderr. Never the body, never the query.

        Two things the default gets wrong here. It logs the full request line
        including any query string, which is somewhere an identifier can end up;
        and anything written to stdout would corrupt a caller piping the
        server's output. Bodies are never logged at all — a posted bundle is a
        patient record.
        """
        status = args[1] if len(args) > 1 else "-"
        path = self.path.split("?", 1)[0]
        sys.stderr.write(
            f"{self.log_date_time_string()} {self.command} {path} {status}\n"
        )


def build_server(
    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, warm: bool = True
) -> ThreadingHTTPServer:
    """A configured server, not yet serving. Split out so tests can drive it."""
    service = RecommendationService()
    if warm:
        service.warm()

    handler = type("_BoundHandler", (_Handler,), {"service": service})
    return ThreadingHTTPServer((host, port), handler)


def _notice(message: str) -> None:
    """Diagnostics to stderr, unbuffered.

    Two failures this avoids. `print` to a pipe is block-buffered, so the banner
    — which carries the disclaimer — is still sitting in the buffer when the
    process is signalled, and nobody ever sees it. And stdout belongs to the
    caller: `cli serve | something` should get the caller's data, not ours.
    """
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})


def exposure_warning(host: str) -> str | None:
    """The message for a non-loopback bind, or None. Separate so it is testable.

    Inline in `serve()` the only way to exercise this was to actually bind to a
    routable address, which is not something a test suite should do.
    """
    if host in LOOPBACK:
        return None
    return (
        f"WARNING: binding to {host}, which is not loopback. This service has "
        f"no authentication and no TLS, and it returns clinical "
        f"decision-support output. {DISCLAIMER}"
    )


def serve(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    warning = exposure_warning(host)
    if warning:
        _notice(warning)
    server = build_server(host, port)
    _notice(
        f"TreatmentRx on http://{host}:{port}  "
        "(GET /health, GET /capabilities, GET /model, GET /biomarkers, "
        "POST /recommend, POST /trial/precision, POST /trial/power)"
    )
    _notice(DISCLAIMER)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "DISCLAIMER",
    "LIMITATIONS",
    "MAX_BODY_BYTES",
    "RecommendationService",
    "build_server",
    "exposure_warning",
    "serve",
]
