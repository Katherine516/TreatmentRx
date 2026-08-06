from __future__ import annotations

from precisionrx_agent.shared.models import (
    DAGValidationResult,
    DataContractReport,
    MethodResult,
    RecommendationStatus,
    SafetyFinding,
    SafetyResult,
    StageRecord,
)


class SafetyGate:
    """Synchronous hard gate between statistical engine and explanation layer."""

    def evaluate(
        self,
        stages: list[StageRecord],
        result: MethodResult,
        allergies: list[str],
        dag_validation: DAGValidationResult | None = None,
        data_contract: DataContractReport | None = None,
    ) -> SafetyResult:
        findings: list[SafetyFinding] = []
        recommended = result.recommended_action.lower()

        if data_contract:
            for issue in data_contract.issues:
                if issue.severity == "error":
                    findings.append(
                        SafetyFinding(
                            code="data_contract_error",
                            severity="block",
                            message=issue.message,
                        )
                    )

        if dag_validation and not dag_validation.identified:
            findings.append(
                SafetyFinding(
                    code="causal_identifiability_failed",
                    severity="block",
                    message=dag_validation.blocked_reason or "The active causal graph did not identify the treatment effect.",
                )
            )

        for allergy in allergies:
            if allergy.lower() in recommended:
                findings.append(
                    SafetyFinding(
                        code="allergy_contraindication",
                        severity="block",
                        message=f"Recommended action conflicts with recorded allergy: {allergy}.",
                    )
                )

        latest_features = stages[-1].features
        alt = self._feature(latest_features, "alt", default=25)
        egfr = self._feature(latest_features, "egfr", default=90)
        pregnancy = bool(latest_features.get("pregnant", False))
        if "JAK" in result.recommended_action and (alt > 120 or egfr < 30 or pregnancy):
            findings.append(
                SafetyFinding(
                    code="jak_safety_review",
                    severity="block",
                    message="JAK inhibitor recommendation requires review due to organ function or pregnancy flag.",
                )
            )

        ordered = sorted(result.q_values.values(), reverse=True)
        confidence_gap = round(ordered[0] - ordered[1], 3) if len(ordered) > 1 else 0.0
        if confidence_gap < 0.04:
            findings.append(
                SafetyFinding(
                    code="clinical_equipoise",
                    severity="warn",
                    message="Top treatment is not meaningfully separated from the next-best option.",
                )
            )

        ood_flag = self._ood(latest_features)
        if ood_flag:
            findings.append(
                SafetyFinding(
                    code="outside_training_support",
                    severity="warn",
                    message="Patient covariates are outside the current synthetic training support.",
                )
            )

        delayed_tox = self._delayed_toxicity(stages, result)
        if delayed_tox:
            findings.append(delayed_tox)

        if any(finding.severity == "block" for finding in findings):
            status = RecommendationStatus.BLOCKED
        elif confidence_gap < 0.04:
            status = RecommendationStatus.EQUIPOISE
        elif ood_flag:
            status = RecommendationStatus.REVIEW
        else:
            status = RecommendationStatus.RECOMMEND

        return SafetyResult(status=status, findings=findings, ood_flag=ood_flag, confidence_gap=confidence_gap)

    def _feature(self, features: dict[str, object], key: str, default: float) -> float:
        value = features.get(key, default)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default

    def _delayed_toxicity(self, stages: list[StageRecord], result: MethodResult) -> SafetyFinding | None:
        """v5.1 #1 — trajectory-level check: a sequence safe at each visit but
        accumulating toward a delayed adverse event (e.g. cumulative hepatotoxic
        exposure with a rising ALT trend) is caught here, not at a single visit."""
        hepatotoxic = ("methotrexate", "jak", "tofacitinib", "baricitinib", "upadacitinib", "leflunomide")
        exposure = sum(
            1 for stage in stages if any(token in stage.treatment.lower() for token in hepatotoxic)
        )
        alts = [
            self._feature(stage.features, "alt", default=0.0)
            for stage in stages
            if isinstance(stage.features.get("alt"), (int, float)) and not isinstance(stage.features.get("alt"), bool)
        ]
        rising_alt = len(alts) >= 2 and alts[-1] > alts[0] and alts[-1] > 60
        recommending_hepatotoxic = any(token in result.recommended_action.lower() for token in hepatotoxic)
        if exposure >= 2 and rising_alt and recommending_hepatotoxic:
            return SafetyFinding(
                code="delayed_toxicity_accumulation",
                severity="warn",
                message=(
                    "Cumulative hepatotoxic exposure with a rising ALT trend across visits; "
                    "monitor for delayed toxicity within the assessment window before continuing."
                ),
            )
        return None

    def _ood(self, features: dict[str, object]) -> bool:
        das28 = self._feature(features, "das28", default=4.0)
        crp = self._feature(features, "crp", default=8.0)
        egfr = self._feature(features, "egfr", default=90.0)
        return das28 > 8.0 or crp > 120.0 or egfr < 15.0
