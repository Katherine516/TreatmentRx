from __future__ import annotations

from treatmentrx.contracts import Decision, PatientState, RecommendationStatus, SafeDecision, SafetyFlag


class SafetyLayer:
    """Layer 4: code-enforced feasible-set and safety filter."""

    def apply(self, decision: Decision, state: PatientState) -> SafeDecision:
        feasible, removed, flags = self._build_feasible_set(state)
        flags.extend(self._diagnostic_flags(state))

        status = decision.status
        final_decision = decision
        provenance = {
            "safety_layer": "treatmentrx.safety",
            "patient_hash": state.patient_hash,
            "history_summary": state.history_summary,
            "original_recommended_arm": decision.recommended_arm,
            "feasible_set_size": len(feasible),
            "removed_arms": removed,
            "versions": state.versions.__dict__,
        }

        if any(flag.severity == "block" for flag in flags):
            status = RecommendationStatus.BLOCKED
        elif not feasible:
            flags.append(SafetyFlag("empty_feasible_set", "block", "No safe treatment arm remains after filtering."))
            status = RecommendationStatus.BLOCKED
        elif decision.recommended_arm not in feasible:
            replacement = self._best_feasible(decision, feasible)
            if replacement is None:
                flags.append(
                    SafetyFlag(
                        "recommended_arm_infeasible",
                        "block",
                        f"{decision.recommended_arm} is infeasible and no scored feasible alternative exists.",
                        decision.recommended_arm,
                    )
                )
                status = RecommendationStatus.BLOCKED
            else:
                flags.append(
                    SafetyFlag(
                        "feasible_set_rewrite",
                        "warn",
                        f"Safety layer replaced {decision.recommended_arm} with {replacement}.",
                        decision.recommended_arm,
                    )
                )
                final_decision = Decision(
                    recommended_arm=replacement,
                    q_values=decision.q_values,
                    model_weights=decision.model_weights,
                    uncertainty=decision.uncertainty,
                    status=RecommendationStatus.REVIEW,
                    rationale=decision.rationale + " Safety layer selected the best feasible scored alternative.",
                    estimates=decision.estimates,
                )
                status = RecommendationStatus.REVIEW

        return SafeDecision(
            decision=final_decision,
            feasible_arms=feasible,
            removed_arms=removed,
            safety_flags=flags,
            status=status,
            provenance=provenance | {"final_recommended_arm": final_decision.recommended_arm},
        )

    def _build_feasible_set(self, state: PatientState) -> tuple[list[str], dict[str, str], list[SafetyFlag]]:
        latest = state.stages[-1].features if state.stages else {}
        alt = self._feature(latest, "alt", 25.0)
        egfr = self._feature(latest, "egfr", 90.0)
        pregnant = bool(latest.get("pregnant", False) or latest.get("pregnancy", False))
        allergies = [allergy.lower() for allergy in state.allergies]
        feasible: list[str] = []
        removed: dict[str, str] = {}
        flags: list[SafetyFlag] = []

        for arm in state.feasible_arms:
            reason = self._unsafe_reason(arm, alt, egfr, pregnant, allergies)
            if reason:
                removed[arm] = reason
                flags.append(SafetyFlag("arm_removed", "warn", reason, arm))
            else:
                feasible.append(arm)
        return feasible, removed, flags

    def _diagnostic_flags(self, state: PatientState) -> list[SafetyFlag]:
        return [
            SafetyFlag("diagnostic_failed", "block", diagnostic.message)
            for diagnostic in state.diagnostics
            if diagnostic.severity == "error" and not diagnostic.passed
        ]

    def _unsafe_reason(
        self,
        arm: str,
        alt: float,
        egfr: float,
        pregnant: bool,
        allergies: list[str],
    ) -> str | None:
        lower = arm.lower()
        for allergy in allergies:
            if allergy and allergy in lower:
                return f"{arm} removed due to allergy conflict: {allergy}"
        if "jak" in lower:
            if pregnant:
                return "JAK inhibitor removed due to pregnancy flag."
            if alt > 120:
                return "JAK inhibitor removed due to ALT > 120."
            if egfr < 30:
                return "JAK inhibitor removed due to eGFR < 30."
        if "csdmard" in lower or "methotrexate" in lower:
            if pregnant:
                return "Methotrexate/csDMARD optimization removed due to pregnancy flag."
            if egfr < 30:
                return "Methotrexate/csDMARD optimization removed due to eGFR < 30."
        return None

    def _best_feasible(self, decision: Decision, feasible: list[str]) -> str | None:
        candidates = {arm: value for arm, value in decision.q_values.items() if arm in feasible}
        if not candidates:
            return None
        return max(candidates, key=candidates.get)

    def _feature(self, features: dict[str, object], key: str, default: float) -> float:
        value = features.get(key, default)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default
