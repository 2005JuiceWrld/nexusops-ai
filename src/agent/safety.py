from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RiskClass(str, Enum):
    """Deterministic risk classification for agent tools."""

    READ_ONLY = "read_only"
    LOW_RISK = "low_risk"
    PRODUCTION_CHANGE = "production_change"
    DESTRUCTIVE = "destructive"


@dataclass(frozen=True)
class AuthorizationDecision:
    """Result of a deterministic authorization check."""

    allowed: bool
    requires_approval: bool
    risk_class: RiskClass
    reason: str


class SafetyPolicy:
    """
    Deterministic safety and authorization policy.

    The LLM may recommend an action, but it cannot authorize
    or execute a production-impacting action.
    """

    TOOL_RISK: dict[str, RiskClass] = {
        "search_incidents": RiskClass.READ_ONLY,
        "get_recent_deployments": RiskClass.READ_ONLY,
        "get_service_status": RiskClass.READ_ONLY,
        "retrieve_runbook": RiskClass.READ_ONLY,
        "get_production_policy": RiskClass.READ_ONLY,
        "calculate_metrics": RiskClass.READ_ONLY,
        "restart_service": RiskClass.PRODUCTION_CHANGE,
    }

    @classmethod
    def classify_tool(cls, tool_name: str) -> RiskClass:
        """
        Return the deterministic risk class for a tool.

        Unknown tools fail closed.
        """
        return cls.TOOL_RISK.get(
            tool_name,
            RiskClass.DESTRUCTIVE,
        )

    @classmethod
    def authorize(
        cls,
        tool_name: str,
        approved: bool = False,
    ) -> AuthorizationDecision:
        """
        Determine whether a tool may execute.

        Approval is supplied externally by an authorized human
        or approval mechanism. It is never inferred from the LLM.
        """

        risk_class = cls.classify_tool(tool_name)

        if risk_class == RiskClass.READ_ONLY:
            return AuthorizationDecision(
                allowed=True,
                requires_approval=False,
                risk_class=risk_class,
                reason="Read-only operation.",
            )

        if risk_class == RiskClass.LOW_RISK:
            return AuthorizationDecision(
                allowed=True,
                requires_approval=False,
                risk_class=risk_class,
                reason="Low-risk operation.",
            )

        if risk_class == RiskClass.PRODUCTION_CHANGE:
            if not approved:
                return AuthorizationDecision(
                    allowed=False,
                    requires_approval=True,
                    risk_class=risk_class,
                    reason=(
                        "Production-impacting operation requires "
                        "explicit human approval."
                    ),
                )

            return AuthorizationDecision(
                allowed=True,
                requires_approval=True,
                risk_class=risk_class,
                reason=(
                    "Production-impacting operation approved "
                    "by an external approval signal."
                ),
            )

        return AuthorizationDecision(
            allowed=False,
            requires_approval=True,
            risk_class=risk_class,
            reason=(
                "Unknown or destructive operation is denied "
                "by default."
            ),
        )

    @classmethod
    def validate_recommendation(
        cls,
        recommendation: str,
    ) -> AuthorizationDecision:
        """
        Classify an LLM recommendation without executing it.

        This method deliberately does not treat language such as
        'approved' or 'execute immediately' as authorization.
        """

        text = recommendation.lower().strip()

        production_keywords = (
            "rollback",
            "roll back",
            "rolling back",
            "restart",
            "deploy",
            "production",
            "delete",
            "disable",
            "terminate",
            "scale",
        )

        if any(
            keyword in text
            for keyword in production_keywords
        ):
            return AuthorizationDecision(
                allowed=False,
                requires_approval=True,
                risk_class=RiskClass.PRODUCTION_CHANGE,
                reason=(
                    "Recommendation contains a potentially "
                    "production-impacting action. Explicit "
                    "authorization is required."
                ),
            )

        return AuthorizationDecision(
            allowed=True,
            requires_approval=False,
            risk_class=RiskClass.READ_ONLY,
            reason=(
                "Recommendation does not contain a detected "
                "production-impacting action."
            ),
        )