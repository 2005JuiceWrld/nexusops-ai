from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InvestigationState:
    """State carried throughout a NexusOps investigation."""

    question: str

    plan: list[str] = field(default_factory=list)

    current_step: str = "PLAN"

    evidence: list[dict[str, Any]] = field(default_factory=list)

    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    hypotheses: list[str] = field(default_factory=list)

    verification: list[dict[str, Any]] = field(default_factory=list)

    risk_level: str = "low"

    approval_required: bool = False

    recommendation: str = ""

    final_answer: str = ""

    audit_log: list[dict[str, Any]] = field(default_factory=list)

    def log_event(
        self,
        event: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.audit_log.append(
            {
                "event": event,
                "step": self.current_step,
                "details": details or {},
            }
        )

    def add_evidence(
        self,
        source: str,
        evidence: Any,
    ) -> None:
        self.evidence.append(
            {
                "source": source,
                "evidence": evidence,
            }
        )

    def record_tool_call(
        self,
        tool: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        self.tool_calls.append(
            {
                "tool": tool,
                "arguments": arguments,
                "result": result,
            }
        )


if __name__ == "__main__":
    state = InvestigationState(
        question=(
            "Investigate payment failures after the latest deployment."
        )
    )

    state.plan = [
        "Investigate incidents",
        "Check recent deployments",
        "Check service dependencies",
        "Retrieve remediation runbook",
        "Verify likely cause",
        "Determine remediation risk",
    ]

    state.log_event(
        "investigation_created",
        {"question": state.question},
    )

    print("\nNEXUSOPS INVESTIGATION STATE")
    print("=" * 60)
    print(f"Question: {state.question}")
    print(f"Current step: {state.current_step}")
    print("\nPlan:")

    for index, step in enumerate(state.plan, start=1):
        print(f"{index}. {step}")

    print(f"\nAudit events: {len(state.audit_log)}")