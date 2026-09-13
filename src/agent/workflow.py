from __future__ import annotations
from src.agent.safety import SafetyPolicy

import json
from typing import Any

from src.agent.llm import LLMReasoner
from src.agent.state import InvestigationState
from src.agent.tools import (
    get_production_policy,
    get_recent_deployments,
    get_service_status,
    retrieve_runbook,
    search_incidents,
)
from src.agent.verifier import EvidenceVerifier
from src.graph.service_graph import ServiceDependencyGraph

class InvestigationWorkflow:
    """Investigation workflow for NexusOps AI."""

    def __init__(self, state: InvestigationState) -> None:
        self.state = state
        self.reasoner = LLMReasoner()
        self.verifier = EvidenceVerifier()
        self.graph = ServiceDependencyGraph()
        self.llm_result: dict[str, Any] = {}

    def run(self) -> InvestigationState:
        self.plan()
        self.retrieve()
        self.tool_calls()
        self.analyze()
        self.verify()
        self.answer()

        return self.state

    # ---------------------------------------------------------
    # PLAN
    # ---------------------------------------------------------

    def plan(self) -> None:
        self.state.current_step = "PLAN"

        self.state.plan = [
            "Search recent payment incidents",
            "Inspect recent payment-service deployments",
            "Check payment-service status",
            "Check fraud-service status",
            "Retrieve payment remediation runbook",
            "Check production rollback policy",
            "Correlate evidence and determine likely cause",
        ]

        self.state.log_event(
            "plan_created",
            {
                "steps": self.state.plan,
            },
        )

    # ---------------------------------------------------------
    # RETRIEVE
    # ---------------------------------------------------------

    def retrieve(self) -> None:
        self.state.current_step = "RETRIEVE"

        self.state.log_event(
            "retrieval_started",
            {
                "query": self.state.question,
            },
        )

    # ---------------------------------------------------------
    # TOOL CALL HELPER
    # ---------------------------------------------------------

    def _call_tool(
        self,
        tool_name: str,
        tool_function: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        result = tool_function(**kwargs)

        self.state.record_tool_call(
            tool=tool_name,
            arguments=kwargs,
            result=result,
        )

        return result

    # ---------------------------------------------------------
    # TOOL CALLS
    # ---------------------------------------------------------

    def tool_calls(self) -> None:
        self.state.current_step = "TOOL_CALL"

        incidents = self._call_tool(
            "search_incidents",
            search_incidents,
            service="payment-service",
        )

        deployments = self._call_tool(
            "get_recent_deployments",
            get_recent_deployments,
            service="payment-service",
        )

        payment_status = self._call_tool(
            "get_service_status",
            get_service_status,
            service="payment-service",
        )

        fraud_status = self._call_tool(
            "get_service_status",
            get_service_status,
            service="fraud-service",
        )

        runbook = self._call_tool(
            "retrieve_runbook",
            retrieve_runbook,
            topic="payment",
        )
        
        policy = self._call_tool(
            "get_production_policy",
            get_production_policy,
        )

        self.state.add_evidence(
            "search_incidents",
            incidents,
        )

        self.state.add_evidence(
            "get_recent_deployments",
            deployments,
        )

        self.state.add_evidence(
            "payment_service_status",
            payment_status,
        )

        self.state.add_evidence(
            "fraud_service_status",
            fraud_status,
        )

        self.state.add_evidence(
            "payment_runbook",
            runbook,
        )

        self.state.add_evidence(
            "production_policy",
            policy,
        )

        graph_reasoning = self.graph.explain_service("payment-service")

        self.state.add_evidence(
            "service_dependency_graph",
            graph_reasoning,
        )

        self.state.log_event(
            "tool_calls_completed",
            {
                "tool_count": 6,
                "graph_reasoning": graph_reasoning,
            },
        )

     
    # ---------------------------------------------------------
    # ANALYZE
    # ---------------------------------------------------------

    def analyze(self) -> None:
        self.state.current_step = "ANALYZE"

        incident_ids: list[str] = []
        deployment_ids: list[str] = []

        incidents_evidence = self.state.evidence[0]["evidence"]
        deployments_evidence = self.state.evidence[1]["evidence"]

        # Extract incident records.
        if isinstance(incidents_evidence, dict):
            incident_items = incidents_evidence.get(
                "incidents",
                incidents_evidence.get("results", []),
            )
        elif isinstance(incidents_evidence, list):
            incident_items = incidents_evidence
        else:
            incident_items = []

        # Extract deployment records.
        if isinstance(deployments_evidence, dict):
            deployment_items = deployments_evidence.get(
                "deployments",
                deployments_evidence.get("results", []),
            )
        elif isinstance(deployments_evidence, list):
            deployment_items = deployments_evidence
        else:
            deployment_items = []

        if isinstance(incident_items, list):
            incident_ids = [
                item["id"]
                for item in incident_items
                if isinstance(item, dict) and "id" in item
            ]

        if isinstance(deployment_items, list):
            deployment_ids = [
                item["id"]
                for item in deployment_items
                if isinstance(item, dict) and "id" in item
            ]

        fraud_evidence = self.state.evidence[3]["evidence"]

        if isinstance(fraud_evidence, dict):
            fraud_status = fraud_evidence.get("status")
        else:
            fraud_status = None

        # Deterministic hypothesis generated before LLM reasoning.
        hypothesis = (
            "The most likely cause is increased fraud-service latency "
            "combined with changes introduced by payment-service "
            "version 2.4.1. The dependency relationship provides a "
            "plausible mechanism for payment-service timeouts."
        )

        self.state.hypotheses.append(hypothesis)

        self.state.log_event(
            "hypothesis_generated",
            {
                "incident_ids": incident_ids,
                "deployment_ids": deployment_ids,
                "fraud_status": fraud_status,
            },
        )

        # Send evidence and hypothesis to the local LLM.
        llm_result = self.reasoner.analyze(
            question=self.state.question,
            evidence=self.state.evidence,
            hypotheses=self.state.hypotheses,
        )

        # Preserve the complete structured LLM result for deterministic
        # verification in the next workflow stage.
        self.llm_result = llm_result

        self.state.log_event(
            "llm_analysis_completed",
            {
                "summary": llm_result.get("summary"),
                "likely_cause": llm_result.get("likely_cause"),
                "confidence": llm_result.get("confidence"),
            },
        )

        # Store the LLM recommendation in workflow state.
        self.state.recommendation = llm_result.get(
            "recommended_action",
            "",
        )

        # Preserve the complete LLM reasoning output in the audit log.
        self.state.log_event(
            "llm_reasoning_recorded",
            {
                "llm_result": llm_result,
            },
        )

    # ---------------------------------------------------------
    # VERIFY
    # ---------------------------------------------------------

    def verify(self) -> None:
        self.state.current_step = "VERIFY"

        verification_result = self.verifier.verify(
            question=self.state.question,
            evidence=self.state.evidence,
            hypotheses=self.state.hypotheses,
            llm_result=self.llm_result,
        )

        self.state.verification = verification_result

        if verification_result.get("passed", False):
            self.state.risk_level = "high"
            self.state.approval_required = True
        else:
            self.state.risk_level = "unknown"
            self.state.approval_required = True

        self.state.log_event(
            "verification_completed",
            verification_result,
        )

    # ---------------------------------------------------------
    # ANSWER
    # ---------------------------------------------------------

    def answer(self) -> None:
        self.state.current_step = "ANSWER"

        if not self.state.verification.get("passed", False):
            self.state.final_answer = (
                "The available evidence is insufficient to confidently "
                "determine the root cause of the payment failures."
            )

            self.state.log_event(
                "answer_generated",
                {
                    "status": "insufficient_evidence",
                },
            )

            return

        likely_cause = self.llm_result.get(
            "likely_cause",
            "",
        )

        recommended_action = self.llm_result.get(
            "recommended_action",
            self.state.recommendation,
        )

        self.state.final_answer = (
            f"Payment failures were investigated using operational "
            f"evidence from incidents, deployments, service status, "
            f"runbooks, and production policy. The verified likely "
            f"cause is: {likely_cause} "
            f"The recommended action is: {recommended_action} "
            f"Because the recommended remediation is potentially "
            f"production-impacting, human approval is required."
        )

        self.state.log_event(
            "answer_generated",
            {
                "status": "verified",
                "risk_level": self.state.risk_level,
                "approval_required": self.state.approval_required,
            },
        )


# =============================================================
# DEMO
# =============================================================

def main() -> None:
    question = (
        "Payment failures increased after the latest deployment. "
        "Investigate the likely cause using incidents, deployment "
        "history, service dependencies and runbooks. Recommend a "
        "remediation and determine whether approval is required."
    )

    state = InvestigationState(
        question=question,
    )

    workflow = InvestigationWorkflow(state)

    result = workflow.run()

    print("\nQUESTION")
    print("=" * 60)
    print(result.question)

    print("\nPLAN")
    print("=" * 60)
    print(json.dumps(result.plan, indent=2))

    print("\nHYPOTHESES")
    print("=" * 60)
    print(json.dumps(result.hypotheses, indent=2))

    print("\nVERIFICATION")
    print("=" * 60)
    print(json.dumps(result.verification, indent=2))

    print("\nRISK")
    print("=" * 60)
    print(result.risk_level)

    print("\nAPPROVAL REQUIRED")
    print("=" * 60)
    print(result.approval_required)

    print("\nRECOMMENDATION")
    print("=" * 60)
    print(result.recommendation)

    print("\nFINAL ANSWER")
    print("=" * 60)
    print(result.final_answer)

    print("\nAUDIT LOG")
    print("=" * 60)
    print(json.dumps(result.audit_log, indent=2))


if __name__ == "__main__":
    main()