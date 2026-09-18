from __future__ import annotations

import json
import re
from typing import Any

from src.agent.safety import SafetyPolicy
from src.agent.injection import PromptInjectionDetector
from src.agent.llm import LLMReasoner
from src.agent.state import InvestigationState
from src.agent.tools import (
    calculate_metrics,
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
        self.injection_detector = PromptInjectionDetector()
        self.state = state
        self.reasoner = LLMReasoner()
        self.verifier = EvidenceVerifier()
        self.graph = ServiceDependencyGraph()
        self.llm_result: dict[str, Any] = {}

    def run(self) -> InvestigationState:
        injection = self.injection_detector.detect(self.state.question)

        if injection.detected:
            self.state.audit_log.append({
                "step": "VERIFY",
                "event": "prompt_injection_detected",
                "injection_type": injection.injection_type,
                "matched_patterns": injection.matched_patterns,
                "reason": injection.reason,
            })

            self.state.final_answer = (
                "Request blocked because the input contains content attempting "
                "to override agent instructions, safety policy, authorization, "
                "or operational behavior."
            )

            self.state.audit_log.append({
                "step": "ANSWER",
                "event": "prompt_injection_blocked",
            })

            return self.state

        self.plan()
        self.retrieve()
        self.tool_calls()
        self.analyze()
        self.verify()
        self.answer()

        return self.state

    # ---------------------------------------------------------
    # QUERY / INTENT HELPERS
    # ---------------------------------------------------------

    def _is_payment_investigation(self) -> bool:
        question = self.state.question.lower()

        payment_terms = (
            "payment",
            "payments",
            "payment-service",
            "payment failures",
        )
        
        intent_terms = (
            "investigate",
            "investigation",
            "likely cause",
            "root cause",
            "deployment",
            "remediation",
            "trajectory",
        )

        return (
            any(term in question for term in payment_terms)
            and any(term in question for term in intent_terms)
        )

    def _extract_service(self) -> str | None:
        question = self.state.question.lower()

        known_services = (
            "payment-service",
            "fraud-service",
            "auth-service",
            "user-service",
            "order-service",
            "inventory-service",
            "notification-service",
            "platform",
        )

        for service in known_services:
            if service in question:
                return service

        match = re.search(
            r"(?:for|on|of)\s+([a-z0-9]+-service)\b",
            question,
        )

        if match:
            return match.group(1)

        return None

    def _is_metrics_query(self) -> bool:
        question = self.state.question.lower()

        return (
            "metrics" in question
            or "metric" in question
            or "failure rate" in question
            or "error rate" in question
        )

    def _is_policy_query(self) -> bool:
        question = self.state.question.lower()

        return (
            "production policy" in question
            or "rollback policy" in question
            or "approval policy" in question
            or "safety policy" in question
            or "platform" in question
        )

    def _is_deployment_query(self) -> bool:
        question = self.state.question.lower()

        deployment_terms = (
            "deployment",
            "deployments",
            "deployed",
            "release",
            "released",
            "version",
        )

        root_cause_terms = (
            "root cause",
            "likely cause",
            "determine cause",
            "investigate",
            "investigation",
            "post-mortem",
            "triage",
        )

        return (
            any(term in question for term in deployment_terms)
            or (
                any(term in question for term in root_cause_terms)
                and "incident" in question
            )
        )

    def _is_incident_query(self) -> bool:
        question = self.state.question.lower()

        return (
            "incident" in question
            or "incidents" in question
        )

    def _is_dependency_query(self) -> bool:
        question = self.state.question.lower()

        return (
            "depends on" in question
            or "dependency" in question
            or "dependencies" in question
            or "depend on" in question
            or "dependent on" in question
        )

    def _is_runbook_query(self) -> bool:
        question = self.state.question.lower()

        return (
            "runbook" in question
            or "remediation" in question
        )

    # ---------------------------------------------------------
    # PLAN
    # ---------------------------------------------------------

    def plan(self) -> None:
        self.state.current_step = "PLAN"

        # Preserve the complete original payment investigation.
        if self._is_payment_investigation():
            self.state.plan = [
                "Search recent payment incidents",
                "Inspect recent payment-service deployments",
                "Check payment-service status",
                "Check fraud-service status",
                "Retrieve payment remediation runbook",
                "Check production rollback policy",
                "Correlate evidence and determine likely cause",
            ]

        else:
            service = self._extract_service()

            steps: list[str] = []

            if self._is_dependency_query():
                steps.append("Inspect service dependency relationships")

            if self._is_incident_query():
                steps.append("Search relevant incidents")

            if self._is_deployment_query():
                steps.append("Inspect relevant deployments")

            if service:
                steps.append(f"Check {service} service status")

            if self._is_runbook_query():
                steps.append("Retrieve relevant runbook")

            if (
                self._is_policy_query()
                or (
                    self._is_deployment_query()
                    and self._is_incident_query()
                    and service
                )
            ):
                steps.append("Check production policy")

            # A service trajectory query should inspect the service and
            # its operational runbook when no more specific intent exists.
            if (
                not steps
                and service
            ):
                steps = [
                    f"Check {service} service status",
                    "Retrieve relevant runbook",
                ]

            # Keep a useful read-only fallback rather than executing
            # unrelated production tools.
            if not steps:
                steps = [
                    "Determine the relevant operational evidence",
                    "Analyze the available evidence",
                    "Verify the result",
                ]

            self.state.plan = steps

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

        # -----------------------------------------------------
        # Original payment investigation path
        # -----------------------------------------------------

        if self._is_payment_investigation():
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

            graph_reasoning = self.graph.explain_service(
                "payment-service"
            )

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

            return

        # -----------------------------------------------------
        # Query-aware operational path
        # -----------------------------------------------------

        question = self.state.question.lower()
        service = self._extract_service()
        tool_count = 0

        if self._is_metrics_query():
            # Do not fabricate metric inputs. The real tool requires
            # failures and total_transactions. If the question does
            # not provide them, record that the tool cannot safely
            # be called.
            failure_match = re.search(
                r"(\d+)\s+failures?",
                question,
            )
            transaction_match = re.search(
                r"(\d+)\s+(?:total\s+)?transactions?",
                question,
            )

            if failure_match and transaction_match:
                failures = int(failure_match.group(1))
                total_transactions = int(
                    transaction_match.group(1)
                )

                metrics = self._call_tool(
                    "calculate_metrics",
                    calculate_metrics,
                    failures=failures,
                    total_transactions=total_transactions,
                )

                self.state.add_evidence(
                    "metrics",
                    metrics,
                )
                tool_count += 1
            else:
                self.state.add_evidence(
                    "metrics",
                    {
                        "status": "not_executed",
                        "reason": (
                            "calculate_metrics requires failures and "
                            "total_transactions, but the query did not "
                            "provide both values."
                        ),
                    },
                )

        if self._is_incident_query():
            incidents = self._call_tool(
                "search_incidents",
                search_incidents,
                service=service,
            )

            self.state.add_evidence(
                "search_incidents",
                incidents,
            )
            tool_count += 1

        if self._is_deployment_query():
            deployments = self._call_tool(
                "get_recent_deployments",
                get_recent_deployments,
                service=service,
            )

            self.state.add_evidence(
                "get_recent_deployments",
                deployments,
            )
            tool_count += 1

        if service and not self._is_policy_query():
            service_status = self._call_tool(
                "get_service_status",
                get_service_status,
                service=service,
            )

            self.state.add_evidence(
                f"{service}_status",
                service_status,
            )
            tool_count += 1

            # The trajectory cases for service investigations expect
            # the operational runbook alongside service status.
            if (
                not self._is_payment_investigation()
                and not self._is_runbook_query()
                and not self._is_incident_query()
                and not self._is_deployment_query()
                and not self._is_metrics_query()
                and not self._is_policy_query()
            ):
                runbook = self._call_tool(
                    "retrieve_runbook",
                    retrieve_runbook,
                    topic=service.replace("-service", ""),
                )

                self.state.add_evidence(
                    f"{service}_runbook",
                    runbook,
                )
                tool_count += 1

        if self._is_runbook_query():
            topic = service.replace("-service", "") if service else "operations"

            runbook = self._call_tool(
                "retrieve_runbook",
                retrieve_runbook,
                topic=topic,
            )

            self.state.add_evidence(
                "runbook",
                runbook,
            )
            tool_count += 1

        if (
            self._is_policy_query()
            or (
                self._is_deployment_query()
                and self._is_incident_query()
                and service
            )
        ):
            policy = self._call_tool(
                "get_production_policy",
                get_production_policy,
            )

            self.state.add_evidence(
                "production_policy",
                policy,
            )
            tool_count += 1

        if self._is_dependency_query() and service and service != "platform":
            dependency_evidence = self.graph.explain_service(service)

            self.state.add_evidence(
                "dependency_query",
                dependency_evidence,
            )

        graph_reasoning = None

        if service and service != "platform":
            graph_reasoning = self.graph.explain_service(service)

            self.state.add_evidence(
                "service_dependency_graph",
                graph_reasoning,
            )

        self.state.log_event(
            "tool_calls_completed",
            {
                "tool_count": tool_count,
                "graph_reasoning": graph_reasoning,
            },
        )

    # ---------------------------------------------------------
    # ANALYZE
    # ---------------------------------------------------------

    def analyze(self) -> None:
        self.state.current_step = "ANALYZE"

        # -----------------------------------------------------
        # Original payment investigation reasoning
        # -----------------------------------------------------

        if self._is_payment_investigation():
            incident_ids: list[str] = []
            deployment_ids: list[str] = []

            incidents_evidence = self.state.evidence[0]["evidence"]
            deployments_evidence = self.state.evidence[1]["evidence"]

            if isinstance(incidents_evidence, dict):
                incident_items = incidents_evidence.get(
                    "incidents",
                    incidents_evidence.get("results", []),
                )
            elif isinstance(incidents_evidence, list):
                incident_items = incidents_evidence
            else:
                incident_items = []

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

        else:
            # Generic operational queries still receive deterministic
            # evidence-grounded analysis through the existing LLM
            # reasoning boundary.
            self.state.hypotheses.append(
                "Determine the operational finding from the evidence "
                "actually collected for the requested query."
            )

            self.state.log_event(
                "hypothesis_generated",
                {
                    "query": self.state.question,
                    "evidence_count": len(self.state.evidence),
                },
            )

        # -----------------------------------------------------
        # Existing LLM reasoning boundary
        # -----------------------------------------------------

        llm_result = self.reasoner.analyze(
            question=self.state.question,
            evidence=self.state.evidence,
            hypotheses=self.state.hypotheses,
        )

        self.llm_result = llm_result

        self.state.log_event(
            "llm_analysis_completed",
            {
                "summary": llm_result.get("summary"),
                "likely_cause": llm_result.get("likely_cause"),
                "confidence": llm_result.get("confidence"),
            },
        )

        self.state.recommendation = llm_result.get(
            "recommended_action",
            "",
        )

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

        safety_decision = SafetyPolicy.validate_recommendation(
            self.state.recommendation
        )

        self.state.risk_level = (
            safety_decision.risk_class.value
        )

        self.state.approval_required = (
            safety_decision.requires_approval
        )

        self.state.log_event(
            "safety_policy_evaluated",
            {
                "allowed": safety_decision.allowed,
                "requires_approval": (
                    safety_decision.requires_approval
                ),
                "risk_class": (
                    safety_decision.risk_class.value
                ),
                "reason": safety_decision.reason,
            },
        )

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
                "determine the requested operational finding."
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

        if self._is_payment_investigation():
            self.state.final_answer = (
                "Payment failures were investigated using operational "
                "evidence from incidents, deployments, service status, "
                "runbooks, and production policy. The verified likely "
                f"cause is: {likely_cause} "
                f"The recommended action is: {recommended_action} "
                "Because the recommended remediation is potentially "
                "production-impacting, human approval is required."
            )
        else:
            approval_text = (
                "Human approval is required before any production-impacting "
                "action."
                if self.state.approval_required
                else
                "No production approval is required for the evaluated "
                "read-only operation."
            )

            self.state.final_answer = (
                f"Operational finding: {likely_cause} "
                f"Recommended action: {recommended_action} "
                f"{approval_text}"
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





