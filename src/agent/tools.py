from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DATA_PATH = Path("data/raw/operations.json")


def load_documents() -> list[dict[str, Any]]:
    with DATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _documents_by_type(document_type: str) -> list[dict[str, Any]]:
    return [
        doc
        for doc in load_documents()
        if doc.get("type") == document_type
    ]


def search_incidents(
    service: str | None = None,
) -> dict[str, Any]:
    """Find operational incidents."""

    incidents = _documents_by_type("incident")

    if service:
        incidents = [
            incident
            for incident in incidents
            if incident.get("service") == service
        ]

    return {
        "tool": "search_incidents",
        "success": True,
        "count": len(incidents),
        "results": incidents,
    }


def get_recent_deployments(
    service: str | None = None,
) -> dict[str, Any]:
    """Return deployment history."""

    deployments = _documents_by_type("deployment")

    if service:
        deployments = [
            deployment
            for deployment in deployments
            if deployment.get("service") == service
        ]

    return {
        "tool": "get_recent_deployments",
        "success": True,
        "count": len(deployments),
        "results": deployments,
    }


def get_service_status(
    service: str,
) -> dict[str, Any]:
    """Return current service status from the operational dataset."""

    services = _documents_by_type("service")

    matches = [
        item
        for item in services
        if item.get("service") == service
    ]

    if not matches:
        return {
            "tool": "get_service_status",
            "success": False,
            "error": f"Unknown service: {service}",
        }

    service_doc = matches[0]

    # Deterministic simulated operational state.
    status = {
        "service": service,
        "status": "degraded"
        if service == "fraud-service"
        else "operational",
        "p95_latency_ms": 620
        if service == "fraud-service"
        else 180,
        "error_rate_percent": 8.4
        if service == "fraud-service"
        else 0.7,
    }

    return {
        "tool": "get_service_status",
        "success": True,
        "service_profile": service_doc,
        "status": status,
    }


def retrieve_runbook(
    topic: str,
) -> dict[str, Any]:
    """Retrieve runbooks relevant to an operational topic."""

    runbooks = _documents_by_type("runbook")

    topic_lower = topic.lower()

    matches = [
        runbook
        for runbook in runbooks
        if (
            topic_lower in runbook.get("title", "").lower()
            or topic_lower in runbook.get("content", "").lower()
            or topic_lower in runbook.get("service", "").lower()
        )
    ]

    return {
        "tool": "retrieve_runbook",
        "success": True,
        "count": len(matches),
        "results": matches,
    }


def get_production_policy() -> dict[str, Any]:
    """Retrieve production action approval policy."""

    policies = _documents_by_type("policy")

    return {
        "tool": "get_production_policy",
        "success": True,
        "results": policies,
    }


def calculate_metrics(
    failures: int,
    total_transactions: int,
) -> dict[str, Any]:
    """Calculate a basic failure rate."""

    if total_transactions <= 0:
        return {
            "tool": "calculate_metrics",
            "success": False,
            "error": "total_transactions must be greater than zero",
        }

    failure_rate = failures / total_transactions

    return {
        "tool": "calculate_metrics",
        "success": True,
        "failures": failures,
        "total_transactions": total_transactions,
        "failure_rate": failure_rate,
        "failure_rate_percent": failure_rate * 100,
    }


def restart_service(
    service: str,
    approved: bool = False,
) -> dict[str, Any]:
    """
    Simulated high-risk action.

    No real service is restarted.
    """

    if not approved:
        return {
            "tool": "restart_service",
            "success": False,
            "requires_approval": True,
            "message": (
                f"Restart of {service} requires authorized "
                "human approval."
            ),
        }

    return {
        "tool": "restart_service",
        "success": True,
        "simulated": True,
        "requires_approval": False,
        "message": (
            f"Simulated restart request approved for {service}."
        ),
    }


TOOL_REGISTRY = {
    "search_incidents": search_incidents,
    "get_recent_deployments": get_recent_deployments,
    "get_service_status": get_service_status,
    "retrieve_runbook": retrieve_runbook,
    "get_production_policy": get_production_policy,
    "calculate_metrics": calculate_metrics,
    "restart_service": restart_service,
}


if __name__ == "__main__":
    print("\nNEXUSOPS TOOL TEST")
    print("=" * 60)

    print("\n1. Incidents")
    print(search_incidents("payment-service"))

    print("\n2. Deployments")
    print(get_recent_deployments("payment-service"))

    print("\n3. Service status")
    print(get_service_status("fraud-service"))

    print("\n4. Runbook")
    print(retrieve_runbook("payment"))

    print("\n5. Production policy")
    print(get_production_policy())

    print("\n6. High-risk action")
    print(restart_service("payment-service"))