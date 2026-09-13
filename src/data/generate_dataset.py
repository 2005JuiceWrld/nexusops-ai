from __future__ import annotations

import json
from pathlib import Path

OUTPUT = Path("data/raw/operations.json")


DOCUMENTS = [
    {
        "id": "INC-001",
        "type": "incident",
        "title": "Elevated payment failures after payment-service deployment",
        "service": "payment-service",
        "severity": "SEV-1",
        "content": (
            "Payment failures increased significantly after deployment "
            "payment-service version 2.4.1. The incident began shortly "
            "after the deployment completed. Error responses from the "
            "payment service increased while successful transactions "
            "decreased. Initial investigation identified elevated latency "
            "between payment-service and fraud-service."
        ),
    },
    {
        "id": "DEP-001",
        "type": "deployment",
        "title": "payment-service version 2.4.1 deployment",
        "service": "payment-service",
        "version": "2.4.1",
        "content": (
            "payment-service version 2.4.1 was deployed to production. "
            "The deployment introduced changes to fraud verification "
            "request handling and timeout configuration."
        ),
    },
    {
        "id": "INC-002",
        "type": "incident",
        "title": "Fraud service latency increase",
        "service": "fraud-service",
        "severity": "SEV-2",
        "content": (
            "fraud-service experienced elevated response latency. "
            "The service remained available but p95 latency increased "
            "above the normal operating range. Dependent services "
            "reported increased timeout rates."
        ),
    },
    {
        "id": "SVC-001",
        "type": "service",
        "title": "payment-service service profile",
        "service": "payment-service",
        "content": (
            "payment-service processes customer payment transactions. "
            "It depends on fraud-service for transaction risk verification "
            "before completing payment authorization. Increased latency "
            "from fraud-service can cause payment-service timeouts."
        ),
    },
    {
        "id": "SVC-002",
        "type": "service",
        "title": "fraud-service service profile",
        "service": "fraud-service",
        "content": (
            "fraud-service performs fraud and risk verification for "
            "payment transactions. payment-service is a primary upstream "
            "consumer. Normal p95 latency is below 250 milliseconds."
        ),
    },
    {
        "id": "RUN-001",
        "type": "runbook",
        "title": "Payment failure investigation runbook",
        "service": "payment-service",
        "content": (
            "When payment failures increase, first compare failure rate "
            "before and after the most recent deployment. Check payment "
            "service latency and downstream dependency latency. Inspect "
            "recent deployment configuration changes. If a deployment "
            "correlates strongly with the incident, rollback should be "
            "considered according to the deployment rollback policy."
        ),
    },
    {
        "id": "RUN-002",
        "type": "runbook",
        "title": "Fraud service latency investigation",
        "service": "fraud-service",
        "content": (
            "For elevated fraud-service latency, inspect p95 and p99 "
            "latency, request volume, error rate, and downstream database "
            "latency. Check recent deployments before making infrastructure "
            "changes."
        ),
    },
    {
        "id": "POL-001",
        "type": "policy",
        "title": "Production rollback approval policy",
        "service": "platform",
        "content": (
            "Production deployments may be rolled back when an incident "
            "is strongly associated with a recent release. Automated "
            "rollback actions require an authorized human approval unless "
            "the incident is covered by a pre-approved emergency policy."
        ),
    },
    {
        "id": "DEP-002",
        "type": "deployment",
        "title": "fraud-service version 5.8.0 deployment",
        "service": "fraud-service",
        "version": "5.8.0",
        "content": (
            "fraud-service version 5.8.0 was deployed before the payment "
            "incident. The release modified request batching behavior."
        ),
    },
]


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(DOCUMENTS, f, indent=2)

    print(f"Created {len(DOCUMENTS)} documents")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()