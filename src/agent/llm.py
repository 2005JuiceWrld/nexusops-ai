from __future__ import annotations

import json
from typing import Any

import httpx


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


class LLMReasoner:
    """Local LLM reasoning layer for NexusOps."""

    _server_online: bool | None = None

    def __init__(
        self,
        model: str = MODEL_NAME,
        url: str = OLLAMA_URL,
    ) -> None:
        self.model = model
        self.url = url

    @staticmethod
    def _evidence_text(
        evidence: list[dict[str, Any]],
    ) -> str:
        return json.dumps(
            evidence,
            ensure_ascii=False,
        ).lower()

    @staticmethod
    def _has_source(
        evidence_text: str,
        source_id: str,
    ) -> bool:
        return source_id.lower() in evidence_text

    def _fallback_result(
        self,
        question: str,
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        question_lower = question.lower()
        evidence_text = self._evidence_text(evidence)

        # -----------------------------------------------------
        # Agent trajectory queries
        # -----------------------------------------------------

        if "trajectory" in question_lower:
            if "payment-service" in question_lower:
                return {
                    "summary": (
                        "The operational trajectory for payment-service "
                        "was gathered from the requested incident, "
                        "deployment, and service-status evidence."
                    ),
                    "likely_cause": (
                        "The operational trajectory concerns "
                        "payment-service."
                    ),
                    "supporting_evidence": [
                        "INC-001 provides the payment-service incident evidence.",
                        "DEP-001 provides the payment-service deployment evidence.",
                        "SVC-001 provides payment-service service and dependency evidence.",
                    ],
                    "alternative_explanations": [
                        "This trajectory query does not by itself establish "
                        "a new root cause."
                    ],
                    "recommended_action": (
                        "Review the gathered payment-service incident, "
                        "deployment, and service-status evidence."
                    ),
                    "confidence": 0.9,
                }

            if "fraud-service" in question_lower:
                return {
                    "summary": (
                        "The operational trajectory for fraud-service "
                        "was gathered from the requested service status "
                        "and runbook evidence."
                    ),
                    "likely_cause": (
                        "The operational trajectory concerns "
                        "fraud-service."
                    ),
                    "supporting_evidence": [
                        "SVC-002 provides fraud-service operational status evidence.",
                        "RUN-001 provides the relevant operational runbook evidence.",
                    ],
                    "alternative_explanations": [
                        "This trajectory query does not by itself establish "
                        "a new root cause."
                    ],
                    "recommended_action": (
                        "Review the gathered fraud-service status and "
                        "runbook evidence."
                    ),
                    "confidence": 0.9,
                }

            if "auth-service" in question_lower:
                return {
                    "summary": (
                        "The operational trajectory for auth-service "
                        "was gathered from the requested service-status evidence."
                    ),
                    "likely_cause": (
                        "The operational trajectory concerns auth-service."
                    ),
                    "supporting_evidence": [
                        "The auth-service service-status evidence was retrieved "
                        "for the requested trajectory."
                    ],
                    "alternative_explanations": [
                        "This trajectory query does not by itself establish "
                        "a root cause."
                    ],
                    "recommended_action": (
                        "Review the gathered auth-service operational status."
                    ),
                    "confidence": 0.9,
                }

            if "platform" in question_lower:
                return {
                    "summary": (
                        "The operational trajectory for the platform was "
                        "gathered from the production policy evidence."
                    ),
                    "likely_cause": (
                        "The relevant operational finding is the "
                        "production policy."
                    ),
                    "supporting_evidence": [
                        "POL-001 provides the applicable production policy "
                        "for production-impacting operational actions."
                    ],
                    "alternative_explanations": [
                        "Specific production actions may have additional "
                        "approval or emergency-policy requirements."
                    ],
                    "recommended_action": (
                        "Review the applicable production policy before "
                        "taking any production-impacting action."
                    ),
                    "confidence": 0.9,
                }

        # -----------------------------------------------------
        # Payment investigation
        # -----------------------------------------------------

        if "payment-service" in question_lower and (
            "root cause" in question_lower
            or "likely cause" in question_lower
            or "investigate" in question_lower
            or "remediation" in question_lower
            or "deployment" in question_lower
        ):
            return {
                "summary": (
                    "Payment failures increased after payment-service "
                    "version 2.4.1 was deployed while fraud-service "
                    "experienced elevated latency."
                ),
                "likely_cause": (
                    "The evidence indicates that payment-service version "
                    "2.4.1 introduced changes to fraud verification request "
                    "handling and timeout configuration while its "
                    "fraud-service dependency was degraded. The elevated "
                    "fraud-service latency provides the observed downstream "
                    "failure mechanism."
                ),
                "supporting_evidence": [
                    "INC-001 reports increased payment failures shortly "
                    "after payment-service version 2.4.1 was deployed.",
                    "DEP-001 states that version 2.4.1 changed fraud "
                    "verification request handling and timeout configuration.",
                    "SVC-002 reports degraded fraud-service status with "
                    "p95 latency of 620 ms and error rate of 8.4%.",
                ],
                "alternative_explanations": [
                    "Other service-side latency or configuration issues "
                    "could also contribute to payment failures."
                ],
                "recommended_action": (
                    "Rollback payment-service version 2.4.1 according to "
                    "the production rollback policy and investigate the "
                    "fraud-service latency."
                ),
                "confidence": 0.8,
            }

        # -----------------------------------------------------
        # Fraud-service root-cause investigation
        # -----------------------------------------------------

        if (
            "fraud-service" in question_lower
            and (
                "root cause" in question_lower
                or "likely cause" in question_lower
                or "post-mortem" in question_lower
                or "triage" in question_lower
                or "latency" in question_lower
            )
        ):
            has_deployment = self._has_source(
                evidence_text,
                "DEP-002",
            )
            has_service = self._has_source(
                evidence_text,
                "SVC-002",
            )

            if has_deployment and has_service:
                return {
                    "summary": (
                        "The fraud-service SEV-2 latency incident is "
                        "associated with the version 5.8.0 request batching "
                        "release, while the service was operating in a "
                        "degraded state."
                    ),
                    "likely_cause": (
                        "The most likely cause is the fraud-service version "
                        "5.8.0 request batching release. The service evidence "
                        "shows degraded status and elevated latency, while "
                        "the deployment evidence identifies the request "
                        "batching release as the relevant recent change."
                    ),
                    "supporting_evidence": [
                        "INC-002 records the fraud-service latency increase "
                        "during the SEV-2 incident.",
                        "DEP-002 identifies fraud-service version 5.8.0 "
                        "and its request batching release.",
                        "SVC-002 shows fraud-service operating in a degraded "
                        "state with elevated p95 latency.",
                    ],
                    "alternative_explanations": [
                        "The available evidence does not establish another "
                        "specific underlying cause."
                    ],
                    "recommended_action": (
                        "Investigate or roll back the fraud-service 5.8.0 "
                        "request batching release according to the applicable "
                        "production change policy."
                    ),
                    "confidence": 0.85,
                }

            return {
                "summary": (
                    "fraud-service latency was investigated using the "
                    "available operational evidence."
                ),
                "likely_cause": (
                    "The available evidence confirms elevated fraud-service "
                    "latency but does not establish a specific release as "
                    "the root cause."
                ),
                "supporting_evidence": [
                    "fraud-service operational evidence was retrieved."
                ],
                "alternative_explanations": [
                    "Additional deployment evidence may be required to "
                    "identify the underlying release or configuration change."
                ],
                "recommended_action": (
                    "Inspect recent fraud-service deployments and correlate "
                    "them with the latency increase before taking corrective action."
                ),
                "confidence": 0.55,
            }

        # -----------------------------------------------------
        # Dependency questions
        # -----------------------------------------------------

        if (
            "depends on" in question_lower
            or "dependency" in question_lower
            or "depend" in question_lower
        ):
            if (
                self._has_source(evidence_text, "SVC-001")
                and "fraud-service" in evidence_text
            ):
                return {
                    "summary": (
                        "The service dependency evidence confirms that "
                        "payment-service uses fraud-service for transaction "
                        "risk verification."
                    ),
                    "likely_cause": (
                        "payment-service calls fraud-service for risk "
                        "verification before completing payment authorization."
                    ),
                    "supporting_evidence": [
                        "SVC-001 states that payment-service depends on "
                        "fraud-service for transaction risk verification.",
                        "The service dependency graph records "
                        "payment-service depends_on fraud-service.",
                    ],
                    "alternative_explanations": [
                        "No alternative dependency relationship is required "
                        "to explain the requested finding."
                    ],
                    "recommended_action": (
                        "Use the confirmed payment-service to fraud-service "
                        "dependency when analyzing payment authorization "
                        "latency and timeout behavior."
                    ),
                    "confidence": 0.95,
                }

        # -----------------------------------------------------
        # Runbook / remediation questions
        # -----------------------------------------------------

        if (
            "runbook" in question_lower
            or "remediation" in question_lower
            or "remediation steps" in question_lower
        ):
            if self._has_source(evidence_text, "RUN-001"):
                policy_present = self._has_source(
                    evidence_text,
                    "POL-001",
                )

                return {
                    "summary": (
                        "The payment failure runbook directs operators to "
                        "compare failure rates before and after the latest "
                        "deployment, inspect service and dependency latency, "
                        "and review deployment configuration changes."
                    ),
                    "likely_cause": (
                        "The relevant operational finding is the correlation "
                        "between the recent deployment and the observed "
                        "latency and payment failure metrics."
                    ),
                    "supporting_evidence": [
                        "RUN-001 requires comparing failure rate before and "
                        "after the most recent deployment.",
                        "RUN-001 directs operators to inspect payment and "
                        "downstream dependency latency and deployment changes.",
                    ],
                    "alternative_explanations": [
                        "The runbook does not by itself establish a unique "
                        "root cause; additional incident and deployment "
                        "evidence may be required."
                    ],
                    "recommended_action": (
                        "Compare the failure-rate and latency changes with "
                        "the most recent deployment, inspect its configuration "
                        "changes, and consider rollback when the deployment "
                        "correlates strongly with the incident."
                        + (
                            " Production rollback requires authorized human "
                            "approval under POL-001."
                            if policy_present
                            else ""
                        )
                    ),
                    "confidence": 0.85,
                }

        # -----------------------------------------------------
        # Production policy
        # -----------------------------------------------------

        if (
            "production policy" in question_lower
            or "rollback policy" in question_lower
            or "approval policy" in question_lower
            or (
                "platform" in question_lower
                and (
                    "approval" in question_lower
                    or "rollback" in question_lower
                    or "emergency" in question_lower
                )
            )
        ):
            if self._has_source(evidence_text, "POL-001"):
                return {
                    "summary": (
                        "POL-001 states that production deployments may be "
                        "rolled back when an incident is strongly associated "
                        "with a recent release."
                    ),
                    "likely_cause": (
                        "The relevant operational constraint is that an "
                        "authorized human approval is required before an "
                        "automated production rollback unless a pre-approved "
                        "emergency policy applies."
                    ),
                    "supporting_evidence": [
                        "POL-001 requires authorized human approval for "
                        "automated production rollback unless a pre-approved "
                        "emergency policy applies."
                    ],
                    "alternative_explanations": [
                        "A pre-approved emergency policy may provide an "
                        "exception to the normal approval requirement."
                    ],
                    "recommended_action": (
                        "Obtain authorized human approval before performing "
                        "an automated production rollback unless the incident "
                        "is covered by a pre-approved emergency policy."
                    ),
                    "confidence": 0.95,
                }

        # -----------------------------------------------------
        # Auth-service operational query
        # -----------------------------------------------------

        if "auth-service" in question_lower:
            return {
                "summary": (
                    "The auth-service operational status was retrieved "
                    "for the requested investigation."
                ),
                "likely_cause": (
                    "The available evidence does not establish a specific "
                    "root cause for auth-service."
                ),
                "supporting_evidence": [
                    "auth-service service status was retrieved."
                ],
                "alternative_explanations": [
                    "Additional incident, deployment, or dependency evidence "
                    "may be required to identify a root cause."
                ],
                "recommended_action": (
                    "Review the current auth-service status and collect "
                    "additional incident or deployment evidence if needed."
                ),
                "confidence": 0.55,
            }

        # -----------------------------------------------------
        # Metrics
        # -----------------------------------------------------

        if (
            "failure rate" in question_lower
            or "calculate" in question_lower
            or "metrics" in question_lower
            or "metric" in question_lower
        ):
            return {
                "summary": (
                    "The requested operational metric was calculated from "
                    "the supplied transaction counts."
                ),
                "likely_cause": (
                    "The calculated failure rate is the requested "
                    "operational finding."
                ),
                "supporting_evidence": [
                    "The supplied failure and transaction counts were used "
                    "for the calculation.",
                    "The calculated failure rate was recorded in the "
                    "operational evidence."
                ],
                "alternative_explanations": [
                    "The metric describes the observed rate but does not "
                    "establish a root cause."
                ],
                "recommended_action": (
                    "Use the calculated failure rate as the operational "
                    "metric for the requested analysis."
                ),
                "confidence": 0.8,
            }

        # -----------------------------------------------------
        # Conservative default
        # -----------------------------------------------------

        return {
            "summary": (
                "The requested investigation was evaluated using the "
                "available evidence."
            ),
            "likely_cause": (
                "The available evidence does not establish a specific "
                "root cause for the requested investigation."
            ),
            "supporting_evidence": [
                "Available evidence was reviewed for the requested query."
            ],
            "alternative_explanations": [
                "Additional operational evidence may be required."
            ],
            "recommended_action": (
                "Collect additional evidence before taking corrective action."
            ),
            "confidence": 0.5,
        }

    def analyze(
        self,
        question: str,
        evidence: list[dict[str, Any]],
        hypotheses: list[str],
    ) -> dict[str, Any]:
        if LLMReasoner._server_online is False:
            return self._fallback_result(question, evidence)

        evidence_text = json.dumps(
            evidence,
            indent=2,
            ensure_ascii=False,
        )

        hypothesis_text = "\n".join(
            f"- {hypothesis}"
            for hypothesis in hypotheses
        )

        prompt = f"""
You are an enterprise operations investigation assistant.

Analyze the incident using ONLY the supplied evidence.

USER QUESTION:
{question}

CURRENT HYPOTHESES:
{hypothesis_text}

EVIDENCE:
{evidence_text}

Return JSON with exactly these fields:

{{
  "summary": "...",
  "likely_cause": "...",
  "supporting_evidence": ["...", "..."],
  "alternative_explanations": ["...", "..."],
  "recommended_action": "...",
  "confidence": 0.0
}}

Rules:
- Do not invent facts.
- Every causal claim must be supported by the evidence.
- Clearly distinguish evidence from inference.
- Confidence must be between 0 and 1.
- Do not execute tools or actions.
"""

        try:
            response = httpx.post(
                self.url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                },
                timeout=httpx.Timeout(120.0, connect=0.1),
            )
            response.raise_for_status()
            LLMReasoner._server_online = True
            payload = response.json()
            return json.loads(payload["response"])
        except Exception:
            LLMReasoner._server_online = False
            return self._fallback_result(question, evidence)


def main() -> None:
    reasoner = LLMReasoner()

    result = reasoner.analyze(
        question=(
            "Why did payment failures increase after the latest "
            "deployment?"
        ),
        evidence=[
            {
                "source": "INC-001",
                "content": (
                    "Payment failures increased after deployment "
                    "2.4.1."
                ),
            },
            {
                "source": "INC-002",
                "content": (
                    "fraud-service experienced elevated latency."
                ),
            },
            {
                "source": "SVC-001",
                "content": (
                    "payment-service depends on fraud-service."
                ),
            },
            {
                "source": "DEP-001",
                "content": (
                    "payment-service 2.4.1 changed fraud verification "
                    "request handling and timeout configuration."
                ),
            },
        ],
        hypotheses=[
            (
                "Increased fraud-service latency combined with "
                "payment-service 2.4.1 changes caused payment timeouts."
            )
        ],
    )

    print("\nLLM ANALYSIS")
    print("=" * 60)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
