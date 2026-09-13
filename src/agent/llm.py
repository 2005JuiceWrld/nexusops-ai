from __future__ import annotations

import json
from typing import Any

import httpx


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


class LLMReasoner:
    """Local LLM reasoning layer for NexusOps."""

    def __init__(
        self,
        model: str = MODEL_NAME,
        url: str = OLLAMA_URL,
    ) -> None:
        self.model = model
        self.url = url

    def analyze(
        self,
        question: str,
        evidence: list[dict[str, Any]],
        hypotheses: list[str],
    ) -> dict[str, Any]:

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

        response = httpx.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
            timeout=120.0,
        )

        response.raise_for_status()

        payload = response.json()

        return json.loads(payload["response"])


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