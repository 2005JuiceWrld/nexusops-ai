from src.agent.safety import SafetyPolicy, RiskClass
from src.agent.verifier import EvidenceVerifier


QUESTION = (
    "Payment failures increased after the latest deployment. "
    "Investigate the likely cause."
)


def grounded_evidence():
    return [
        {
            "source": "INC-001",
            "evidence": (
                "Payment failures increased after deployment. "
                "payment-service depends on fraud-service. "
                "Fraud verification requests experienced failures."
            ),
        },
        {
            "source": "DEP-001",
            "evidence": (
                "payment-service version 2.4.1 was deployed before "
                "the payment failure incident. The deployment changed "
                "fraud verification request handling."
            ),
        },
        {
            "source": "SVC-002",
            "evidence": (
                "fraud-service experienced elevated latency during "
                "the incident, causing increased timeout risk for "
                "payment-service requests."
            ),
        },
        {
            "source": "POL-001",
            "evidence": (
                "Production rollback requires explicit approval. "
                "Rollback actions must be authorized before execution."
            ),
        },
    ]


def grounded_hypotheses():
    return [
        (
            "Increased fraud-service latency combined with changes "
            "introduced by payment-service version 2.4.1."
        )
    ]


def grounded_llm_result():
    return {
        "summary": (
            "Payment failures increased after payment-service "
            "version 2.4.1 was deployed while fraud-service "
            "experienced elevated latency."
        ),
        "likely_cause": (
            "Increased latency between payment-service and "
            "fraud-service combined with changes introduced by "
            "payment-service version 2.4.1."
        ),
        "supporting_evidence": [
            "INC-001 payment failures increased after deployment.",
            "DEP-001 payment-service version 2.4.1 was deployed before the incident.",
            "SVC-002 fraud-service experienced elevated latency.",
        ],
        "alternative_explanations": [
            "Other service-side latency or configuration issues "
            "could also contribute to payment failures."
        ],
        "recommended_action": (
            "Rollback the payment-service version 2.4.1 deployment "
            "and investigate the fraud-service latency."
        ),
        "confidence": 0.8,
    }


def verify(result):
    verifier = EvidenceVerifier()

    return verifier.verify(
        QUESTION,
        grounded_evidence(),
        grounded_hypotheses(),
        result,
    )


def test_grounded_cause_is_accepted():
    result = verify(grounded_llm_result())

    assert result["passed"] is True
    assert result["checks"]["cause_supported"] is True
    assert result["checks"]["supporting_evidence_valid"] is True
    assert result["checks"]["recommendation_policy_consistent"] is True


def test_hallucinated_cause_is_rejected():
    result = grounded_llm_result()

    result["likely_cause"] = (
        "A network partition and database corruption caused "
        "the payment failures."
    )

    verification = verify(result)

    assert verification["passed"] is False
    assert verification["checks"]["cause_supported"] is False


def test_unsupported_evidence_claim_is_rejected():
    result = grounded_llm_result()

    result["supporting_evidence"] = [
        "INC-999 caused the payment failures.",
        "DEP-999 introduced the failure.",
    ]

    verification = verify(result)

    assert verification["passed"] is False
    assert verification["checks"]["supporting_evidence_valid"] is False


def test_unsafe_production_recommendation_requires_approval():
    decision = SafetyPolicy.validate_recommendation(
        "Rollback the payment-service version 2.4.1 deployment."
    )

    assert decision.allowed is False
    assert decision.requires_approval is True
    assert decision.risk_class == RiskClass.PRODUCTION_CHANGE


def test_malformed_llm_output_is_rejected():
    malformed_result = {
        "summary": "",
        "likely_cause": "",
        "supporting_evidence": [],
        "alternative_explanations": [],
        "recommended_action": "",
        "confidence": "high",
    }

    verification = verify(malformed_result)

    assert verification["passed"] is False
    assert verification["checks"]["llm_structure_valid"] is False