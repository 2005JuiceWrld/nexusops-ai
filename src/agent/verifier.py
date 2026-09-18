from __future__ import annotations

import re
from typing import Any


class EvidenceVerifier:
    """Deterministic verifier for LLM-generated investigation reasoning.

    The verifier does not decide whether the LLM is intelligent.
    It checks whether the generated reasoning is sufficiently grounded
    in authoritative evidence and whether the recommendation is
    consistent with the available operational policy.
    """

    def verify(
        self,
        question: str,
        evidence: list[dict[str, Any]],
        hypotheses: list[str],
        llm_result: dict[str, Any],
    ) -> dict[str, Any]:
        evidence_text = self._flatten_evidence(evidence)

        checks = {
            "evidence_available": bool(evidence),
            "llm_structure_valid": self._valid_llm_structure(
                llm_result
            ),
            "confidence_valid": self._valid_confidence(
                llm_result
            ),
            "cause_supported": self._cause_supported(
                llm_result.get("likely_cause", ""),
                evidence,
            ),
            "supporting_evidence_valid": self._supporting_evidence_valid(
                llm_result.get("supporting_evidence", []),
                evidence,
            ),
            "recommendation_present": bool(
                str(llm_result.get("recommended_action", "")).strip()
            ),
            "recommendation_policy_consistent": (
                self._recommendation_policy_consistent(
                    llm_result.get("recommended_action", ""),
                    evidence,
                )
            ),
        }

        unsupported_claims = self._find_unsupported_claims(
            llm_result,
            evidence,
        )

        checks["unsupported_claims_valid"] = not unsupported_claims

        passed = all(checks.values())

        return {
            "passed": passed,
            "checks": checks,
            "unsupported_claims": unsupported_claims,
            "confidence": llm_result.get("confidence"),
            "likely_cause": llm_result.get(
                "likely_cause",
                "",
            ),
            "recommended_action": llm_result.get(
                "recommended_action",
                "",
            ),
        }

    @staticmethod
    def _flatten_evidence(
        evidence: list[dict[str, Any]],
    ) -> str:
        parts: list[str] = []

        for item in evidence:
            source = str(
                item.get("source", "")
            )

            value = item.get(
                "evidence",
                "",
            )

            parts.append(
                f"{source}: {value}"
            )

        return "\n".join(parts).lower()

    @staticmethod
    def _valid_llm_structure(
        result: dict[str, Any],
    ) -> bool:
        required_fields = {
            "summary",
            "likely_cause",
            "supporting_evidence",
            "alternative_explanations",
            "recommended_action",
            "confidence",
        }

        if not isinstance(result, dict):
            return False

        if not required_fields.issubset(result.keys()):
            return False

        if not isinstance(result.get("summary"), str):
            return False

        if not result["summary"].strip():
            return False

        if not isinstance(result.get("likely_cause"), str):
            return False

        if not result["likely_cause"].strip():
            return False

        if not isinstance(result.get("supporting_evidence"), list):
            return False

        if not result["supporting_evidence"]:
            return False

        if not isinstance(
            result.get("alternative_explanations"),
            list,
        ):
            return False

        if not isinstance(result.get("recommended_action"), str):
            return False

        if not result["recommended_action"].strip():
            return False

        confidence = result.get("confidence")

        if isinstance(confidence, bool):
            return False

        if not isinstance(confidence, (int, float)):
            return False

        return True

    @staticmethod
    def _valid_confidence(
        result: dict[str, Any],
    ) -> bool:
        confidence = result.get("confidence")

        if isinstance(
            confidence,
            bool,
        ):
            return False

        if not isinstance(
            confidence,
            (int, float),
        ):
            return False

        return 0.0 <= float(confidence) <= 1.0

    @classmethod
    def _cause_supported(
        cls,
        likely_cause: str,
        evidence: list[dict[str, Any]],
    ) -> bool:
        if not likely_cause.strip() or not evidence:
            return False

        evidence_text = cls._flatten_evidence(evidence)

        if not evidence_text.strip():
            return False

        claim = likely_cause.lower()

        claim_words = {
            word
            for word in re.findall(
                r"[a-z0-9.]+",
                claim,
            )
            if len(word) >= 4
        }

        evidence_words = set(
            re.findall(
                r"[a-z0-9.]+",
                evidence_text,
            )
        )

        generic_terms = {
            "caused",
            "cause",
            "causes",
            "because",
            "combined",
            "combining",
            "combination",
            "increased",
            "increase",
            "decreased",
            "decrease",
            "after",
            "before",
            "during",
            "issue",
            "issues",
            "problem",
            "problems",
            "service",
            "services",
            "production",
            "system",
            "failure",
            "failures",
            "error",
            "errors",
            "changes",
            "changed",
            "introduced",
            "introduction",
            "likely",
            "most",
            "reason",
            "impact",
            "result",
            "results",
            "related",
            "relationship",
            "with",
            "from",
            "the",
            "and",
            "that",
            "operational",
            "finding",
            "check",
            "checks",
            "analysis",
            "investigation",
        }

        specific_terms = {
            word
            for word in claim_words
            if word not in generic_terms
        }

        if not specific_terms:
            return False

        unsupported_terms = (
            specific_terms - evidence_words
        )

        if unsupported_terms:
            return False

        # Require meaningful grounding rather than accepting a
        # generic sentence that merely mentions the evidence.
        overlap = specific_terms.intersection(
            evidence_words
        )

        if len(overlap) >= 2:
            return True

        # A strong compound identifier such as "fraud-service"
        # can itself be sufficient when directly present.
        compound_terms = {
            word
            for word in specific_terms
            if "-" in word or "." in word
        }

        return bool(
            compound_terms.intersection(evidence_words)
        )

    @classmethod
    def _cause_supported(
        cls,
        likely_cause: str,
        evidence: list[dict[str, Any]],
    ) -> bool:
        if not likely_cause.strip() or not evidence:
            return False

        evidence_text = cls._flatten_evidence(evidence)

        if not evidence_text:
            return False

        claim = likely_cause.lower()
        evidence_lower = evidence_text.lower()

        # Important technical concepts that must be grounded.
        technical_terms = {
            "service",
            "services",
            "payment-service",
            "fraud-service",
            "auth-service",
            "latency",
            "timeout",
            "timeouts",
            "deployment",
            "deployed",
            "version",
            "fraud",
            "verification",
            "request",
            "requests",
            "configuration",
            "transaction",
            "transactions",
            "error",
            "errors",
            "failure",
            "failures",
            "incident",
            "incidents",
            "policy",
            "approval",
            "rollback",
            "restart",
            "metric",
            "metrics",
            "rate",
            "production",
            "runbook",
            "operational",
            "status",
        }

        claim_terms = {
            term
            for term in technical_terms
            if term in claim
        }

        grounded_terms = {
            term
            for term in claim_terms
            if term in evidence_lower
        }

        # A legitimate causal explanation must contain
        # multiple technical concepts grounded in evidence.
        if len(grounded_terms) < 2:
            return False

        # Explicitly reject claims introducing unsupported
        # infrastructure or failure mechanisms.
        unsupported_mechanisms = {
            "network partition",
            "network outage",
            "database corruption",
            "database failure",
            "disk failure",
            "memory leak",
            "hardware failure",
            "dns failure",
            "dns outage",
            "certificate failure",
        }

        for mechanism in unsupported_mechanisms:
            if mechanism in claim and mechanism not in evidence_lower:
                return False

        return True

    @classmethod
    def _supporting_evidence_valid(
        cls,
        supporting_evidence: Any,
        evidence: list[dict[str, Any]],
    ) -> bool:
        if not isinstance(supporting_evidence, list):
            return False

        if not supporting_evidence or not evidence:
            return False

        valid_sources = {
            str(item.get("source", "")).lower()
            for item in evidence
            if item.get("source")
        }

        evidence_texts = []

        for item in evidence:
            source = str(item.get("source", "")).lower()
            value = item.get("evidence", "")

            if isinstance(value, dict):
                value_text = " ".join(
                    f"{key} {val}"
                    for key, val in value.items()
                )
            elif isinstance(value, list):
                value_text = " ".join(
                    str(entry)
                    for entry in value
                )
            else:
                value_text = str(value)

            evidence_texts.append(
                f"{source} {value_text}".lower()
            )

        combined_evidence = "\n".join(evidence_texts)

        if not evidence_texts:
            return False

        for claim in supporting_evidence:
            if not isinstance(claim, str) or not claim.strip():
                return False

            claim_lower = claim.lower()

            # ---------------------------------------------------------
            # Case 1: Explicit evidence source reference
            # ---------------------------------------------------------
            referenced_sources = {
                source
                for source in valid_sources
                if source and source in claim_lower
            }

            if referenced_sources:
                if not any(
                    cls._text_overlap(
                        claim_lower,
                        source_text,
                    )
                    for source_text in evidence_texts
                ):
                    # If exact wording does not overlap, require
                    # meaningful factual words from the claim to appear
                    # in the referenced evidence.
                    claim_words = {
                        word
                        for word in re.findall(
                            r"[a-z0-9.-]+",
                            claim_lower,
                        )
                        if len(word) >= 4
                    }

                    if not any(
                        len(
                            claim_words.intersection(
                                set(
                                    re.findall(
                                        r"[a-z0-9.-]+",
                                        source_text,
                                    )
                                )
                            )
                        ) >= 1
                        for source_text in evidence_texts
                    ):
                        return False

                continue

            # ---------------------------------------------------------
            # Case 2: Explicit operational record ID
            # ---------------------------------------------------------
            referenced_record_ids = {
                token.lower()
                for token in re.findall(
                    r"\b(?:inc|dep|svc|run|pol)-\d+\b",
                    claim_lower,
                )
                if token.lower() in combined_evidence
            }

            if referenced_record_ids:
                if not any(
                    record_id in source_text
                    and (
                        cls._text_overlap(
                            claim_lower,
                            source_text,
                        )
                        or len(
                            {
                                word
                                for word in re.findall(
                                    r"[a-z0-9.-]+",
                                    claim_lower,
                                )
                                if len(word) >= 4
                            }.intersection(
                                set(
                                    re.findall(
                                        r"[a-z0-9.-]+",
                                        source_text,
                                    )
                                )
                            )
                        ) >= 1
                    )
                    for record_id in referenced_record_ids
                    for source_text in evidence_texts
                ):
                    return False

                continue

            # ---------------------------------------------------------
            # Case 3: Direct textual grounding
            # ---------------------------------------------------------
            if any(
                cls._text_overlap(
                    claim_lower,
                    source_text,
                )
                for source_text in evidence_texts
            ):
                continue

            # ---------------------------------------------------------
            # Case 4: Concise factual summary
            #
            # Require at least one meaningful claim word to be grounded
            # in the supplied evidence. This works across services,
            # metrics, policies, incidents, deployments, and runbooks.
            # ---------------------------------------------------------
            claim_words = {
                word
                for word in re.findall(
                    r"[a-z0-9.-]+",
                    claim_lower,
                )
                if len(word) >= 4
            }

            stop_words = {
                "this",
                "that",
                "with",
                "from",
                "into",
                "using",
                "were",
                "was",
                "the",
                "and",
                "for",
                "requested",
                "available",
                "operational",
                "investigation",
                "evidence",
            }

            meaningful_words = claim_words - stop_words

            evidence_words = set(
                re.findall(
                    r"[a-z0-9.-]+",
                    combined_evidence,
                )
            )

            grounded_words = meaningful_words.intersection(
                evidence_words
            )

            if not grounded_words:
                return False

        return True

    @staticmethod
    def _text_overlap(
        claim: str,
        source: str,
    ) -> bool:
        claim_words = {
            word
            for word in re.findall(
                r"[a-z0-9.]+",
                claim.lower(),
            )
            if len(word) >= 4
        }

        source_words = {
            word
            for word in re.findall(
                r"[a-z0-9.]+",
                source.lower(),
            )
            if len(word) >= 4
        }

        if not claim_words:
            return False

        overlap = claim_words.intersection(
            source_words
        )

        if len(overlap) < 3:
            return False

        generic_terms = {
            "caused",
            "cause",
            "causes",
            "because",
            "combined",
            "combining",
            "increased",
            "increase",
            "decreased",
            "decrease",
            "after",
            "before",
            "during",
            "issue",
            "issues",
            "problem",
            "problems",
            "service",
            "production",
            "system",
            "failure",
            "failures",
            "error",
            "errors",
            "changes",
            "changed",
            "introduced",
            "introduction",
            "likely",
            "most",
            "reason",
            "impact",
            "result",
            "results",
            "related",
            "relationship",
            "combined",
            "combination",
        }

        specific_claim_terms = {
            word
            for word in claim_words
            if word not in generic_terms
        }

        if not specific_claim_terms:
            return False

        # Important technical concepts introduced by the LLM
        # must be grounded in the supplied evidence.
        unsupported_specific_terms = (
            specific_claim_terms - source_words
        )

        if unsupported_specific_terms:
            return False

        technical_terms = {
            "deployment",
            "deployed",
            "version",
            "latency",
            "timeout",
            "timeouts",
            "rollback",
            "restart",
            "incident",
            "policy",
            "approval",
            "fraud",
            "payment",
            "database",
            "network",
            "partition",
            "configuration",
            "config",
            "verification",
            "transaction",
        }

        return bool(
            overlap.intersection(
                technical_terms
            )
        )

    @classmethod
    def _claim_has_support(
        cls,
        claim: str,
        evidence: list[dict[str, Any]],
    ) -> bool:
        claim_lower = claim.lower()

        for item in evidence:
            source = str(
                item.get("source", "")
            ).lower()

            source_text = str(
                item.get("evidence", "")
            ).lower()

            combined = (
                f"{source} {source_text}"
            )

            if cls._text_overlap(
                claim_lower,
                combined,
            ):
                return True

        return False

    @classmethod
    def _recommendation_policy_consistent(
        cls,
        recommendation: str,
        evidence: list[dict[str, Any]],
    ) -> bool:
        if not recommendation.strip():
            return False

        recommendation_lower = (
            recommendation.lower()
        )

        evidence_text = cls._flatten_evidence(
            evidence
        )

        production_action_terms = [
            "rollback",
            "roll back",
            "rolling back",
            "restart",
            "disable",
            "deploy",
            "change configuration",
            "change config",
            "modify production",
        ]

        requires_policy = any(
            term in recommendation_lower
            for term in production_action_terms
        )

        # Investigation-only recommendations do not
        # require production authorization evidence.
        if not requires_policy:
            return True

        # Production actions must have policy evidence.

        policy_evidence = [
            item
            for item in evidence
            if (
                "policy" in str(
                    item.get("source", "")
                ).lower()
                or str(
                    item.get("source", "")
                ).lower().startswith("pol-")
                or "policy" in str(
                    item.get("evidence", "")
                ).lower()
            )
        ]

        if not policy_evidence:
            return False

        policy_text = " ".join(
            str(
                item.get(
                    "evidence",
                    "",
                )
            ).lower()
            for item in policy_evidence
        )

        # Rollback recommendation must have rollback
        # authorization evidence.
        if "rollback" in recommendation_lower or "roll back" in recommendation_lower:
            if "rollback" not in policy_text:
                return False

        # Actions affecting production must have approval
        # evidence unless policy explicitly indicates that
        # approval is not required.
        production_terms = [
            "production",
            "prod",
        ]

        affects_production = any(
            term in recommendation_lower
            for term in production_terms
        ) or any(
            term in recommendation_lower
            for term in [
                "rollback",
                "roll back",
                "rolling back",
                "restart",
                "deploy",
                "change configuration",
                "modify production",
            ]
        )

        if affects_production:
            if (
                "approval" not in policy_text
                and "approve" not in policy_text
            ):
                return False

        # The policy itself must actually be present in
        # the supplied evidence universe.
        return bool(
            evidence_text.strip()
        )

    @classmethod
    def _find_unsupported_claims(
        cls,
        llm_result: dict[str, Any],
        evidence: list[dict[str, Any]],
    ) -> list[str]:
        """
        Identify claims that introduce unsupported factual mechanisms.

        Causal explanations may legitimately combine facts from multiple
        evidence sources, so likely_cause is validated using the same
        multi-source grounding logic as _cause_supported().
        """

        unsupported: list[str] = []

        likely_cause = str(
            llm_result.get("likely_cause", "")
        ).strip()

        if likely_cause:
            if not cls._cause_supported(
                likely_cause,
                evidence,
            ):
                unsupported.append(likely_cause)

        return list(dict.fromkeys(unsupported))

    @staticmethod
    def _split_sentences(
        text: str,
    ) -> list[str]:
        return [
            sentence.strip()
            for sentence in re.split(
                r"(?<=[.!?])\s+",
                text.strip(),
            )
            if sentence.strip()
        ]