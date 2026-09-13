from __future__ import annotations

from typing import Any

from src.retrieval.bm25 import BM25Retriever, load_documents
from src.retrieval.dense import DenseRetriever


class HybridRetriever:
    """Hybrid BM25 + dense retrieval using Reciprocal Rank Fusion."""

    def __init__(
        self,
        documents: list[dict[str, Any]],
        rrf_k: int = 60,
    ) -> None:
        self.documents = documents
        self.rrf_k = rrf_k

        self.bm25 = BM25Retriever(documents)
        self.dense = DenseRetriever(documents)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        bm25_results = self.bm25.search(
            query,
            top_k=len(self.documents),
        )

        dense_results = self.dense.search(
            query,
            top_k=len(self.documents),
        )

        fused_scores: dict[str, float] = {}
        documents_by_id: dict[str, dict[str, Any]] = {}

        for result in bm25_results:
            document = result["document"]
            doc_id = document["id"]

            documents_by_id[doc_id] = document

            rank = result["rank"]

            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + (
                1.0 / (self.rrf_k + rank)
            )

        for result in dense_results:
            document = result["document"]
            doc_id = document["id"]

            documents_by_id[doc_id] = document

            rank = result["rank"]

            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + (
                1.0 / (self.rrf_k + rank)
            )

        ranked = sorted(
            fused_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        results = []

        for rank, (doc_id, score) in enumerate(
            ranked[:top_k],
            start=1,
        ):
            results.append(
                {
                    "rank": rank,
                    "score": score,
                    "document": documents_by_id[doc_id],
                }
            )

        return results


def main() -> None:
    documents = load_documents()

    retriever = HybridRetriever(documents)

    query = (
        "Payment failures increased after the latest deployment. "
        "Investigate the likely cause using deployment history, "
        "service dependencies and operational runbooks."
    )

    results = retriever.search(query, top_k=5)

    print("\nHYBRID SEARCH — BM25 + DENSE + RRF")
    print("=" * 60)

    for result in results:
        document = result["document"]

        print(
            f"\n#{result['rank']} "
            f"{document['id']} "
            f"(RRF={result['score']:.6f})"
        )

        print(f"Title:   {document['title']}")
        print(f"Type:    {document['type']}")
        print(f"Service: {document.get('service', 'N/A')}")


if __name__ == "__main__":
    main()
