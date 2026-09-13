from __future__ import annotations

from typing import Any

from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Cross-encoder reranker for retrieved operational evidence."""

    def __init__(self) -> None:
        self.model = CrossEncoder(MODEL_NAME)

    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if not results:
            return []

        pairs = [
            (
                query,
                (
                    f"{result['document'].get('title', '')}. "
                    f"{result['document'].get('content', '')}"
                ),
            )
            for result in results
        ]

        scores = self.model.predict(pairs)

        reranked = []

        for result, score in zip(results, scores):
            item = dict(result)
            item["reranker_score"] = float(score)
            reranked.append(item)

        reranked.sort(
            key=lambda item: item["reranker_score"],
            reverse=True,
        )

        for rank, result in enumerate(
            reranked[:top_k],
            start=1,
        ):
            result["rank"] = rank

        return reranked[:top_k]


def main() -> None:
    from src.retrieval.bm25 import load_documents
    from src.retrieval.hybrid import HybridRetriever

    documents = load_documents()

    hybrid = HybridRetriever(documents)
    reranker = Reranker()

    query = (
        "Payment failures increased after the latest deployment. "
        "Investigate the likely cause using deployment history, "
        "service dependencies and operational runbooks."
    )

    candidates = hybrid.search(
        query,
        top_k=len(documents),
    )

    results = reranker.rerank(
        query,
        candidates,
        top_k=5,
    )

    print("\nRERANKED RESULTS")
    print("=" * 70)

    for result in results:
        document = result["document"]

        print(
            f"\n#{result['rank']} "
            f"{document['id']} "
            f"(reranker={result['reranker_score']:.4f})"
        )

        print(f"Title:   {document['title']}")
        print(f"Type:    {document['type']}")
        print(f"Service: {document.get('service', 'N/A')}")


if __name__ == "__main__":
    main()