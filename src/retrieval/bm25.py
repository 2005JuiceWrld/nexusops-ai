from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


DATA_PATH = Path("data/raw/operations.json")


def tokenize(text: str) -> list[str]:
    """Simple deterministic tokenizer for BM25."""
    return re.findall(r"\b[a-zA-Z0-9_.-]+\b", text.lower())


class BM25Retriever:
    """Keyword-based retrieval over NexusCorp operational documents."""

    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents

        corpus = [
            tokenize(
                f"{doc.get('title', '')} "
                f"{doc.get('type', '')} "
                f"{doc.get('service', '')} "
                f"{doc.get('content', '')}"
            )
            for doc in documents
        ]

        self.index = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return the top-k documents ranked by BM25 score."""

        query_tokens = tokenize(query)
        scores = self.index.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []

        for rank, index in enumerate(ranked_indices, start=1):
            document = dict(self.documents[index])

            results.append(
                {
                    "rank": rank,
                    "score": float(scores[index]),
                    "document": document,
                }
            )

        return results


def load_documents() -> list[dict[str, Any]]:
    with DATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    documents = load_documents()
    retriever = BM25Retriever(documents)

    query = (
        "Payment failures increased after latest deployment. "
        "Investigate payment service and fraud service latency."
    )

    results = retriever.search(query, top_k=5)

    print("\nBM25 SEARCH RESULTS")
    print("=" * 60)

    for result in results:
        document = result["document"]

        print(
            f"\n#{result['rank']} "
            f"{document['id']} "
            f"(score={result['score']:.3f})"
        )

        print(f"Title: {document['title']}")
        print(f"Type:  {document['type']}")
        print(f"Service: {document.get('service', 'N/A')}")


if __name__ == "__main__":
    main()