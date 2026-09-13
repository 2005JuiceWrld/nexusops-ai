from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import ndcg_score

from src.retrieval.bm25 import BM25Retriever, load_documents
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever


QUERIES: list[dict[str, Any]] = [
    {
        "query": "payment failures after latest deployment",
        "relevant": ["INC-001", "DEP-001", "RUN-001"],
    },
    {
        "query": "fraud service latency causing payment timeouts",
        "relevant": ["INC-002", "SVC-001", "SVC-002"],
    },
    {
        "query": "payment service dependency on fraud service",
        "relevant": ["SVC-001", "SVC-002"],
    },
    {
        "query": "production rollback approval requirements",
        "relevant": ["POL-001", "RUN-001"],
    },
    {
        "query": "recent fraud service deployment",
        "relevant": ["DEP-002", "INC-002"],
    },
]


def recall_at_k(
    ranked_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    retrieved = set(ranked_ids[:k])
    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    return len(retrieved & relevant) / len(relevant)


def reciprocal_rank(
    ranked_ids: list[str],
    relevant_ids: list[str],
) -> float:
    relevant = set(relevant_ids)

    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    ranked_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    relevant = set(relevant_ids)

    y_true = [
        1 if doc_id in relevant else 0
        for doc_id in ranked_ids[:k]
    ]

    if not any(y_true):
        return 0.0

    # Ideal ranking puts every relevant document first.
    ideal = sorted(y_true, reverse=True)

    return float(
        ndcg_score(
            np.asarray([ideal]),
            np.asarray([y_true]),
        )
    )


def evaluate_retriever(
    name: str,
    retriever: Any,
) -> dict[str, float]:

    recall5 = []
    recall10 = []
    mrr = []
    ndcg10 = []

    for item in QUERIES:
        results = retriever.search(
            item["query"],
            top_k=10,
        )

        ranked_ids = [
            result["document"]["id"]
            for result in results
        ]

        recall5.append(
            recall_at_k(
                ranked_ids,
                item["relevant"],
                5,
            )
        )

        recall10.append(
            recall_at_k(
                ranked_ids,
                item["relevant"],
                10,
            )
        )

        mrr.append(
            reciprocal_rank(
                ranked_ids,
                item["relevant"],
            )
        )

        ndcg10.append(
            ndcg_at_k(
                ranked_ids,
                item["relevant"],
                10,
            )
        )

    return {
        "system": name,
        "Recall@5": float(np.mean(recall5)),
        "Recall@10": float(np.mean(recall10)),
        "MRR": float(np.mean(mrr)),
        "NDCG@10": float(np.mean(ndcg10)),
    }


def main() -> None:
    documents = load_documents()

    systems = [
        (
            "BM25",
            BM25Retriever(documents),
        ),
        (
            "Dense",
            DenseRetriever(documents),
        ),
        (
            "Hybrid + RRF",
            HybridRetriever(documents),
        ),
    ]

    print("\nNEXUSOPS RETRIEVAL EVALUATION")
    print("=" * 70)

    results = []

    for name, retriever in systems:
        print(f"\nEvaluating: {name}")

        metrics = evaluate_retriever(
            name,
            retriever,
        )

        results.append(metrics)

    print("\nRESULTS")
    print("=" * 70)

    print(
        f"{'System':<18}"
        f"{'Recall@5':<12}"
        f"{'Recall@10':<12}"
        f"{'MRR':<12}"
        f"{'NDCG@10':<12}"
    )

    print("-" * 70)

    for result in results:
        print(
            f"{result['system']:<18}"
            f"{result['Recall@5']:<12.3f}"
            f"{result['Recall@10']:<12.3f}"
            f"{result['MRR']:<12.3f}"
            f"{result['NDCG@10']:<12.3f}"
        )


if __name__ == "__main__":
    main()