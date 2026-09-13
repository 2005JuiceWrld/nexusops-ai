from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DATA_PATH = Path("data/raw/operations.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class DenseRetriever:
    """Semantic retrieval using Sentence Transformers and FAISS."""

    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents

        self.model = SentenceTransformer(MODEL_NAME)

        self.texts = [
            self._document_to_text(document)
            for document in documents
        ]

        embeddings = self.model.encode(
            self.texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        self.embeddings = embeddings.astype("float32")

        dimension = self.embeddings.shape[1]

        # Inner product on normalized vectors = cosine similarity.
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)

    @staticmethod
    def _document_to_text(document: dict[str, Any]) -> str:
        return (
            f"Title: {document.get('title', '')}. "
            f"Type: {document.get('type', '')}. "
            f"Service: {document.get('service', '')}. "
            f"Content: {document.get('content', '')}"
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return top-k documents ranked by semantic similarity."""

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(
            query_embedding,
            min(top_k, len(self.documents)),
        )

        results = []

        for rank, (score, index) in enumerate(
            zip(scores[0], indices[0]),
            start=1,
        ):
            document = dict(self.documents[int(index)])

            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "document": document,
                }
            )

        return results


def load_documents() -> list[dict[str, Any]]:
    with DATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    documents = load_documents()
    retriever = DenseRetriever(documents)

    query = (
        "Payment failures increased after the latest release. "
        "Investigate the cause by checking downstream dependencies."
    )

    results = retriever.search(query, top_k=5)

    print("\nDENSE SEARCH RESULTS")
    print("=" * 60)

    for result in results:
        document = result["document"]

        print(
            f"\n#{result['rank']} "
            f"{document['id']} "
            f"(similarity={result['score']:.4f})"
        )

        print(f"Title: {document['title']}")
        print(f"Type: {document['type']}")
        print(f"Service: {document.get('service', 'N/A')}")


if __name__ == "__main__":
    main()