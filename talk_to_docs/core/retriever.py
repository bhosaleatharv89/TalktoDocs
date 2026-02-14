"""Retriever for similarity search and relevance filtering."""
from __future__ import annotations

from dataclasses import dataclass

from config import settings
from core.embedder import BaseEmbedder
from core.vector_store import FaissVectorStore


@dataclass
class RetrievedChunk:
    """Retrieved chunk with a similarity score."""

    text: str
    source_file: str
    score: float


class Retriever:
    """Embedding + vector store retrieval orchestration."""

    def __init__(self, embedder: BaseEmbedder, vector_store: FaissVectorStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Search and return filtered chunks ordered by relevance."""
        k = top_k or settings.retriever_top_k
        query_vector = self.embedder.encode([query])
        raw_results = self.vector_store.search(query_vector, k)

        filtered = [
            RetrievedChunk(
                text=item["text"],
                source_file=item["source_file"],
                score=score,
            )
            for item, score in raw_results
            if score >= settings.retriever_score_threshold
        ]
        return sorted(filtered, key=lambda x: x.score, reverse=True)
