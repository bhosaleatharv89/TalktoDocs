"""FAISS vector storage with disk persistence."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import faiss
import numpy as np

from core.chunker import Chunk
from config import settings
from utils.logger import get_logger


logger = get_logger(__name__)


class FaissVectorStore:
    """Thin wrapper around FAISS with metadata persistence."""

    def __init__(self, index_name: str | None = None) -> None:
        self.index_name = index_name or settings.index_name
        self.index_path = settings.index_dir / f"{self.index_name}.faiss"
        self.meta_path = settings.index_dir / f"{self.index_name}.json"
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: list[dict] = []

    def add(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        """Add embedding vectors and chunk metadata to index."""
        if len(chunks) != len(embeddings):
            raise ValueError("Embeddings/chunks length mismatch")

        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)

        self.index.add(embeddings)
        self.metadata.extend(asdict(chunk) for chunk in chunks)
        logger.info("Indexed %s chunks", len(chunks))

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[dict, float]]:
        """Return top-k metadata entries with similarity score."""
        if self.index is None or self.index.ntotal == 0:
            return []

        query = np.asarray(query_vector, dtype="float32")
        if query.ndim == 1:
            query = np.expand_dims(query, axis=0)

        scores, indices = self.index.search(query, top_k)
        results: list[tuple[dict, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((self.metadata[idx], float(score)))
        return results

    def save(self) -> None:
        """Persist FAISS index and metadata to disk."""
        if self.index is None:
            return
        faiss.write_index(self.index, str(self.index_path))
        self.meta_path.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")
        logger.info("Persisted index at %s", self.index_path)

    def load(self) -> bool:
        """Load existing index and metadata from disk."""
        if not self.index_path.exists() or not self.meta_path.exists():
            return False

        self.index = faiss.read_index(str(self.index_path))
        self.metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))
        logger.info("Loaded index with %s vectors", self.index.ntotal)
        return True

    def clear(self) -> None:
        """Remove index from memory and disk."""
        self.index = None
        self.metadata = []
        if self.index_path.exists():
            self.index_path.unlink()
        if self.meta_path.exists():
            self.meta_path.unlink()
        logger.info("Cleared vector store %s", self.index_name)
