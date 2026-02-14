"""Embedding providers with caching and batching support."""
from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from config import settings
from utils.helpers import batch_items


class BaseEmbedder(ABC):
    """Abstract interface for embedding providers."""

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into an array of embeddings."""


class SentenceTransformerEmbedder(BaseEmbedder):
    """SentenceTransformer-backed local embedding implementation."""

    def __init__(self) -> None:
        self.model = SentenceTransformer(settings.sentence_transformer_model)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype="float32")


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embeddings implementation with response caching."""

    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")
        self.client = OpenAI(api_key=settings.openai_api_key)

    @lru_cache(maxsize=4096)
    def _embed_one(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=settings.openai_embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for group in batch_items(texts, 64):
            vectors.extend([self._embed_one(item) for item in group])
        return np.asarray(vectors, dtype="float32")


def get_embedder() -> BaseEmbedder:
    """Factory returning configured embedding provider."""
    if settings.embedding_provider.lower() == "openai":
        return OpenAIEmbedder()
    return SentenceTransformerEmbedder()
