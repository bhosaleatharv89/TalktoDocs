"""Service that orchestrates upload-to-index ingestion workflow."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.chunker import TextChunker
from core.cleaner import TextCleaner
from core.embedder import BaseEmbedder
from core.loader import DocumentLoader
from core.vector_store import FaissVectorStore
from utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class IngestionResult:
    """Structured output for ingestion operations."""

    file_name: str
    chunks_indexed: int


class IngestionService:
    """Pipeline service for validating, parsing, chunking and indexing files."""

    def __init__(
        self,
        loader: DocumentLoader,
        cleaner: TextCleaner,
        chunker: TextChunker,
        embedder: BaseEmbedder,
        vector_store: FaissVectorStore,
    ) -> None:
        self.loader = loader
        self.cleaner = cleaner
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    def ingest_uploaded_file(self, file_name: str, file_bytes: bytes) -> IngestionResult:
        """Ingest a file from Streamlit upload object bytes."""
        self.loader.validate_file(file_name, len(file_bytes))
        saved_path = self.loader.save_uploaded_file(file_name, _BytesReader(file_bytes))
        return self.ingest_path(saved_path)

    def ingest_path(self, file_path: Path) -> IngestionResult:
        """Ingest file from path and persist updated vector index."""
        raw_text = self.loader.load_text(file_path)
        cleaned_text = self.cleaner.clean(raw_text)

        chunks = self.chunker.chunk_document(cleaned_text, source_file=file_path.name)
        if not chunks:
            raise ValueError(f"No text chunks generated for {file_path.name}")

        embeddings = self.embedder.encode([chunk.text for chunk in chunks])
        self.vector_store.add(embeddings, chunks)
        self.vector_store.save()

        logger.info("Ingestion complete for %s (%s chunks)", file_path.name, len(chunks))
        return IngestionResult(file_name=file_path.name, chunks_indexed=len(chunks))


class _BytesReader:
    """Minimal file-like wrapper to support save_uploaded_file method."""

    def __init__(self, data: bytes) -> None:
        self.data = data

    def read(self) -> bytes:
        return self.data
