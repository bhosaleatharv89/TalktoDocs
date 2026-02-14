"""Chunking logic for splitting documents into overlapping windows."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    """A chunk and associated metadata."""

    chunk_id: str
    text: str
    source_file: str
    start_char: int
    end_char: int


class TextChunker:
    """Simple character-based chunker with overlap."""

    def __init__(self, chunk_size: int, overlap: int) -> None:
        if overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, text: str, source_file: str) -> list[Chunk]:
        """Split cleaned text into overlapping chunks."""
        chunks: list[Chunk] = []
        step = self.chunk_size - self.overlap
        index = 0

        for start in range(0, len(text), step):
            end = min(start + self.chunk_size, len(text))
            window = text[start:end].strip()
            if not window:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{source_file}-{index}",
                    text=window,
                    source_file=source_file,
                    start_char=start,
                    end_char=end,
                )
            )
            index += 1
            if end == len(text):
                break

        return chunks
