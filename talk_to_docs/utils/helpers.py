"""Generic helper functions used across modules."""
from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable, TypeVar


ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}
T = TypeVar("T")


def is_allowed_file(file_name: str) -> bool:
    """Validate file extension against supported formats."""
    return Path(file_name).suffix.lower() in ALLOWED_EXTENSIONS


def mb_to_bytes(size_mb: int) -> int:
    """Convert MB to bytes."""
    return size_mb * 1024 * 1024


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace while preserving sentence readability."""
    return re.sub(r"\s+", " ", text).strip()


def batch_items(items: list[T], batch_size: int) -> Iterable[list[T]]:
    """Yield list batches for efficient processing."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]
