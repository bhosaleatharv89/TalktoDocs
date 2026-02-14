"""Document loading and text extraction utilities."""
from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

from PyPDF2 import PdfReader
from docx import Document

from config import settings
from utils.helpers import is_allowed_file, mb_to_bytes
from utils.logger import get_logger


logger = get_logger(__name__)


class FileValidationError(ValueError):
    """Raised when an uploaded file does not pass validation."""


class DocumentLoader:
    """Handle file validation, persistence, and text extraction."""

    def validate_file(self, file_name: str, file_size: int) -> None:
        """Check file extension and maximum size constraints."""
        if not is_allowed_file(file_name):
            raise FileValidationError(
                f"Unsupported file type: {Path(file_name).suffix}. Allowed: PDF, TXT, DOCX"
            )

        max_size = mb_to_bytes(settings.max_file_size_mb)
        if file_size > max_size:
            raise FileValidationError(
                f"File '{file_name}' exceeds max size of {settings.max_file_size_mb} MB"
            )

    def save_uploaded_file(self, file_name: str, file_stream: BinaryIO) -> Path:
        """Persist uploaded file into temporary uploads directory."""
        destination = settings.upload_dir / Path(file_name).name
        with destination.open("wb") as output:
            output.write(file_stream.read())
        logger.info("Saved uploaded file to %s", destination)
        return destination

    def load_text(self, file_path: Path) -> str:
        """Extract text content based on file extension."""
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return self._read_pdf(file_path)
        if suffix == ".txt":
            return file_path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".docx":
            return self._read_docx(file_path)
        raise FileValidationError(f"Unsupported file extension: {suffix}")

    def _read_pdf(self, file_path: Path) -> str:
        """Read PDF content by concatenating all page text."""
        reader = PdfReader(str(file_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    def _read_docx(self, file_path: Path) -> str:
        """Read DOCX content by concatenating paragraph text."""
        doc = Document(str(file_path))
        paragraphs = [paragraph.text for paragraph in doc.paragraphs]
        return "\n".join(paragraphs)
