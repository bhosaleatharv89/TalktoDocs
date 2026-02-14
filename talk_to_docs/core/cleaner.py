"""Text cleaning and normalization component."""
from __future__ import annotations

import re

from utils.helpers import normalize_whitespace


class TextCleaner:
    """Apply lightweight text normalization suitable for embeddings."""

    def clean(self, text: str) -> str:
        """Remove noisy characters and normalize whitespace."""
        text = text.replace("\x00", " ")
        text = re.sub(r"[\u200b-\u200f\ufeff]", "", text)  # zero-width / BOM chars
        text = re.sub(r"\n{3,}", "\n\n", text)
        return normalize_whitespace(text)
