"""Application configuration management."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Strongly-typed runtime settings read from environment variables."""

    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "sentence_transformer")

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_llm_model: str = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    sentence_transformer_model: str = os.getenv(
        "SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    data_dir: Path = Path(os.getenv("DATA_DIR", "./data"))
    upload_dir: Path = Path(os.getenv("UPLOAD_DIR", "./data/uploads"))
    index_dir: Path = Path(os.getenv("INDEX_DIR", "./data/indexes"))
    index_name: str = os.getenv("INDEX_NAME", "default_index")

    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "700"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    retriever_top_k: int = int(os.getenv("RETRIEVER_TOP_K", "5"))
    retriever_score_threshold: float = float(os.getenv("RETRIEVER_SCORE_THRESHOLD", "0.30"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    def prepare_directories(self) -> None:
        """Ensure all runtime directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.prepare_directories()
