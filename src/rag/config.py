"""Central configuration.

All tunable settings live here so the rest of the code stays clean.
Values are read from environment variables (see .env.example) with
sensible defaults, so the project runs even without a .env file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Project root = two levels up from this file (src/rag/config.py -> project/).
ROOT_DIR = Path(__file__).resolve().parents[2]


def _path(env_key: str, default: str) -> Path:
    """Resolve a path from an env var, relative to the project root."""
    value = os.getenv(env_key, default)
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


@dataclass(frozen=True)
class Settings:
    raw_data_dir: Path = _path("RAW_DATA_DIR", "data/raw")
    chroma_dir: Path = _path("CHROMA_DIR", "data/chroma")
    collection_name: str = os.getenv("COLLECTION_NAME", "documents")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )


settings = Settings()
