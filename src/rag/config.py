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

    # The document trees (src/doctree): one JSON file per PDF, holding its
    # sections in the shape the author wrote them. Separate from chroma_dir
    # because it is the source of truth and the Chroma side is a derived index —
    # deleting data/chroma/ costs a re-embed, deleting this costs a re-parse.
    tree_dir: Path = _path("TREE_DIR", "data/tree")

    # Chroma collection holding one embedding per tree node, over its HEADING
    # PATH rather than its body. Kept apart from `collection_name` because the
    # two answer different questions — "which passage is about this" versus
    # "which section is this" — and mixing rows of both kinds in one collection
    # would make every distance meaningless.
    heading_collection_name: str = os.getenv("HEADING_COLLECTION_NAME", "headings")


settings = Settings()
