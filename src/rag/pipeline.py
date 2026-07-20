"""Ingestion pipeline: PDF files -> chunks -> embeddings -> Chroma."""

from __future__ import annotations

from pathlib import Path

from .config import settings
from .extractor import extract_chunks
from .vector_store import add_chunks, get_collection


def ingest_pdf(pdf_path: Path | str) -> int:
    """Ingest a single PDF. Returns the number of chunks stored."""
    chunks = extract_chunks(pdf_path)
    collection = get_collection()
    add_chunks(collection, chunks)
    return len(chunks)


def ingest_directory(directory: Path | str | None = None) -> int:
    """Ingest every PDF in a directory. Returns total chunks stored."""
    directory = Path(directory) if directory else settings.raw_data_dir
    pdfs = sorted(directory.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in: {directory}")

    total = 0
    for pdf in pdfs:
        count = ingest_pdf(pdf)
        total += count
        print(f"  {pdf.name}: {count} chunks")

    return total
