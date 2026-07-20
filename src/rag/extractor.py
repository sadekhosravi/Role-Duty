"""PDF extraction and chunking using Docling.

Docling parses the PDF into a structured document, and its HybridChunker
splits that document into coherent, retrieval-friendly text chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter


@dataclass
class Chunk:
    """A single piece of text extracted from a document."""

    text: str
    source: str  # filename the chunk came from
    index: int  # position of the chunk within that document


def extract_chunks(pdf_path: Path | str) -> list[Chunk]:
    """Convert a PDF into a list of text chunks."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document = DocumentConverter().convert(pdf_path).document
    chunker = HybridChunker()

    chunks: list[Chunk] = []
    for i, chunk in enumerate(chunker.chunk(document)):
        text = chunk.text.strip()
        if text:
            chunks.append(Chunk(text=text, source=pdf_path.name, index=i))

    return chunks
