"""Ingest: PDF -> tree on disk -> heading index in Chroma.

Two writes per document, in that order and not the other, because the tree is
the source of truth and the index is derived from it. A crash between them
leaves a tree with no index — a document that cannot be found but can still be
read by id, which is recoverable by re-running this. The reverse would leave an
index full of ids that resolve to nothing, and a search returning citations to
sections that are not there is the failure this project has spent the most time
removing.

This replaces the chunk-and-embed pipeline that used to live in rag/pipeline.py.
Same Docling parse, same citation labels, same Chroma directory — what changed is
the unit. The store now holds one row per section the author wrote, instead of
one row per fixed-size window of their prose.
"""

from __future__ import annotations

from pathlib import Path

from rag.config import settings

from rag.vector_store import delete_source

from . import search, store
from .index import get_heading_collection, index_document
from .tree import build_from_pdf


def ingest_pdf(pdf_path: Path | str, directory: Path | None = None) -> int:
    """Parse one PDF into a tree, persist it, and index its headings.

    Returns the number of sections indexed, which is not the number of nodes in
    the tree: container headings carry no text of their own and are addressable
    without being findable. See index._indexable.
    """
    document = build_from_pdf(pdf_path)
    store.save(document, directory)
    indexed = index_document(get_heading_collection(), document)
    # A long-lived process — the HTTP server — can ingest and then answer a
    # question in the same breath. Both caches were built before this write.
    store.reset()
    search.reset()
    return indexed


def remove_document(source: str, directory: Path | None = None) -> int:
    """Drop one document's index rows and its tree. Returns the rows removed.

    Rows first, tree second, and the order is the mirror of the ingest's for the
    same reason. A tree left behind after its rows are gone is a document that
    can still be read by id and never found; rows left behind after the tree is
    gone are search results that resolve to nothing. Both are bad, and only the
    first is recoverable by re-running the ingest.
    """
    removed = delete_source(get_heading_collection(), source)
    store.tree_path(source, directory).unlink(missing_ok=True)
    store.reset()
    search.reset()
    return removed


def ingest_directory(directory: Path | str | None = None) -> int:
    """Ingest every PDF in a directory. Returns the total sections indexed."""
    directory = Path(directory) if directory else settings.raw_data_dir
    pdfs = sorted(directory.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in: {directory}")

    total = 0
    for pdf in pdfs:
        count = ingest_pdf(pdf)
        total += count
        print(f"  {pdf.name}: {count} sections")
    return total
