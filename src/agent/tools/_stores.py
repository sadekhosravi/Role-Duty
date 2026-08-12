"""Lazily-opened, process-wide handles to the backing stores.

Both stores are expensive to open — LightRAG reloads the whole graph from disk,
Chroma spins up a client — and a node may call several tools, several times, in
one run. Opening them per call would pay that cost every time, so each is built
on first use and reused for the life of the process.

They live here rather than in a tool module because more than one caller needs
them and none owns them: graph_search reads the LightRAG store, section_search
reads the heading index, and the HTTP API re-exports both so that a request and
an agent run are the same read of the same files (see api/stores.py).

The document trees themselves are not here. They are plain JSON and their cache
lives in doctree.store, which is where the ingest that invalidates it can reach
it — a handle held here would have to be invalidated from a package that does
not import this one.
"""

from __future__ import annotations

from doctree.index import get_heading_collection as _open_headings
from graph_rag.graph_rag import build_rag

_rag = None
_headings = None


async def get_rag():
    """The shared LightRAG instance, initialized on first use."""
    global _rag
    if _rag is None:
        _rag = await build_rag()
    return _rag


def get_heading_collection():
    """The shared Chroma collection of section headings, opened on first use."""
    global _headings
    if _headings is None:
        _headings = _open_headings()
    return _headings
