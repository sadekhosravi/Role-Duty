"""Lazily-opened, process-wide handles to the backing stores.

Both stores are expensive to open — LightRAG reloads the whole graph from disk,
Chroma spins up a client — and a node may call several tools, several times, in
one run. Opening them per call would pay that cost every time, so each is built
on first use and reused for the life of the process.

They live here rather than in a tool module because two tools need them and
neither owns them: graph_search and keyword_search both read the LightRAG store
(keyword_search reads its persisted chunk file directly, so it needs no handle),
and vector_search needs the Chroma collection.
"""

from __future__ import annotations

from graph_rag.graph_rag import build_rag
from rag.vector_store import get_collection as _open_collection

_rag = None
_collection = None


async def get_rag():
    """The shared LightRAG instance, initialized on first use."""
    global _rag
    if _rag is None:
        _rag = await build_rag()
    return _rag


def get_collection():
    """The shared Chroma collection, opened on first use."""
    global _collection
    if _collection is None:
        _collection = _open_collection()
    return _collection
