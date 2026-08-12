"""Chroma wrapper: opening a collection, and the reads that work on any of them.

A thin layer over a persistent Chroma collection so the rest of the code doesn't
deal with client setup directly.

Nothing here writes any more. The chunk pipeline this module used to serve —
`add_chunks`, and the `Chunk` dataclass in extractor.py — is gone: the unit of
retrieval is now a document section, written by `doctree.index` which owns both
the rows and what goes in them. What is left is the client and three reads that
are indifferent to what a row represents, which is why they are still shared
rather than copied into doctree.
"""

from __future__ import annotations

import chromadb

from .config import settings
from .embeddings import get_embedding_function


def get_collection(name: str | None = None):
    """Open (or create) a persistent Chroma collection in CHROMA_DIR.

    Named rather than fixed because there are now two: the heading index
    doctree writes, and whatever else this project grows. They must not be one
    collection — a distance between a query and a heading key means something
    different from a distance between a query and a passage, and a store holding
    both would rank them against each other as if it did not.
    """
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    return client.get_or_create_collection(
        name=name or settings.collection_name,
        embedding_function=get_embedding_function(),
    )


def source_counts(collection) -> dict[str, int]:
    """How many rows each ingested document holds, keyed by file name.

    Chroma has no "distinct values of a metadata field" query, so this reads
    every row's metadata and tallies it. That is fine at this corpus's size and
    would not be at a large one — the fix then is a separate manifest written at
    ingest, not a cleverer query here.
    """
    rows = collection.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in rows["metadatas"] or []:
        source = (meta or {}).get("source")
        if source:
            counts[source] = counts.get(source, 0) + 1
    return counts


def delete_source(collection, source: str) -> int:
    """Drop every row that came from one document. Returns how many went.

    The count is taken before the delete because Chroma's `delete` returns
    nothing useful, and "removed 0" is the answer a caller needs when it asked
    about a document that was never ingested.
    """
    removed = source_counts(collection).get(source, 0)
    if removed:
        collection.delete(where={"source": source})
    return removed


def similarity_search(collection, query: str, n_results: int = 5) -> list[dict]:
    """Return the rows most similar to the query prompt."""
    result = collection.query(query_texts=[query], n_results=n_results)

    # Chroma returns parallel lists nested one level deep (one per query).
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]
