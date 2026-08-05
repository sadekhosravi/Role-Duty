"""Chroma vector store wrapper.

A thin layer over a persistent Chroma collection so the rest of the code
doesn't deal with client setup directly.
"""

from __future__ import annotations

import chromadb

from graph_rag.extraction import LABEL_SEP

from .config import settings
from .embeddings import get_embedding_function
from .extractor import Chunk


def get_collection():
    """Open (or create) the persistent Chroma collection."""
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    return client.get_or_create_collection(
        name=settings.collection_name,
        embedding_function=get_embedding_function(),
    )


def _metadata(chunk: Chunk) -> dict:
    """Chroma metadata for one chunk.

    `label` is the citation string, built at ingest time and stored rather than
    reassembled at query time — the same thing the graph ingest does when it
    hands LightRAG a file_path. Storing it means the retrieval tool prints a
    label it did not have to construct, which is the only way a citation can be
    checked against what a tool actually returned.

    Chroma metadata values must be scalars, so the heading tuple is joined and
    an unknown page is omitted rather than written as None.
    """
    meta = {
        "source": chunk.source,
        "index": chunk.index,
        "label": chunk.label,
        "headings": LABEL_SEP.join(chunk.headings),
    }
    if chunk.page is not None:
        meta["page"] = chunk.page
    return meta


def add_chunks(collection, chunks: list[Chunk]) -> None:
    """Embed and store chunks. Chroma computes the vectors on add().

    Each document's existing chunks are deleted first. Ids are `source:index`,
    so re-ingesting used to overwrite chunk-for-chunk and that was enough — but
    only while the chunking produced the same count every time. It no longer
    does: sections group several old chunks into one, so a re-ingest yields
    fewer ids than the previous run wrote, and every surplus id would linger as
    an orphan holding text from the old scheme. Those orphans are still
    retrievable and still cite themselves, which makes them worse than useless.
    """
    if not chunks:
        return

    for source in {chunk.source for chunk in chunks}:
        collection.delete(where={"source": source})

    collection.add(
        ids=[f"{c.source}:{c.index}" for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[_metadata(c) for c in chunks],
    )


def similarity_search(collection, query: str, n_results: int = 5) -> list[dict]:
    """Return the chunks most similar to the query prompt."""
    result = collection.query(query_texts=[query], n_results=n_results)

    # Chroma returns parallel lists nested one level deep (one per query).
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]
