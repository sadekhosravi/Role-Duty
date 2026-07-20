"""Chroma vector store wrapper.

A thin layer over a persistent Chroma collection so the rest of the code
doesn't deal with client setup directly.
"""

from __future__ import annotations

import chromadb

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


def add_chunks(collection, chunks: list[Chunk]) -> None:
    """Embed and store chunks. Chroma computes the vectors on add()."""
    if not chunks:
        return

    collection.add(
        ids=[f"{c.source}:{c.index}" for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[{"source": c.source, "index": c.index} for c in chunks],
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
