"""Plain semantic search over the Chroma vector store (src/rag)."""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field

from rag.vector_store import similarity_search

from ._stores import get_collection


class VectorSearchInput(BaseModel):
    """Arguments for naive_rag_search."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        min_length=1,
        description="The question to search for.",
    )
    # Bounded because the value lands in a store query and every chunk returned
    # is spent from the node's context: 0 would retrieve nothing and a large
    # number would bury the useful hits among weak ones.
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="How many chunks to return (1-20).",
    )


@tool(args_schema=VectorSearchInput)
async def naive_rag_search(question: str, top_k: int = 5) -> str:
    """Plain semantic search over document chunks (no graph, no reranking).

    Returns the chunks whose embeddings are closest to the question, each with
    its source file and a distance (lower is closer). Useful as an independent
    check on the graph — it can surface a passage the graph never connected —
    and for questions answered by one self-contained piece of text.
    """
    hits = similarity_search(get_collection(), question, n_results=top_k)
    if not hits:
        return "[no vector matches — is the Chroma store ingested?]"
    return "\n\n".join(
        f"[{i}] {hit['metadata']['source']} (chunk {hit['metadata']['index']}) "
        f"- distance {hit['distance']:.4f}\n{hit['text']}"
        for i, hit in enumerate(hits, 1)
    )
