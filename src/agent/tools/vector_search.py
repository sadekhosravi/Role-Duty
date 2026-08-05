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

    The cheapest and fastest retriever, and the one to try FIRST. It makes no
    model calls of its own: it embeds the question and returns the chunks whose
    embeddings are closest, each with its source file and a distance (lower is
    closer). For any question that one self-contained passage answers, this is
    also the last tool you need.

    Its labels stop at the chunk index — this store keeps no page or section —
    so copy what it gives you and do not complete it from memory.

    Escalate to graph_rag_search only when this did not settle the question:
    nothing relevant came back, or the answer needs a connection between roles
    that no single passage states.
    """
    hits = similarity_search(get_collection(), question, n_results=top_k)
    if not hits:
        return "[no vector matches — is the Chroma store ingested?]"
    # Labels use the same " › " shape as the graph and keyword tools, so a
    # citation copied out of any of the three reads the same way. This store
    # holds only a file and a chunk index — the Chroma ingest keeps no page or
    # section — so the label stops there rather than inventing the rest.
    return "\n\n".join(
        f"[{i}] {hit['metadata']['source']} › chunk {hit['metadata']['index']} "
        f"(distance {hit['distance']:.4f})\n{hit['text']}"
        for i, hit in enumerate(hits, 1)
    )
