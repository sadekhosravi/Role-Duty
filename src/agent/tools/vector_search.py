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
    # The distance goes BEFORE the label, never after. A label is harvested from
    # tool output by reading from the filename to the end of the line, so a
    # trailing "(distance 0.9850)" gets absorbed into it: the same section then
    # appears under two different labels depending on which tool returned it,
    # and a citation carrying the score matches the harvested string exactly and
    # passes the verifier's check. Putting the score in front removes it from
    # the label without hiding it — the same shape keyword_search already uses.
    return "\n\n".join(
        f"[{i}] (distance {hit['distance']:.4f}) {_label(hit['metadata'])}\n{hit['text']}"
        for i, hit in enumerate(hits, 1)
    )


def _label(metadata: dict) -> str:
    """The citation label for one hit.

    Read from metadata rather than assembled here: it was built at ingest time
    by the same function the graph ingest uses, so a citation copied out of this
    tool is character-identical to the same section cited from graph_rag_search.

    The fallback is for a store written before labels existed. Without it those
    rows would cite the string "None", and the verifier — which compares a
    citation against what a tool actually returned — would reject every answer
    resting on one, giving no hint that the cause is a stale ingest rather than
    a bad answer.
    """
    label = metadata.get("label")
    if label:
        return label
    return f"{metadata['source']} › chunk {metadata['index']} [stale ingest — re-run scripts/ingest.py]"
