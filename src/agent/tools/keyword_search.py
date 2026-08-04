"""BM25 keyword search over LightRAG's indexed chunks (src/graph_rag).

The underlying search reads the persisted chunk store directly, so this tool
needs no open handle and no API key — it is the one retriever that works with
the network down.
"""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field

from graph_rag.keyword_search import keyword_search as _bm25_search


class KeywordSearchInput(BaseModel):
    """Arguments for keyword_search."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        min_length=1,
        description="The words to search for.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="How many sections to return (1-20).",
    )


@tool(args_schema=KeywordSearchInput)
def keyword_search(question: str, top_k: int = 5) -> str:
    """Exact keyword (BM25) search over the indexed sections.

    Ranks sections purely by term overlap — no embeddings, no LLM. Use it when
    the question turns on a literal string that semantic search rounds off: a
    full role title, an acronym, a document code, a numeric threshold. Returns
    section labels with a relevance score and a short snippet.
    """
    hits = _bm25_search(question, top_k=top_k)
    if not hits:
        return "[no keyword matches]"
    return "\n".join(f"[{h.score:.2f}] {h.label}\n    {h.snippet}" for h in hits)
