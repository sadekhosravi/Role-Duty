"""The three retrieval tools the RAG node can call.

Each one wraps a retriever that already exists in this repo and returns a plain
string — the node reads tool output as text, so every tool formats its hits with
the same kind of citable label the answer prompt expects.

  graph_rag_search   src/graph_rag  — LightRAG graph + vector, retrieve-only
  naive_rag_search   src/rag        — Chroma similarity search
  keyword_search     src/graph_rag  — BM25 over LightRAG's indexed chunks

The retrievers are deliberately different in kind, not just in tuning: the graph
finds multi-hop connections between roles, the vector stores find semantically
similar passages, and BM25 finds exact terms that embeddings blur (titles, codes,
thresholds). The node is expected to use more than one and reconcile them.
"""

from __future__ import annotations

from langchain_core.tools import tool
from lightrag import QueryParam

from graph_rag.graph_rag import build_rag
from graph_rag.keyword_search import keyword_search as _bm25_search
from rag.vector_store import get_collection, similarity_search

# Both stores are expensive to open (LightRAG reloads the whole graph from disk;
# Chroma spins up a client) and the node may call a tool several times in one
# run, so each is built once and reused for the life of the process.
_rag = None
_collection = None


async def _get_rag():
    global _rag
    if _rag is None:
        _rag = await build_rag()
    return _rag


def _get_collection():
    global _collection
    if _collection is None:
        _collection = get_collection()
    return _collection


@tool
async def graph_rag_search(question: str, mode: str = "mix") -> str:
    """Search the knowledge graph of roles, duties and reporting lines.

    Returns the retrieved context — the matched entities, the relationships
    connecting them (the multi-hop paths), and the source chunks with their
    citation labels. Best for questions about how roles relate: who reports to
    whom, where something escalates, what a role may not do, and anything that
    needs following a chain across sections or documents.

    Args:
        question: The question to retrieve context for.
        mode: Retrieval strategy — "mix" (graph + vector, the default),
            "hybrid" (graph only), "local" (entity-centric),
            "global" (relationship-centric), or "naive" (vector only).
    """
    rag = await _get_rag()
    # only_need_context=True makes LightRAG stop after retrieval and hand back
    # the context it assembled instead of writing an answer with it. Reasoning
    # over that context is this node's job, not LightRAG's.
    context = await rag.aquery(
        question,
        param=QueryParam(mode=mode, only_need_context=True),
    )
    return context or "[no graph context found]"


@tool
async def naive_rag_search(question: str, top_k: int = 5) -> str:
    """Plain semantic search over document chunks (no graph, no reranking).

    Returns the chunks whose embeddings are closest to the question, each with
    its source file and a distance (lower is closer). Useful as an independent
    check on the graph — it can surface a passage the graph never connected —
    and for questions answered by one self-contained piece of text.

    Args:
        question: The question to search for.
        top_k: How many chunks to return.
    """
    hits = similarity_search(_get_collection(), question, n_results=top_k)
    if not hits:
        return "[no vector matches — is the Chroma store ingested?]"
    return "\n\n".join(
        f"[{i}] {hit['metadata']['source']} (chunk {hit['metadata']['index']}) "
        f"- distance {hit['distance']:.4f}\n{hit['text']}"
        for i, hit in enumerate(hits, 1)
    )


@tool
def keyword_search(question: str, top_k: int = 5) -> str:
    """Exact keyword (BM25) search over the indexed sections.

    Ranks sections purely by term overlap — no embeddings, no LLM. Use it when
    the question turns on a literal string that semantic search rounds off: a
    full role title, an acronym, a document code, a numeric threshold. Returns
    section labels with a relevance score and a short snippet.

    Args:
        question: The words to search for.
        top_k: How many sections to return.
    """
    hits = _bm25_search(question, top_k=top_k)
    if not hits:
        return "[no keyword matches]"
    return "\n".join(f"[{h.score:.2f}] {h.label}\n    {h.snippet}" for h in hits)


TOOLS = [graph_rag_search, naive_rag_search, keyword_search]
