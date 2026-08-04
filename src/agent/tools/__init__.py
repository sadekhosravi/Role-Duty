"""The retrieval tools a node can be given — one module per tool.

    graph_search.py    graph_rag_search   src/graph_rag  — LightRAG graph only
    vector_search.py   naive_rag_search   src/rag        — Chroma similarity
    keyword_search.py  keyword_search     src/graph_rag  — BM25 over the chunks
    _stores.py                                           — shared store handles

Each tool wraps a retriever that already exists in this repo and returns a plain
string, since a node reads tool output as text. They are deliberately different
in kind, not just in tuning: the graph finds multi-hop connections between roles,
the vector store finds semantically similar passages, and BM25 finds exact terms
that embeddings blur (titles, codes, thresholds). A node is expected to use more
than one and reconcile them.

Import a tool from this package to hand it to a node:

    from agent.tools import graph_rag_search, keyword_search

This package defines what the tools ARE. Which node may call which is decided in
graph.py, where each node lists the tools it is allowed.
"""

from .graph_search import graph_rag_search
from .keyword_search import keyword_search
from .vector_search import naive_rag_search

# Every tool that exists. A convenience for handing a node the full set — it is
# not a default: a node gets exactly the tools its NodeSpec lists.
ALL_TOOLS = [graph_rag_search, naive_rag_search, keyword_search]

__all__ = ["graph_rag_search", "naive_rag_search", "keyword_search", "ALL_TOOLS"]
