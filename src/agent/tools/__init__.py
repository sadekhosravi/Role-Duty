"""The tools a node can be given — one module per tool.

    graph_search.py    graph_rag_search   src/graph_rag  — LightRAG graph only
    vector_search.py   naive_rag_search   src/rag        — Chroma similarity
    keyword_search.py  keyword_search     src/graph_rag  — BM25 over the chunks
    tickets.py         create_ticket      MCP server     — writes a .docx ticket
    _stores.py                                           — shared store handles

The three retrievers wrap something that already exists in this repo and return
a plain string, since a node reads tool output as text. They are deliberately
different in kind, not just in tuning: the graph finds multi-hop connections
between roles, the vector store finds semantically similar passages, and BM25
finds exact terms that embeddings blur (titles, codes, thresholds). A node is
expected to use more than one and reconcile them.

`tickets.py` is not a retriever and is not declared here at all: it is
discovered from an MCP server (see src/ticket_mcp), which owns its schema. It is
also the only tool with a side effect — everything else in this package reads.

Import a tool from this package to hand it to a node:

    from agent.tools import graph_rag_search, keyword_search

This package defines what the tools ARE. Which node may call which is decided by
each node's own NodeSpec, and that list is the access boundary: `create_ticket`
appears in exactly one of them.
"""

from .graph_search import graph_rag_search
from .keyword_search import keyword_search
from .tickets import ticket_tools
from .vector_search import naive_rag_search

# Every retriever that exists. A convenience for handing a node the full set —
# it is not a default: a node gets exactly the tools its NodeSpec lists. The
# ticket tool is deliberately not in here; it writes, and a list named "all"
# is exactly how a read-only node would end up able to write.
ALL_RETRIEVERS = [graph_rag_search, naive_rag_search, keyword_search]

__all__ = [
    "graph_rag_search",
    "naive_rag_search",
    "keyword_search",
    "ticket_tools",
    "ALL_RETRIEVERS",
]
