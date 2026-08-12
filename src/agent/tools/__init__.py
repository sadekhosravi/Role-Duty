"""The tools a node can be given — one module per tool.

    section_search.py  find_section     src/doctree    — shortlist sections
                       read_section     src/doctree    — read one whole
    graph_search.py    graph_rag_search src/graph_rag  — LightRAG graph only
    tickets.py         create_ticket    MCP server     — writes a .docx ticket
    _stores.py                                         — shared store handles

The retrievers wrap something that already exists in this repo and return a
plain string, since a node reads tool output as text. They are deliberately
different in kind, not just in tuning, and they form a cascade rather than a
menu:

  find_section + read_section    the ordinary path. Find the sections, read the
                                ones that matter, whole. No model calls, one
                                embedding call, and it is the only path that
                                returns a section as the author wrote it.
  graph_rag_search              the escalation. The only tool that returns
                                RELATIONSHIPS, so the only one that can answer a
                                question no single section states. It makes
                                several model calls, so it is the expensive one.

Exact-term matching is not on that list because it is not a tool any more. BM25
runs inside find_section on every call, fused with the heading search — see
doctree/search.py. It was a separate tool and got skipped, which is what a
free-and-useful thing should never be allowed to be.

`tickets.py` is not a retriever and is not declared here at all: it is
discovered from an MCP server (see src/ticket_mcp), which owns its schema. It is
also the only tool with a side effect — everything else in this package reads.

Import a tool from this package to hand it to a node:

    from agent.tools import find_section, read_section

This package defines what the tools ARE. Which node may call which is decided by
each node's own NodeSpec, and that list is the access boundary: `create_ticket`
appears in exactly one of them.
"""

from .graph_search import graph_rag_search
from .section_search import find_section, read_section
from .tickets import ticket_tools

# Every retriever that exists. A convenience for handing a node the full set —
# it is not a default: a node gets exactly the tools its NodeSpec lists. The
# ticket tool is deliberately not in here; it writes, and a list named "all"
# is exactly how a read-only node would end up able to write.
ALL_RETRIEVERS = [find_section, read_section, graph_rag_search]

__all__ = [
    "find_section",
    "read_section",
    "graph_rag_search",
    "ticket_tools",
    "ALL_RETRIEVERS",
]
