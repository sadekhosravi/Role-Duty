"""The stores this process shares, and the lock that guards writing to one.

One process, one LightRAG, one Chroma client. The handles are not built here —
they are `agent.tools._stores`, which the agent's retrieval tools already use —
and that is the point rather than an accident of import. A request to
`/query/graph` and a `graph_rag_search` inside an agent run are the same read of
the same files, and if each built its own LightRAG the process would hold two
instances over one on-disk store, each finalizing storages the other still had
open. Reaching into a private-by-name module is the smaller sin.

Concurrency, stated rather than hoped for:

* Reads run concurrently. `aquery` on one initialized instance is what LightRAG
  is built for, and the graph query spends its time waiting on OpenRouter.
* Writes take `graph_write_lock()`. Two ingests interleaving over one NetworkX
  graph and one NanoVectorDB is not a race the storage layer resolves; nor is an
  ingest running against a document a delete is halfway through removing.
* A read during a write sees whatever is committed at that moment. It is not
  serialized against writers, because holding a lock across a query means one
  ingest blocks every question for as long as the LLM takes to extract a corpus.

The lock is per process. Run more than one worker and it guards nothing — see
scripts/serve.py, which is why that script starts exactly one.
"""

from __future__ import annotations

import asyncio

# Importing this package starts the ticket MCP server to discover its tools (see
# agent/tools/tickets.py). That happens once, at import, and it is why the app
# takes a moment to come up.
from agent.tools import _stores

# Re-exported under their own names so the routers read as if these were ours,
# and so a future move to a real store registry is one edit in one file.
get_heading_collection = _stores.get_heading_collection
get_rag = _stores.get_rag

_graph_write_lock = asyncio.Lock()


def graph_write_lock() -> asyncio.Lock:
    """The lock every graph mutation must hold: ingest, and delete."""
    return _graph_write_lock
