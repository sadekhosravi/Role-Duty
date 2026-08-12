"""Graph retrieval over the LightRAG knowledge graph (src/graph_rag)."""

from __future__ import annotations

from typing import Literal

from langchain_core.tools import tool
from lightrag import QueryParam
from pydantic import BaseModel, ConfigDict, Field

from ._stores import get_rag

# Graph-only retrieval strategies. LightRAG also offers "naive" (vector search
# over its own chunks) and "mix" (graph + vector), and `graph_rag.query` defaults
# to mix because pure graph retrieval returns nothing when a question's entities
# are not matched in the graph — a silent miss. Neither belongs here. LightRAG's
# vector half indexes ITS chunks, which are not the sections find_section returns,
# so turning it on would put a second, differently-cut view of the same corpus
# into the same result and leave the node reconciling two vocabularies. The
# backstop moves up a level instead — when the graph comes back empty, the node
# goes back to find_section itself.
GraphMode = Literal["hybrid", "local", "global"]


class GraphSearchInput(BaseModel):
    """Arguments for graph_rag_search."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        min_length=1,
        description="The question to retrieve context for.",
    )
    mode: GraphMode = Field(
        default="hybrid",
        description=(
            "Graph strategy. 'hybrid' (entities + relationships) suits most "
            "questions; 'local' is entity-centric, for one role's immediate "
            "neighbourhood; 'global' is relationship-centric, for broad themes."
        ),
    )


@tool(args_schema=GraphSearchInput)
async def graph_rag_search(question: str, mode: GraphMode = "hybrid") -> str:
    """Search the knowledge graph of roles, duties and reporting lines.

    The ESCALATION tool, not the opening move. It runs its own model calls to
    match entities and rank relationships, which makes it far slower and far
    more expensive than find_section and read_section — read the sections first,
    and come here only when they did not settle the question.

    Graph retrieval only — no vector search. Returns the matched entities, the
    relationships connecting them (the multi-hop paths), and the source chunks
    behind them with their citation labels. It is the only tool that returns
    relationships, so it is the one that answers how roles relate: who reports
    to whom, where something escalates, what a role may not do, and anything
    that needs following a chain across sections or documents.

    If this returns nothing useful, the question's roles were not matched in the
    graph. Do not retry it with the same wording — go back to find_section with
    the exact title or phrase instead; its keyword half matches literal strings
    the graph's entity matcher missed.
    """
    # No manual clamp on `mode`: the Literal reaches the model as a JSON-schema
    # enum, and anything outside it is rejected before this function runs, with
    # the reason handed back so the model can correct it. That is stricter than
    # silently falling back to "hybrid", which would hide the mistake — and it
    # is what keeps "naive" and "mix" from sneaking vector search in here.
    rag = await get_rag()
    # only_need_context=True makes LightRAG stop after retrieval and hand back
    # the context it assembled instead of writing an answer with it. Reasoning
    # over that context is this node's job, not LightRAG's.
    context = await rag.aquery(
        question,
        param=QueryParam(mode=mode, only_need_context=True),
    )
    return context or "[no graph context found]"
