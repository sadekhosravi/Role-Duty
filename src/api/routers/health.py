"""Is the server up, and what does it have to work with.

Two endpoints rather than one, split on cost. `/health` touches only the local
disk, so it is safe to poll; `/health/tracing` makes a real network call to
Langfuse, so it is asked for deliberately. Folding the second into the first
would make the cheap check as slow and as failable as the expensive one.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from agent import observability
from graph_rag.keyword_search import WORKING_DIR as GRAPH_DIR
from graph_rag.keyword_search import indexed_chunk_count
from rag.config import settings as rag_settings
from rag.vector_store import source_counts
from ticket_mcp.document import tickets_dir

from ..paths import relative
from ..schemas import GraphStoreHealth, HealthResponse, TracingHealth, VectorStoreHealth
from ..stores import get_collection

router = APIRouter(tags=["health"])


def _read_health() -> HealthResponse:
    """Gather the whole picture. Synchronous — every call here hits the disk."""
    collection = get_collection()
    counts = source_counts(collection)
    tickets = tickets_dir()

    return HealthResponse(
        raw_dir=relative(rag_settings.raw_data_dir),
        raw_pdfs=len(list(rag_settings.raw_data_dir.glob("*.pdf")))
        if rag_settings.raw_data_dir.exists()
        else 0,
        vector=VectorStoreHealth(
            directory=relative(rag_settings.chroma_dir),
            exists=rag_settings.chroma_dir.exists(),
            collection=rag_settings.collection_name,
            chunks=sum(counts.values()),
            documents=len(counts),
        ),
        graph=GraphStoreHealth(
            directory=relative(GRAPH_DIR),
            exists=GRAPH_DIR.exists(),
            indexed_chunks=indexed_chunk_count(),
        ),
        tickets_dir=relative(tickets),
        tickets=len(list(tickets.glob("*.docx"))) if tickets.exists() else 0,
        tracing_enabled=observability.is_enabled(),
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Server status and what is in each store",
    description=(
        "Answers without calling any model provider, so it is cheap enough to "
        "poll. Use it to check that an ingest actually landed: `vector.chunks` "
        "and `graph.indexed_chunks` are the two stores, filled separately."
    ),
)
async def health() -> HealthResponse:
    # Chroma and the chunk store are both blocking reads. In a threadpool they
    # cost a thread; on the event loop they would stall every request in flight.
    return await run_in_threadpool(_read_health)


@router.get(
    "/health/tracing",
    response_model=TracingHealth,
    summary="Check the Langfuse connection",
    description=(
        "The same check as `python scripts/agent.py --check-tracing`. Makes a "
        "network call, so it is a separate endpoint from /health. Tracing being "
        "off is reported, not treated as a failure — the agent runs untraced."
    ),
)
async def tracing() -> TracingHealth:
    ok, detail = await run_in_threadpool(observability.auth_check)
    return TracingHealth(enabled=observability.is_enabled(), ok=ok, detail=detail)
