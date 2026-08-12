"""The two ingests, submitted as background jobs.

They are separate endpoints because they are separate stores, filled by separate
pipelines, at wildly different cost:

    /ingest/sections  Docling parses the PDF into a section tree, written as
                      JSON under data/tree/, then Chroma embeds one row per
                      section keyed on its heading path. Seconds to a minute per
                      document, one embedding call.
    /ingest/graph     The same parse, then an LLM reads every section to extract
                      entities and relationships and merge them into the graph.
                      Minutes per document, and it is the expensive one.

Both return `202 Accepted` with a job id rather than an answer. See api/jobs.py
for why, and for what this job runner is not.

Both are idempotent per document: the section ingest rewrites a document's tree
whole and deletes its existing index rows before writing, and the graph ingest
deletes its existing sections. So re-ingesting a changed PDF replaces it rather
than doubling it.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, status
from fastapi.concurrency import run_in_threadpool

from doctree import ingest_pdf
from graph_rag import graph_rag
from rag.config import settings as rag_settings

from ..deps import JobsDep
from ..paths import pdfs_at, relative, under_root
from ..schemas import ErrorResponse, IngestRequest, JobAccepted
from ..stores import get_rag, graph_write_lock

router = APIRouter(prefix="/ingest", tags=["ingest"])

_ERRORS = {
    status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
    status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
}


def _target(request: IngestRequest) -> Path:
    """Resolve what to ingest, defaulting to RAW_DATA_DIR.

    Resolved in the request, not in the job: a path that does not exist or sits
    outside the project should be a 400 the caller sees, not a job that is
    accepted and then fails somewhere they have to go looking for it.
    """
    if request.path is None:
        return rag_settings.raw_data_dir
    return under_root(request.path)


@router.post(
    "/sections",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    responses=_ERRORS,
    summary="Section ingest — PDFs into document trees and the heading index",
    description=(
        "Parses each PDF into the tree of sections its author wrote, saves that "
        "tree as JSON under data/tree/, and writes one Chroma row per section "
        "keyed on its heading path.\n\n"
        "The tree is the source of truth and the index is derived from it, so "
        "data/chroma/ can be deleted and rebuilt without re-running Docling.\n\n"
        "Returns a job id immediately. Poll `GET /jobs/{id}` until status is "
        "`succeeded` — `result.documents` then lists what each PDF contributed. "
        "The count is indexed sections, which is fewer than the tree's nodes: "
        "a heading with no text of its own is addressable but not findable.\n\n"
        "Re-ingesting a document replaces its tree and its rows."
    ),
)
async def ingest_sections(request: IngestRequest, jobs: JobsDep) -> JobAccepted:
    target = _target(request)
    pdfs = pdfs_at(target)

    async def work() -> dict:
        # One call per PDF rather than `ingest_directory`, which is this loop
        # plus a print. The loop is here so the job can report per-document
        # counts; a single total tells you nothing about which file was empty.
        # Docling is synchronous and slow, so each parse goes to a thread.
        documents = {}
        for pdf in pdfs:
            documents[pdf.name] = await run_in_threadpool(ingest_pdf, pdf)
        return {
            "store": "sections",
            "path": relative(target),
            "documents": documents,
            "sections": sum(documents.values()),
        }

    return JobAccepted.of(jobs.submit("ingest.sections", relative(target), work))


@router.post(
    "/graph",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    responses=_ERRORS,
    summary="Graph RAG ingest — PDFs into the LightRAG knowledge graph",
    description=(
        "Parses each PDF into sections, then has the LLM extract roles, duties "
        "and the limits on them from every section and merge them into one "
        "persistent graph.\n\n"
        "This is the slow, expensive one: minutes per document and many LLM "
        "calls. Poll `GET /jobs/{id}`.\n\n"
        "Also fills the corpus `POST /query/keyword` searches. Ingests are "
        "serialized against each other — one graph, one writer."
    ),
)
async def ingest_graph(request: IngestRequest, jobs: JobsDep) -> JobAccepted:
    target = _target(request)
    pdfs = pdfs_at(target)

    async def work() -> dict:
        rag = await get_rag()
        documents = {}
        # Held across the whole job, not per document: a second ingest starting
        # midway through this one would interleave writes into one NetworkX
        # graph. Queries are deliberately not blocked by it — see api/stores.py.
        async with graph_write_lock():
            for pdf in pdfs:
                documents[pdf.name] = await graph_rag.ingest(pdf, rag=rag)
        return {
            "store": "graph",
            "path": relative(target),
            "documents": documents,
            "sections": sum(documents.values()),
        }

    return JobAccepted.of(jobs.submit("ingest.graph", relative(target), work))
