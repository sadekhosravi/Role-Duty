"""The PDFs: getting them onto the server, seeing where they got to, removing them.

Upload exists so the API can be tested end to end from a browser. Without it,
"ingest the PDFs" means "first go and copy files into data/raw on the machine
running this", which is not something a Swagger page can do and not something a
client should have to.

Upload and ingest are deliberately two calls. A PDF on disk and a PDF in a store
are different states, ingesting is minutes of work that belongs in a job, and
the same file is ingested into two stores by two different pipelines — so an
upload that silently indexed something would have to pick one, and would be
wrong either way.

The listing is one row per document across all three places it can live: the raw
directory, the vector store, the graph. A file present in one and absent from
another is the normal state while setting up, not an anomaly, and seeing that at
a glance is the point of merging them.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from doctree.ingest import remove_document
from graph_rag import graph_rag
from lightrag.base import DocStatus
from rag.config import settings as rag_settings
from rag.vector_store import source_counts

from ..paths import plain_name, relative
from ..schemas import (
    DeleteDocumentResponse,
    DocumentList,
    DocumentStatus,
    ErrorResponse,
    StoreName,
    UploadResponse,
)
from ..stores import get_heading_collection, get_rag, graph_write_lock

router = APIRouter(prefix="/documents", tags=["documents"])

# Read in 1 MiB pieces rather than `await file.read()`. A PDF is small enough
# that it would not matter until the day someone posts one that is not, and a
# whole file in memory per concurrent upload is an easy thing to avoid.
UPLOAD_CHUNK = 1024 * 1024


async def _graph_sections() -> dict[str, int]:
    """How many section-documents the graph holds, per source PDF.

    Section ids are "{pdf}#{n}" (see graph_rag.ingest), so the PDF a section
    came from is the id up to the first '#'. Every status is counted, including
    the failed and pending ones — a half-ingested document is still occupying
    the store, and reporting it as absent is how someone ends up re-ingesting on
    top of it.
    """
    rag = await get_rag()
    counts: dict[str, int] = {}
    for state in DocStatus:
        for doc_id in await rag.get_docs_by_status(state):
            if "#" in doc_id:
                name = doc_id.rsplit("#", 1)[0]
                counts[name] = counts.get(name, 0) + 1
    return counts


@router.get(
    "",
    response_model=DocumentList,
    summary="Every document, and which stores hold it",
    description=(
        "One row per PDF, merged across the raw directory, the section index "
        "and the LightRAG graph. `sections: 0` with `graph_sections: 12` means "
        "that document has had the graph ingest run on it and not the section "
        "one.\n\n"
        "Opens the graph store on first call, which takes a moment."
    ),
)
async def list_documents() -> DocumentList:
    raw_dir = rag_settings.raw_data_dir
    sections = await run_in_threadpool(source_counts, get_heading_collection())
    graph = await _graph_sections()

    files = {path.name: path for path in raw_dir.glob("*.pdf")} if raw_dir.exists() else {}

    documents = []
    for name in sorted(set(files) | set(sections) | set(graph)):
        path = files.get(name)
        stat = path.stat() if path else None
        documents.append(
            DocumentStatus(
                name=name,
                raw_path=relative(path) if path else None,
                size_bytes=stat.st_size if stat else None,
                modified_at=(
                    datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc) if stat else None
                ),
                sections=sections.get(name, 0),
                graph_sections=graph.get(name, 0),
            )
        )
    return DocumentList(raw_dir=relative(raw_dir), documents=documents)


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_409_CONFLICT: {"model": ErrorResponse},
    },
    summary="Upload a PDF into the raw directory",
    description=(
        "Writes the file into RAW_DATA_DIR (data/raw) and stops there — it does "
        "not ingest. Follow it with `POST /ingest/naive`, `POST /ingest/graph`, "
        "or both.\n\n"
        "Refuses to overwrite unless `overwrite=true`, because replacing the "
        "file under an already-ingested document leaves the stores describing a "
        "PDF that is no longer there."
    ),
)
async def upload_document(
    file: UploadFile = File(description="A PDF."),
    overwrite: bool = Query(default=False, description="Replace a file of the same name."),
) -> UploadResponse:
    name = plain_name(file.filename or "", suffix=".pdf")
    raw_dir = rag_settings.raw_data_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / name

    replaced = target.exists()
    if replaced and not overwrite:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"{name} already exists — pass overwrite=true to replace it",
        )

    size = 0
    with target.open("wb") as out:
        while piece := await file.read(UPLOAD_CHUNK):
            size += len(piece)
            await run_in_threadpool(out.write, piece)

    if size == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "the uploaded file was empty")

    return UploadResponse(
        name=name, path=relative(target), size_bytes=size, replaced=replaced
    )


@router.delete(
    "/{name}",
    response_model=DeleteDocumentResponse,
    responses={status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse}},
    summary="Remove a document from the stores",
    description=(
        "Deletes the document's tree and its rows from the section index, and "
        "its sections from the graph. The graph delete also prunes entities and "
        "relationships that lose their last supporting section, while keeping "
        "those other documents still cite.\n\n"
        "The PDF itself stays in data/raw — removing it from an index and "
        "deleting the source are different intentions, and only one of them is "
        "reversible by re-ingesting.\n\n"
        "Removing a document that was never ingested is not an error: the "
        "counts come back zero."
    ),
)
async def delete_document(
    name: str,
    store: StoreName = Query(default="all", description="Which store(s) to remove it from."),
) -> DeleteDocumentResponse:
    name = plain_name(name, suffix=".pdf")

    removed = 0
    if store in {"all", "vector"}:
        # Both halves in one call: the index rows and the tree they resolve to
        # have to go together, and doctree owns the order they go in.
        removed = await run_in_threadpool(remove_document, name)

    graph_result = "not requested"
    if store in {"all", "graph"}:
        rag = await get_rag()
        # A delete mutates the graph, so it takes the same lock an ingest does.
        async with graph_write_lock():
            graph_result = await graph_rag.remove(name, rag=rag)

    return DeleteDocumentResponse(
        name=name, store=store, sections_removed=removed, graph=graph_result
    )
