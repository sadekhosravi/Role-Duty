"""The three retrieval paths, as three endpoints.

They are not three tunings of one thing and the API keeps them apart on purpose,
because they fail differently and comparing them is most of what this project is
for:

    /query/section   The document trees. Returns whole sections, ranked by
                     heading search fused with BM25. No LLM, no answer —
                     retrieval is the whole of it.
    /query/graph     LightRAG. Returns a written answer with a `### References`
                     section. Costs LLM calls and takes seconds.
    /query/keyword   BM25 over the chunks the GRAPH ingest indexed. A different
                     corpus from /query/section — LightRAG cuts the documents
                     its own way — and it is here to see what the graph is
                     working from, not as a second section search.

Each calls the same function the matching CLI calls, so `POST /query/graph` and
`python src/graph_rag/graph_rag.py query "..."` cannot diverge.
"""

from __future__ import annotations

from fastapi import APIRouter, status
from fastapi.concurrency import run_in_threadpool

from doctree import corpus, find_sections
from doctree.index import search_headings
from graph_rag import graph_rag
from graph_rag.keyword_search import keyword_search

from ..schemas import (
    ErrorResponse,
    GraphQueryRequest,
    GraphQueryResponse,
    KeywordHit,
    KeywordQueryRequest,
    KeywordQueryResponse,
    SectionHit,
    SectionQueryRequest,
    SectionQueryResponse,
)
from ..stores import get_heading_collection, get_rag

router = APIRouter(prefix="/query", tags=["query"])

PREVIEW_CHARS = 300


@router.post(
    "/section",
    response_model=SectionQueryResponse,
    summary="Chunkless retrieval — find the sections that answer a question",
    description=(
        "Ranked sections, straight from the document trees. There is no "
        "generation step: this is what the agent's researcher sees, which is "
        "what you want when an answer looks wrong and the question is whether "
        "retrieval or the model produced it.\n\n"
        "Two retrievers run on every call and their rankings are fused — the "
        "heading index (embeddings, one API call) and BM25 over the full "
        "section text (free). `found_by` says which found each hit.\n\n"
        "What comes back is the section as the author wrote it, with its "
        "sub-sections, not a fixed-size window of it. Set `include_text: false` "
        "for the shortlist alone.\n\n"
        "Needs `POST /ingest/sections` to have run."
    ),
)
async def query_section(request: SectionQueryRequest) -> SectionQueryResponse:
    # The embedding half first, because it is the only part that leaves the
    # process. Everything after it is local and goes to a thread together.
    headings = await run_in_threadpool(
        search_headings, get_heading_collection(), request.query, request.top_k * 3
    )

    def search() -> list[SectionHit]:
        store = corpus()
        hits = find_sections(
            request.query, top_k=request.top_k, headings=[h["id"] for h in headings]
        )
        return [
            SectionHit(
                rank=rank,
                id=hit.id,
                # Built at ingest by the function every store here shares, not
                # reassembled — a section cited from this endpoint and the same
                # section cited by the agent have to be the same string.
                label=hit.label,
                source=hit.node.source,
                page=hit.node.page,
                score=hit.score,
                found_by=list(hit.found_by),
                words=len(hit.node.text.split()),
                text=store.read(hit.id) if request.include_text else None,
                preview=" ".join(hit.node.text.split())[:PREVIEW_CHARS],
            )
            for rank, hit in enumerate(hits, start=1)
        ]

    # Reading the trees and building the BM25 index are both blocking, and BM25
    # rebuilds over the whole corpus the first time it is asked.
    results = await run_in_threadpool(search)
    return SectionQueryResponse(query=request.query, top_k=request.top_k, results=results)


@router.post(
    "/graph",
    response_model=GraphQueryResponse,
    responses={status.HTTP_502_BAD_GATEWAY: {"model": ErrorResponse}},
    summary="Graph RAG — an answer from the LightRAG graph, with citations",
    description=(
        "The full path: graph traversal plus vector search, reranked, then an "
        "answer written against a prompt that requires a `### References` "
        "section naming the exact PDF sections it used.\n\n"
        "This one costs money and takes seconds — it makes several LLM calls "
        "and an embedding call per request. `mode` chooses how much of the "
        "graph is used; the default `mix` is the one that does not silently "
        "return nothing when a question's entities are missing from the graph.\n\n"
        "Needs `POST /ingest/graph` to have run."
    ),
)
async def query_graph(request: GraphQueryRequest) -> GraphQueryResponse:
    # The shared instance, not a fresh one: see api/stores.py on why a second
    # LightRAG over the same files is a way to lose the store. Reads are not
    # serialized — this awaits OpenRouter for most of its life.
    rag = await get_rag()
    answer = await graph_rag.query(
        request.question,
        mode=request.mode,
        user_prompt=request.user_prompt,
        rag=rag,
    )
    return GraphQueryResponse(question=request.question, mode=request.mode, answer=answer)


@router.post(
    "/keyword",
    response_model=KeywordQueryResponse,
    summary="BM25 — lexical search over the GRAPH store's chunks",
    description=(
        "Pure keyword overlap, no embeddings and no LLM, so it is free and "
        "instant. Scores are BM25 scores: comparable within one result set and "
        "meaningless between two. Sections with no overlap at all are dropped, "
        "so an empty list means no term matched — not that the store is empty.\n\n"
        "Reads the chunks `POST /ingest/graph` wrote, so it needs that ingest "
        "rather than the section one. This is deliberately NOT the keyword half "
        "of `/query/section`: that one runs over the document trees and returns "
        "section ids, and it is fused into the search rather than exposed. "
        "This endpoint exists to show what the graph is working from."
    ),
)
async def query_keyword(request: KeywordQueryRequest) -> KeywordQueryResponse:
    # Blocking, and not trivially so: BM25 builds its index over the whole
    # corpus on every call.
    hits = await run_in_threadpool(keyword_search, request.query, request.top_k)
    return KeywordQueryResponse(
        query=request.query,
        top_k=request.top_k,
        results=[
            KeywordHit(rank=rank, label=hit.label, score=hit.score, snippet=hit.snippet)
            for rank, hit in enumerate(hits, start=1)
        ],
    )
