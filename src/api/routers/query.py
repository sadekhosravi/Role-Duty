"""The three retrieval paths, as three endpoints.

They are not three tunings of one thing and the API keeps them apart on purpose,
because they fail differently and comparing them is most of what this project is
for:

    /query/naive     Chroma similarity. Returns chunks, ranked. No LLM, no
                     answer — retrieval is the whole of it.
    /query/graph     LightRAG. Returns a written answer with a `### References`
                     section. Costs LLM calls and takes seconds.
    /query/keyword   BM25 over the chunks the graph ingest indexed. Exact terms
                     — titles, codes, thresholds — that embeddings blur. No LLM.

Each calls the same function the matching CLI calls, so `POST /query/graph` and
`python src/graph_rag/graph_rag.py query "..."` cannot diverge.
"""

from __future__ import annotations

from fastapi import APIRouter, status
from fastapi.concurrency import run_in_threadpool

from graph_rag import graph_rag
from graph_rag.keyword_search import keyword_search
from rag.vector_store import similarity_search

from ..schemas import (
    ErrorResponse,
    GraphQueryRequest,
    GraphQueryResponse,
    KeywordHit,
    KeywordQueryRequest,
    KeywordQueryResponse,
    VectorHit,
    VectorQueryRequest,
    VectorQueryResponse,
)
from ..stores import get_collection, get_rag

router = APIRouter(prefix="/query", tags=["query"])


@router.post(
    "/naive",
    response_model=VectorQueryResponse,
    summary="Naive RAG — similarity search over the Chroma store",
    description=(
        "Ranked chunks and their distances, straight from the vector store. "
        "There is no generation step: this is what the retriever saw, which is "
        "what you want when an answer looks wrong and the question is whether "
        "retrieval or the model produced it.\n\n"
        "Needs `POST /ingest/naive` to have run."
    ),
)
async def query_naive(request: VectorQueryRequest) -> VectorQueryResponse:
    collection = get_collection()
    results = await run_in_threadpool(
        similarity_search, collection, request.query, request.top_k
    )
    return VectorQueryResponse(
        query=request.query,
        top_k=request.top_k,
        results=[
            VectorHit(
                rank=rank,
                # The label was built at ingest and stored, not reassembled
                # here. The fallback is for a store written before labels
                # existed — re-ingest and it goes away.
                label=hit["metadata"].get("label")
                or f"{hit['metadata']['source']} (chunk {hit['metadata']['index']})",
                source=hit["metadata"]["source"],
                page=hit["metadata"].get("page"),
                distance=hit["distance"],
                text=hit["text"],
            )
            for rank, hit in enumerate(results, start=1)
        ],
    )


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
    summary="BM25 — lexical search over the indexed chunks",
    description=(
        "Pure keyword overlap, no embeddings and no LLM, so it is free and "
        "instant. Scores are BM25 scores: comparable within one result set and "
        "meaningless between two. Sections with no overlap at all are dropped, "
        "so an empty list means no term matched — not that the store is empty.\n\n"
        "Reads the chunks `POST /ingest/graph` wrote, so it needs that ingest "
        "rather than the naive one."
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
