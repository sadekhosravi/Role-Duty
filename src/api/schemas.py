"""Every request and response body, in one file.

These are not a translation layer over the pipelines — they are the API's
contract, and FastAPI builds the OpenAPI schema (and therefore the Swagger page)
entirely out of them. Two consequences worth stating, because they are the
reason this file is as verbose as it is:

* A `description` here is what a person testing the endpoint reads. It is the
  documentation, not a comment about it.
* Request models set `extra="forbid"`, so a misspelled field is a 422 naming
  it rather than a silently ignored key and a surprising default. The same
  reasoning as `AgentState`'s, applied at the edge instead of between nodes.

Requests carry an `examples` block so the Swagger "Try it out" body arrives
pre-filled with something that actually works against this corpus, which is the
difference between an API you can test in a browser and one you can read about.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .jobs import Job, JobStatus

# LightRAG's retrieval strategies, as `graph_rag.query` documents them. A
# Literal rather than a string, so Swagger renders a dropdown and an unknown
# mode is rejected at the edge instead of inside LightRAG.
GraphMode = Literal["naive", "local", "global", "hybrid", "mix"]

StoreName = Literal["all", "vector", "graph"]


class Request(BaseModel):
    """Base for anything a client sends: unknown fields are an error."""

    model_config = ConfigDict(extra="forbid")


# --- Ingestion -----------------------------------------------------------------


class IngestRequest(Request):
    """What to ingest. Used by both the naive and the graph ingest."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"path": "data/raw"},
                {"path": "data/raw/sample_role_duties_hotel.pdf"},
            ]
        },
    )

    path: str | None = Field(
        default=None,
        description=(
            "A PDF file, or a folder of PDFs, relative to the project root. "
            "Defaults to RAW_DATA_DIR (data/raw). Must resolve inside the "
            "project."
        ),
    )


class JobView(BaseModel):
    """A background job as a client sees it.

    `result` is whatever the job returned — for an ingest, what it stored and
    where. It is a free-form object because the two ingests report different
    units (chunks, sections) and flattening that into a shared shape would make
    both of them read as neither.
    """

    id: str
    kind: str = Field(description='What was submitted, e.g. "ingest.graph".')
    target: str = Field(description="What it was submitted against.")
    status: JobStatus
    submitted_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_seconds: float | None = None
    result: dict[str, Any] | None = Field(
        default=None, description="Set once the job succeeds."
    )
    error: str | None = Field(default=None, description="Set only if it failed.")

    @classmethod
    def of(cls, job: Job) -> JobView:
        return cls(
            id=job.id,
            kind=job.kind,
            target=job.target,
            status=job.status,
            submitted_at=job.submitted_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            duration_seconds=job.duration_seconds,
            result=job.result,
            error=job.error,
        )


class JobAccepted(BaseModel):
    """The 202 an ingest answers with: the job, and where to watch it."""

    job: JobView
    poll: str = Field(description="GET this until status is succeeded or failed.")

    @classmethod
    def of(cls, job: Job) -> JobAccepted:
        return cls(job=JobView.of(job), poll=f"/jobs/{job.id}")


class JobList(BaseModel):
    jobs: list[JobView]


# --- Querying ------------------------------------------------------------------


class SectionQueryRequest(Request):
    """A search over the document trees — retrieval only, no answer."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [{"query": "who authorizes a UPS bypass?", "top_k": 5}]
        },
    )

    query: str = Field(min_length=1, description="The question or literal phrase to match.")
    top_k: int = Field(default=5, ge=1, le=50, description="How many sections to return.")
    include_text: bool = Field(
        default=True,
        description=(
            "Return each section's full text. False gives the shortlist alone — "
            "labels, ids and previews — which is what the agent sees before it "
            "decides what to read."
        ),
    )


class SectionHit(BaseModel):
    """One retrieved section, whole."""

    rank: int
    id: str = Field(
        description=(
            'How to address the section, e.g. "sample-role-duties#shift-'
            'supervisor". Stable across a re-ingest. Not a citation.'
        )
    )
    label: str = Field(
        description=(
            'The citation, e.g. "x.pdf › page 2 › Duty Manager › Escalation '
            'Path". This is what an answer quotes, never the id.'
        )
    )
    source: str
    page: int | None = None
    score: float = Field(
        description=(
            "Reciprocal-rank-fusion score. Comparable within one result set and "
            "meaningless between two — it is built from ranks, not distances."
        )
    )
    found_by: list[str] = Field(
        description=(
            "Which retrievers found it: 'headings' (embeddings), 'keyword' "
            "(BM25), or both. Both agreeing is a stronger signal than either "
            "ranking it first."
        )
    )
    words: int
    text: str | None = Field(
        default=None,
        description="The section and its sub-sections. Null when include_text is false.",
    )
    preview: str


class SectionQueryResponse(BaseModel):
    query: str
    top_k: int
    results: list[SectionHit]


class GraphQueryRequest(Request):
    """A question answered from the LightRAG graph, with citations."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "question": "Who authorizes a UPS bypass, and who may not?",
                    "mode": "mix",
                }
            ]
        },
    )

    question: str = Field(min_length=1)
    mode: GraphMode = Field(
        default="mix",
        description=(
            "Retrieval strategy. naive = vector only; local = entity-centric; "
            "global = relationship-centric; hybrid = both graph paths; "
            "mix = hybrid plus vector search, which backstops the graph when a "
            "question's entities are not matched."
        ),
    )
    user_prompt: str | None = Field(
        default=None,
        description=(
            "Per-call steering, appended to the answer template's additional "
            "instructions. For narrowing one question — standing rules belong "
            "in the system prompt."
        ),
    )


class GraphQueryResponse(BaseModel):
    question: str
    mode: GraphMode
    answer: str = Field(
        description="The answer, ending in a `### References` section naming its sources."
    )


class KeywordQueryRequest(Request):
    """BM25 lexical search over the chunks the graph ingest indexed."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"query": "UPS bypass authorization", "top_k": 5}]},
    )

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class KeywordHit(BaseModel):
    rank: int
    label: str
    score: float = Field(description="BM25 score — higher is more keyword overlap.")
    snippet: str


class KeywordQueryResponse(BaseModel):
    query: str
    top_k: int
    results: list[KeywordHit]


# --- The agent -----------------------------------------------------------------


class AgentChatRequest(Request):
    """One turn of a conversation with the workflow."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "message": "I found a knife in a guest room, what should I do?",
                    "session_id": None,
                },
                {"message": "yes please, file it", "session_id": "api-1a2b3c4d"},
            ]
        },
    )

    message: str = Field(min_length=1, description="What to say to the agent.")
    session_id: str | None = Field(
        default=None,
        description=(
            "The conversation this turn belongs to. Omit to start one — the id "
            "comes back in the response, and sending it on the next turn is "
            "what lets 'yes, file it' refer to anything. An unknown id starts a "
            "new conversation rather than failing."
        ),
    )
    user_id: str | None = Field(
        default=None, description="Attributes the run to a user in Langfuse."
    )
    tags: list[str] | None = Field(
        default=None, description="Langfuse tags for this run — use them to mark experiments."
    )
    include_trace: bool = Field(
        default=False,
        description="Also return the route the run took and every tool call it made.",
    )


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any]


class AgentChatResponse(BaseModel):
    """The agent's reply, and enough of the run to tell whether to trust it."""

    session_id: str
    turn: int = Field(description="Which turn of this conversation this was.")
    answer: str = Field(description="The answer alone.")
    reply: str = Field(
        description=(
            "What a client should show: the answer, plus the ticket offer when "
            "there is one. This is also what the transcript records."
        )
    )
    ticket_offer: str | None = Field(
        default=None,
        description="The offer, if the run earned one. Null means it was not offered.",
    )
    ticket_recipient: str = Field(
        default="",
        description="The role the answer says to raise the matter with. Empty if none.",
    )
    verdict: Literal["pass", "fail", "ungraded"] | None = Field(
        default=None,
        description=(
            "The verifier's judgement. 'fail' means the answer is returned but "
            "was not fully grounded; 'ungraded' means the verifier itself did "
            "not run, so nothing checked it. Neither is 'pass'."
        ),
    )
    revisions: int = Field(description="How many times the answer was sent back.")
    trace_url: str | None = Field(default=None, description="The Langfuse trace, if tracing is on.")
    route: list[str] | None = Field(
        default=None, description="Workers the orchestrator ran, in order. Needs include_trace."
    )
    tool_calls: list[ToolCall] | None = Field(
        default=None, description="Every tool call the run made. Needs include_trace."
    )


class TranscriptTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class SessionSummary(BaseModel):
    id: str
    created_at: datetime
    updated_at: datetime
    turns: int


class SessionDetail(SessionSummary):
    transcript: list[TranscriptTurn]


class SessionList(BaseModel):
    sessions: list[SessionSummary]


# --- Documents -----------------------------------------------------------------


class DocumentStatus(BaseModel):
    """One document, across all three places it can exist.

    A single row per name rather than three separate listings, because the
    question worth answering is "is this PDF ingested, and into what" — and the
    two stores are ingested into separately, so a file present in one and absent
    from the other is the normal state during setup, not an anomaly.
    """

    name: str
    raw_path: str | None = Field(default=None, description="Null if the PDF is not on disk.")
    size_bytes: int | None = None
    modified_at: datetime | None = None
    sections: int = Field(default=0, description="Sections in the heading index.")
    graph_sections: int = Field(default=0, description="Section documents in the LightRAG graph.")


class DocumentList(BaseModel):
    raw_dir: str
    documents: list[DocumentStatus]


class UploadResponse(BaseModel):
    name: str
    path: str
    size_bytes: int
    replaced: bool = Field(description="Whether a file of that name was overwritten.")
    note: str = Field(
        default="Uploaded only. Call /ingest/sections or /ingest/graph to index it.",
        description="Uploading does not ingest — the two are separate on purpose.",
    )


class DeleteDocumentResponse(BaseModel):
    name: str
    store: StoreName
    sections_removed: int
    graph: str = Field(description="What the graph store reported, verbatim.")


# --- Tickets -------------------------------------------------------------------


class TicketFile(BaseModel):
    name: str
    path: str
    size_bytes: int
    created_at: datetime


class TicketList(BaseModel):
    tickets_dir: str
    tickets: list[TicketFile]


# --- Health --------------------------------------------------------------------


class SectionStoreHealth(BaseModel):
    """The chunkless store: trees on disk, and the heading index over them.

    Both counts, because they are written by the same ingest and can still
    disagree — and when they do, which one is behind says what went wrong.
    """

    tree_dir: str
    trees: int = Field(description="Documents parsed into data/tree/ as JSON.")
    directory: str = Field(description="The Chroma directory holding the heading index.")
    collection: str
    indexed_sections: int = Field(description="Sections findable by search.")
    documents: int


class GraphStoreHealth(BaseModel):
    directory: str
    exists: bool
    indexed_chunks: int = Field(
        description="Chunks in LightRAG's persisted chunk store — the BM25 corpus."
    )


class HealthResponse(BaseModel):
    """Whether the server is up, and what it has to work with.

    Deliberately answers without touching the LLM provider: this is what a load
    balancer polls, and a health check that costs an API call is one nobody can
    afford to run often enough to be useful.
    """

    status: Literal["ok"] = "ok"
    raw_dir: str
    raw_pdfs: int
    sections: SectionStoreHealth
    graph: GraphStoreHealth
    tickets_dir: str
    tickets: int
    tracing_enabled: bool


class TracingHealth(BaseModel):
    """The one check that does go over the network — hence its own endpoint."""

    enabled: bool
    ok: bool
    detail: str


class ErrorResponse(BaseModel):
    """The shape of every error this API returns."""

    detail: str
