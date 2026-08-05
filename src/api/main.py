"""The app: what it is called, what it holds, and how it fails.

Built by a factory rather than assembled at module level, so the app can be
constructed twice — a test client beside the running server — without the two
sharing a job registry or each other's conversations. `app` at the bottom is the
one instance uvicorn imports.

Read this file for three things and the routers for everything else:

* **Lifespan.** What is opened before the first request, and what is cancelled
  before the process exits. A background ingest left running while the
  interpreter tears down is how a LightRAG store ends up half written.
* **Error handling.** Which exception from the pipelines becomes which status
  code, in one place, so no router has to wrap a call in try/except to turn a
  missing PDF into a 404.
* **The OpenAPI description.** It is the front page of the Swagger UI, and for
  an API whose endpoints cost real money and take real minutes, the order to
  call them in is the first thing a reader needs.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .jobs import JobRegistry
from .paths import PathRejected
from .routers import agent, documents, health, ingest, jobs, query, tickets
from .sessions import SessionStore
from .settings import settings

log = logging.getLogger(__name__)

TITLE = "Role-Duty RAG API"
VERSION = "0.1.0"

DESCRIPTION = """
Three retrieval paths over a corpus of organizational role documents, and an
agent that uses all three — the same pipelines the `scripts/` CLIs drive.

### The order to call these in

1. `POST /documents/upload` — put a PDF where the server can read it, or skip
   this if `data/raw` is already populated.
2. `POST /ingest/naive` and `POST /ingest/graph` — fill the two stores. They are
   separate stores filled by separate pipelines; doing one does not do the
   other. Both return a **job id**: poll `GET /jobs/{id}` until it succeeds.
3. `POST /query/naive`, `POST /query/graph`, `POST /query/keyword` — the three
   retrieval paths, deliberately kept apart so they can be compared.
4. `POST /agent/chat` — the whole workflow, as a conversation.

`GET /health` tells you which of those have actually happened.

### What costs what

`/query/naive` and `/query/keyword` are local and instant. `/query/graph` makes
several LLM calls. `/agent/chat` is several models in a loop and takes tens of
seconds. `/ingest/graph` has an LLM read every section of every document and is
the expensive one by a wide margin.

### One process

Jobs and conversations live in this process's memory and are gone when it
restarts, and the lock that stops two ingests corrupting the graph is a lock
within it. Run one worker — see `scripts/serve.py`.
"""

TAGS = [
    {"name": "health", "description": "Is the server up, and what is in each store."},
    {"name": "documents", "description": "The PDFs: upload, list, remove from the stores."},
    {"name": "ingest", "description": "Fill the stores. Slow, so both return a job."},
    {"name": "jobs", "description": "Watch what an ingest is doing."},
    {"name": "query", "description": "The three retrieval paths: vector, graph, BM25."},
    {"name": "agent", "description": "The workflow, as a multi-turn conversation."},
    {"name": "tickets", "description": "The incident tickets the agent filed."},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open what the process holds, and close it properly on the way out.

    The stores are deliberately *not* opened here. Chroma and LightRAG are both
    expensive to open and neither is needed to serve `/health`, so each is built
    on first use (see api/stores.py) — the alternative is every restart paying
    for a graph reload whether or not anyone asks the graph anything.

    What does happen here is the shutdown half, and it earns the lifespan on its
    own: a background ingest still running when the interpreter starts tearing
    down is a write to the LightRAG store that may not finish.
    """
    app.state.jobs = JobRegistry(max_jobs=settings.max_jobs)
    app.state.sessions = SessionStore(
        max_sessions=settings.max_sessions, max_turns=settings.max_session_turns
    )
    try:
        yield
    finally:
        await app.state.jobs.shutdown()


def create_app() -> FastAPI:
    """Build the application. Called once below; called again by tests."""
    app = FastAPI(
        title=TITLE,
        version=VERSION,
        description=DESCRIPTION,
        openapi_tags=TAGS,
        lifespan=lifespan,
        # Swagger keeps its "Try it out" bodies and the tags collapsed between
        # reloads, which matters when the thing being tried takes a minute.
        swagger_ui_parameters={"docExpansion": "list", "persistAuthorization": True},
    )

    if settings.cors_origins:
        # Only when configured. Swagger is served by this app and needs no CORS
        # at all, so a default wildcard would exist purely to let a page on some
        # other origin drive an API that writes files.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    for module in (health, documents, ingest, jobs, query, agent, tickets):
        app.include_router(module.router)

    _install_error_handlers(app)
    return app


def _install_error_handlers(app: FastAPI) -> None:
    """Map the pipelines' exceptions onto status codes, once, for every route.

    These are the failures the library layer raises by design — a PDF that is
    not there, a folder with no PDFs in it, a path pointing outside the project.
    Handling them here rather than in each router is what keeps the routers as
    a call and a response model; the alternative is the same try/except copied
    into nine endpoints, where it will be forgotten in the tenth.
    """

    @app.exception_handler(FileNotFoundError)
    async def _not_found(request: Request, error: FileNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(error)}
        )

    @app.exception_handler(PathRejected)
    async def _rejected(request: Request, error: PathRejected) -> JSONResponse:
        # A client asking for something it is not allowed to have, not a bug.
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(error)}
        )

    @app.exception_handler(ValueError)
    async def _bad_request(request: Request, error: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(error)}
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, error: Exception) -> JSONResponse:
        # Everything else — an OpenRouter outage, a corrupt store. The type and
        # message go to the client because this API is a development tool and a
        # bare "internal server error" would send someone to the logs for
        # something the response could have told them. Behind a public front
        # door, this handler is the one to change.
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"{type(error).__name__}: {error}"},
        )


app = create_app()
