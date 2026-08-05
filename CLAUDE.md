# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A deliberately minimal RAG ingestion + retrieval pipeline: PDF → Docling extraction → embeddings → Chroma vector store → similarity search. It is a learning foundation meant to grow one step at a time, so favor small, readable changes over abstraction.

## Commands

Dependencies are managed with **uv** (`uv.lock`, `pyproject.toml`, Python 3.14 pinned in `.python-version`), though a `requirements.txt` also exists.

```bash
# Setup (uv)
uv sync

# Setup (plain venv)
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell/cmd)
pip install -r requirements.txt

# Ingest PDFs (chunks -> embeddings -> Chroma)
python scripts/ingest.py                 # all PDFs in data/raw
python scripts/ingest.py data/raw/x.pdf  # single file
python scripts/ingest.py path/to/folder  # a folder

# Query (similarity search only — no LLM answer step yet)
python scripts/query.py "your question"
python scripts/query.py "your question" --top-k 3

# The agent (a conversation, not a one-shot query)
python scripts/agent.py "I found a knife in a guest room, what should I do?"
python scripts/agent.py --once "..."     # answer and exit
python scripts/check_agent.py            # the wiring, checked offline

# The ticket MCP server, standalone
python -m ticket_mcp                     # waits on stdin; Ctrl-C to stop

# The HTTP API (Swagger at /docs, ReDoc at /redoc)
python scripts/serve.py
python scripts/serve.py --port 9000 --reload
```

There is currently **no test suite, linter, or formatter** configured.
`scripts/check_agent.py` is the closest thing: ~150 offline assertions over the
agent's graph shape, tool boundaries, prompts, stores and the ticket server. Run
it after touching anything under `src/agent/` or `src/ticket_mcp/`.

One check is expected to fail today — `every graph chunk has a citable label`,
four `Hospitalist (Internal Medicine)` chunks with no file prefix. It predates
the current work and is out of scope; everything else should pass.

## Architecture

Two layers: a `src/rag/` library and thin `scripts/` CLIs. The scripts prepend `src/` to `sys.path` at runtime (`sys.path.insert(...)`) so `rag` imports resolve without installing the package — keep that shim if you add new scripts.

Ingestion data flow (`pipeline.py` orchestrates):
1. `extractor.extract_chunks()` — delegates to `graph_rag.extraction.parse_pdf()`, the **same** parser the graph ingest uses, and wraps each `Section` in a `Chunk` dataclass (`text`, `source`, `index`, `label`, `page`, `headings`). Sharing that call is deliberate: it carries the owning role into every sub-section's text and label, and it was added here only after the two ingest paths drifted and the vector store spent a while returning role-less chunks that the agent then mis-attributed. Do not reintroduce a separate chunker here.
2. `vector_store.add_chunks()` — writes chunks to a persistent Chroma collection. **Chroma computes the embeddings at `add()` time** via the collection's embedding function; the code never embeds text itself.
3. Chunk IDs are `"{source}:{index}"`. `add_chunks` deletes a document's existing rows before writing, because a re-ingest can now produce fewer sections than a previous run did and the surplus ids would otherwise survive as retrievable orphans.

Querying (`query.py`) calls `vector_store.similarity_search()` and prints ranked chunks with source/distance. There is no generation/answer step — retrieval is the end of the pipeline today.

### Configuration & embeddings — read this before changing either

- `config.py` centralizes all settings in a frozen `Settings` dataclass, read from env vars (via `python-dotenv`) with defaults; paths resolve relative to `ROOT_DIR` (two levels up from `src/rag/config.py`). Env keys: `RAW_DATA_DIR`, `CHROMA_DIR`, `COLLECTION_NAME`, `EMBEDDING_MODEL`.
- `embeddings.py` is the **single place** that turns text into vectors. Despite the README mentioning sentence-transformers, the actual code path uses Chroma's `OpenAIEmbeddingFunction` pointed at **OpenRouter** via `OPENROUTER_API_KEY` and `OPENROUTER_API_BASE_URL` (also from `.env`). Swap the model with `EMBEDDING_MODEL`, or replace this one function to change providers.
- After changing the embedding model, **re-ingest everything** — vectors from different models are not comparable, and the persisted store in `data/chroma/` will hold stale ones.

### The agent, and the one tool that writes

`src/agent/` is a LangGraph orchestrator-workers workflow — see `graph.py` for
the wiring and `nodes/` for one file per node. Five nodes: orchestrator,
researcher, responder, verifier, filer.

The **filer** is the only node with a side effect. It writes an incident ticket
as a Word document, and its tool comes from an MCP server (`src/ticket_mcp/`)
started as a stdio subprocess. Several things about it are load-bearing:

- The **server owns the tool's schema**. Nothing in `src/agent/` declares what a
  ticket looks like; `tools/tickets.py` starts the server, asks what it offers,
  and raises if `create_ticket` is not among them. Do not hand-write a mirror of
  the schema on the client side — that duplication is the exact drift this
  arrangement exists to prevent.
- Discovery is **synchronous, at import**, because `NodeSpec`, `NODE_SPECS` and
  `WORKFLOW` are module constants and a node's tool list has to exist by the time
  its spec does. It costs one short-lived subprocess per process. The tools do
  not hold that session open — each call opens its own — which is what makes it
  safe to discover in one event loop and call from another.
- A discovered tool **captures the environment** the server will be spawned
  with. Changing `TICKETS_DIR` afterwards has no effect on an existing handle;
  call `load_tools()` again.
- `create_ticket` appears in **exactly one** node's tool list, and `check_agent.py`
  asserts that. Tool access is per node, so that list is the whole boundary
  between "the agent can answer" and "the agent can write files".
- The filer routes to **END**, not back to the orchestrator: it acts rather than
  gathers, and its confirmation is the user-facing reply. It must not reach the
  verifier, which requires a `### References` section that a ticket confirmation
  has no business carrying.

The conversation lives in `scripts/agent.py`, not in a LangGraph checkpointer.
`AgentState` is the work on **one** request — `delegations` accumulates under
`operator.add`, `revisions` counts up, `verdict` describes the current answer —
so checkpointing it would start turn two with turn one's budget spent. The CLI
keeps the transcript (questions and replies only) and passes it to
`run_workflow(history=...)`; the working state starts fresh each turn.

### The HTTP layer

`src/api/` is FastAPI over everything above — see `main.py` for the app and
`routers/` for one module per resource. It implements no retrieval, no ingestion
and no agent: each route calls the same function the matching CLI calls, which
is the only thing keeping the two surfaces from answering differently.

What it *does* own is the difference between a process that runs once and one
that stays up, and that is where the load-bearing decisions are:

- **One LightRAG per process, and it is the agent's.** `api/stores.py` re-exports
  `agent.tools._stores`, deliberately. A router that built its own instance would
  put two LightRAGs over one on-disk store, each finalizing storages the other
  still had open. This is also why `graph_rag.ingest/query/remove` now take an
  optional `rag=` — passing one means "used as-is, left open"; passing nothing
  keeps the build-and-finalize the CLIs rely on.
- **Writes take `graph_write_lock()`, reads do not.** Two ingests interleaving
  over one NetworkX graph is not a race the storage layer resolves. Holding the
  lock across a query, on the other hand, would mean one ingest blocks every
  question for as long as an LLM takes to read a corpus.
- **Ingest returns a job, not an answer.** `api/jobs.py` — asyncio tasks in the
  serving process, bounded and in memory. Read its docstring before reaching for
  it: no queue, no worker, nothing survives a restart.
- **The agent is multi-turn and the server keeps the transcript.**
  `api/sessions.py`, one lock per session. It stores exactly what the CLI stores
  — questions and the replies as the user saw them, offer included — for the
  same reason `run_workflow` takes `history` rather than checkpointing.
- **Blocking calls go to a threadpool.** Chroma, BM25 and Docling are all
  synchronous; `run_in_threadpool` around them is not optional in an async
  server. `graph_rag.ingest` offloads `parse_pdf` itself for the same reason.
- **One worker.** All of that state is per process, so `scripts/serve.py` starts
  exactly one and says why.

The ticket-offer rule lives in `src/agent/conversation.py` because both the CLI
and `/agent/chat` apply it. `check_agent.py` still reaches it through
`scripts/agent.py`, which re-exports it.

### Known drift to be aware of

The README and `requirements.txt` describe a self-contained sentence-transformers setup, but `embeddings.py` and `pyproject.toml` (`openai`, `docling`, `chromadb`) reflect the current OpenRouter-based reality. When touching embeddings or dependencies, treat the code as the source of truth and reconcile the docs.

## Data & secrets

- `data/raw/` holds input PDFs (gitignored); `data/chroma/` is the persisted vector DB, created automatically (gitignored); `data/tickets/` holds filed tickets as `.docx`, created on the first ticket (gitignored via `data/*`). Override with `TICKETS_DIR`.
- `.env` holds real credentials and is gitignored. No `.env.example` currently exists despite the README referencing one.
