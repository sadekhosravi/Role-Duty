# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An agentic RAG system over role-and-duty documents: PDF → Docling extraction → a
tree of the sections the author wrote → find-then-read retrieval, with a LightRAG
knowledge graph behind it as the escalation tier, driven by a LangGraph
orchestrator-workers workflow. It is a learning foundation meant to grow one step
at a time, so favor small, readable changes over abstraction.

Nothing is chunked. That is the design decision most of the code follows from —
see "Chunkless retrieval" below before changing anything under `src/doctree/`.

## Commands

Dependencies are managed with **uv** (`uv.lock`, `pyproject.toml`, Python 3.14 pinned in `.python-version`), though a `requirements.txt` also exists.

```bash
# Setup (uv)
uv sync

# Setup (plain venv)
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell/cmd)
pip install -r requirements.txt

# Ingest PDFs (section tree -> data/tree, heading index -> Chroma)
python scripts/ingest.py                 # all PDFs in data/raw
python scripts/ingest.py data/raw/x.pdf  # single file
python scripts/ingest.py path/to/folder  # a folder

# Query the sections directly — retrieval only, no answer step
python scripts/query.py "your question"
python scripts/query.py "your question" --top-k 3
python scripts/query.py --outline sample_role_duties.pdf
python scripts/query.py --read sample-role-duties#shift-supervisor

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
`scripts/check_agent.py` is the closest thing: ~185 offline assertions over the
agent's graph shape, tool boundaries, prompts, stores and the ticket server. Run
it after touching anything under `src/agent/`, `src/doctree/` or
`src/ticket_mcp/`.

One check is expected to fail today — `every graph chunk has a citable label`,
four `Hospitalist (Internal Medicine)` chunks with no file prefix. It predates
the current work and is out of scope; everything else should pass.

## Architecture

Layers: `src/doctree/` (chunkless retrieval), `src/graph_rag/` (LightRAG),
`src/agent/` (the LangGraph workflow), `src/api/` (FastAPI over all of it),
`src/rag/` (settings, embeddings, the Chroma client), and thin `scripts/` CLIs.
The scripts prepend `src/` to `sys.path` at runtime (`sys.path.insert(...)`) so
the packages resolve without installing them — keep that shim if you add a script.

### Chunkless retrieval — read this before touching src/doctree/

The corpus is never cut into fixed-size windows. `graph_rag.extraction.parse_pdf`
already recovers each piece of text's heading path; `doctree.tree` keeps that
structure as a tree instead of flattening it to a label string.

1. `tree.build_from_pdf()` — one `Node` per heading path, nested by prefix, ids
   from the slugified path (`sample-role-duties#shift-supervisor/out-of-scope`).
   A heading that spans a page break becomes ONE node here, where `parse_pdf`
   yields two — the tree's unit is the section the author wrote.
2. `store.save()` — one JSON file per document under `data/tree/`. **This is the
   source of truth.** `data/chroma/` is derived and can be deleted and rebuilt
   from it without re-running Docling, which is slow and occasionally faults
   inside a native library.
3. `index.index_document()` — one Chroma row per section that has text, embedded
   over its heading path **plus its first 400 characters**. Not the heading path
   alone: half the headings here are structural ("Out of Scope"), so a pure
   heading key has nothing for a content question to match. The row is only how
   a section is *found*; what is returned is always the whole section from the
   tree, so this is not a chunk by another name.
4. `search.find_sections()` — BM25 over full section text, fused with the
   heading hits by **Reciprocal Rank Fusion** (ranks only — a cosine distance
   and a BM25 score share no scale). BM25 is fused *inside* this function
   rather than offered as its own tool, because as a tool it got skipped, and
   it is free.

Things that will break if you change them carelessly:

- **Both halves must return tree node ids.** BM25 used to run over LightRAG's
  chunk store, which is cut differently and identifies rows differently. Fusing
  those two lists would be arithmetic over two vocabularies.
- **The ingest writes the tree first and the index second**, and the delete does
  the reverse. Index rows that resolve to nothing are search results that lead
  to a citation to nowhere; a tree with no rows is merely unfindable, and one
  re-ingest fixes it.
- **`find_section` returns previews; `read_section` returns evidence.** The
  researcher's prompt and the tool descriptions both say so in as many words,
  and `check_agent.py` asserts both strings. A model handed text will use it.

Querying (`scripts/query.py`) fuses the same two retrievers and can also print a
document's outline or read one section by id. There is no answer step — that is
the agent's job.

### Configuration & embeddings — read this before changing either

- `config.py` centralizes all settings in a frozen `Settings` dataclass, read from env vars (via `python-dotenv`) with defaults; paths resolve relative to `ROOT_DIR` (two levels up from `src/rag/config.py`). Env keys: `RAW_DATA_DIR`, `TREE_DIR`, `CHROMA_DIR`, `COLLECTION_NAME`, `HEADING_COLLECTION_NAME`, `EMBEDDING_MODEL`.
- `embeddings.py` is the **single place** that turns text into vectors. Despite the README mentioning sentence-transformers, the actual code path uses Chroma's `OpenAIEmbeddingFunction` pointed at **OpenRouter** via `OPENROUTER_API_KEY` and `OPENROUTER_API_BASE_URL` (also from `.env`). Swap the model with `EMBEDDING_MODEL`, or replace this one function to change providers.
- After changing the embedding model, **re-index everything** — vectors from different models are not comparable, and the persisted store in `data/chroma/` will hold stale ones. `data/tree/` is unaffected: it holds no vectors, so the re-index does not need a re-parse.

### The agent, and the one tool that writes

`src/agent/` is a LangGraph orchestrator-workers workflow — see `graph.py` for
the wiring and `nodes/` for one file per node. Five nodes: orchestrator,
researcher, responder, verifier, filer.

**The orchestrator is the only node that decides where control goes**, including
whether the run is over. The verifier used to have a conditional edge of its own,
which made two nodes able to terminate and put the loop bounds in two files; it
now reports its verdict back like any other worker. Load-bearing details:

- `state.py` splits the vocabulary in two. `WorkerName` is what the *model* may
  be asked to pick and is the only one that reaches it as a JSON-schema enum;
  `RouteName` adds `finish`, which the graph maps to `END`. The model therefore
  cannot choose to stop — `orchestrator._is_done()` decides that from the verdict
  and the counters, in Python, before any model call.
- **That terminal check runs before the delegation cap**, and the order is not
  cosmetic. A graded run comes back through the orchestrator; a cap read first
  would send it to the responder again on every pass and never reach the exit.
- The finish route writes `state.reply`, never `state.answer`. `answer` is the
  exact string the verifier graded, and the ticket offer appended for the user
  was not part of what it graded. Front ends read `conversation.final_reply()`.
- The filer keeps its own edge to `END` and sets `reply` itself, since the
  orchestrator is never reached from there.

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
- **Blocking calls go to a threadpool.** Chroma, BM25, the tree JSON reads and
  Docling are all synchronous; `run_in_threadpool` around them is not optional
  in an async server. `graph_rag.ingest` offloads `parse_pdf` itself for the
  same reason.
- **The section caches are reset by the ingest, not by the routers.**
  `doctree.store` and `doctree.search` each cache one process-wide object, and a
  server that ingests and then answers would otherwise reply from the corpus it
  had at startup. `doctree.ingest` calls both `reset()`s, so nothing else has to
  remember to.
- **One worker.** All of that state is per process, so `scripts/serve.py` starts
  exactly one and says why.

The ticket-offer rule lives in `src/agent/conversation.py` because both the CLI
and `/agent/chat` apply it. `check_agent.py` still reaches it through
`scripts/agent.py`, which re-exports it.

### Known drift to be aware of

The README and `requirements.txt` describe a self-contained sentence-transformers setup, but `embeddings.py` and `pyproject.toml` (`openai`, `docling`, `chromadb`) reflect the current OpenRouter-based reality. When touching embeddings or dependencies, treat the code as the source of truth and reconcile the docs.

## Data & secrets

- `data/raw/` holds input PDFs (gitignored); `data/tree/` holds the section trees as JSON and is the source of truth for retrieval; `data/chroma/` is the derived heading index, created automatically; `data/tickets/` holds filed tickets as `.docx`, created on the first ticket. All gitignored via `data/*`. Override the tickets location with `TICKETS_DIR`, the trees with `TREE_DIR`.
- `.env` holds real credentials and is gitignored. No `.env.example` currently exists despite the README referencing one.
