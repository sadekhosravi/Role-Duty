# Role-Duty — an agentic RAG system over role and duty documents

Ask who is responsible for something, get a cited answer, and — when the answer
establishes someone to raise the matter with — have a ticket filed for them as a
Word document.

The corpus is a set of role and duty reference guides for several unrelated
organisations (a hotel, a hospital, an airport, a data centre, a museum, a
police department, a school, a shop). The questions that matter about documents
like these are rarely lookups. They are *"a housekeeper found a knife — who does
that go to, and does it stop there?"*, which turns on an escalation chain, on
what a role is explicitly **not** allowed to do, and on not confusing two
organisations that both have a "Duty Manager".

Built on [LangGraph](https://github.com/langchain-ai/langgraph), with chunkless
retrieval over the corpus, an evaluator loop, MCP for the one action that writes
to disk, and Langfuse for tracing.

```
START ─> orchestrator ─> researcher ─> back to orchestrator
                      ─> filer ─> END
                      ─> responder ─> verifier ─> back to orchestrator
                      ─> finish ─> END
```

**Nothing is chunked.** The documents are not cut into fixed-size windows —
they are kept as the tree of sections their author wrote, one JSON file per PDF
under `data/tree/`. Retrieval is then the two steps a person would take:
`find_section` shortlists (heading embeddings fused with BM25, both on every
call), and `read_section` returns one section whole, sub-sections included. A
role's duties, its exclusions and its escalation path arrive together, which is
the combination this corpus punishes you for splitting up. `graph_rag_search`
([LightRAG](https://github.com/HKUDS/LightRAG)) stays as the escalation tier:
the only retriever that returns *relationships*, and the expensive one.

**Retrieval is deterministic — the model only decides what to look for.** Docling
parses the PDF, `tree.py` assembles the sections from its heading paths, BM25
ranks over their text and RRF fuses the two lists; none of that calls a language
model. The one model in the retrieval path is the embedding model behind the
heading index. The chat model reads the question, picks the search and writes the
answer, but it never decides what a section says.

**The orchestrator decides when to stop.** The verifier grades and hands its
verdict back like any other worker; nothing but the orchestrator can end a run.
An evaluator that could also halt would be a second controller, and the loop
bounds would have to be read in two places — which is how they were, and they
disagreed.

**Grounding is checked, not requested.** The responder cannot retrieve, so the
verifier grades its claims against the same evidence the responder had. Citation
labels are validated by string comparison against what the tools actually
returned — prompting for correct citations failed twice, first by inventing
plausible labels with shifted page numbers, then by stripping the file and page
instead of copying them. String comparison is not persuadable.

**Every loop is bounded.** Delegations, revisions and retrievals each have a cap,
because a disagreement between two nodes otherwise runs until the recursion limit
kills the request and returns nothing at all.

> Built as a learning project, one step at a time. Most comments in the code
> explain *why* something is the way it is, usually because the obvious
> alternative was tried first and broke.

## Project structure

```
Role-Duty/
├── data/
│   ├── raw/            # drop your PDFs here
│   ├── tree/           # the section trees, one JSON per PDF — source of truth
│   ├── chroma/         # heading index, derived from the trees
│   ├── tickets/        # filed tickets, as .docx (created automatically)
│   └── val/            # the validation corpus and its own stores, never mixed in
├── scripts/
│   ├── ingest.py       # PDF -> section tree -> heading index
│   ├── query.py        # find sections, read one, or print an outline
│   ├── agent.py        # a conversation with the agent
│   ├── serve.py        # the HTTP API (Swagger at /docs)
│   ├── check_agent.py  # the agent's wiring, checked offline
│   └── eval/           # the scored validation run — see its own README
├── src/doctree/
│   ├── tree.py         # Node/Document — the section tree itself
│   ├── store.py        # JSON per document, and reading a section whole
│   ├── index.py        # one Chroma embedding per section, over its headings
│   ├── search.py       # heading search fused with BM25, always both
│   └── ingest.py       # PDF -> tree -> index
├── src/rag/            # settings, the embedding model, the Chroma client
├── src/graph_rag/      # Docling extraction (both stores use it), and LightRAG
├── src/agent/          # the LangGraph workflow — see src/agent/graph.py
├── src/api/            # FastAPI over all of the above — see src/api/main.py
└── src/ticket_mcp/     # MCP server: writes a ticket as a Word document
```

## Setup

Python 3.14. [uv](https://docs.astral.sh/uv/) is the supported path:

```bash
uv sync
cp .env.example .env          # then put your OpenRouter key in it
```

Or with a plain venv:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
cp .env.example .env
```

You need an [OpenRouter](https://openrouter.ai/keys) key — one key serves the
chat model, the embeddings and the reranker. Everything else in `.env.example`
has a working default; see [The models](#the-models) for what each one drives.

> With uv there is nothing to activate: prefix every command below with
> `uv run` (`uv run python scripts/agent.py …`). With the venv, activate it once
> and run `python` directly.

## Usage

1. Put one or more PDFs in `data/raw/`.

2. Ingest them — twice, once into each store:

   ```bash
   python scripts/ingest.py                       # -> section trees + heading index
   python src/graph_rag/graph_rag.py ingest data/raw   # -> LightRAG graph
   ```

3. Ask it something:

   ```bash
   python scripts/agent.py "Who authorises a refund above the threshold?"
   ```

   Or search the sections directly, with no agent and no answer step:

   ```bash
   python scripts/query.py "What is the refund policy?" --top-k 3
   python scripts/query.py --outline sample_role_duties.pdf   # the headings
   python scripts/query.py --read sample-role-duties#shift-supervisor
   ```

4. Check the wiring at any point. 187 assertions, offline, no API key needed —
   one of them is a known failure, see [Known rough edges](#known-rough-edges):

   ```bash
   python scripts/check_agent.py
   ```

## Talking to the agent

`scripts/agent.py` is a conversation, not a one-shot query. Ask something, get a
cited answer, and then keep going — the follow-up turns can refer back to what
was just said.

```bash
python scripts/agent.py                                  # start empty
python scripts/agent.py "I found a knife in a guest room, what should I do?"
python scripts/agent.py "..." --once                     # answer and exit
python scripts/agent.py "..." --trace                    # show the tool calls
```

Only the questions and the replies are carried between turns — not the evidence
behind them. Each turn's retrieval budget, revision count and verdict start
fresh, so a long conversation cannot use up a later question's turns.

## Filing a ticket (MCP)

When an answer establishes someone to raise a matter with, the agent offers to
file them a ticket. Say yes and it writes a Word document to `data/tickets/`,
then asks what else you need.

```
you> i am a housekeeper, i found a knife in the hotel room, what should i do?

The Housekeeping Supervisor (Housekeeping) escalates to the Duty Manager
(Front of House) for anything found in a room that requires police
notification [1].

### References
- [1] sample_role_duties_hotel.pdf › page 3 › Housekeeping Supervisor (Housekeeping) › Escalation Path:

Would you like me to file a ticket for the Housekeeping Supervisor
(Housekeeping)? Say yes and I will write it to data/tickets/.

you> yes

I created a ticket for the Housekeeping Supervisor (Housekeeping). It is saved
to data/tickets/TKT-20260805-125055-knife-found-in-hotel-room.docx …

What else do you need?
```

The ticket carries who it is addressed to, what happened, a summary of the
situation, the next steps, and the source labels the answer cited.

**When it does not offer.** Only a situation with somebody to take it to gets
the offer — not a greeting, not a general question about how the organisation
works, not an answer that could not find a responsible role, and not an answer
the verifier rejected. The verifier decides, as a by-product of the grading call
it already makes, and the offer names the role so the question can be answered
without re-reading the answer.

The writing is done by an **MCP server** in `src/ticket_mcp/`, started as a
stdio subprocess. It exposes exactly one tool, `create_ticket`, and it owns that
tool's schema — nothing in `src/agent/` declares what a ticket looks like, it
just asks the server what it offers. Run it standalone to check it starts:

```bash
python -m ticket_mcp        # waits on stdin for MCP traffic; Ctrl-C to stop
```

Two constraints are worth knowing, because both were mistakes waiting to happen:

- **Only the `filer` node has that tool.** Tool access is per node, so no other
  node can write a file, and `scripts/check_agent.py` asserts the tool appears in
  exactly one node's list.
- **One ticket per turn, and only when asked.** Past the cap the tool is
  unbound rather than discouraged, and the orchestrator is told that an
  unasked-for ticket is a real document in someone's queue.

`TICKETS_DIR` moves where tickets are written (default `data/tickets`).

## The HTTP API

Everything above, over HTTP, with a Swagger page to drive it from:

```bash
python scripts/serve.py            # http://127.0.0.1:8000/docs
python scripts/serve.py --port 9000 --reload
```

Nothing in `src/api/` implements retrieval, ingestion or the agent — each route
calls the same function the matching CLI calls, so the two surfaces cannot
answer the same question differently.

| Endpoint | What it is |
| --- | --- |
| `POST /documents/upload` | Put a PDF in `data/raw` from the browser. Does not ingest. |
| `GET /documents` | One row per PDF: on disk, in the section index, in the graph. |
| `DELETE /documents/{name}` | Drop it from either store, or both. The PDF stays. |
| `POST /ingest/sections` | PDFs → section trees + heading index. Returns a **job id**. |
| `POST /ingest/graph` | PDFs → LightRAG graph. Returns a **job id**. The slow, expensive one. |
| `GET /jobs/{id}` | Poll an ingest until `succeeded` or `failed`. |
| `POST /query/section` | Whole sections, ranked. No LLM, no answer. |
| `POST /query/graph` | An answer from the graph, with a `### References` section. |
| `POST /query/keyword` | BM25 over the *graph* store's chunks. A different corpus — it is there to show what the graph is working from. |
| `POST /agent/chat` | The whole workflow, as a conversation. |
| `GET /agent/sessions/{id}` | What a conversation remembers. |
| `GET /tickets` | What the agent filed, and a download for each. |
| `GET /health` | Which stores are actually populated. Costs nothing. |

**Ingest returns a job, not an answer.** Parsing a PDF takes a minute and the
graph ingest has an LLM read every section of it, which is not work that fits
inside a request. `POST /ingest/graph` answers `202` with a job id; poll
`GET /jobs/{id}`. Jobs run as asyncio tasks in the serving process — there is no
queue and no worker, so a restart loses them.

**The agent is multi-turn, and the server holds the transcript.** Omit
`session_id` on the first call, then send back the one you get:

```bash
curl -X POST localhost:8000/agent/chat -H 'content-type: application/json' \
  -d '{"message":"I found a knife in a guest room. What should I do?"}'
# -> {"session_id":"api-3f90a947", ..., "ticket_offer":"Would you like me to …"}

curl -X POST localhost:8000/agent/chat -H 'content-type: application/json' \
  -d '{"message":"yes please, file it","session_id":"api-3f90a947"}'
# -> the ticket, written to data/tickets/
```

Without the session id the second call has nothing to agree to, and the filer
has nothing to file. Sessions are in memory and bounded, like the jobs.

**Check `verdict` on the reply.** The workflow returns rejected answers rather
than nothing — the revision cap ends the loop whether or not the answer got
better. `pass` means the verifier found it grounded; `fail` means it did not and
the answer came back anyway; `ungraded` means the verifier never ran. The CLI
prints a warning at that point, and over HTTP the field is the warning.

**Run one worker.** The job registry, the conversations and the lock that keeps
two ingests from writing over each other are objects in this process. A second
worker would give each of them a second copy and the lock would guard nothing.
`scripts/serve.py` starts one on purpose; scaling past it means moving that
state out, not adding a flag.

Optional settings, all with working defaults: `API_HOST`, `API_PORT`,
`API_CORS_ORIGINS` (comma-separated; the middleware is not installed unless you
set it), `API_MAX_JOBS`, `API_MAX_SESSIONS`, `API_MAX_SESSION_TURNS`.

## Observability (Langfuse)

The agent traces itself to [Langfuse](https://langfuse.com) — one trace per run,
with a span per node, a generation per model call (model, tokens, cost, latency)
and a span per retrieval. It is **off unless configured**, so nothing here is
needed to run the agent.

1. Start Langfuse (self-hosted, via its own `docker compose up`) and open
   <http://localhost:3000>. Create an account, then an organisation and project.

2. In **Project Settings → API Keys**, create a key pair and put it in `.env`:

   ```ini
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_BASE_URL=http://localhost:3000   # the default; omit if unchanged
   ```

3. Check the connection, then ask something:

   ```bash
   python scripts/agent.py --check-tracing
   python scripts/agent.py "Who authorizes a UPS bypass?"          # prints a trace link
   python scripts/agent.py "..." --session demo --tag baseline      # group and label runs
   ```

Each run is scored automatically, so the dashboard can be grouped and filtered
on them: `grounded` (the verifier's verdict — `ungraded` stays distinct from
`pass`, because an unchecked answer is not a verified one), `answered`, and the
three loop counters `delegations`, `revisions` and `retrievals`.

Other settings, all optional:

| Variable | Default | What it does |
| --- | --- | --- |
| `LANGFUSE_TRACING_ENABLED` | on | Set `false` to keep the keys but stop sending. |
| `LANGFUSE_TRACING_ENVIRONMENT` | `development` | Separates experiments from real runs. |
| `LANGFUSE_TRACE_OPENAI_SDK` | `false` | See below. |

> **Why that last one exists.** LightRAG turns on its own Langfuse integration
> as soon as it sees the two keys, and the `langfuse.openai` import it uses
> patches the OpenAI SDK *globally* — so our own LangChain calls get traced
> twice, once by the callback handler and once by the patch. Measured: one call,
> two generations, the same tokens counted in both. The duplicates are therefore
> dropped by default. Set this to `true` to see LightRAG's internal calls, and
> read cost and token charts knowing they are inflated.

All of this lives in `src/agent/observability.py`; no node knows about it.

## The models

Three of them, all reached through OpenRouter on the one key:

| Variable | Default | What it drives |
| --- | --- | --- |
| `LLM_MODEL` | `openai/gpt-4o-mini` | Every node of the agent, and LightRAG's entity extraction. |
| `EMBEDDING_MODEL` | none that works — set it | The heading index, and the graph's vector half. Nothing else. |
| `RERANK_MODEL` | `cohere/rerank-v3.5` | Re-scores graph candidates against the question. `RERANK=false` skips the step. |

`src/rag/embeddings.py` is the single place text becomes vectors — Chroma's
`OpenAIEmbeddingFunction` pointed at OpenRouter — so `EMBEDDING_MODEL` picks the
model and replacing that one function changes the provider. It has no usable
fallback: `src/rag/config.py` still defaults to a sentence-transformers name
OpenRouter cannot serve, left over from an earlier local setup. `.env.example`
carries a working value; keep one there.

Two things to know before you change either model:

> **Changing the embedding model means re-ingesting.** Vectors from different
> models are not comparable and `data/chroma/` still holds the old ones.
> `data/tree/` is unaffected — it stores no vectors, so the re-index does not
> need another Docling parse.

> **Changing the chat model means deleting LightRAG's cache first.** Its
> response cache in `data/graph_rag/kv_store_llm_response_cache.json` keys
> extractions by prompt, without the model name, so a re-ingest replays the
> *previous* model's work and reports success. Delete the file, then re-ingest.

## Measuring it

`scripts/eval/` scores the current configuration out of 100 against a
purpose-built corpus of five fictional organisations sharing one world, so the
cross-document questions are real rather than decorative. The traps are planted
deliberately: three roles whose titles all end in "Duty Manager" with different
authority, a Battalion Chief who outranks the Fire Marshal on scene but cannot
close an occupancy, one question resting on a false premise, and one the
documents simply do not answer.

```bash
uv run --with reportlab python scripts/eval/make_val_pdfs.py   # build the corpus
uv run python scripts/eval/run_eval.py --ingest                # score the graph
uv run python scripts/eval/run_eval.py --via agent             # score the agent
```

`--via` chooses what is being measured, and the two numbers do not compare:
`graph` asks LightRAG directly, `agent` runs the whole workflow — find, read,
answer, verify, revise. Every store the harness touches is redirected under
`data/val/`, so the fixtures never mix with your corpus. `--no-judge` drops the
LLM judge, which makes a run free and fast enough to use while editing prompts,
at the cost of being generous.

Recorded baseline: **87.9 / 100**, `--via graph`, `google/gemini-2.5-flash`,
2026-07-23 — no trap tripped. There is no recorded `--via agent` run: the mode
is newer than the baseline, and a full pass is many model calls per question and
minutes on a hard one.

Scoring weights, what each trap is for, and how to read a result:
`scripts/eval/README.md`.

## Known rough edges

Written down rather than hidden, because they are the honest state of it:

- `scripts/check_agent.py` has one expected failure — four
  `Hospitalist (Internal Medicine)` chunks in the graph store carry no file
  prefix, so they cannot be cited. An ingest-side bug, not a retrieval one.
- The corpus holds eight organisations and a search routinely returns sections
  from several. The prompts work hard to keep them apart; a question naming an
  organisation the corpus also documents can still burn most of the delegation
  budget on the wrong one.
- Docling's heading detection is not reliable, and the tree inherits whatever it
  decides. Two failure shapes, one fixed and one not:
  - **Fixed.** It sometimes reads a role's `Reports To: …` line as a heading at
    the role's own level, which made the chunker drop the role's title
    entirely — five roles across the validation corpus were addressable only as
    "Reports To: Fire Chief" and the like. `extraction.py` now recovers the
    owner from the document item stream, where both headings still exist in
    order.
  - **Not fixed.** Where a heading is missed altogether, that role's
    sub-sections stay at the top of the tree with no owner rather than being
    attributed to the previous role — uninformative, which is the right
    failure, but `python scripts/query.py --outline` will show the odd bare
    `Key Responsibilities` node because of it.
- The heading index embeds each section's heading path *plus its opening
  words*, not the heading path alone. Half the headings in this corpus are
  structural ("Out of Scope"), so a pure-heading key has nothing for a question
  about a *thing* to match. BM25 over the full text covers the rest.
- The API keeps its jobs and its conversations in the serving process, so a
  restart loses both and a second worker would break the graph write lock. Fine
  for one node; not a deployment.

## Where to go next

- Cut the cost of cross-organisation questions — filter retrieval by document
  once the question names one. The tree makes this cheap: a section id already
  begins with its document.
- Let the responder cite by section *id* rather than by typing the label, so a
  fabricated citation is unrepresentable instead of merely detectable. The ids
  now exist; what is missing is rendering them back into labels at the end.
- Give the researcher the document outline as a tool, so it can navigate a
  document it has already found instead of searching it again.
- Stream `/agent/chat` over SSE, so a client can watch the workflow rather than
  wait on it.
- Expose the graph itself — nodes, edges, and the subgraph behind an answer —
  which is what would make the retrieval visible instead of merely cited.
