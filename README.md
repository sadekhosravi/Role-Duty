# Role-Duty — Simple RAG Pipeline

A minimal Retrieval-Augmented Generation (RAG) ingestion pipeline built on
[Docling](https://github.com/docling-project/docling) (PDF extraction) and
[Chroma](https://www.trychroma.com/) (vector database).

It does four things:

1. **Extract** text from PDFs (Docling).
2. **Embed** the text with a pluggable embedding model.
3. **Store** the vectors in a persistent Chroma collection.
4. **Query** with a prompt and run a similarity search.

This is deliberately small — a foundation to build a more advanced RAG
system on, one step at a time.

## Project structure

```
Role-Duty/
├── data/
│   ├── raw/            # drop your PDFs here
│   ├── chroma/         # persisted vector DB (created automatically)
│   └── tickets/        # filed tickets, as .docx (created automatically)
├── scripts/
│   ├── ingest.py       # PDF -> chunks -> embeddings -> Chroma
│   ├── query.py        # prompt -> similarity search
│   ├── agent.py        # a conversation with the agent
│   └── check_agent.py  # the agent's wiring, checked offline
├── src/rag/
│   ├── config.py       # all settings in one place
│   ├── extractor.py    # Docling: PDF -> text chunks
│   ├── embeddings.py   # the embedding model (swap it here)
│   ├── vector_store.py # Chroma wrapper
│   └── pipeline.py     # ingestion orchestration
├── src/agent/          # the LangGraph workflow — see src/agent/graph.py
└── src/ticket_mcp/     # MCP server: writes a ticket as a Word document
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
cp .env.example .env          # then edit if you want to change defaults
```

## Usage

1. Put one or more PDFs in `data/raw/`.

2. Ingest them:

   ```bash
   python scripts/ingest.py                  # all PDFs in data/raw
   python scripts/ingest.py data/raw/mydoc.pdf
   ```

3. Query them:

   ```bash
   python scripts/query.py "What is the refund policy?"
   python scripts/query.py "What is the refund policy?" --top-k 3
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

## Choosing / changing the embedding model

The embedding model is decoupled on purpose. Set `EMBEDDING_MODEL` in `.env`
to any [sentence-transformers](https://www.sbert.net/docs/pretrained_models.html)
model name. To use a completely different provider (OpenAI, Cohere, a local
model), edit the single function in `src/rag/embeddings.py`.

> Note: after changing the embedding model, re-ingest your PDFs — vectors from
> different models are not comparable.

## Where to go next

Good next steps as you expand this:

- Add an LLM step to generate answers from the retrieved chunks.
- Tune chunking (size / overlap) in `extractor.py`.
- Store richer metadata (page numbers, section headings) for better citations.
- Add a small API or UI on top of `query.py`.
