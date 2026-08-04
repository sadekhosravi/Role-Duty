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
│   └── chroma/         # persisted vector DB (created automatically)
├── scripts/
│   ├── ingest.py       # PDF -> chunks -> embeddings -> Chroma
│   └── query.py        # prompt -> similarity search
└── src/rag/
    ├── config.py       # all settings in one place
    ├── extractor.py    # Docling: PDF -> text chunks
    ├── embeddings.py   # the embedding model (swap it here)
    ├── vector_store.py # Chroma wrapper
    └── pipeline.py     # ingestion orchestration
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
