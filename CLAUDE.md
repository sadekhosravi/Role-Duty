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
```

There is currently **no test suite, linter, or formatter** configured.

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

### Known drift to be aware of

The README and `requirements.txt` describe a self-contained sentence-transformers setup, but `embeddings.py` and `pyproject.toml` (`openai`, `docling`, `chromadb`) reflect the current OpenRouter-based reality. When touching embeddings or dependencies, treat the code as the source of truth and reconcile the docs.

## Data & secrets

- `data/raw/` holds input PDFs (gitignored); `data/chroma/` is the persisted vector DB, created automatically (gitignored).
- `.env` holds real credentials and is gitignored. No `.env.example` currently exists despite the README referencing one.
