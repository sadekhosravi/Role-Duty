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
