"""GraphRAG over PDFs: Docling (StandardPdfPipeline) -> LightRAG graph -> answer.

Everything lives in this one file on purpose — parsing, ingestion, and querying —
so it reads top to bottom as a single, learnable example.

The pipeline has three steps, one function each:
  1. parse_pdf() : PDF -> Docling StandardPdfPipeline -> HybridChunker chunks
  2. ingest()    : chunks -> LightRAG (builds an entity/relationship graph)
  3. query()     : a natural-language question -> answer grounded in the graph

Storage is LightRAG's default file-based stack — NanoVectorDB for vectors,
NetworkX for the graph, JSON for key/value — under GRAPH_RAG_WORKING_DIR.
No database server, no Docker.

All LLM + embedding calls go through OpenRouter via an OpenAI-compatible client,
configured entirely from environment variables (.env). See the note on
embeddings below: OpenRouter itself does not serve an /embeddings endpoint.

Usage:
    python src/graph_rag/graph_rag.py ingest data/raw/mydoc.pdf   # one file
    python src/graph_rag/graph_rag.py ingest data/raw             # every PDF in a folder
    python src/graph_rag/graph_rag.py remove AMG.pdf              # drop one source PDF
    python src/graph_rag/graph_rag.py query "What is the refund policy?"
    python src/graph_rag/graph_rag.py query "..." --mode local
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.rerank import cohere_rerank
from lightrag.utils import EmbeddingFunc, setup_logger

load_dotenv()
setup_logger("lightrag", level="INFO")


# --- Configuration (all from environment / .env) -------------------------------

# LLM calls -> OpenRouter's OpenAI-compatible chat completions endpoint.
LLM_API_KEY = os.environ["OPENROUTER_API_KEY"]
LLM_BASE_URL = os.environ.get("OPENROUTER_API_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-4o-mini")

# Embedding calls. NOTE: OpenRouter does NOT expose an /embeddings endpoint, so
# these default to the OpenRouter vars (which will 404 on embeddings) but can be
# pointed at any OpenAI-compatible embeddings server via EMBEDDING_API_* in .env.
# We strip a leading "openai/" so an OpenRouter-style id also works with real OpenAI.
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY") or LLM_API_KEY
EMBEDDING_BASE_URL = os.environ.get("EMBEDDING_API_BASE_URL") or LLM_BASE_URL
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small").removeprefix("openai/")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "1536"))
EMBEDDING_MAX_TOKENS = int(os.environ.get("EMBEDDING_MAX_TOKEN_SIZE", "8192"))

# Where LightRAG persists its graph + NanoVectorDB files (created automatically).
WORKING_DIR = Path(os.environ.get("GRAPH_RAG_WORKING_DIR", "data/graph_rag"))

# Rerank calls. AFTER graph + vector search gathers candidate chunks, a reranker
# re-scores each chunk against the actual question and reorders them, so the most
# relevant text reaches the LLM first (and weak chunks can be dropped). Unlike
# embeddings, OpenRouter DOES serve reranking via a Cohere-compatible /rerank
# endpoint, so we reuse the same OPENROUTER_API_KEY and base host by default.
#   - RERANK=false turns the whole step off (retrieval order is used as-is).
#   - MIN_RERANK_SCORE drops chunks scoring below it (0..1). Start at 0.0 while
#     testing (reorder only, drop nothing); raise it later to prune weak chunks.
RERANK_ENABLED = os.environ.get("RERANK", "true").lower() == "true"
RERANK_MODEL = os.environ.get("RERANK_MODEL", "cohere/rerank-v3.5")
RERANK_API_KEY = os.environ.get("RERANK_API_KEY") or LLM_API_KEY
RERANK_BASE_URL = os.environ.get(
    "RERANK_API_BASE_URL", "https://openrouter.ai/api/v1/rerank"
)
MIN_RERANK_SCORE = float(os.environ.get("MIN_RERANK_SCORE", "0.0"))

# Citations. When on, answers end with a `### References` section naming the
# source PDF(s) each fact came from (see query() for how this works).
CITE_SOURCES = os.environ.get("CITE_SOURCES", "true").lower() == "true"


# --- Extraction guidance: what the LLM should pull out during ingest -----------
#
# On every chunk during `ingest`, LightRAG asks the LLM to extract entities and
# relationships. This text is passed via LightRAG's `entity_types_guidance` knob
# (see build_rag) and tells the model WHAT to look for. Edit it to match your
# document; keep it to plain guidance (no braces `{}`) and let LightRAG handle
# the output format.
EXTRACTION_GUIDANCE = """\
This document is about organizational roles and their duties. From the input text, extract:

- As ENTITIES: the roles, positions, or titles mentioned (e.g. Manager, Auditor,
  Coordinator), the duties or responsibilities they hold, and the teams or
  departments they belong to. Treat each distinct role as its own entity.
- As RELATIONSHIPS: how these roles relate to one another and to their duties —
  who reports to or supervises whom, who is responsible for or performs which
  duty, who collaborates with, delegates to, or depends on whom, and which
  department each role belongs to.

Focus on roles, their duties, and how they connect. Ignore incidental details
that are not about roles or their responsibilities."""


# --- 1. Parse: PDF -> Docling StandardPdfPipeline -> structure-aware chunks -----

def parse_pdf(pdf_path: str | Path) -> list[str]:
    """Extract a PDF into a list of structure-aware text chunks.

    Uses Docling's StandardPdfPipeline (deterministic text extraction plus
    layout + table-detection models — no OCR, no vision model, no GPU), then
    HybridChunker to split the document along its natural structure. Each chunk
    is 'contextualized' so it carries its section headings, which gives the
    graph-extraction step cleaner, self-contained text to work from.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False             # deterministic text, no OCR/VLM
    pipeline_options.do_table_structure = True  # detect tables + layout

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )

    document = converter.convert(pdf_path).document

    chunker = HybridChunker()
    chunks: list[str] = []
    for chunk in chunker.chunk(document):
        text = chunker.contextualize(chunk=chunk).strip()
        if text:
            chunks.append(text)
    return chunks


# --- LightRAG wiring: route LLM + embeddings through OpenRouter -----------------

async def _llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs) -> str:
    """LightRAG's LLM hook -> OpenRouter chat completion (OpenAI-compatible)."""
    return await openai_complete_if_cache(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        **kwargs,
    )


async def _embedding_func(texts: list[str]):
    """LightRAG's embedding hook -> OpenAI-compatible /embeddings endpoint."""
    return await openai_embed(
        texts,
        model=EMBEDDING_MODEL,
        api_key=EMBEDDING_API_KEY,
        base_url=EMBEDDING_BASE_URL,
    )


async def _rerank_model_func(query, documents, top_n=None, **kwargs):
    """LightRAG's rerank hook -> OpenRouter's Cohere-compatible /rerank endpoint.

    LightRAG calls this after retrieval with the candidate chunk texts. The
    built-in cohere_rerank helper POSTs {query, documents, top_n} and returns
    [{"index", "relevance_score"}], which LightRAG uses to reorder and trim the
    chunks (dropping any below MIN_RERANK_SCORE) before building the prompt.
    """
    return await cohere_rerank(
        query=query,
        documents=documents,
        top_n=top_n,
        model=RERANK_MODEL,
        api_key=RERANK_API_KEY,
        base_url=RERANK_BASE_URL,
        **kwargs,
    )


async def build_rag() -> LightRAG:
    """Create + initialize a LightRAG instance backed by default file storage.

    The defaults are NanoVectorDB (vectors), NetworkX (graph) and JSON (KV) —
    all file-based under WORKING_DIR, so there is no server to run.
    """
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    rag = LightRAG(
        working_dir=str(WORKING_DIR),
        llm_model_func=_llm_model_func,
        llm_model_name=LLM_MODEL,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=EMBEDDING_MAX_TOKENS,
            func=_embedding_func,
        ),
        # Rerank retrieved chunks by relevance before answering (see config above).
        # When RERANK is off we pass None, so LightRAG skips the step entirely.
        rerank_model_func=_rerank_model_func if RERANK_ENABLED else None,
        min_rerank_score=MIN_RERANK_SCORE,
        # Tell the extractor what entities/relationships to pull out (see above).
        addon_params={"entity_types_guidance": EXTRACTION_GUIDANCE},
    )
    # Both awaits are required before the instance can be used.
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


# --- 2. Ingest: chunks -> LightRAG graph ---------------------------------------

async def ingest(path: str | Path) -> int:
    """Ingest a single PDF or every PDF in a directory. Returns total chunks added.

    Pass a .pdf file to ingest just that file, or a directory to ingest every
    top-level *.pdf inside it. The graph is built once and all PDFs are inserted
    into it. During insertion LightRAG uses the LLM to extract entities and
    relationships from each chunk and merges them into a persistent graph on disk.
    """
    path = Path(path)
    if path.is_dir():
        pdfs = sorted(path.glob("*.pdf"))
        if not pdfs:
            raise FileNotFoundError(f"No PDF files found in: {path}")
    elif path.is_file():
        pdfs = [path]
    else:
        raise FileNotFoundError(f"Path not found: {path}")

    rag = await build_rag()
    total = 0
    try:
        for pdf in pdfs:
            chunks = parse_pdf(pdf)
            if not chunks:
                print(f"  {pdf.name}: no text extracted, skipping")
                continue
            # Insert the PDF as ONE document, tagged with its file name as both
            # the id and file_path so it has a stable identity for removal.
            # We hand LightRAG the text assembled from our structure-aware
            # HybridChunker chunks; LightRAG runs its full extract-and-merge
            # pipeline over it to build the graph. (Passing the chunks as a list
            # instead makes each a separate same-named document, and LightRAG
            # drops all but the first as duplicates — which leaves the graph
            # almost empty. Custom-chunk insertion extracts but never merges into
            # the graph in this version, so it can't be used here either.)
            full_text = "\n\n".join(chunks)
            await rag.ainsert(full_text, ids=pdf.name, file_paths=pdf.name)
            print(f"  {pdf.name}: {len(chunks)} chunks")
            total += len(chunks)
    finally:
        await rag.finalize_storages()
    return total


# --- 3. Query: question -> answer from the graph -------------------------------

async def query(question: str, mode: str = "hybrid") -> str:
    """Answer a natural-language question using the graph.

    mode is one of LightRAG's retrieval strategies:
      naive  - plain vector search over chunks
      local  - entity-centric (neighborhood of matched entities)
      global - relationship-centric (broad themes)
      hybrid - local + global combined (default)
      mix    - hybrid graph retrieval plus naive vector search

    Citations: LightRAG tags every retrieved chunk with the source PDF (the
    file_paths we set at ingest) and its default answer prompt ends the reply
    with a `### References` section listing those sources. `include_references`
    turns that reference flow on so the answer says which document each fact
    came from. To answer with no citations, set CITE_SOURCES=false.
    """
    rag = await build_rag()
    try:
        return await rag.aquery(
            question,
            param=QueryParam(mode=mode, include_references=CITE_SOURCES),
        )
    finally:
        await rag.finalize_storages()


# --- Remove: drop everything that came from a given PDF ------------------------

async def remove(pdf_name: str) -> str:
    """Remove the document ingested from a given PDF. Returns a status message.

    We ingest each PDF as one document whose id is its file name, so removal is
    a direct delete by that id (a bare name like "AMG.pdf" or a path — only the
    file name is used). LightRAG prunes any entities/relationships that no longer
    have a supporting source, and keeps those still cited by other documents.
    """
    doc_id = Path(pdf_name).name
    rag = await build_rag()
    try:
        result = await rag.adelete_by_doc_id(doc_id)
    finally:
        await rag.finalize_storages()
    return f"{result.status}: {result.message}"


# --- CLI -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GraphRAG over PDFs (Docling + LightRAG).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Add a PDF (or a folder of PDFs) to the graph.")
    p_ingest.add_argument("path", help="A PDF file, or a directory containing PDFs.")

    p_remove = sub.add_parser("remove", help="Remove everything ingested from a PDF.")
    p_remove.add_argument("name", help="The PDF file name to remove, e.g. AMG.pdf")

    p_query = sub.add_parser("query", help="Ask a question against the graph.")
    p_query.add_argument("question", help="Your natural-language question.")
    p_query.add_argument(
        "--mode",
        default="hybrid",
        choices=["naive", "local", "global", "hybrid", "mix"],
        help="Retrieval strategy (default: hybrid).",
    )

    args = parser.parse_args()

    if args.command == "ingest":
        total = asyncio.run(ingest(args.path))
        print(f"Done: {total} chunks added to the graph.")
    elif args.command == "remove":
        status = asyncio.run(remove(args.name))
        print(f"{Path(args.name).name}: {status}")
    elif args.command == "query":
        answer = asyncio.run(query(args.question, mode=args.mode))
        print(answer)


if __name__ == "__main__":
    main()
