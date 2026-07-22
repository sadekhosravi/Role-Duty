"""GraphRAG over PDFs: Docling (StandardPdfPipeline) -> LightRAG graph -> answer.

Everything lives in this one file on purpose — parsing, ingestion, and querying —
so it reads top to bottom as a single, learnable example.

The pipeline has three steps, one function each:
  1. parse_pdf() : PDF -> Docling StandardPdfPipeline -> Sections (per heading)
  2. ingest()    : Sections -> LightRAG (each a citable doc in one shared graph)
  3. query()     : a natural-language question -> answer grounded in the graph,
                   with a `### References` section citing the exact PDF sections

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
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

from lightrag import LightRAG, QueryParam
from lightrag.base import DocStatus
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
# source PDF(s) each fact came from (see query() for how this works). With
# section referencing (see ingest), each reference is a PDF section, e.g.
# "AMG.pdf › 2. Roles › 2.1 Manager".
CITE_SOURCES = os.environ.get("CITE_SOURCES", "true").lower() == "true"

# --- Optional: lift the "max 5 citations" cap in the answer prompt -------------
#
# LightRAG's default answer prompt instructs the model to list AT MOST 5
# citations. That limit lives only in the prompt text (not in code), and the
# reference list handed to the model is already uncapped — so raising it is a
# one-line patch to the template, applied once here at import (before any query).
# The model still lists only sections that actually supported the answer; you're
# lifting the ceiling, not forcing a count. Uncomment to enable. The assert
# guards against a silent no-op if a LightRAG upgrade changes the wording.
#
# from lightrag.prompt import PROMPTS
# _CITATION_CAP = "Provide maximum of 5 most relevant citations."
# assert _CITATION_CAP in PROMPTS["rag_response"], (
#     "citation-cap text not found — LightRAG may have changed the prompt wording"
# )
# PROMPTS["rag_response"] = PROMPTS["rag_response"].replace(
#     _CITATION_CAP, "Provide all relevant citations."
# )


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


# --- 1. Parse: PDF -> Docling StandardPdfPipeline -> per-section text -----------

@dataclass
class Section:
    """One citable unit of a PDF: text under a single heading path, on one page.

    `headings` is the Docling heading hierarchy for this section, outermost
    first, e.g. ("2. Roles", "2.1 Manager"). `page` is the (1-based) PDF page the
    text sits on. `text` is every chunk under that heading+page, joined. At
    ingest time (heading + page) becomes the citation label, so answers can cite
    "sample.pdf › page 25 › Shift Supervisor" rather than just the whole PDF.
    Grouping by page means a heading that spans pages yields one section per page.
    """

    headings: tuple[str, ...]
    page: int | None
    text: str


def _normalize_heading(heading: str) -> str:
    """Collapse whitespace so the same heading always yields the same label.

    Citations are deduplicated on the exact label string, so stray spacing or
    line breaks in a heading would otherwise split one section into two
    references. Normalizing here keeps a section's citation stable.
    """
    return " ".join(heading.split())


def _chunk_start_page(chunk) -> int | None:
    """The first PDF page a chunk's content appears on, or None if unknown.

    Docling records provenance for each chunk in `meta.doc_items[].prov[]`, and
    each provenance entry carries a 1-based `page_no`. A chunk can straddle a
    page break, so we take the smallest page number as its starting page.
    """
    pages = [
        prov.page_no
        for item in (chunk.meta.doc_items or [])
        for prov in (item.prov or [])
    ]
    return min(pages) if pages else None


def parse_pdf(pdf_path: str | Path) -> list[Section]:
    """Extract a PDF into a list of Sections, one per heading path.

    Uses Docling's StandardPdfPipeline (deterministic text extraction plus
    layout + table-detection models — no OCR, no vision model, no GPU), then
    HybridChunker to split the document along its natural structure. Chunks are
    'contextualized' (each carries its section headings) and then grouped by
    that heading path: every chunk sharing the same headings becomes one
    Section. HybridChunker yields chunks in document order, so sections keep
    their original order too. Chunks with no heading fall under a single
    empty-path section for that PDF.
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
    # Group by (heading path, page) — one citable section per heading per page.
    # dict preserves first-seen order, so sections stay in document order.
    grouped: dict[tuple[tuple[str, ...], int | None], list[str]] = {}
    for chunk in chunker.chunk(document):
        text = chunker.contextualize(chunk=chunk).strip()
        if not text:
            continue
        headings = tuple(
            h for h in (_normalize_heading(x) for x in (chunk.meta.headings or [])) if h
        )
        page = _chunk_start_page(chunk)
        grouped.setdefault((headings, page), []).append(text)

    return [
        Section(headings=headings, page=page, text="\n\n".join(parts))
        for (headings, page), parts in grouped.items()
    ]


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


# --- 2. Ingest: sections -> LightRAG graph -------------------------------------

# Separator between the PDF name and its heading path in a citation label, e.g.
# "AMG.pdf › 2. Roles › 2.1 Manager". Purely cosmetic — pick any glyph you like.
LABEL_SEP = " › "


def _section_label(pdf_name: str, page: int | None, headings: tuple[str, ...]) -> str:
    """The citation label (LightRAG file_path) for one section of a PDF.

    e.g. "sample.pdf › page 25 › Shift Supervisor". The page and heading parts
    are each omitted when unknown, so a heading-less first page is just
    "sample.pdf › page 1", and a page-less chunk is "sample.pdf › Overview".
    """
    parts = [pdf_name]
    if page is not None:
        parts.append(f"page {page}")
    parts.extend(headings)
    return LABEL_SEP.join(parts)


async def _delete_pdf(rag: LightRAG, pdf_name: str) -> int:
    """Delete every section-document ingested from a PDF. Returns the count.

    Each PDF is stored as many section-documents with ids "{pdf}#0", "{pdf}#1",
    … (see ingest), so removing a PDF means deleting all ids with that prefix.
    We look them up from LightRAG's doc-status store across every status (a
    half-processed doc still needs cleaning) and delete each one. LightRAG then
    prunes entities/relationships that lose their last supporting source and
    keeps those still cited by other sections or documents.
    """
    prefix = f"{pdf_name}#"
    doc_ids: set[str] = set()
    for status in DocStatus:
        docs = await rag.get_docs_by_status(status)
        doc_ids.update(doc_id for doc_id in docs if doc_id.startswith(prefix))
    for doc_id in doc_ids:
        await rag.adelete_by_doc_id(doc_id)
    return len(doc_ids)


async def ingest(path: str | Path) -> int:
    """Ingest a single PDF or every PDF in a directory. Returns total sections added.

    Pass a .pdf file to ingest just that file, or a directory to ingest every
    top-level *.pdf inside it. Each PDF is split into Sections (one per heading
    path) and every section is inserted as its own LightRAG document, tagged with
    a section-qualified file_path so answers can cite the exact section. During
    insertion LightRAG uses the LLM to extract entities and relationships from
    each section and merges them into one persistent graph on disk — the graph
    stays global across sections, so cross-section reasoning is preserved.
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
            sections = parse_pdf(pdf)
            if not sections:
                print(f"  {pdf.name}: no text extracted, skipping")
                continue
            # Re-ingesting? Drop this PDF's previous sections first. Section ids
            # are positional, so if the PDF changed and produced fewer sections,
            # old trailing ids would otherwise linger. A clean delete-then-insert
            # keeps the overwrite semantics the single-doc version had.
            await _delete_pdf(rag, pdf.name)
            # Insert each section as its own document: unique positional id for
            # removal, and a section-qualified file_path that becomes the
            # citation label. Passing parallel lists to ainsert inserts them as
            # distinct documents (distinct ids => no duplicate-drop), and
            # LightRAG runs its full extract-and-merge pipeline over each.
            texts = [section.text for section in sections]
            ids = [f"{pdf.name}#{i}" for i in range(len(sections))]
            file_paths = [_section_label(pdf.name, s.page, s.headings) for s in sections]
            await rag.ainsert(texts, ids=ids, file_paths=file_paths)
            print(f"  {pdf.name}: {len(sections)} sections")
            total += len(sections)
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
    """Remove every section ingested from a given PDF. Returns a status message.

    Each PDF is ingested as many section-documents (see ingest), so removal
    deletes all of them by their shared "{pdf}#" id prefix (a bare name like
    "AMG.pdf" or a path — only the file name is used).
    """
    doc_name = Path(pdf_name).name
    rag = await build_rag()
    try:
        count = await _delete_pdf(rag, doc_name)
    finally:
        await rag.finalize_storages()
    if count == 0:
        return f"no sections found for {doc_name}"
    return f"removed {count} section(s) from {doc_name}"


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
        print(f"Done: {total} sections added to the graph.")
    elif args.command == "remove":
        status = asyncio.run(remove(args.name))
        print(f"{Path(args.name).name}: {status}")
    elif args.command == "query":
        answer = asyncio.run(query(args.question, mode=args.mode))
        print(answer)


if __name__ == "__main__":
    main()
