"""Extraction: PDF -> Docling StandardPdfPipeline -> per-section text.

Turns a PDF into a list of `Section`s — one citable unit per heading path per
page — which the graph_rag ingest step then inserts into LightRAG. This module
is pure Docling: no LLM, no network, no LightRAG, so it can be tested on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline


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
