"""Extraction: PDF -> Docling StandardPdfPipeline -> per-section text.

Turns a PDF into a list of `Section`s — one citable unit per heading path per
page — which the graph_rag ingest step then inserts into LightRAG. This module
is pure Docling: no LLM, no network, no LightRAG, so it can be tested on its own.
"""

from __future__ import annotations

import re
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


# Separator between the PDF name and its heading path in a citation label, e.g.
# "AMG.pdf › 2. Roles › 2.1 Manager". Purely cosmetic — pick any glyph you like.
LABEL_SEP = " › "


def section_label(pdf_name: str, page: int | None, headings: tuple[str, ...]) -> str:
    """The citation label for one section of a PDF.

    e.g. "sample.pdf › page 25 › Shift Supervisor". The page and heading parts
    are each omitted when unknown, so a heading-less first page is just
    "sample.pdf › page 1", and a page-less chunk is "sample.pdf › Overview".

    Lives here, next to Section, rather than in graph_rag.py, because BOTH
    stores label their sections with it — the graph passes it to LightRAG as a
    file_path, the vector store keeps it in Chroma metadata. Sharing one
    implementation is what makes a citation copied out of either tool read the
    same, and it is deliberate insurance: this module already learned once what
    happens when the two ingest paths are maintained separately.
    """
    parts = [pdf_name]
    if page is not None:
        parts.append(f"page {page}")
    parts.extend(headings)
    return LABEL_SEP.join(parts)


def _normalize_heading(heading: str) -> str:
    """Collapse whitespace so the same heading always yields the same label.

    Citations are deduplicated on the exact label string, so stray spacing or
    line breaks in a heading would otherwise split one section into two
    references. Normalizing here keeps a section's citation stable.
    """
    return " ".join(heading.split())


# Sub-headings that recur under every role in a Role & Duty document. They name a
# KIND of section, never its owner, so on their own they cannot identify a role.
STRUCTURAL_HEADINGS = frozenset(
    {
        "key responsibilities",
        "responsibilities",
        "out of scope",
        "escalation path",
        "reports to",
        "summary",
    }
)


def _is_structural(heading: str) -> bool:
    """True if a heading labels a sub-section rather than naming its owner."""
    return heading.strip().rstrip(":").lower() in STRUCTURAL_HEADINGS


# A reporting line — "Reports To: Site Duty Manager" — which is a FIELD of the
# role above it and never a section of its own.
#
# It needs its own rule because Docling is inconsistent about it within a single
# document: the same line, in the same layout, comes back as body text under
# four roles and as a `section_header` under the fifth. Where it comes back as a
# header it sits at the role's own level, so it replaces the role as the current
# heading and the role's title disappears from the parse entirely. Four roles in
# the validation corpus were reachable only as "Reports To: Fire Chief" and the
# like — a question naming the Fire Marshal, the Facilities Supervisor or the
# Critical Facilities Engineer found nothing, and the agent correctly reported
# that the document gives the role no title.
#
# Matching the shape rather than the exact string, because what follows the
# colon is a different role name every time. STRUCTURAL_HEADINGS already covers
# a bare "Reports To:"; this covers the same label with its value attached.
_REPORTS_TO = re.compile(r"^reports?\s+to\b\s*[:–—-]", re.IGNORECASE)


def _is_reports_to(heading: str) -> bool:
    """True if a heading is a reporting-line field rather than a section name."""
    return bool(_REPORTS_TO.match(heading.strip()))


def _owner_by_item(document) -> dict[str, str]:
    """Which role really owns each item, read from the document item stream.

    The chunker's heading path cannot answer this. When Docling emits a
    reporting line as a `section_header` at the role's own level, the chunker
    treats it as a *replacement* for the role rather than a child of it, so the
    role's title never enters the path at all — demoting the item's label
    afterwards does not help, because the chunker has already decided.

    The item stream still has both, in order, so the owner is recoverable there:
    walk it, and remember the last section header that actually named something.
    Items are keyed by `self_ref`, which is what a chunk's `doc_items` carry
    back, so a chunk can be asked which role it came from regardless of what its
    heading path says.
    """
    owners: dict[str, str] = {}
    current: str | None = None
    for item, _level in document.iterate_items():
        text = (getattr(item, "text", "") or "").strip()
        ref = getattr(item, "self_ref", None)
        if text and str(getattr(item, "label", "")) == "section_header":
            heading = _normalize_heading(text)
            if not _is_reports_to(heading) and not _is_structural(heading):
                current = heading
        if ref and current:
            owners[ref] = current
    return owners


def _repair_path(headings: tuple[str, ...], owner: str | None) -> tuple[str, ...]:
    """Put the real role back where a reporting line displaced it.

    Surgical on purpose: a path with no reporting line in it is returned
    untouched. Rewriting every path from the item stream would look tidier and
    would flatten genuinely nested documents — a path like
    ("2. Roles", "2.1 Manager") would collapse to just the deepest header — so
    only the broken shape is touched.
    """
    if not any(_is_reports_to(heading) for heading in headings):
        return headings

    repaired: list[str] = []
    for heading in headings:
        if not _is_reports_to(heading):
            repaired.append(heading)
        elif owner and owner not in repaired:
            repaired.append(owner)
    return tuple(repaired)


def _chunk_ref(chunk) -> str | None:
    """The `self_ref` of a chunk's first item, or None if it carries no items.

    The link back from a chunk to its position in the document item stream,
    which is how a chunk can be asked which role it sits under when its own
    heading path does not say.
    """
    items = chunk.meta.doc_items or []
    return getattr(items[0], "self_ref", None) if items else None


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
    # Read the true owner of every item before chunking, because the chunker is
    # where the information gets lost. See _owner_by_item.
    owner_of = _owner_by_item(document)
    # Group by (heading path, page) — one citable section per heading per page.
    # dict preserves first-seen order, so sections stay in document order.
    grouped: dict[tuple[tuple[str, ...], int | None], list[str]] = {}
    # The role a bare sub-section belongs to. In these PDFs the role title and its
    # "Out of Scope" sub-heading sit at the SAME heading level, so Docling reports
    # a flat path — ("Out of Scope",) — with the role nowhere in it. Two roles'
    # exclusion lists on one page then share a grouping key and merge into a
    # single chunk that names neither of them, and the ingest-time extractor has
    # to guess the owner. It guesses wrong, writing edges that invert the source
    # ("Approving refunds above $500 is out of scope for the Store Manager" — that
    # is the Shift Supervisor's ceiling; the Store Manager is who approves above
    # it). Chunks yield in document order, so remembering the last heading that
    # actually named something restores the owner for the sub-sections under it.
    owner: str | None = None
    seen_under_owner: set[str] = set()
    dropped_headings = 0
    for chunk in chunker.chunk(document):
        text = chunker.contextualize(chunk=chunk).strip()
        if not text:
            continue
        headings = tuple(
            h for h in (_normalize_heading(x) for x in (chunk.meta.headings or [])) if h
        )
        # Repair a path whose role was displaced by a reporting line, using the
        # item stream rather than the path itself. The reporting line stays in
        # the chunk TEXT either way — it is a real line of the document, and it
        # sits in the body for every role Docling read correctly. What changes
        # is only who the section is filed under.
        headings = _repair_path(headings, owner_of.get(_chunk_ref(chunk)))

        # Deepest heading that names something is the owner of what follows. When
        # a PDF *does* nest properly, the role is already in the path and this
        # just tracks it; nothing is prepended.
        named = [h for h in headings if not _is_structural(h)]
        if named:
            owner = named[-1]
            seen_under_owner = set()
        elif headings:
            key = headings[-1].strip().rstrip(":").lower()
            # A sub-section repeating under one owner means the layout model
            # missed a heading and a new, unnamed role has started. (Docling does
            # exactly this on sample_role_duties.pdf: "Store Manager" never
            # appears in the item stream, so its three sub-sections trail the
            # Shift Supervisor.) Carrying the owner across that boundary would
            # relabel one role's rules as another's — the failure this whole
            # change exists to remove — so drop the attribution instead. The
            # section stays unowned, which is merely uninformative, not false.
            if key in seen_under_owner:
                owner = None
                seen_under_owner = set()
                dropped_headings += 1
            seen_under_owner.add(key)
            if owner:
                headings = (owner, *headings)
                # The label alone is not enough: the extraction and answering
                # models read `text`, not the citation. Put the owner in the text
                # too, so a retrieved exclusion list carries the role it binds.
                if owner.lower() not in text.lower():
                    text = f"{owner}\n{text}"
        page = _chunk_start_page(chunk)
        grouped.setdefault((headings, page), []).append(text)

    if dropped_headings:
        print(
            f"  {pdf_path.name}: {dropped_headings} section(s) left unattributed — "
            f"the PDF has a heading Docling did not detect, so the role that owns "
            f"them is unknown. Their text is still ingested and citable."
        )

    return [
        Section(headings=headings, page=page, text="\n\n".join(parts))
        for (headings, page), parts in grouped.items()
    ]
