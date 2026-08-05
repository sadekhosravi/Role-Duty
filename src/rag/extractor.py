"""PDF extraction and chunking for the vector store.

Delegates the parsing to `graph_rag.extraction.parse_pdf` — the same function
the graph ingest uses. That is the point of this module: both stores now see
the corpus split the same way, labelled the same way, with the same role
attached to every sub-section.

They did not always. This module used to run a bare HybridChunker and keep only
`chunk.text`, which drops the heading path — so an "Escalation Path" section
arrived in Chroma as a naked sentence with no indication of whose escalation
path it was. Retrieval worked; attribution did not. Asked who a housekeeper
tells about an item needing police notification, the agent retrieved exactly
the right sentence

    Escalate to the Executive Housekeeper for staffing shortfalls ... and to the
    Duty Manager (Front of House) for anything found in a room that requires
    police notification.

and, with no role named anywhere in that chunk, bound it to the only role noun
in the neighbouring chunks — "room attendants" — and reported a reporting line
the document does not contain. The graph store had been fixed for this a week
earlier and the fix never crossed over. Hence the shared call: a heuristic that
has to be maintained in two places is one that will be maintained in one.

The unit stored here is therefore a Section: one heading path, on one page,
carrying its owner's name in both its text and its citation label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from graph_rag.extraction import parse_pdf, section_label


@dataclass
class Chunk:
    """A single citable piece of text extracted from a document."""

    text: str
    source: str  # filename the chunk came from
    index: int  # position of the section within that document
    label: str  # citation string, e.g. "x.pdf › page 2 › Role › Escalation Path"
    page: int | None = None
    headings: tuple[str, ...] = field(default_factory=tuple)


def extract_chunks(pdf_path: Path | str) -> list[Chunk]:
    """Convert a PDF into a list of citable chunks.

    `index` is the section's position in the document rather than the raw
    chunker's, because sections are what get stored. It only has to be stable
    and unique within one PDF — it is half of the Chroma id — and the grouping
    is deterministic, so the same PDF yields the same numbering every time.

    Note that a re-ingest can now produce FEWER units than the old chunking did.
    Ids are `source:index`, so the surplus ids from a previous ingest would
    survive as orphans; `vector_store.add_chunks` clears a document's existing
    chunks before writing, which is what keeps that from happening.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    return [
        Chunk(
            text=section.text,
            source=pdf_path.name,
            index=i,
            label=section_label(pdf_path.name, section.page, section.headings),
            page=section.page,
            headings=section.headings,
        )
        for i, section in enumerate(parse_pdf(pdf_path))
    ]
