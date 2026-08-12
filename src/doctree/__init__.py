"""Chunkless retrieval: keep the document's structure, search it, read a section.

    tree.py     the Node/Document model, built from graph_rag's Docling parse
    store.py    one JSON file per document under data/tree/, and reading a section
    index.py    one Chroma embedding per section, over its heading path
    search.py   heading search fused with BM25, always both
    ingest.py   PDF -> tree -> index

The idea is smaller than the name suggests. A chunker takes prose and cuts it
into windows of roughly N tokens, because that is what fits an embedding and a
context. The cut has nothing to do with the document: it lands mid-sentence,
mid-list, and — the expensive case in this corpus — it separates a role's rules
from the role's name, after which the only thing that can reattach them is a
guess. Everything else in this repo exists to work around that: the label
carrying the heading path, the owner prepended to the text, the verifier
checking every title against retrieved text.

So do not cut. The author already divided the document, into sections with
headings; keep those as the unit, keep the nesting, and address a section by
where it sits rather than by which window it fell into. Retrieval becomes two
steps a person would recognise — find the section, then read it — and what comes
back is a whole section, in order, with its sub-sections under it.

What this does not replace is the graph. Multi-hop questions ("who approves what
the Shift Supervisor cannot?") are answered by connections between sections, and
no amount of returning whole sections finds a connection that no single section
states. LightRAG stays as the escalation tier behind this one.
"""

from .ingest import ingest_directory, ingest_pdf
from .search import Hit, find_sections
from .store import Corpus, corpus, load_corpus
from .tree import Document, Node, build_document, build_from_pdf, outline

__all__ = [
    "Corpus",
    "Document",
    "Hit",
    "Node",
    "build_document",
    "build_from_pdf",
    "corpus",
    "find_sections",
    "ingest_directory",
    "ingest_pdf",
    "load_corpus",
    "outline",
]
