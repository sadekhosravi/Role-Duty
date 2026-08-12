"""The heading index: one embedding per section, over what the section IS.

The naive store embeds passages, so a query is matched against prose and the
winner is whichever 500 tokens happened to phrase things most like the question.
This indexes the *navigation layer* instead — one row per tree node — so a hit
names a section, and the section is then read whole from the tree.

What gets embedded is the section's heading path together with its opening
words, and the second half of that is a considered compromise rather than an
oversight. A pure heading-path embedding is the cleaner idea and it fails on
this corpus in a specific way: half the headings here are structural ("Out of
Scope", "Escalation Path"), so the only content in the key is the role name,
and a question phrased about a *thing* — a refund limit, a chlorine spill —
has nothing to match. The opening words give it something, without turning the
row back into a chunk: the row is not what gets returned, it is only how the
section is found, and what comes back is always the whole section from
store.py. BM25 over the full text covers what the opening words miss.

Rows are ids, headings and a label — never the body. Chroma is an index here,
not a second copy of the corpus, and the tree on disk stays the one source of
truth.
"""

from __future__ import annotations

from rag.config import settings
from rag.vector_store import get_collection

from .tree import Document, Node

# How much of a section's own text joins its heading path in the embedded key.
# Enough to carry what the section is about, short enough that the key stays a
# description of the section rather than a sample of it.
PREVIEW_CHARS = 400


def get_heading_collection():
    """Open (or create) the persistent collection of section headings."""
    return get_collection(settings.heading_collection_name)


def index_key(node: Node) -> str:
    """The text embedded for one section: what it is, then how it opens."""
    path = " › ".join(node.headings) or node.source
    preview = " ".join(node.text.split())[:PREVIEW_CHARS]
    return f"{path}\n{preview}".strip()


def _indexable(document: Document) -> list[Node]:
    """The nodes worth a row.

    Container headings with no text of their own are skipped: retrieving one
    would return a section that says nothing, and its children are indexed in
    their own right anyway. The container is still addressable by id — a caller
    that wants the whole role reads the parent — it just is not something a
    search can land on by accident.
    """
    return [node for node in document.walk() if node.text]


def index_document(collection, document: Document) -> int:
    """Write one document's sections to the heading index. Returns the count.

    The document's existing rows go first. Section ids come from heading paths,
    so a corrected parse — one where Docling finally detects a heading it was
    missing — produces a different set of ids, and the old ones would otherwise
    survive as rows pointing at sections the tree no longer has. A search could
    then return an id that `read_section` cannot resolve, which is the worst
    shape of failure available here: a citation to nowhere.
    """
    collection.delete(where={"source": document.source})
    nodes = _indexable(document)
    if not nodes:
        return 0

    collection.add(
        ids=[node.id for node in nodes],
        documents=[index_key(node) for node in nodes],
        metadatas=[
            {
                "source": node.source,
                "label": node.label,
                "headings": " › ".join(node.headings),
                "words": len(node.text.split()),
            }
            for node in nodes
        ],
    )
    return len(nodes)


def search_headings(collection, question: str, top_k: int = 8) -> list[dict]:
    """Rank sections by how well their heading key matches the question.

    Returns dicts rather than Nodes: this module knows about Chroma and ids, and
    resolving an id to a section is the corpus's job. Keeping the two apart is
    what lets the fusion in search.py treat this and BM25 as the same kind of
    thing — two ranked lists of ids.
    """
    if collection.count() == 0:
        return []
    result = collection.query(query_texts=[question], n_results=top_k)
    return [
        {"id": node_id, "label": (meta or {}).get("label", node_id), "distance": distance}
        for node_id, meta, distance in zip(
            result["ids"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


def delete_source(collection, source: str) -> int:
    """Drop one document's rows. Returns how many went."""
    rows = collection.get(where={"source": source})
    removed = len(rows["ids"])
    if removed:
        collection.delete(where={"source": source})
    return removed
