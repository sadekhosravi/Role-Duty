"""Where the trees live: one JSON file per document under data/tree/.

JSON, on disk, next to the PDFs rather than inside a database, and that is a
choice worth defending. The tree is the *parse* — the expensive, deterministic
part — and everything else in this repo is derived from it: the heading
embeddings, the BM25 index, the graph. Keeping it in a plain file means the
derived stores can be thrown away and rebuilt without re-running Docling, which
takes minutes per corpus and occasionally falls over inside a native library.
It also means a person can open one and read it, which is not nothing when the
question is "why did it cite that".

The corpus is loaded once per process and cached. These files are small and
read on every retrieval; re-reading them per call would be the sort of thing
that looks free until the eval runs 25 questions.
"""

from __future__ import annotations

import json
from pathlib import Path

from rag.config import settings

from .tree import Document, Node

# Bumped when the on-disk shape changes in a way an older file cannot satisfy.
# A tree written by a previous version is refused rather than half-read: a
# missing field would surface as an empty section, which reads exactly like a
# document that genuinely has one.
FORMAT_VERSION = 1


def tree_path(source: str, directory: Path | None = None) -> Path:
    """Where one document's tree is written. `source` is the PDF's file name."""
    return (directory or settings.tree_dir) / f"{Path(source).stem}.json"


def save(document: Document, directory: Path | None = None) -> Path:
    """Write one document's tree, replacing any previous one.

    Written whole rather than merged. A re-parse is authoritative about the
    document it just read, and merging would keep sections that a corrected
    parse no longer produces — the same orphan problem `add_chunks` solves by
    deleting a source's rows before writing.
    """
    path = tree_path(document.source, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": FORMAT_VERSION,
        "source": document.source,
        "root": document.root,
        "nodes": [
            {
                "id": node.id,
                "headings": list(node.headings),
                "page": node.page,
                "text": node.text,
                "parent": node.parent,
                "children": list(node.children),
            }
            for node in document.nodes.values()
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load(path: Path) -> Document:
    """Read one document's tree back."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"{path.name} was written by tree format v{version}, this is "
            f"v{FORMAT_VERSION} — re-run: python scripts/ingest.py"
        )
    source = payload["source"]
    nodes = {
        entry["id"]: Node(
            id=entry["id"],
            source=source,
            headings=tuple(entry["headings"]),
            page=entry["page"],
            text=entry["text"],
            parent=entry["parent"],
            children=list(entry["children"]),
        )
        for entry in payload["nodes"]
    }
    return Document(source=source, root=payload["root"], nodes=nodes)


class Corpus:
    """Every document tree on disk, addressable by node id.

    One flat id -> node map across all documents, which works because an id
    begins with the document's own slug. That is the property `find_section`
    depends on: a hit from the heading index and a hit from BM25 name the same
    thing in the same words, so fusing the two lists is a matter of counting
    ids rather than reconciling two vocabularies.
    """

    def __init__(self, documents: list[Document]):
        self.documents = {document.source: document for document in documents}
        self.nodes: dict[str, Node] = {
            node_id: node
            for document in documents
            for node_id, node in document.nodes.items()
        }

    def __len__(self) -> int:
        return len(self.nodes)

    def document_of(self, node_id: str) -> Document:
        """The document a node belongs to."""
        return self.documents[self.nodes[node_id].source]

    def get(self, node_id: str) -> Node | None:
        return self.nodes.get(node_id)

    def read(self, node_id: str, with_children: bool = True) -> str:
        """One section as prose: its own text, then its sub-sections' in order.

        This is the chunkless read. A section is returned whole — no window, no
        overlap, no truncation at a token count — and its sub-sections come with
        it by default, because "the Shift Supervisor section" means the duties
        AND the exclusions AND the escalation path, and an answer built from any
        one of those alone is the failure mode this corpus is full of.

        Each part is headed by its own title so a model reading the result can
        tell which sub-section a sentence came from, and cite the right one.
        """
        node = self.nodes[node_id]
        if not with_children:
            return node.text

        parts = []
        for descendant in self.document_of(node_id).walk(node_id):
            if not descendant.text:
                continue
            heading = " › ".join(descendant.headings) or descendant.source
            parts.append(f"## {heading}\n{descendant.text}")
        return "\n\n".join(parts)


_corpus: Corpus | None = None


def load_corpus(directory: Path | None = None) -> Corpus:
    """Every tree in `directory`, parsed once and cached for the process.

    An empty directory yields an empty Corpus rather than an error. That is a
    real state — nothing ingested yet — and the tools report it as such, which
    is more use to whoever is running them than a traceback.
    """
    directory = directory or settings.tree_dir
    if not directory.exists():
        return Corpus([])
    return Corpus([load(path) for path in sorted(directory.glob("*.json"))])


def corpus() -> Corpus:
    """The process-wide corpus. Built on first use, like the other stores."""
    global _corpus
    if _corpus is None:
        _corpus = load_corpus()
    return _corpus


def reset() -> None:
    """Drop the cache, so the next read sees what ingest just wrote.

    Needed because a long-lived process — the HTTP server — can ingest and then
    query, and a corpus cached before the ingest would answer from the corpus
    that existed when the process started.
    """
    global _corpus
    _corpus = None
