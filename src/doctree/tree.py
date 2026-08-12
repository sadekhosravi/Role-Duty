"""The document tree: what a PDF looks like when nothing is chunked.

`graph_rag.extraction.parse_pdf` already recovers a document's structure — it
reads the heading path of every piece of text Docling emits — and then throws
the structure away, keeping the path only as a string to print in a citation.
Everything downstream sees a flat list of sections and has to find the right one
by embedding similarity over its body text.

This module keeps the structure instead. The same sections come in; a tree of
`Node`s comes out, each one addressable by an id derived from its heading path,
each one knowing its parent and its children. Retrieval can then do what a person
does with a document: look at the headings, pick the section, read it whole,
follow it down into its sub-sections. Nothing is split at a token boundary,
because nothing is split at all — the unit is the section the author wrote.

Two things are deliberately NOT here. There is no embedding, no store and no
search: those are index.py, store.py and search.py, and keeping them out means
this file can be exercised on a PDF with no API key and no database. And there is
no second parser — `parse_pdf` stays the one place a PDF becomes text, for the
same reason src/rag/extractor.py stopped having its own. Two parsers is how the
two ingests drifted the first time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from graph_rag.extraction import Section, parse_pdf, section_label

# Separates a document's id from the heading path within it, and the path's parts
# from each other. Both are characters a slug can never contain, so an id splits
# back into its parts unambiguously.
DOC_SEP = "#"
PATH_SEP = "/"

_UNSLUGGABLE = re.compile(r"[^a-z0-9]+")


def slug(text: str) -> str:
    """A heading reduced to something that can live in an id.

    Lowercased, with every run of non-alphanumerics collapsed to a hyphen. It
    only has to be stable and readable: stable so an id survives a re-ingest of
    an unchanged document, readable because these ids are what the agent asks
    for by name and what a citation will carry, and an opaque hash would make
    both of those worse.

    Lossy on purpose, and collisions are handled where ids are assigned rather
    than by making this injective — a slug nobody can read would defeat the
    point.
    """
    return _UNSLUGGABLE.sub("-", text.lower()).strip("-") or "section"


@dataclass
class Node:
    """One section of one document: its own text, and where it sits.

    `text` is this section's text and nothing else — a parent does not contain
    its children's prose. That separation is what lets a caller ask for a
    heading on its own (cheap, and often enough) or for the whole subtree
    (complete, and what "read the section" means to a person). `read` in
    store.py assembles the second from the first.

    `page` is the page the section starts on, kept for the citation label and
    for nothing else. A container heading that has no text of its own inherits
    the page of its first descendant that does, so its label still points
    somewhere real.
    """

    id: str
    source: str
    headings: tuple[str, ...]
    page: int | None = None
    text: str = ""
    parent: str | None = None
    children: list[str] = field(default_factory=list)

    @property
    def title(self) -> str:
        """The deepest heading — what this section is called, on its own."""
        return self.headings[-1] if self.headings else self.source

    @property
    def label(self) -> str:
        """The citation label, built by the function both other stores use.

        Shared rather than reimplemented: a section cited out of this tree and
        the same section cited out of the graph have to read identically, or the
        verifier's check against harvested labels turns into a check on which
        tool happened to find it.
        """
        return section_label(self.source, self.page, self.headings)

    @property
    def depth(self) -> int:
        """How far below the document root this sits. The root itself is 0."""
        return len(self.headings)


@dataclass
class Document:
    """One PDF as a tree: every node, plus which one is the root.

    Nodes are held flat and refer to each other by id rather than nested by
    reference. Flat is what a store wants (one row per node, addressable), what
    JSON wants (no cycles to encode), and what a lookup by id wants; the tree
    shape is recovered from `parent`/`children` whenever it is needed, which is
    less often than you would think.
    """

    source: str
    root: str
    nodes: dict[str, Node]

    def __len__(self) -> int:
        return len(self.nodes)

    def child_nodes(self, node_id: str) -> list[Node]:
        """The children of one node, in document order."""
        return [self.nodes[child] for child in self.nodes[node_id].children]

    def walk(self, node_id: str | None = None):
        """Every node from `node_id` down, depth-first, in document order."""
        current = self.nodes[node_id or self.root]
        yield current
        for child in current.children:
            yield from self.walk(child)


def _unique_id(base: str, taken: set[str]) -> str:
    """`base`, or the first numbered variant of it that is free.

    Slugs collide: "Out of Scope" and "Out of Scope:" reduce to the same string,
    and a document can carry both under one owner. Numbering the second is
    deterministic because sections arrive in document order, so a re-ingest of
    an unchanged PDF assigns exactly the same ids.
    """
    if base not in taken:
        return base
    n = 2
    while f"{base}-{n}" in taken:
        n += 1
    return f"{base}-{n}"


def build_document(source: str, sections: list[Section]) -> Document:
    """Assemble a tree from the flat sections of one PDF.

    A section's heading path IS its position in the tree, so the tree is built
    by walking each path from the root and creating whatever is not there yet.
    Headings that Docling reported as a parent but that carry no text of their
    own — the common case for a role title whose prose all lives under "Key
    Responsibilities" — become container nodes: real, addressable, empty.

    Two sections can share a heading path when one section spans a page break,
    since `parse_pdf` groups by (path, page). Here they merge back into the one
    section the author wrote, which is the whole point of this module, and the
    node keeps the earlier page for its label.
    """
    root_id = slug(Path(source).stem)
    root = Node(id=root_id, source=source, headings=())
    nodes: dict[str, Node] = {root_id: root}
    # Path -> id, so the second section under a heading finds the node the first
    # one created instead of making a rival with the same name.
    by_path: dict[tuple[str, ...], str] = {(): root_id}

    for section in sections:
        parent_id = root_id
        for depth in range(1, len(section.headings) + 1):
            path = tuple(section.headings[:depth])
            node_id = by_path.get(path)
            if node_id is None:
                node_id = _unique_id(
                    f"{root_id}{DOC_SEP}{PATH_SEP.join(slug(h) for h in path)}",
                    set(nodes),
                )
                nodes[node_id] = Node(
                    id=node_id, source=source, headings=path, parent=parent_id
                )
                nodes[parent_id].children.append(node_id)
                by_path[path] = node_id
            parent_id = node_id

        node = nodes[parent_id]
        node.text = f"{node.text}\n\n{section.text}".strip() if node.text else section.text
        if node.page is None or (section.page is not None and section.page < node.page):
            node.page = section.page

    _inherit_pages(nodes, root_id)
    return Document(source=source, root=root_id, nodes=nodes)


def _inherit_pages(nodes: dict[str, Node], node_id: str) -> int | None:
    """Give every page-less container the page its first real descendant is on.

    A container has no text and therefore no provenance of its own, so its label
    would otherwise be "file.pdf › Shift Supervisor" — correct, but pointing at
    a document rather than a place in one. Depth-first, so a container two
    levels up still ends up with the earliest page beneath it.
    """
    pages = [_inherit_pages(nodes, child) for child in nodes[node_id].children]
    known = [page for page in (nodes[node_id].page, *pages) if page is not None]
    if known:
        nodes[node_id].page = min(known)
    return nodes[node_id].page


def build_from_pdf(pdf_path: str | Path) -> Document:
    """Parse a PDF and build its tree. The only entry point that touches Docling."""
    pdf_path = Path(pdf_path)
    return build_document(pdf_path.name, parse_pdf(pdf_path))


def outline(document: Document, node_id: str | None = None, max_depth: int = 3) -> str:
    """The document's headings as an indented, addressable list.

    Written for a model to read and act on, which is why every line carries the
    id: an outline that shows what exists but not how to ask for it forces a
    second search to get back to something already found. Bodies are not
    included at any depth — the outline is the map, `read` is the territory, and
    conflating them is how a "cheap navigation step" becomes the expensive one.
    """
    lines = []
    root = node_id or document.root
    for node in document.walk(root):
        relative = node.depth - document.nodes[root].depth
        if relative > max_depth:
            continue
        size = f" ({len(node.text.split())} words)" if node.text else ""
        lines.append(f"{'  ' * relative}- {node.title}{size}  [{node.id}]")
    return "\n".join(lines)
