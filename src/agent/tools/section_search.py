"""Find a section, then read it: the two halves of chunkless retrieval.

These replace `naive_rag_search`, and the split is the whole idea. That tool did
one thing — return the passages nearest the question — which meant every result
was both the search result AND the evidence, so the amount retrieved and the
amount read were the same number, and tuning `top_k` traded recall against
context with no way to have both.

Here they come apart. `find_section` is cheap and wide: it returns a list of
sections with a line of preview each, enough to choose from and not enough to
answer from. `read_section` is the expensive one, and it returns a whole
section — the author's unit, its sub-sections under it, nothing truncated.

A section is addressed by an id derived from its heading path
(`sample-role-duties#shift-supervisor/out-of-scope`), which is readable, stable
across a re-ingest, and — the part that matters for citations — NOT a citation.
The citation is the label, printed on its own line exactly as every other tool
prints it. The id is how you ask for something; the label is how you refer to it.
"""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field

from doctree import corpus, find_sections
from doctree.index import search_headings

from ._stores import get_heading_collection

# How much of a section shows up in a search result. A choosing aid: enough to
# tell whether this is the section you want, short enough that reading twenty of
# them is not an alternative to reading one properly.
PREVIEW_CHARS = 300

# The heading index contributes more candidates than the caller asked for, so
# that fusion has something to fuse. A section both retrievers rank mid-list
# should be able to beat one only the embeddings liked, and it can only do that
# if it survives to be counted twice.
CANDIDATE_FACTOR = 3


class FindSectionInput(BaseModel):
    """Arguments for find_section."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        min_length=1,
        description="What you are looking for. A question or a literal phrase.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="How many sections to list (1-20).",
    )


class ReadSectionInput(BaseModel):
    """Arguments for read_section."""

    model_config = ConfigDict(extra="forbid")

    section_id: str = Field(
        min_length=1,
        description=(
            "The id of a section, exactly as find_section printed it — e.g. "
            "sample-role-duties#shift-supervisor. Not a citation label."
        ),
    )
    include_subsections: bool = Field(
        default=True,
        description=(
            "Include the section's sub-sections. Leave this true unless you "
            "want one sub-section on its own."
        ),
    )


@tool(args_schema=FindSectionInput)
async def find_section(question: str, top_k: int = 5) -> str:
    """Find the document sections most likely to hold an answer. START HERE.

    Searches two ways at once and merges the results: by meaning, over each
    section's heading and opening lines, and by exact keyword, over its full
    text. Both run on every call, so a literal string — a job title, a
    threshold, an acronym — is matched even when the question is phrased
    loosely, and you never have to ask for the keyword half separately.

    It returns a SHORTLIST, not evidence: an id, a citation label, and a few
    lines of preview per section. Do not answer or cite from the preview. Pick
    the sections that matter and call read_section on each — that is where the
    text you are allowed to rely on comes from.

    Escalate to graph_rag_search only when a question turns on a connection
    between sections that no single section states.
    """
    store = corpus()
    if not len(store):
        return "[no sections indexed — run: python scripts/ingest.py]"

    headings = await _heading_candidates(question, top_k * CANDIDATE_FACTOR)
    hits = find_sections(question, top_k=top_k, headings=headings)
    if not hits:
        return "[no sections matched]"

    blocks = []
    for i, hit in enumerate(hits, 1):
        node = hit.node
        preview = " ".join(node.text.split())[:PREVIEW_CHARS]
        children = ", ".join(node.children)
        # The label goes on its own line and nothing follows it there. A label is
        # harvested from tool output by reading from the file name to the end of
        # the line, so anything trailing it — a score, an id, a word count — is
        # absorbed into the label and the citation stops matching.
        lines = [
            f"[{i}] {node.label}",
            f"    id: {node.id}",
            f"    found by: {'+'.join(hit.found_by)} | {len(node.text.split())} words",
        ]
        if children:
            lines.append(f"    sub-sections: {children}")
        lines.append(f"    preview: {preview}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


@tool(args_schema=ReadSectionInput)
async def read_section(section_id: str, include_subsections: bool = True) -> str:
    """Read one section of a document in full, with its sub-sections.

    The evidence step. Nothing is truncated, summarised or windowed: you get the
    section as the author wrote it, and by default its sub-sections underneath —
    so reading a role gives you its responsibilities AND its exclusions AND its
    escalation path together, which is the combination this corpus punishes you
    for splitting up.

    Take the `section_id` from a find_section result, or from the `sub-sections`
    line of one to go deeper. Cite the label printed at the top of the result,
    not the id.
    """
    store = corpus()
    node = store.get(section_id)
    if node is None:
        near = [
            known
            for known in store.nodes
            if section_id.lower() in known.lower() or known.lower() in section_id.lower()
        ]
        hint = f" Did you mean: {', '.join(near[:5])}" if near else ""
        return f"[no section with id {section_id!r} — use find_section first]{hint}"

    body = store.read(section_id, with_children=include_subsections)
    if not body.strip():
        children = ", ".join(node.children)
        return (
            f"{node.label}\n\n[this heading has no text of its own]"
            + (f"\nSub-sections: {children}" if children else "")
        )
    return f"{node.label}\n\n{body}"


async def _heading_candidates(question: str, limit: int) -> list[str]:
    """Ids from the heading index, or none if it cannot be reached.

    A failure here degrades the search to keyword-only rather than ending the
    run. That is a real trade and worth stating: keyword-only still finds exact
    strings, which is most of what this corpus is asked for, and a researcher
    that gets nothing has no way to report a gap either.
    """
    import logging

    try:
        hits = search_headings(get_heading_collection(), question, top_k=limit)
    except Exception as error:  # noqa: BLE001 - logged, then degraded
        logging.getLogger(__name__).warning(
            "heading search failed (%s); falling back to keyword-only", error
        )
        return []
    return [hit["id"] for hit in hits]
