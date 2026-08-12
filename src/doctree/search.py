"""Finding a section: heading search and keyword search, fused, always both.

Two retrievers over the same units, run together on every call:

  - the heading index (index.py) — embeddings, so it finds a section that means
    what the question means even when it shares no words with it;
  - BM25 over each section's full text — no embeddings and no model calls, so it
    finds the exact strings embeddings blur: a full job title, a threshold, an
    acronym, a room number.

They are fused here rather than offered as two tools, and that is deliberate.
Offered separately, the cheap exact-match half gets skipped — it reads as a
confirmation step, so it runs after a mistake rather than before one, if at all.
It costs nothing: no API call, no model, a few milliseconds over a corpus this
size. Something that cheap and that useful should not be a decision, so it is
not one.

Fusing needs the two halves to name the same things. They do, because both run
over tree nodes and return node ids — which is why BM25 was moved off LightRAG's
chunk store before this function existed. Fused over the old arrangement it
would have merged two different identifier vocabularies into one ranked list,
and the ranking would have been arithmetic on nonsense.

The fusion itself is Reciprocal Rank Fusion: score a document by 1/(k+rank) in
each list and add. RRF uses only the ranks, never the scores, which is the
property that matters here — a cosine distance and a BM25 score are not on the
same scale, share no units, and cannot be normalised into each other without
inventing a relationship between them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .store import Corpus, corpus
from .tree import Node

# RRF's damping constant. 60 is the value from the original paper and the one
# every implementation uses; it is not tuned here because tuning it on eight
# documents would fit the corpus rather than the problem.
RRF_K = 60

# How many candidates each retriever contributes before fusion. Wider than the
# result set on purpose: a section that both retrievers rank 6th deserves to
# beat one that only the vector half ranks 2nd, and it can only do that if it
# survives long enough to be counted twice.
CANDIDATES = 12


@dataclass
class Hit:
    """One found section, and why it was found.

    `found_by` is carried through to the tool output rather than kept as
    diagnostics. A section both retrievers agree on is worth more attention than
    one only the embeddings liked, and saying which is which lets the model
    weigh a keyword-only hit — usually an exact-string match, usually the thing
    it was asked to confirm — differently from a semantic near-miss.
    """

    node: Node
    score: float
    found_by: tuple[str, ...]

    @property
    def id(self) -> str:
        return self.node.id

    @property
    def label(self) -> str:
        return self.node.label


def _tokenize(text: str) -> list[str]:
    """Lowercased word tokens for BM25 (runs of letters/digits/underscore)."""
    return re.findall(r"\w+", text.lower())


class KeywordIndex:
    """BM25 over the sections of the tree, built once per corpus.

    Over section text, not chunk text. A section is a longer document than a
    chunk, which BM25 handles natively — length normalisation is what the b
    parameter is for — and it means a hit is already the unit that gets
    returned, with no aggregating step to map chunks back up to their section.
    That step used to exist, and it is where the identifier vocabularies parted
    company.
    """

    def __init__(self, corpus: Corpus):
        # Only sections with text of their own. A container heading has nothing
        # for BM25 to match and would rank on its title alone.
        self.nodes = [node for node in corpus.nodes.values() if node.text]
        self.bm25 = (
            BM25Okapi([_tokenize(f"{' '.join(n.headings)} {n.text}") for n in self.nodes])
            if self.nodes
            else None
        )

    def search(self, question: str, top_k: int) -> list[str]:
        """Node ids, best keyword overlap first. Zero-scoring sections dropped."""
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(_tokenize(question))
        ranked = sorted(
            ((score, node.id) for score, node in zip(scores, self.nodes) if score > 0),
            key=lambda pair: pair[0],
            reverse=True,
        )
        return [node_id for _, node_id in ranked[:top_k]]


_keyword_index: KeywordIndex | None = None


def keyword_index() -> KeywordIndex:
    """The process-wide BM25 index. Built on first use, like the other stores."""
    global _keyword_index
    if _keyword_index is None:
        _keyword_index = KeywordIndex(corpus())
    return _keyword_index


def reset() -> None:
    """Drop the built index, so the next search sees what ingest just wrote."""
    global _keyword_index
    _keyword_index = None


def fuse(ranked_lists: dict[str, list[str]], k: int = RRF_K) -> list[tuple[str, float, tuple[str, ...]]]:
    """Reciprocal Rank Fusion over several ranked lists of ids.

    Returns (id, score, which lists it came from), best first. A list that came
    back empty contributes nothing and costs nothing — which is the case that
    keeps this honest when one retriever is unavailable, rather than the fused
    ranking silently becoming the other retriever's ranking under a new name.
    """
    scores: dict[str, float] = {}
    sources: dict[str, list[str]] = {}
    for name, ids in ranked_lists.items():
        for rank, node_id in enumerate(ids, start=1):
            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (k + rank)
            sources.setdefault(node_id, []).append(name)
    return sorted(
        ((node_id, score, tuple(sources[node_id])) for node_id, score in scores.items()),
        key=lambda row: row[1],
        reverse=True,
    )


def find_sections(question: str, top_k: int = 5, headings: list[str] | None = None) -> list[Hit]:
    """Find the sections most likely to answer a question.

    `headings` is the heading-index result, passed in rather than fetched: that
    half needs an embedding call and therefore an API key, and keeping the call
    outside lets everything else here — the BM25 half, the fusion, the ordering
    — be exercised offline. Pass None and the search is keyword-only, which is
    degraded but honest, and is what happens when nothing has been embedded yet.
    """
    store = corpus()
    ranked = {"keyword": keyword_index().search(question, CANDIDATES)}
    if headings:
        ranked["headings"] = headings[:CANDIDATES]

    hits = []
    for node_id, score, found_by in fuse(ranked):
        node = store.get(node_id)
        # A row can outlive the section it names if the index and the tree were
        # written at different times. Skipping it beats returning an id that
        # read_section will not resolve.
        if node is None:
            continue
        hits.append(Hit(node=node, score=score, found_by=found_by))
        if len(hits) == top_k:
            break
    return hits
