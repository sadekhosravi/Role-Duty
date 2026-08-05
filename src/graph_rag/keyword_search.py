"""Keyword search: BM25 over the chunks LightRAG already indexed.

GraphRAG gives us semantic (embedding) + graph retrieval, but not exact-term
matching — the names, codes, and acronyms that vector search can miss. BM25
(via rank_bm25) fills that gap: it ranks chunks purely by keyword overlap with
the query, no embeddings and no LLM. It runs over the very chunks graph_rag
already indexed (LightRAG's persisted text-chunk store), so every hit carries the
same section + page citation label for free. This is the "keyword" half of
hybrid search; fusing it with the graph/semantic answer is a later step.

Standalone by design: it only reads the persisted chunk store, so it doesn't
import graph_rag (and needs no API keys). It mirrors graph_rag's GRAPH_RAG_WORKING_DIR
default so both point at the same store.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

# LightRAG persists every indexed chunk here at ingest time (id -> content + our
# section-label file_path). This is the BM25 corpus. Kept in sync with
# graph_rag.WORKING_DIR via the same env var + default.
WORKING_DIR = Path(os.environ.get("GRAPH_RAG_WORKING_DIR", "data/graph_rag"))
CHUNK_STORE = WORKING_DIR / "kv_store_text_chunks.json"


@dataclass
class KeywordHit:
    """One BM25 keyword-search result: a section and its best-matching snippet."""

    label: str    # the section + page citation label (e.g. "x.pdf › page 3 › Manager")
    score: float  # BM25 relevance score (higher = more keyword overlap)
    snippet: str  # a short excerpt of the best-matching chunk in that section


def _tokenize(text: str) -> list[str]:
    """Lowercased word tokens for BM25 (runs of letters/digits/underscore)."""
    return re.findall(r"\w+", text.lower())


def _load_chunks() -> list[dict[str, str]]:
    """Load LightRAG's indexed chunks as [{content, label}], or [] if none yet.

    Reads the persisted chunk store written at ingest time. Each entry carries
    the chunk `content` and its `file_path` — which is our section + page
    citation label — so BM25 results are citable without any extra bookkeeping.
    """
    if not CHUNK_STORE.exists():
        return []
    data = json.loads(CHUNK_STORE.read_text(encoding="utf-8"))
    chunks: list[dict[str, str]] = []
    for entry in data.values():
        content = (entry.get("content") or "").strip()
        if content:
            chunks.append(
                {"content": content, "label": entry.get("file_path", "unknown_source")}
            )
    return chunks


def indexed_chunk_count() -> int:
    """How many chunks the graph ingest has indexed. 0 when nothing has been.

    The BM25 corpus size, which is also the cheapest honest answer to "is the
    graph store populated" — cheaper than opening LightRAG, which reloads the
    whole graph to tell you the same thing.
    """
    return len(_load_chunks())


def keyword_search(question: str, top_k: int = 5) -> list[KeywordHit]:
    """Rank indexed sections by BM25 keyword overlap with the question.

    Pure lexical search — strong on exact terms (names, IDs, acronyms) that
    semantic search can miss. Chunk scores are aggregated up to their section
    (a section's score is its best chunk's), so results align with the citation
    labels. Returns up to top_k sections, highest score first; sections with no
    keyword overlap (BM25 score 0) are dropped.
    """
    chunks = _load_chunks()
    if not chunks:
        return []

    bm25 = BM25Okapi([_tokenize(c["content"]) for c in chunks])
    scores = bm25.get_scores(_tokenize(question))

    # Keep, per section label, the highest-scoring chunk (score + its text).
    best: dict[str, tuple[float, str]] = {}
    for chunk, score in zip(chunks, scores):
        label = chunk["label"]
        if label not in best or score > best[label][0]:
            best[label] = (float(score), chunk["content"])

    hits = [
        KeywordHit(label=label, score=score, snippet=" ".join(content.split())[:200])
        for label, (score, content) in best.items()
        if score > 0
    ]
    hits.sort(key=lambda hit: hit.score, reverse=True)
    return hits[:top_k]
