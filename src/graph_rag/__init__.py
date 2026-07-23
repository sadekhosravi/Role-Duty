"""GraphRAG over PDFs using Docling + LightRAG.

Layout: extraction.py (PDF -> Sections), graph_rag.py (graph ingest/query),
keyword_search.py (BM25). See graph_rag.py for the overview.
"""

from .extraction import Section, parse_pdf
from .graph_rag import ingest, query, remove
from .keyword_search import KeywordHit, keyword_search

__all__ = [
    "Section",
    "parse_pdf",
    "ingest",
    "query",
    "remove",
    "KeywordHit",
    "keyword_search",
]
