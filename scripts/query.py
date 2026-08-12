"""CLI: find the sections that answer a question, and optionally read one.

Usage:
    python scripts/query.py "your question here"
    python scripts/query.py "your question here" --top-k 3
    python scripts/query.py --read sample_role_duties#shift-supervisor
    python scripts/query.py --outline sample_role_duties.pdf

Retrieval only — there is no generation step here, and that is the point: this
is what the agent's researcher sees before any model reads it, which is what you
want when an answer looks wrong and the question is whether retrieval or the
model produced it.

Both halves of the search run on every query: the heading index (embeddings, one
API call) and BM25 over the section text (free). Each hit says which of the two
found it.
"""

import argparse
import sys
from pathlib import Path

# Make the src/ package importable when running as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doctree import corpus, find_sections, outline
from doctree.index import get_heading_collection, search_headings

PREVIEW_CHARS = 400


def _read(node_id: str) -> int:
    store = corpus()
    if store.get(node_id) is None:
        print(f"No section with id: {node_id}")
        return 1
    print(f"\n{store.get(node_id).label}\n{'-' * 60}")
    print(store.read(node_id))
    return 0


def _outline(source: str) -> int:
    store = corpus()
    document = store.documents.get(source) or store.documents.get(f"{source}.pdf")
    if document is None:
        print(f"No document called {source}. Have you ingested it?")
        print("Ingested: " + (", ".join(sorted(store.documents)) or "nothing"))
        return 1
    print(f"\n{document.source}\n{'-' * 60}")
    print(outline(document, max_depth=9))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the document trees.")
    parser.add_argument("query", nargs="?", help="The question to search for.")
    parser.add_argument("--top-k", type=int, default=5, help="How many sections to return.")
    parser.add_argument("--read", metavar="ID", help="Print one section whole, by id.")
    parser.add_argument("--outline", metavar="PDF", help="Print a document's headings.")
    args = parser.parse_args()

    if args.read:
        raise SystemExit(_read(args.read))
    if args.outline:
        raise SystemExit(_outline(args.outline))
    if not args.query:
        parser.error("give a question, or --read an id, or --outline a document")

    # The embedding half is fetched here rather than inside find_sections, which
    # is what keeps the fusion and the keyword half runnable without an API key.
    headings = search_headings(get_heading_collection(), args.query, top_k=args.top_k * 3)
    hits = find_sections(args.query, top_k=args.top_k, headings=[h["id"] for h in headings])

    if not hits:
        print("No results. Have you ingested any PDFs yet? (python scripts/ingest.py)")
        return

    print(f'\nTop {len(hits)} sections for: "{args.query}"\n')
    for rank, hit in enumerate(hits, start=1):
        found = "+".join(hit.found_by)
        print(f"[{rank}] {hit.label}  ({found}, score {hit.score:.4f})")
        print(f"    {hit.id}")
        print("    " + " ".join(hit.node.text.split())[:PREVIEW_CHARS])
        print("-" * 60)


if __name__ == "__main__":
    main()
