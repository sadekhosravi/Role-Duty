"""CLI: run a similarity search against the ingested documents.

Usage:
    python scripts/query.py "your question here"
    python scripts/query.py "your question here" --top-k 3
"""

import argparse
import sys
from pathlib import Path

# Make the src/ package importable when running as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.vector_store import get_collection, similarity_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the vector store.")
    parser.add_argument("query", help="The prompt / question to search for.")
    parser.add_argument(
        "--top-k", type=int, default=5, help="How many results to return."
    )
    args = parser.parse_args()

    collection = get_collection()
    results = similarity_search(collection, args.query, n_results=args.top_k)

    if not results:
        print("No results. Have you ingested any PDFs yet?")
        return

    print(f'\nTop {len(results)} results for: "{args.query}"\n')
    for rank, item in enumerate(results, start=1):
        meta = item["metadata"]
        print(f"[{rank}] {meta['source']} (chunk {meta['index']}) "
              f"- distance {item['distance']:.4f}")
        print(item["text"])
        print("-" * 60)


if __name__ == "__main__":
    main()
