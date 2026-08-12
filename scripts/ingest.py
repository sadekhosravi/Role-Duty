"""CLI: parse PDFs into section trees and index their headings.

Usage:
    python scripts/ingest.py                 # ingest all PDFs in data/raw
    python scripts/ingest.py path/to/file.pdf
    python scripts/ingest.py path/to/folder

Writes two things per document: the tree itself, as JSON under data/tree/, and
one Chroma row per section keyed on its heading path. The tree is the source of
truth and the index is derived from it, so data/chroma/ can be deleted and
rebuilt from data/tree/ without re-running Docling.

This is not the graph ingest. That one reads every section with an LLM and takes
minutes per document: python src/graph_rag/graph_rag.py ingest
"""

import argparse
import sys
from pathlib import Path

# Make the src/ package importable when running as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doctree import ingest_directory, ingest_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs as section trees.")
    parser.add_argument(
        "path",
        nargs="?",
        help="A PDF file or a folder of PDFs. Defaults to data/raw.",
    )
    args = parser.parse_args()

    if args.path and Path(args.path).is_file():
        total = ingest_pdf(args.path)
        print(f"Done: {Path(args.path).name} -> {total} sections")
    else:
        total = ingest_directory(args.path)
        print(f"Done: {total} sections indexed")


if __name__ == "__main__":
    main()
