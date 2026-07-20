"""CLI: extract PDFs, vectorize them, and store them in Chroma.

Usage:
    python scripts/ingest.py                 # ingest all PDFs in data/raw
    python scripts/ingest.py path/to/file.pdf
    python scripts/ingest.py path/to/folder
"""

import argparse
import sys
from pathlib import Path

# Make the src/ package importable when running as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.pipeline import ingest_directory, ingest_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into the vector store.")
    parser.add_argument(
        "path",
        nargs="?",
        help="A PDF file or a folder of PDFs. Defaults to data/raw.",
    )
    args = parser.parse_args()

    if args.path and Path(args.path).is_file():
        total = ingest_pdf(args.path)
        print(f"Done: {Path(args.path).name} -> {total} chunks")
    else:
        total = ingest_directory(args.path)
        print(f"Done: {total} chunks stored")


if __name__ == "__main__":
    main()
