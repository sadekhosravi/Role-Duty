"""Turning a path a client sent into one the server is allowed to touch.

Two endpoints take a filesystem path from the outside — ingest takes what to
read, upload takes what to write — and both are the kind of parameter that ends
up reading `../../.env` if nobody says otherwise. Every path from a request goes
through this module, and every one of them is resolved and then checked to be
inside the project root before anything opens it.

`resolve()` before the check, not after: the containment test is on the real
path, so `data/raw/../../..` and a symlink out of the tree are both caught by
the same comparison rather than by a list of patterns to reject.
"""

from __future__ import annotations

from pathlib import Path

from rag.config import ROOT_DIR


class PathRejected(ValueError):
    """A path that resolved outside the project, or a name that is not one."""


def under_root(raw: str, *, must_exist: bool = True) -> Path:
    """Resolve a client-supplied path, relative to the project root.

    Absolute paths are accepted and then held to the same rule, so a caller can
    paste the path a CLI printed without it silently meaning something else.
    """
    path = Path(raw)
    path = path if path.is_absolute() else ROOT_DIR / path
    path = path.resolve()

    if path != ROOT_DIR and ROOT_DIR not in path.parents:
        raise PathRejected(f"path is outside the project: {raw}")
    if must_exist and not path.exists():
        raise FileNotFoundError(f"path not found: {raw}")
    return path


def plain_name(name: str, *, suffix: str) -> str:
    """Validate a bare file name — no directories, no traversal, right suffix.

    Used where the client names a file rather than locating one: uploading and
    deleting a document, downloading a ticket. `Path(name).name` would quietly
    turn "../x.pdf" into "x.pdf" and act on a different file than the one
    asked for, so the difference is rejected instead of normalized away.
    """
    if not name or name != Path(name).name or name in {".", ".."}:
        raise PathRejected(f"not a plain file name: {name!r}")
    if not name.lower().endswith(suffix.lower()):
        raise PathRejected(f"not a {suffix} file: {name!r}")
    return name


def pdfs_at(path: Path) -> list[Path]:
    """The PDFs a path names: itself if a file, its top-level *.pdf if a folder.

    Both ingests accept either, and both want the list rather than the path so
    the job they run can report per-document counts instead of one total. The
    same rule the CLIs apply — top level only, sorted for a stable order.
    """
    if path.is_dir():
        pdfs = sorted(path.glob("*.pdf"))
        if not pdfs:
            raise FileNotFoundError(f"no PDFs in: {relative(path)}")
        return pdfs
    if path.suffix.lower() != ".pdf":
        raise PathRejected(f"not a PDF or a folder: {relative(path)}")
    return [path]


def relative(path: Path) -> str:
    """A path as the project writes it — "data/raw/x.pdf" — for a response.

    Absolute paths leak the server's directory layout into every response and
    read differently on every machine. Anything genuinely outside the project
    falls back to the absolute form, which by then can only be a configured
    location like TICKETS_DIR.
    """
    try:
        return path.resolve().relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return str(path)
