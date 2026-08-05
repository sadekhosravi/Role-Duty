"""What the server reads from the environment.

Deliberately separate from `rag.config.Settings`, which configures the
pipelines. This is about the process serving them — where it listens, who may
call it, how much it keeps in memory — and none of it changes an answer.

A frozen dataclass read from `os.getenv`, to match the rest of the project
rather than introducing a second configuration style for one module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, "") or default)
    except ValueError:
        return default


def _origins() -> list[str]:
    """CORS origins, comma-separated. Empty means the middleware is not added.

    Not defaulted to "*": Swagger is served by this app and needs no CORS at
    all, so the only thing a wildcard default would enable is a page on another
    origin driving an API that can write files. A browser front end is a
    deliberate act — name its origin.
    """
    raw = os.getenv("API_CORS_ORIGINS", "")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


@dataclass(frozen=True)
class ApiSettings:
    host: str = os.getenv("API_HOST", "127.0.0.1")
    port: int = _int("API_PORT", 8000)
    cors_origins: list[str] = field(default_factory=_origins)

    # How many finished jobs and idle conversations to keep. Both stores live in
    # this process's memory, so both need a ceiling: an API that remembers every
    # session forever is a memory leak with a REST interface.
    max_jobs: int = _int("API_MAX_JOBS", 100)
    max_sessions: int = _int("API_MAX_SESSIONS", 100)

    # Turns kept per conversation. The history is replayed into every run, so
    # this is a token budget as much as a memory one; the ticket flow needs only
    # the previous turn or two to make "yes, file it" mean something.
    max_session_turns: int = _int("API_MAX_SESSION_TURNS", 20)


settings = ApiSettings()
