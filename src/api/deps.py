"""The per-app objects a router asks for, as FastAPI dependencies.

The job registry and the session store hang off `app.state` rather than being
module globals, so two apps in one process — a test app beside the real one —
do not share a job list or each other's conversations. These two aliases are how
a router reaches them without knowing that, and without importing `main` (which
imports the routers, so it could not import back).

Read as `jobs: JobsDep` in a signature. Nothing else in this file has to happen.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from .jobs import JobRegistry
from .sessions import SessionStore


def job_registry(request: Request) -> JobRegistry:
    return request.app.state.jobs


def session_store(request: Request) -> SessionStore:
    return request.app.state.sessions


JobsDep = Annotated[JobRegistry, Depends(job_registry)]
SessionsDep = Annotated[SessionStore, Depends(session_store)]
