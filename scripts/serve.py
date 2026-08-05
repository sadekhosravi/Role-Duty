"""CLI: run the HTTP API.

Usage:
    python scripts/serve.py                  # http://127.0.0.1:8000/docs
    python scripts/serve.py --port 9000
    python scripts/serve.py --host 0.0.0.0   # reachable from other machines
    python scripts/serve.py --reload         # restart on a source change

Swagger is at /docs, ReDoc at /redoc, and the raw schema at /openapi.json.

One worker, and not by omission. The job registry, the agent's conversations and
the lock that stops two ingests writing over each other are all objects in this
process — with a second worker each would exist twice, a job submitted to one
would be invisible to the other, and the graph lock would be guarding nothing
while both processes wrote to the same files. Scaling past one worker means
moving that state out (a queue, Redis) rather than adding a flag here.
"""

import argparse
import sys
from pathlib import Path

# Make the src/ packages importable when running as a plain script — the same
# shim scripts/ingest.py and scripts/agent.py use.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn

from api.settings import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Role-Duty RAG API.")
    parser.add_argument(
        "--host", default=settings.host, help=f"Interface to bind (default: {settings.host})."
    )
    parser.add_argument(
        "--port", type=int, default=settings.port, help=f"Port (default: {settings.port})."
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help=(
            "Restart when a source file changes. Development only — a reload "
            "drops every job and conversation the process was holding."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
    )
    args = parser.parse_args()

    print(f"Swagger UI: http://{args.host}:{args.port}/docs\n")

    # The app is passed by import string when reloading (uvicorn has to be able
    # to re-import it in the child process) and as an object otherwise, which
    # avoids importing the whole agent stack twice.
    if args.reload:
        uvicorn.run(
            "api.main:app",
            host=args.host,
            port=args.port,
            reload=True,
            reload_dirs=[str(Path(__file__).resolve().parents[1] / "src")],
            log_level=args.log_level,
        )
        return

    from api.main import app

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
