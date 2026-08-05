"""The ticket tool, discovered from the ticket MCP server.

Unlike the three retrievers, nothing here declares what the tool looks like. The
server owns the schema; this module starts it, asks what it offers, and hands
the result back as ordinary LangChain tools. That is the whole reason to speak
MCP for this rather than importing `ticket_mcp` directly: the description the
model reads and the validation the file writer applies are the same declaration,
so they cannot drift apart the way two hand-written copies would.

The server runs as a subprocess over stdio — the standard transport for a local
MCP server, and the one that needs no port, no daemon and no credentials.

Discovery happens once, at import, and it costs one short-lived subprocess. It
is done synchronously because the workflow it feeds is declared synchronously:
`NodeSpec`, `NODE_SPECS` and `WORKFLOW` are module constants, and a node's tool
list has to exist by the time its spec does. The tools returned do NOT hold that
session open — each call opens its own — so they are safe to build in one event
loop and call from another, which is what lets this work at all.
"""

from __future__ import annotations

import asyncio
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

# The server is a subprocess and reads its settings — TICKETS_DIR — from the
# environment it is handed. Loading .env here rather than relying on some other
# module having done it first: which import runs first is not something this
# should depend on, and the failure would be a ticket quietly written to the
# default directory instead of the configured one.
load_dotenv()

# src/agent/tools/tickets.py -> tools -> agent -> src -> project/
SRC_DIR = Path(__file__).resolve().parents[2]
ROOT_DIR = SRC_DIR.parent

SERVER_NAME = "tickets"
CREATE_TICKET = "create_ticket"


def connection() -> dict:
    """How to start the ticket server.

    `sys.executable` rather than "python": this runs on machines where the venv
    is not on PATH, and the server has to be the same interpreter as the client
    or it will not have python-docx. PYTHONPATH carries src/ so that
    `-m ticket_mcp` resolves without the package being installed — the same
    shim the scripts/ CLIs use, for the same reason.

    The environment is passed in full because the alternative is worse: with no
    `env` the MCP client hands the server a filtered safe-list, which drops
    TICKETS_DIR along with everything else this project sets.

    Note that the result is captured when tools are discovered, not when one is
    called — the tool objects keep the connection they were built with. Changing
    TICKETS_DIR after that point has no effect on an already-discovered tool;
    call `load_tools()` again for a handle that sees the new value.
    """
    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-m", "ticket_mcp"],
        "cwd": str(ROOT_DIR),
        "env": {**os.environ, "PYTHONPATH": str(SRC_DIR)},
    }


async def load_tools() -> list[BaseTool]:
    """Start the server, list what it offers, and wrap it for LangChain."""
    client = MultiServerMCPClient({SERVER_NAME: connection()})
    return await client.get_tools()


def _blocking(coroutine) -> list[BaseTool]:
    """Run `coroutine` to completion from synchronous code, loop or no loop.

    `asyncio.run` is enough at import time, when nothing is running yet. The
    thread is for the other case — an import triggered from inside a running
    loop, which `asyncio.run` refuses outright. Doing it in a worker thread with
    its own loop is the difference between "slower" and "ImportError".
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coroutine).result()


_tools: list[BaseTool] | None = None


def ticket_tools() -> list[BaseTool]:
    """Every tool the ticket server offers. Discovered once, then cached.

    Failure raises rather than returning an empty list. A node declared with no
    tools is a node that quietly cannot act: the model would be told it has
    nothing to file with, apologise, and the run would look like a decision
    rather than a broken install. The same applies to a server that starts but
    no longer offers `create_ticket` — the tool having been renamed is exactly
    the drift this module exists to prevent, so it is checked, not assumed.
    """
    global _tools
    if _tools is None:
        try:
            discovered = _blocking(load_tools())
        except Exception as error:  # noqa: BLE001 - re-raised with the cause named
            raise RuntimeError(
                f"could not start the ticket MCP server "
                f"({sys.executable} -m ticket_mcp): {type(error).__name__}: {error}"
            ) from error

        names = [tool.name for tool in discovered]
        if CREATE_TICKET not in names:
            raise RuntimeError(
                f"the ticket MCP server offers {names or 'no tools'}, "
                f"but the filer node is written against {CREATE_TICKET!r}"
            )
        _tools = discovered
    return _tools
