"""What the agent filed.

The filer is the only node in the workflow with a side effect: it writes a Word
document. Over the CLI you go and look in data/tickets/; over HTTP there has to
be a way to see the same thing, or the one action the agent can take is
invisible to the client that asked for it.

Read-only, and deliberately. Creating a ticket is the agent's job, through the
MCP server that owns the ticket's schema — an HTTP endpoint that wrote one
directly would be a second, hand-written definition of what a ticket is, which
is exactly the drift `src/ticket_mcp` exists to prevent. There is no delete
either: a filed incident ticket is a record.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from ticket_mcp.document import tickets_dir

from ..paths import plain_name, relative
from ..schemas import ErrorResponse, TicketFile, TicketList

router = APIRouter(prefix="/tickets", tags=["tickets"])

DOCX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)


@router.get(
    "",
    response_model=TicketList,
    summary="Tickets the agent has filed, newest first",
    description=(
        "Lists the .docx files in TICKETS_DIR (data/tickets by default). An "
        "empty list on a fresh checkout is normal — the directory is created by "
        "the first ticket."
    ),
)
async def list_tickets() -> TicketList:
    directory = tickets_dir()
    tickets = []
    if directory.exists():
        for path in directory.glob("*.docx"):
            stat = path.stat()
            tickets.append(
                TicketFile(
                    name=path.name,
                    path=relative(path),
                    size_bytes=stat.st_size,
                    created_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                )
            )
    tickets.sort(key=lambda ticket: ticket.created_at, reverse=True)
    return TicketList(tickets_dir=relative(directory), tickets=tickets)


@router.get(
    "/{name}",
    response_class=FileResponse,
    responses={
        status.HTTP_200_OK: {"content": {DOCX_MEDIA_TYPE: {}}},
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
    },
    summary="Download one ticket",
    description="The Word document itself. `name` is a file name from `GET /tickets`.",
)
async def download_ticket(name: str) -> FileResponse:
    # The name comes from the client and it addresses a file, so anything that
    # is not a plain .docx file name is rejected — see api/paths.py. A
    # PathRejected raised here becomes a 400 through the handler in main.py.
    path = tickets_dir() / plain_name(name, suffix=".docx")
    if not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no such ticket: {name}")
    return FileResponse(path, media_type=DOCX_MEDIA_TYPE, filename=name)
