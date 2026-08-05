"""An MCP server that files incident tickets as Word documents.

    document.py   the .docx itself — a Ticket in, a Path out. No MCP, no model.
    server.py     the MCP surface: one tool, `create_ticket`, over stdio.
    __main__.py   `python -m ticket_mcp` starts it.

It is a separate process, spoken to over the Model Context Protocol, and it
knows nothing about the agent. That separation is the point: the agent's only
way to write a file is to ask this server, so the set of files the agent can
create is exactly the set this server is willing to write.

Run it by hand to check it starts:

    python -m ticket_mcp     # then Ctrl-C; it waits on stdin for MCP traffic

Nothing in this package may print to stdout. On a stdio transport stdout *is*
the protocol channel, and a stray print is parsed as a malformed message.
"""

from .document import Ticket, tickets_dir, write_ticket

__all__ = ["Ticket", "write_ticket", "tickets_dir"]
