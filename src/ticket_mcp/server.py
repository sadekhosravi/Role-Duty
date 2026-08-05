"""The MCP surface: one tool, `create_ticket`, spoken over stdio.

This module is the contract with the model. The parameter descriptions below are
not documentation for a human reader — they are shipped to the model as the
tool's JSON schema, and they are the only instructions it gets about what
belongs in each field.

Only one tool is exposed, and it writes one fixed form. The alternative — a
general Word server with a couple of dozen tools for headings, paragraphs and
tables — was rejected: it would need ten calls to produce one ticket, each an
opportunity to drift, and it could not enforce that a ticket has a recipient at
all. One call, one validated form, one file.
"""

from __future__ import annotations

from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from .document import PRIORITIES, ROOT_DIR, Ticket, write_ticket

mcp = FastMCP(
    "role-duty-tickets",
    instructions=(
        "Files incident tickets as Word documents. Call create_ticket once per "
        "incident, only after the person has asked for a ticket, and address it "
        "to a role that the source documents actually make responsible."
    ),
    # One line per request on stderr is noise in the terminal the agent shares.
    # Warnings and errors still come through, which is what anyone debugging a
    # failed ticket actually needs.
    log_level="WARNING",
)


@mcp.tool(
    title="File an incident ticket",
    description=(
        "Write an incident ticket to a Word (.docx) file and return where it "
        "was saved. Call this once, only when the user has asked for a ticket. "
        "Every field must come from the conversation: the incident as the user "
        "described it, and the responsible role and next steps as the answer "
        "established them. Do not invent a recipient — if no role has been "
        "established as responsible, do not call this tool."
    ),
)
def create_ticket(
    addressed_to: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "The exact role title the ticket goes to, as the source "
                "documents write it — e.g. 'Housekeeping Supervisor "
                "(Housekeeping)'. Never a person's name, never a guess."
            ),
        ),
    ],
    subject: Annotated[
        str,
        Field(
            min_length=1,
            description="One line naming the incident. Becomes part of the filename.",
        ),
    ],
    what_happened: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "What the person reported, in their terms: what was found or "
                "observed, where, and when. Facts only — no advice here."
            ),
        ),
    ],
    situation_summary: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Why this needs the recipient: what the documents say about "
                "handling it, which limits apply, and what must not be done. "
                "A short paragraph the recipient can act on without the chat."
            ),
        ),
    ],
    next_steps: Annotated[
        list[str],
        Field(
            min_length=1,
            description=(
                "The actions the answer established, one per item, in order. "
                "Each should be a single instruction. At least one is required "
                "— a ticket that names no action is not actionable."
            ),
        ),
    ],
    raised_by: Annotated[
        str,
        Field(
            description="The role of the person reporting it, if they said — e.g. 'Housekeeper'.",
        ),
    ] = "",
    organisation: Annotated[
        str,
        Field(
            description=(
                "Which organisation this concerns — the corpus holds several "
                "unrelated ones. The site or the source document name."
            ),
        ),
    ] = "",
    priority: Annotated[
        Literal[PRIORITIES],  # type: ignore[valid-type]
        Field(
            description=(
                "How fast it needs attention. Use 'urgent' for anything the "
                "documents route to police, security or a safety escalation."
            ),
        ),
    ] = "normal",
    references: Annotated[
        list[str],
        Field(
            description=(
                "The full source labels behind the next steps, copied exactly "
                "from the answer's References section. Copy them; do not "
                "rebuild them."
            ),
        ),
    ] = [],  # noqa: B006 - pydantic copies the default per call; never mutated here
) -> str:
    """File the ticket and report where it went."""
    ticket = Ticket(
        addressed_to=addressed_to.strip(),
        subject=subject.strip(),
        what_happened=what_happened.strip(),
        situation_summary=situation_summary.strip(),
        next_steps=tuple(step for step in next_steps if step.strip()),
        raised_by=raised_by.strip(),
        organisation=organisation.strip(),
        priority=priority,
        references=tuple(dict.fromkeys(r.strip() for r in references if r.strip())),
    )
    path = write_ticket(ticket)

    # The project-relative path is what the agent should quote back: it is the
    # form the user asked for ("in data/tickets") and it reads the same on every
    # machine. Falls back to the absolute path when tickets are written outside
    # the project, which is what TICKETS_DIR allows.
    try:
        shown = path.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        shown = str(path)
    return (
        f"Ticket {ticket.reference} created for {ticket.addressed_to}.\n"
        f"Saved to: {shown}\n"
        f"Full path: {path}"
    )


def main() -> None:
    """Serve on stdio. Blocks until the client closes the connection."""
    mcp.run(transport="stdio")
