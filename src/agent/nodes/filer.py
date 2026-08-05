"""The filer: turns an answer the user accepted into a ticket document.

The only node that changes anything outside the conversation, and the only one
whose tool comes from an MCP server rather than this repo. It writes a Word
ticket to data/tickets/ addressed to the role the answer established as
responsible.

It is deliberately the end of its own branch: it goes straight to END rather
than back to the orchestrator. Filing is an action, not evidence — there is
nothing for the orchestrator to do with the result, and routing it onward to the
responder would produce a document-grounded answer with a `### References`
section about a file that has no sources. So the filer writes its own
confirmation, and that confirmation is what the user reads.

Two things it must not do, and both have their own guard:

  - File without being asked. The orchestrator is told never to route here on
    its own, and the prompt below tells the node to refuse if the conversation
    holds no request for a ticket.
  - File twice. MAX_TICKETS is enforced by unbinding the tool, not by asking —
    past the cap the model has no way to call it, the same way the researcher's
    retrieval budget works.
"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, ToolMessage

from ..llm import build_llm
from ..spec import NodeSpec
from ..state import FILER, AgentState
from ..tools.tickets import CREATE_TICKET, ticket_tools

# Discovered from the MCP server at import — the server owns the schema, so
# there is nothing to declare here. See tools/tickets.py for why this is
# resolved synchronously.
TOOLS = ticket_tools()

# One ticket per turn. A second call would file a duplicate of the incident that
# is already recorded, and duplicates are worse than nothing in a queue someone
# has to work through.
MAX_TICKETS = 1

PROMPT = """\
---Role---

You file incident tickets. The question has already been answered earlier in
this conversation and the person has asked you to raise a ticket about it.

You have one tool, create_ticket. It writes a Word document to data/tickets/ and
returns where it saved it. You have no retrieval tools: everything the ticket
says has to come from this conversation.

---Instructions---

1. Before you file

  - File only if the person actually asked for a ticket. If they have not, or
if you cannot tell, say what you would file and ask them to confirm. Do not
call the tool.
  - File only if an answer earlier in this conversation established who is
responsible. If no role was established, say that, name what is missing, and do
not call the tool. A ticket addressed to a role that nobody established is worse
than no ticket: it looks like a routed report and it goes nowhere.
  - Call create_ticket ONCE. When the tool has returned, you are done filing.

2. Filling the fields

  - addressed_to: the role the earlier answer identified as responsible for
this, written exactly as the answer wrote it. Not a person, not a department you
inferred, not a more senior role that seems more appropriate. If the answer
named a first point of contact and a body it escalates to, address the ticket to
the first point of contact — that is who acts next.
  - what_happened: what the person told you, in their terms. Where, what, when.
No advice in this field.
  - situation_summary: why it needs that role — what the documents say about
handling this kind of item, which limits apply, and what must not be done.
Written so the recipient can act on it without reading this conversation.
  - next_steps: the actions the earlier answer established, one per item, in
the order they should happen. Take them from the answer. Do not add a step the
answer did not give you, however sensible it seems.
  - raised_by: the role the person said they hold, if they said one.
  - organisation: which organisation this concerns. The corpus holds several
unrelated ones and a ticket that does not say which is unusable.
  - priority: urgent when the documents route the matter to police, security or
a safety escalation; otherwise judge it from how the documents describe it.
  - references: the source labels from the earlier answer's References section,
copied character for character. Copy them. Never rebuild one, never shorten one,
and never write a label the answer did not contain. If the answer cited nothing,
leave this empty.

3. After the tool returns

  - Say plainly that you created it, and give the path exactly as the tool
reported it.
  - Say who it is addressed to and, in one line, what it says.
  - Then ask what else they need. The conversation is continuing — this is not a
sign-off.
  - Do not call the tool again, and do not write a `### References` section.
This is a confirmation of an action, not an answer about the documents.

4. If the tool returns an error

  Read it, say what went wrong in one line, and say what you need in order to
  retry. Do not pretend the ticket was filed.
"""

_with_tool = None
_without_tool = None


def _get_llm(tool_allowed: bool):
    """The model, with or without the ticket tool bound. Built on first use."""
    global _with_tool, _without_tool
    if tool_allowed:
        if _with_tool is None:
            _with_tool = build_llm().bind_tools(TOOLS)
        return _with_tool
    if _without_tool is None:
        _without_tool = build_llm()
    return _without_tool


def _tickets_filed(messages: list) -> int:
    """How many tickets this run has already created.

    Counts the tool's results rather than the requests for it: a call that came
    back as an error filed nothing, but it is also not a reason to let the model
    try forever — so an errored attempt counts here too, and the cap ends it.
    """
    return sum(
        1
        for message in messages
        if isinstance(message, ToolMessage) and message.name == CREATE_TICKET
    )


def _text(content) -> str:
    """One string out of a message's content.

    An MCP tool result arrives as a list of content blocks, not a string —
    `[{"type": "text", "text": ...}]` — so anything reading it has to flatten it
    first. Reading `.content` as a string works right up until it silently does
    not match anything.
    """
    if isinstance(content, str):
        return content
    return "\n".join(
        block.get("text", "") if isinstance(block, dict) else str(block)
        for block in content or []
    )


# The line the server prints the destination on. Pinned by a round-trip check
# against the server's real output, because a regex against another component's
# wording is exactly the kind of thing that stops matching without saying so.
_SAVED_TO = re.compile(r"Saved to:\s*(\S[^\n]*)")


def _saved_path(messages: list) -> str | None:
    """Where the ticket went, read from the tool's own result.

    Read rather than reported: asked to quote the path back, the model gave a
    correct and complete confirmation with the path left out — the same class of
    omission as a dropped citation, and fixable the same way. The path is a
    string that already exists in the conversation, so it is copied from there
    rather than requested again.
    """
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and message.name == CREATE_TICKET:
            match = _SAVED_TO.search(_text(message.content))
            return match.group(1).strip() if match else None
    return None


def _note(filed: int) -> str:
    """What this turn is for, stated to the model every time."""
    if filed >= MAX_TICKETS:
        return (
            "The ticket tool has already run this turn. Do not call it again. "
            "Report what it returned — the path exactly as given, and who the "
            "ticket is addressed to — then ask what else they need."
        )
    return (
        "File the ticket now if the person asked for one and an earlier answer "
        "established who is responsible. If either is missing, say what you "
        "need instead and call nothing."
    )


async def run(state: AgentState) -> dict:
    """File the ticket, or confirm what was filed.

    `answer` is set only on the pass that produces no tool call — that is the
    message the user reads, and it is what the CLI prints. Setting it while the
    model is still mid-tool-call would overwrite it with the empty content that
    accompanies a tool request.

    The path is added here rather than trusted to the prompt. A confirmation
    that a ticket exists but does not say where is not much of a confirmation,
    and telling someone the file is somewhere in data/tickets/ is not the same
    as telling them which file.
    """
    filed = _tickets_filed(state.messages)
    llm = _get_llm(tool_allowed=filed < MAX_TICKETS)
    reply = await llm.ainvoke([("system", PROMPT), *state.messages, ("human", _note(filed))])

    if getattr(reply, "tool_calls", None):
        return {"messages": [reply]}

    text = _text(reply.content)
    path = _saved_path(state.messages)
    if path and path not in text:
        text = f"{text.rstrip()}\n\nSaved to: {path}"
        # The rewritten text replaces the reply rather than being appended
        # beside it, so the conversation holds what the user was actually shown.
        reply = AIMessage(text)
    return {"answer": text, "messages": [reply]}


SPEC = NodeSpec(
    name=FILER,
    system_prompt=PROMPT,
    runner=run,
    tools=TOOLS,
)
