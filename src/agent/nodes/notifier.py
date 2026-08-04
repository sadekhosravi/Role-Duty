"""The notifier: delivers a finding to someone outside this conversation.

Stub, and the only node that needs something the project does not have yet — an
MCP tool to send with. Adding it to `tools` below is the whole change: the node
then gets its own executor and its own tool loop like any other.
"""

from __future__ import annotations

from ..spec import NodeSpec
from ..state import NOTIFIER, AgentState

PROMPT = """\
TODO. Deliver an established finding to the person or channel the request
named. Send nothing that has not been verified, and invent no recipient.
"""


async def run(state: AgentState) -> dict:
    """TODO: needs an MCP tool before it can do anything."""
    return {}


SPEC = NodeSpec(
    name=NOTIFIER,
    system_prompt=PROMPT,
    runner=run,
    # TODO: the MCP send tool goes here.
)
