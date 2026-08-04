"""The responder: writes the answer from evidence already gathered.

Stub. No tools by design — if it could retrieve, it could quietly answer from
something the verifier never saw, and the grading downstream would be checking
the wrong thing.
"""

from __future__ import annotations

from ..spec import NodeSpec
from ..state import RESPONDER, AgentState

PROMPT = """\
TODO. Write the final answer from the gathered evidence, in the shape the
request implies. Sections 2-10 of prompts.RAG_SYSTEM_PROMPT are the answering
rules this should be built from.
"""


async def run(state: AgentState) -> dict:
    """TODO."""
    return {}


SPEC = NodeSpec(
    name=RESPONDER,
    system_prompt=PROMPT,
    runner=run,
    # No tools: it writes from what the researcher already found.
)
