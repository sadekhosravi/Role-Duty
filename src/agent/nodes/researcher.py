"""The researcher: gathers cited evidence, and writes no answer.

Stub. This is the closest node to the single-node version this workflow grew
out of — the difference is that it stops once it has evidence instead of going
on to answer, which is what makes the verifier possible.
"""

from __future__ import annotations

from ..spec import NodeSpec
from ..state import RESEARCHER, AgentState
from ..tools import graph_rag_search, keyword_search, naive_rag_search

PROMPT = """\
TODO. Gather evidence with the retrieval tools and report back cited context,
not prose. Section 1 of prompts.RAG_SYSTEM_PROMPT is the retrieval policy this
should be built from.
"""


async def run(state: AgentState) -> dict:
    """TODO: `llm.chat_runner(PROMPT, TOOLS)` is most of this already."""
    return {}


SPEC = NodeSpec(
    name=RESEARCHER,
    system_prompt=PROMPT,
    runner=run,
    tools=[graph_rag_search, naive_rag_search, keyword_search],
)
