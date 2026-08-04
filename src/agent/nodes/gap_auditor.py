"""The gap auditor: finds structural problems across the whole role set.

Stub. Unowned duties, duties two roles both claim, and escalation targets that
do not exist are properties of the graph's shape rather than of any one
passage, which is why this node gets the graph tool and only the graph tool.
"""

from __future__ import annotations

from ..spec import NodeSpec
from ..state import GAP_AUDITOR, AgentState
from ..tools import graph_rag_search

PROMPT = """\
TODO. Sweep the graph for duties nobody owns, duties two roles both own, and
escalation targets that do not exist. Report findings, not prose.
"""


async def run(state: AgentState) -> dict:
    """TODO."""
    return {}


SPEC = NodeSpec(
    name=GAP_AUDITOR,
    system_prompt=PROMPT,
    runner=run,
    # Graph only: the other two retrievers cannot see the graph's shape, which
    # is the only place these findings exist.
    tools=[graph_rag_search],
)
