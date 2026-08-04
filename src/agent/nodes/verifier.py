"""The verifier: decides whether the answer is grounded in what was retrieved.

Stub. Its rule has to stay narrow enough to be answerable — "every claim traces
to retrieved evidence" is checkable, "is this a good answer" is not, and an
evaluator loop with an unanswerable criterion never terminates.
"""

from __future__ import annotations

from ..spec import NodeSpec
from ..state import VERIFIER, AgentState

PROMPT = """\
TODO. Check every claim in the answer traces to retrieved evidence, and that no
role, threshold or reporting line was invented. Pass or fail, and on a fail name
the specific gap so the orchestrator knows who can close it.
"""


async def run(state: AgentState) -> dict:
    """TODO: set `verdict`, and increment `revisions` on a fail.

    Returning nothing counts as a pass and ends the run.
    """
    return {}


SPEC = NodeSpec(
    name=VERIFIER,
    system_prompt=PROMPT,
    runner=run,
    # TODO: decide whether it may spot-check a claim with keyword_search, or
    # must judge only from what is already in the state.
)
