"""Agentic RAG: an orchestrator delegating to specialist nodes.

    nodes/      one file per node — its prompt, its body, its tools, its spec
    tools/      one file per retriever — graph, vector, keyword
    graph.py    the wiring: who hands to whom, and where the loops stop
    state.py    what flows between nodes, and the node-name vocabulary
    spec.py     how a node is declared
    llm.py      the chat model
    prompts.py  RAG_SYSTEM_PROMPT, from the single-node version — the source
                the researcher's and responder's prompts get written from

See graph.py for the overview. The orchestrator is implemented; the other five
nodes are stubs, so the graph runs end to end but does nothing yet.

Everything declared here — the state, each node, the workflow, the arguments
each tool accepts, and the orchestrator's routing decision — is a Pydantic
model, so a malformed declaration or a bad tool call is caught and named at the
point it happens.
"""

from .graph import WORKFLOW, Workflow, ask, build_graph
from .nodes import NODE_SPECS
from .spec import NodeSpec
from .state import AgentState
from .tools import ALL_TOOLS

__all__ = [
    "ask",
    "build_graph",
    "WORKFLOW",
    "Workflow",
    "NodeSpec",
    "NODE_SPECS",
    "AgentState",
    "ALL_TOOLS",
]
