"""Agentic RAG: an orchestrator delegating to specialist nodes.

    nodes/          one file per node — its prompt, its body, its tools, its spec
    tools/          one per tool — sections, graph, and the MCP ticket writer
    conversation.py what a caller does with a finished run
    graph.py        the wiring: who hands to whom, and where the loop stops
    state.py        what flows between nodes, and the routing vocabulary
    spec.py         how a node is declared
    llm.py          the chat model

See graph.py for the overview. All five nodes are implemented: the run gathers
evidence, writes an answer, grades it, and — when the user asks — files a ticket
about it through the MCP server in src/ticket_mcp.

Everything declared here — the state, each node, the workflow, the arguments
each tool accepts, and the orchestrator's routing decision — is a Pydantic
model, so a malformed declaration or a bad tool call is caught and named at the
point it happens.
"""

from .graph import WORKFLOW, Workflow, ask, build_graph, run_workflow
from .nodes import NODE_SPECS
from .spec import NodeSpec
from .state import AgentState
from .tools import ALL_RETRIEVERS

__all__ = [
    "ask",
    "run_workflow",
    "build_graph",
    "WORKFLOW",
    "Workflow",
    "NodeSpec",
    "NODE_SPECS",
    "AgentState",
    "ALL_RETRIEVERS",
]
