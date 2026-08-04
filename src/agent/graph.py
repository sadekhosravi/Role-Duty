"""The LangGraph workflow: an orchestrator delegating to specialist nodes.

    START -> orchestrator -> researcher   -+
                          -> gap_auditor  -+-> back to orchestrator
                          -> notifier     -+
                          -> responder -> verifier -> END
                                                   -> back to orchestrator

Orchestrator-workers with one evaluator loop. The orchestrator is the only node
that decides what happens next: it picks one worker, that worker reports back,
and it picks again until it routes to the responder. The responder's draft then
goes to the verifier, which either ends the run or sends it back around.

This module owns only the wiring. What each node *is* — its prompt, its body,
its tools — lives in its own file under nodes/, and nothing there knows about
the edges. Implementing a node means replacing one stub; no edges change.

Tool access is per node, not global. A NodeSpec lists the tools its node may
call and it is handed nothing else: only those are bound to its model, so it
cannot name a tool outside its list, and each node gets its own executor, so it
cannot reach another node's either. MCP tools, when they arrive, are ordinary
tools — they go in a node's list and change nothing here.

An executor node (`<name>_tools`) is not a reasoning step; it is LangGraph's
mechanical runner for whatever its node asked for, and it always hands control
straight back. A worker loops on its own tools until it produces a message with
no tool calls, and only then reports to the orchestrator.

Both loops are bounded — MAX_DELEGATIONS on the orchestrator's, MAX_REVISIONS on
the verifier's. Neither bound is decoration: without them a disagreement between
two nodes runs until LangGraph's recursion limit kills the request outright,
which returns nothing at all rather than an imperfect answer.

Usage:
    python scripts/agent.py "Who authorizes a UPS bypass?"
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.nodes import NODE_SPECS
from agent.observability import trace_run
from agent.spec import NodeSpec
from agent.state import (
    GAP_AUDITOR,
    MAX_REVISIONS,
    NOTIFIER,
    ORCHESTRATOR,
    RESEARCHER,
    RESPONDER,
    VERIFIER,
    WORKERS,
    AgentState,
)

# Nodes the wiring below refers to by name. Kept next to the wiring so adding an
# edge and forgetting the node it points at is caught by validation.
WIRED_NODES = (ORCHESTRATOR, RESEARCHER, GAP_AUDITOR, NOTIFIER, RESPONDER, VERIFIER)


def _route_to_worker(state: AgentState) -> str:
    """Send control to the worker the orchestrator chose.

    `next_node` is a Literal, so an invented destination cannot reach here; the
    only case left is no choice at all, which means the orchestrator is done
    delegating and the answer should be written.
    """
    return state.next_node or RESPONDER


def _route_after_verify(state: AgentState) -> str:
    """End the run, or send the answer back for another pass.

    Hitting the revision cap returns the current answer with its failed verdict,
    which is a worse answer but still an answer.
    """
    if state.verdict == "fail" and state.revisions < MAX_REVISIONS:
        return ORCHESTRATOR
    return END


class Workflow(BaseModel):
    """The set of nodes to compile, and the rules they satisfy together.

    NodeSpec validates a node on its own; the checks that need every node at
    once — that none collide, and that the ones the wiring names all exist —
    live here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    nodes: list[NodeSpec] = Field(
        min_length=1,
        description="The workflow's nodes.",
    )

    @model_validator(mode="after")
    def _wiring_is_satisfiable(self) -> Workflow:
        """Catch collisions and missing nodes before they become quiet bugs.

        `add_node` with an existing key replaces the node rather than
        complaining, so a duplicate name silently drops one of them; executor
        names are checked too, since a node called `rag_tools` would collide
        with the executor generated for a node called `rag`. And because the
        edges are written against fixed names, a missing node would otherwise
        fail with a message about an edge rather than about the absent node.
        """
        keys = [key for spec in self.nodes for key in (spec.name, spec.tools_node)]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            raise ValueError(f"nodes would share graph keys: {', '.join(duplicates)}")

        missing = sorted(set(WIRED_NODES) - {spec.name for spec in self.nodes})
        if missing:
            raise ValueError(f"the wiring needs these nodes: {', '.join(missing)}")
        return self

    @property
    def by_name(self) -> dict[str, NodeSpec]:
        """The specs, addressable by node name."""
        return {spec.name: spec for spec in self.nodes}

    def compile(self):
        """Wire the specs into a runnable graph."""
        builder = StateGraph(AgentState)

        # Every node, plus its own executor if it has tools. ToolNode's `name`
        # defaults to "tools" and is what shows up in traces, so it is set to
        # match the graph key — otherwise every executor here would report
        # itself as "tools" and they would be indistinguishable.
        for spec in self.nodes:
            builder.add_node(spec.name, spec.runner)
            if spec.tools:
                node = ToolNode(spec.tools, name=spec.tools_node)
                builder.add_node(spec.tools_node, node)
                builder.add_edge(spec.tools_node, spec.name)

        builder.add_edge(START, ORCHESTRATOR)

        # The orchestrator delegates to exactly one worker at a time. The
        # destination list is WORKERS, so a node not in it can never be routed
        # to no matter what the orchestrator returns.
        builder.add_conditional_edges(ORCHESTRATOR, _route_to_worker, list(WORKERS))

        # Workers report back and let the orchestrator decide again.
        for name in (RESEARCHER, GAP_AUDITOR, NOTIFIER):
            self._continue_from(builder, name, ORCHESTRATOR)

        # The answer is always graded before it can leave.
        self._continue_from(builder, RESPONDER, VERIFIER)
        builder.add_conditional_edges(VERIFIER, _route_after_verify, [ORCHESTRATOR, END])

        return builder.compile()

    def _continue_from(self, builder: StateGraph, name: str, target: str) -> None:
        """Send `name` to `target` once it has finished with its own tools.

        A node with tools cannot simply have an edge out: it has to be able to
        loop on its executor first. tools_condition returns END when the model
        stopped asking for tools, and mapping that to `target` is what turns
        "done calling tools" into "hand over".
        """
        spec = self.by_name[name]
        if not spec.tools:
            builder.add_edge(name, target)
            return
        builder.add_conditional_edges(
            name, tools_condition, {"tools": spec.tools_node, END: target}
        )


WORKFLOW = Workflow(nodes=NODE_SPECS)


def build_graph(workflow: Workflow = WORKFLOW):
    """Compile the workflow. Call once and reuse — it holds the open stores."""
    return workflow.compile()


async def run_workflow(
    question: str,
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
) -> tuple[AgentState, str | None]:
    """Run one question through the workflow. Returns the state and its trace URL.

    The single entry point, so that tracing is attached in one place rather than
    at every call site — a caller that forgets the callbacks does not get a
    degraded trace, it gets no trace, and the gap is invisible until someone
    goes looking for a run that was never recorded.

    Note the explicit HumanMessage: the `("user", "...")` shorthand is a
    convenience of the add_messages reducer, and constructing AgentState
    directly goes straight to Pydantic, which validates against AnyMessage.

    The trace URL is returned rather than printed. This module has no business
    writing to a terminal, and the CLI is the only caller that wants it.
    """
    with trace_run(question, session_id=session_id, user_id=user_id, tags=tags) as trace:
        result = await build_graph().ainvoke(
            AgentState(messages=[HumanMessage(question)]),
            config={"callbacks": trace.callbacks},
        )
        # ainvoke returns a plain dict, not an AgentState — see state.py.
        # Rebuilding the model is what makes the result validated and
        # attribute-addressed, and the verifier's verdict readable as a field.
        state = AgentState(**result)
        trace.record(state)
    return state, trace.url


async def ask(question: str) -> str:
    """Run one question and return just the final answer.

    Returns `answer`, not the last message: the verifier appends its critique
    after the answer, so on a rejected run the final message is the complaint
    rather than the thing the user asked for.
    """
    state, _ = await run_workflow(question)
    return state.answer or "(no answer was produced)"
