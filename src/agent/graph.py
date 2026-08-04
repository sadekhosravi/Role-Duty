"""The LangGraph workflow: reasoning nodes, each with its own tools.

    START -> rag -> (rag_tools -> rag)* -> END

Today there is one node. `rag` reads the question, decides which of its
retrievers to call, reads what comes back, and writes the final answer —
reconciling the graph, vector and keyword views itself rather than trusting any
one of them. Its system prompt (prompts.py) governs both halves of that: which
tool to reach for, and how to reason over the results.

Tool access is per node, not global. A node is defined by a NodeSpec listing the
tools it may call, and it is handed nothing else: only those tools are bound to
its model, so it cannot name a tool outside its list, and each node gets its own
executor, so it cannot reach another node's either. Adding a second node means
adding a NodeSpec to WORKFLOW.

An executor node (`<name>_tools`) is not a reasoning step; it is LangGraph's
mechanical runner for whatever its node asked for, and it always hands control
straight back. The loop runs until the node returns a message with no tool
calls, which is the answer.

The shape of the workflow is itself a Pydantic model, so a mistake in how it is
declared — a node name that cannot be a graph key, two nodes that would collide,
the same tool listed twice — is raised at import with the offending field named,
rather than compiling into a graph that quietly misbehaves.

Usage:
    python scripts/agent.py "Who authorizes a UPS bypass?"
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent.prompts import RAG_SYSTEM_PROMPT
from agent.state import AgentState
from agent.tools import graph_rag_search, keyword_search, naive_rag_search
from graph_rag.graph_rag import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


class NodeSpec(BaseModel):
    """One reasoning node: what it is told, and what it is allowed to call.

    `tools` is the node's whole toolbox. It is bound to the node's model and
    used to build the node's own executor, so the list is the access boundary in
    both directions — nothing outside it can be named, nothing outside it can be
    run. A node with an empty list simply answers without retrieving.
    """

    # Frozen because a spec is a declaration, not a working value: the compiled
    # graph closes over these fields, so mutating one afterwards would leave the
    # declaration and the running graph disagreeing.
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(
        min_length=1,
        description="The node's key in the graph, and the stem of its executor's.",
    )
    system_prompt: str = Field(
        min_length=1,
        description="Instructions prepended to every call this node makes.",
    )
    tools: list[BaseTool] = Field(
        default_factory=list,
        description="The only tools this node may call.",
    )

    @field_validator("name")
    @classmethod
    def _usable_as_a_graph_key(cls, name: str) -> str:
        """Reject names that would break or collide once compiled.

        LangGraph node keys are plain strings, so anything is accepted at
        registration and the damage shows up later. Two are worth catching here:
        a leading underscore risks colliding with LangGraph's own reserved keys
        (`__start__`, `__end__`), and a non-identifier name reads badly in every
        trace and edge definition that mentions it.
        """
        if not name.isidentifier():
            raise ValueError(f"must be a valid Python identifier, got {name!r}")
        if name.startswith("_"):
            raise ValueError(f"must not start with '_' (reserved), got {name!r}")
        return name

    @field_validator("tools")
    @classmethod
    def _distinct_tool_names(cls, tools: list[BaseTool]) -> list[BaseTool]:
        """A duplicate name is unresolvable: the model names the tool it wants.

        Two tools sharing a name — the same tool listed twice, or two different
        ones both called `search` — leave the executor no way to know which was
        meant, and the model no way to say.
        """
        names = [tool.name for tool in tools]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate tool names: {', '.join(duplicates)}")
        return tools

    @property
    def tools_node(self) -> str:
        """Name of this node's paired tool executor."""
        return f"{self.name}_tools"


class Workflow(BaseModel):
    """The set of nodes to compile, and the rules they have to satisfy together.

    NodeSpec validates a node on its own; the checks that need every node at
    once — that no two claim the same graph key — live here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    nodes: list[NodeSpec] = Field(
        min_length=1,
        description="The workflow's nodes. The first is the entry point.",
    )

    @model_validator(mode="after")
    def _distinct_graph_keys(self) -> Workflow:
        """Catch name collisions, which LangGraph would resolve by overwriting.

        `add_node` with an existing key replaces the node rather than
        complaining, so a duplicate name silently drops one of them. Executor
        names are checked too: a node called `rag_tools` would collide with the
        executor generated for a node called `rag`.
        """
        keys = [key for spec in self.nodes for key in (spec.name, spec.tools_node)]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            raise ValueError(f"nodes would share graph keys: {', '.join(duplicates)}")
        return self

    @property
    def entry_point(self) -> NodeSpec:
        """The node START leads to."""
        return self.nodes[0]

    def compile(self):
        """Wire the specs into a runnable graph.

        Each node gets its own executor and its own loop back. Note that nodes
        are not yet connected to *each other* — each loops on its own tools and
        then ends — so a second node also needs an edge or a router saying when
        control passes to it. Until it has one, LangGraph prunes it as
        unreachable and it will not appear in the compiled graph at all.
        """
        builder = StateGraph(AgentState)

        for spec in self.nodes:
            builder.add_node(spec.name, _make_node(spec))
            if not spec.tools:
                builder.add_edge(spec.name, END)
                continue
            # This node's own executor, holding only this node's tools.
            # ToolNode's `name` defaults to "tools" and is what shows up in
            # traces, so it is set to match the graph key — otherwise every
            # executor in a multi-node graph reports itself as "tools" and they
            # are indistinguishable.
            builder.add_node(spec.tools_node, ToolNode(spec.tools, name=spec.tools_node))
            # tools_condition routes to the executor when the model asked for a
            # tool, and to END when it answered instead.
            builder.add_conditional_edges(
                spec.name, tools_condition, {"tools": spec.tools_node, END: END}
            )
            builder.add_edge(spec.tools_node, spec.name)

        builder.add_edge(START, self.entry_point.name)
        return builder.compile()


WORKFLOW = Workflow(
    nodes=[
        NodeSpec(
            name="rag",
            system_prompt=RAG_SYSTEM_PROMPT,
            tools=[graph_rag_search, naive_rag_search, keyword_search],
        ),
    ]
)


def build_llm() -> ChatOpenAI:
    """The chat model, pointed at OpenRouter like the rest of the project.

    temperature=0 because every rule in the system prompt is about being exact:
    the same question over the same documents should not produce a different
    role title or a different threshold on a second run.
    """
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=0,
    )


def _make_node(spec: NodeSpec):
    """Build the async function for one node, closed over its own model."""
    llm = build_llm()
    if spec.tools:
        llm = llm.bind_tools(spec.tools)

    async def node(state: AgentState) -> dict:
        # The system prompt is prepended per call rather than stored in state,
        # so it never accumulates as the tool-call loop appends messages — and
        # so each node reads its own prompt rather than the previous node's.
        messages = [("system", spec.system_prompt), *state.messages]
        return {"messages": [await llm.ainvoke(messages)]}

    return node


def build_graph(workflow: Workflow = WORKFLOW):
    """Compile the workflow. Call once and reuse — it holds the open stores."""
    return workflow.compile()


async def ask(question: str) -> str:
    """Run one question through the workflow and return the final answer.

    Note the explicit HumanMessage: the `("user", "...")` shorthand is a
    convenience of the add_messages reducer, and constructing AgentState
    directly goes straight to Pydantic, which validates against AnyMessage.
    """
    result = await build_graph().ainvoke(AgentState(messages=[HumanMessage(question)]))
    # ainvoke returns a plain dict, not an AgentState — see state.py. Rebuilding
    # the model is what makes the result validated and attribute-addressed.
    return AgentState(**result).messages[-1].content
