"""How a node is declared.

`NodeSpec` lives here rather than in graph.py so that a node module can declare
its own spec without importing the graph that will wire it — graph.py imports
the specs, and nothing imports back.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .state import AgentState

# What a node body is: it reads the state and returns the fields it wants
# changed. Everything it does not mention is left alone.
Runner = Callable[[AgentState], Awaitable[dict]]


class NodeSpec(BaseModel):
    """One node: what it is told, what it is allowed to call, and what it runs.

    `tools` is the node's whole toolbox. It is bound to the node's model and
    used to build the node's own executor, so the list is the access boundary in
    both directions — nothing outside it can be named, nothing outside it can be
    run. A node with an empty list simply thinks without retrieving.
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
    runner: Runner = Field(
        description="The async function that runs when control reaches this node.",
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
