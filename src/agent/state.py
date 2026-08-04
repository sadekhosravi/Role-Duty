"""The agent's state: the single object that flows through the workflow.

A Pydantic model rather than LangGraph's built-in `MessagesState`, which is a
TypedDict and therefore checked only by a type checker — at runtime it is a bare
dict that will accept any key and any value. `AgentState` is checked when it is
actually used, so a malformed update fails where it happens instead of surfacing
later as a confusing error inside a model call.

Two behaviours of Pydantic state are worth knowing, both verified against the
installed LangGraph (1.2.10) rather than taken on faith:

* Nodes receive an `AgentState` instance, not a dict, and LangGraph re-validates
  the state on the way into every node — so read `state.messages`, not
  `state["messages"]`. Note that the docs claim validation happens only on the
  first node; the installed version coerces per node. Treat the extra validation
  as a safety net, not a guarantee, since that is undocumented behaviour.
* The compiled graph still *returns* a plain dict, not an `AgentState`. That is
  a documented LangGraph limitation. Rebuild the model if you want one back:

      result = await graph.ainvoke(AgentState(messages=[...]))
      state = AgentState(**result)
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field


class AgentState(BaseModel):
    """Everything the workflow carries between nodes.

    Only the conversation for now. A node that needs to pass something else
    along — a plan, a retrieval budget, a verdict from an earlier node — gets a
    field here rather than smuggling it through the message list.
    """

    # extra="forbid" turns a typo in a node's return dict into an immediate,
    # named error instead of a key that is silently written and never read.
    model_config = ConfigDict(extra="forbid")

    messages: Annotated[list[AnyMessage], add_messages] = Field(
        default_factory=list,
        description=(
            "The running conversation: the question, the model's replies, and "
            "the output of every tool it called."
        ),
    )
    """The `add_messages` reducer is what makes a node's returned messages
    *append* to this list instead of replacing it — the tool-call loop depends
    on that, since each pass has to see everything retrieved so far. It also
    accepts `("user", "...")` tuples and converts them to message objects."""
