"""The agent's state: the single object that flows through the workflow.

A Pydantic model rather than LangGraph's built-in `MessagesState`, which is a
TypedDict and therefore checked only by a type checker — at runtime it is a bare
dict that will accept any key and any value. `AgentState` is checked when it is
actually used, so a malformed update fails where it happens instead of surfacing
later as a confusing error inside a model call.

This module also owns the workflow's *vocabulary* — the node names, and the two
sets a routing decision is drawn from: the workers a model may be asked to pick,
and the wider set of destinations the graph knows how to follow. They live here
rather than in graph.py because the state is what carries a routing decision
between nodes, and putting them here is what lets `next_node` be a constrained
Literal instead of a free string. graph.py imports them back when it wires the
edges.

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

import operator
from typing import Annotated, Literal, get_args

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field

# Node names. Constants rather than bare strings because each one is written in
# several places — the spec, the edges, the routers — and a typo in any of them
# produces a silently disconnected graph rather than an error.
ORCHESTRATOR = "orchestrator"
RESEARCHER = "researcher"
FILER = "filer"
RESPONDER = "responder"
VERIFIER = "verifier"

# The only nodes the orchestrator may delegate to. Hard-capping the destinations
# is the documented fix for the orchestrator-bottleneck failure mode, where an
# unconstrained router loops between workers until it hits the recursion limit.
# As a Literal it reaches the model as a JSON-schema enum, so an invented
# destination is rejected before it can be routed on.
WorkerName = Literal["researcher", "filer", "responder"]
WORKERS: tuple[str, ...] = get_args(WorkerName)

# Not a node: the destination that ends the run. It exists so that "stop" is a
# thing the orchestrator *returns*, in the same field and through the same edge
# as every other decision, rather than a second router somewhere else deciding
# it. The graph maps it to END.
FINISH = "finish"

# Everywhere control can go from the orchestrator. Deliberately wider than
# WorkerName, and the split is the point: a *worker* is something the model may
# be asked to pick, a *route* is something the graph knows how to follow. Only
# the first set reaches the model as an enum, so `finish` is not a choice the
# model can make — the orchestrator's own code decides that, which is what keeps
# the revision cap enforced in Python rather than by asking nicely.
RouteName = Literal["researcher", "filer", "responder", "finish"]
ROUTES: tuple[str, ...] = get_args(RouteName)

# The two Literals are written out separately because typing cannot subtract
# one from the other, and a duplicated list is a list that drifts. Checked at
# import so it drifts loudly.
if not set(WORKERS) < set(ROUTES):
    raise RuntimeError(
        f"every worker must be routable: {sorted(set(WORKERS) - set(ROUTES))} is not"
    )
if FINISH not in ROUTES:
    raise RuntimeError(f"{FINISH!r} must be a route")

# How many times the verifier may send an answer back for another pass. The loop
# needs a bound: a verifier that keeps failing an answer the workers cannot
# improve would otherwise run until LangGraph's recursion limit kills the whole
# request with nothing to show for it.
MAX_REVISIONS = 2

# How many workers the orchestrator may run before the answer has to be written.
# The same reasoning as MAX_REVISIONS, applied to the other loop: an orchestrator
# that keeps delegating gets cut off with an answer rather than an exception.
MAX_DELEGATIONS = 8

# How many tool calls the researcher may make across the whole run. Its loop is
# the third one that needs a bound, and the least visible: a node that keeps
# calling tools never reaches a router, so nothing else can stop it before
# LangGraph's recursion limit does.
MAX_RETRIEVALS = 8


class AgentState(BaseModel):
    """Everything the workflow carries between nodes.

    Deliberately thin for now: the conversation, plus the two decisions that
    determine where control goes next. Nodes that need to hand each other
    structured work — a plan, graded evidence, a draft — get fields here as they
    are implemented, rather than smuggling it through the message list.
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

    next_node: RouteName | None = Field(
        default=None,
        description=(
            "Where the orchestrator sent control: a worker, or 'finish' to end "
            "the run. None only before it has run at all."
        ),
    )
    """Typed as RouteName rather than WorkerName because the orchestrator can now
    end the run, and ending it is a routing decision like any other. What the
    *model* may pick is still WorkerName — see nodes/orchestrator.py."""
    delegations: Annotated[list[str], operator.add] = Field(
        default_factory=list,
        description="Every worker the orchestrator has run, in order.",
    )
    """The orchestrator's memory of its own decisions, and the only thing that
    stops it choosing the same worker forever. `operator.add` is what makes a
    returned `[name]` append rather than replace — the same role `add_messages`
    plays for the conversation."""

    answer: str | None = Field(
        default=None,
        description="The responder's current answer. Rewritten on each revision.",
    )
    """Kept as its own field rather than read off the end of `messages`, because
    the verifier appends its critique after the answer — so the last message is
    routinely *not* the thing the user asked for. This is what `ask()` returns."""

    reply: str | None = Field(
        default=None,
        description="The finished user-facing reply. Set by whichever node ends the run.",
    )
    """The answer plus the ticket offer, when there is one — see
    agent/conversation.py, which composes it. A second field rather than an
    edit to `answer`, and that separation is load-bearing: `answer` is the
    artifact the verifier graded, and appending an offer to it after grading
    would mean the string that was checked is not the string that goes out.
    Callers read this; nodes write it exactly once, at the end."""

    verdict: Literal["pass", "fail", "ungraded"] | None = Field(
        default=None,
        description="The verifier's judgement on the current answer.",
    )
    """"ungraded" is not a judgement — it means the verifier itself failed and
    the answer went out unchecked. Kept distinct from "pass" so nothing downstream
    can report an answer as verified when nothing verified it."""

    revisions: int = Field(
        default=0,
        ge=0,
        description="How many times the verifier has sent the answer back.",
    )

    ticket_recipient: str = Field(
        default="",
        description=(
            "The role this answer says to raise the matter with, if any. Empty "
            "when there is nobody to raise it with, or nothing to raise."
        ),
    )
    """Set by the verifier, read by the CLI to decide whether to offer a ticket.
    It is deliberately not a boolean: naming the role lets the offer say who the
    ticket would go to, and gives something to check the string against. Written
    on every verifier pass including the failing ones, so a recipient from an
    earlier draft cannot survive into a later one that no longer names it."""
