"""The orchestrator: picks which worker runs next, and nothing else.

It is the only node that decides where control goes. Every worker reports back
here, and the run ends when it routes to the responder and the verifier accepts
what the responder wrote.

Two things keep it from looping, which is the documented failure mode of this
pattern. Its destination is a constrained Literal, so it cannot invent a worker
or name itself; and `state.delegations` records what it has already run and is
fed back to it every turn, so "researcher again" is a choice it has to justify
rather than the default. MAX_DELEGATIONS is the backstop under both.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field

from ..llm import build_structured_llm
from ..spec import NodeSpec
from ..state import (
    MAX_DELEGATIONS,
    MAX_REVISIONS,
    ORCHESTRATOR,
    RESEARCHER,
    RESPONDER,
    WORKERS,
    AgentState,
    WorkerName,
)

log = logging.getLogger(__name__)

PROMPT = """\
You are the orchestrator of a team that answers questions about an
organisation's role and duty documents: reporting lines, key responsibilities,
what each role is explicitly not allowed to do, and the authority thresholds
that decide who has to approve what.

You do not retrieve, you do not answer, and you do not talk to the user. Your
only output is a choice of which worker runs next. Choose one, wait for it to
report back, then choose again.


YOUR WORKERS

researcher
    Searches the documents three ways, cheapest first: plain semantic search
    over passages, then the knowledge graph of roles and reporting lines if
    that did not settle it, plus exact keyword search to confirm titles and
    numbers. Reports back cited evidence.
    Send it: whenever the request needs a fact from the documents. A role's
    duties, who someone reports to, whether an action is out of scope, what a
    threshold is, who approves something. This is almost always the first
    worker to run.
    Do not send it: to re-fetch something already reported in this
    conversation, or with the same wording that already came back empty.

gap_auditor
    Sweeps the knowledge graph for structural problems across the whole role
    set: duties nobody owns, duties two roles both claim, and escalation paths
    that point at a role that does not exist.
    Send it: for questions about the role set as a whole. Coverage, overlap,
    "is anything unassigned", "do these two roles conflict", "audit this".
    Do not send it: for a question about one named role or one specific duty.
    That is a lookup, and the researcher does it better.

notifier
    Delivers a finding to a person or a channel outside this conversation.
    Send it: only when the user actually asked for something to be sent, and
    only after the finding it would send has been established.
    Do not send it: to make progress on a question. It gathers nothing. If you
    are unsure whether the user wanted something sent, they did not.

responder
    Writes the final answer from the evidence already gathered, and hands it to
    the verifier.
    Send it: when the evidence on the table actually answers the request.
    Choosing the responder is how you finish; there is no separate stop signal.
    Do not send it: as the first worker. An answer with no retrieved evidence
    behind it is the single worst outcome this team can produce.


HOW TO CHOOSE

Read the request, then read what has already come back, then pick the one
worker that moves things forward most.

Never pick the responder first. Every answer must rest on evidence somebody
retrieved in this run.

Do not repeat a worker unless you can say what new thing it should look for
this time. Repeating the researcher with the same target is the most common way
this team wastes a turn.

If a worker came back with nothing useful, that is information. Either send a
different worker, or send the same one after a genuinely different angle — a
role title instead of a description, an exact phrase instead of a paraphrase.
If two workers have failed to find something, the documents probably do not
contain it, and the responder should say so plainly rather than keep digging.

If the verifier rejected the last answer, read why. Send the work to whoever
can close that specific gap — usually the researcher with a narrow, named
target. Do not send it straight back to the responder; nothing will have
changed, and the same answer will be rejected again.

When you are told you are near your delegation limit, stop gathering and pick
the responder. An answer with acknowledged gaps beats no answer.


WRITING THE BRIEF

Every delegation carries a brief, and the worker sees only that — not your
reasoning, not the question you were asked. It is an instruction, addressed to
the worker, naming the specific thing to do this turn.

A brief that restates the topic is useless. "Research the refund question" tells
a worker that already reported nothing it did not know, and it will report the
same thing again. Name the missing piece:

  Bad:  Find out about UPS bypass authority.
  Good: Confirm the exact job title of the role that authorises a UPS bypass
        using keyword_search, and find the numeric limit stated in its Out of
        Scope section.

On a REPEAT delegation to a worker that has already run, the brief must name
what is new — what was not established last time, a different tool to try, a
different phrasing to search. A worker re-run without a new target will do
nothing, and the turn is wasted. If you cannot name something new for it to do,
that worker is finished; pick a different one or pick the responder.

If the verifier rejected the answer, put the gap it named into the brief.


EXAMPLES

Request: "Who signs off a 700 dollar refund?"
    -> researcher: "Find the refund approval thresholds and the exact title of
       the role that approves above them."

The researcher has reported the threshold and the approving role.
    -> responder: "Answer the question from the thresholds and titles reported."

The researcher's cheap semantic search returned passages naming a role, but not
the escalation path the question actually turns on.
    -> researcher: "Escalate to graph_rag_search for the escalation path above
       the Store Associate, and get chunk text for each hop."

The researcher reported the approving role but no exact title from chunk text.
    -> researcher: "Confirm the exact title of that role with keyword_search.
       The graph gave a name that has not been checked against document text."

Request: "Is there anything in these documents that nobody owns?"
    -> gap_auditor: "Sweep for duties with no owning role across all
       organisations, and report them grouped by document."

Request: "Check whether anyone owns generator testing, and tell the Site
Operations Manager if not."
    -> gap_auditor first: "Establish whether any role owns generator testing at
       the data centre." The notifier has nothing to send until that exists.

The verifier rejected the answer: it named an "Assistant Manager" that appears
in no cited source.
    -> researcher: "The title 'Assistant Manager' is unsupported. Find the real
       title of the role that approves refunds above the threshold."

The researcher has twice come back empty on a duty the user asked about.
    -> responder: "Say the documents do not cover this duty, and name what was
       searched for." Do not send the researcher a third time.
"""


class Route(BaseModel):
    """The orchestrator's decision for this turn: a task, and who does it."""

    model_config = ConfigDict(extra="forbid")

    # `brief` is declared first on purpose: the model fills fields in order, so
    # it has to work out what needs doing before it picks who does it — which is
    # the right order, and stops the choice being a first impression the rest
    # justifies. Unlike the justification this replaced, the brief is kept: it
    # is handed to the worker, which is the only thing that makes a second
    # delegation to the same worker mean anything.
    brief: str = Field(
        min_length=1,
        description=(
            "The task for the worker, addressed to it. Name the specific thing "
            "to find or do this turn, not the general topic. One or two "
            "sentences."
        ),
    )
    next_node: WorkerName = Field(
        description="The worker to run next.",
    )


_router = None


def _get_router():
    """The model, constrained to return a Route. Built on first use.

    Structured output rather than a free-text answer parsed afterwards: the
    destination is a Literal, so it reaches the model as an enum and an invented
    worker name is rejected before it can be routed on.
    """
    global _router
    if _router is None:
        _router = build_structured_llm(Route)
    return _router


def _fallback(state: AgentState) -> WorkerName:
    """Where to go when the routing call itself fails.

    A flaky structured-output call should cost a turn, not the request. The
    choice is the one that cannot make things worse: gather if nothing has been
    gathered, otherwise write the answer with what is already there.
    """
    return RESEARCHER if not state.delegations else RESPONDER


def _progress(state: AgentState) -> str:
    """What has happened so far, written for the model to read.

    The orchestrator cannot infer its own history from the conversation alone —
    a worker that reports nothing leaves no trace there — so its record of what
    it has already run is stated explicitly every turn.
    """
    lines = []
    if state.delegations:
        lines.append("Workers run so far, in order: " + ", ".join(state.delegations) + ".")
    else:
        lines.append("No worker has run yet.")

    if state.verdict == "fail":
        lines.append(
            f"The verifier rejected the last answer "
            f"(revision {state.revisions} of {MAX_REVISIONS}). "
            f"Send the work to whoever can close the gap it named."
        )

    remaining = MAX_DELEGATIONS - len(state.delegations)
    lines.append(
        f"You may delegate {remaining} more time(s) before the answer must be written."
        if remaining > 1
        else "This is your last delegation. Pick the responder unless something is badly wrong."
    )
    return "\n".join(lines)


def _delegation(target: str, brief: str) -> HumanMessage:
    """The brief, addressed to the worker that is about to run.

    Delegating used to pass nothing but a destination, on the reasoning that
    routing chatter would only clutter the conversation. That was wrong, and
    visibly so: a worker re-invoked with no task saw its own finished report and
    correctly did nothing, seven times in a row. The brief is the difference
    between "run again" and "run again, and this time find X".

    It goes in as a user turn — providers differ on how they treat assistant
    messages the assistant did not write, and this has to be read reliably.
    """
    return HumanMessage(f"[orchestrator → {target}] {brief}")


async def run(state: AgentState) -> dict:
    """Choose the next worker, and tell it what to do."""
    # The cap is enforced here rather than in the router so that hitting it does
    # not cost a model call, and so the decision is recorded like any other.
    if len(state.delegations) >= MAX_DELEGATIONS:
        brief = (
            "The retrieval budget for this question is spent. Write the answer "
            "from what has been gathered, and be explicit about what is missing."
        )
        return {
            "next_node": RESPONDER,
            "delegations": [RESPONDER],
            "messages": [_delegation(RESPONDER, brief)],
        }

    # The progress note goes in as a user turn rather than a second system
    # message: multiple system messages are handled inconsistently across
    # providers, and this has to be read reliably every turn.
    try:
        route: Route = await _get_router().ainvoke(
            [("system", PROMPT), *state.messages, ("human", _progress(state))]
        )
        target, brief = route.next_node, route.brief
    except Exception as error:  # noqa: BLE001 - logged, then recovered from
        target = _fallback(state)
        brief = (
            "Continue with the question as asked; the routing step failed, so "
            "this brief is a default rather than a specific instruction."
        )
        log.warning("routing call failed (%s); falling back to %s", error, target)

    return {
        "next_node": target,
        "delegations": [target],
        "messages": [_delegation(target, brief)],
    }


SPEC = NodeSpec(
    name=ORCHESTRATOR,
    system_prompt=PROMPT,
    runner=run,
    # No tools on purpose: it delegates, it does not do the work. The one thing
    # it is allowed to produce is a destination.
)

# The prompt has to describe every worker it can be asked to choose. A worker
# added to WorkerName but not written up above would be a destination the
# orchestrator has never been told about — it would still be routable, and the
# model would have no idea what it does. Cheap to check, so check it.
_undescribed = [name for name in WORKERS if name not in PROMPT]
if _undescribed:
    raise RuntimeError(
        f"orchestrator prompt does not describe: {', '.join(_undescribed)}"
    )
