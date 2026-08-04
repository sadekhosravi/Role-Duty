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

from pydantic import BaseModel, ConfigDict, Field

from ..llm import build_llm
from ..spec import NodeSpec
from ..state import (
    MAX_DELEGATIONS,
    MAX_REVISIONS,
    ORCHESTRATOR,
    RESPONDER,
    WORKERS,
    AgentState,
    WorkerName,
)

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
    Searches the documents three ways — the knowledge graph of roles and
    reporting lines, semantic search over passages, and exact keyword search —
    and reports back cited evidence.
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


EXAMPLES

Request: "Who signs off a 700 dollar refund?"
    -> researcher. Needs both the threshold and the role that holds it.

The researcher has reported the threshold and the approving role.
    -> responder. The evidence answers the question.

Request: "Is there anything in these documents that nobody owns?"
    -> gap_auditor. It is a question about coverage across the whole role set,
       not about any one role.

Request: "Check whether anyone owns generator testing, and tell the Site
Operations Manager if not."
    -> gap_auditor first. The notifier has nothing to send until the finding
       exists.

The verifier rejected the answer: it named an "Assistant Manager" that appears
in no cited source.
    -> researcher, to establish the real title of that role. Not the responder.

The researcher and the keyword search have both come back empty on a duty the
user asked about.
    -> responder, to say the documents do not cover it.
"""


class Route(BaseModel):
    """The orchestrator's decision for this turn."""

    model_config = ConfigDict(extra="forbid")

    # `reason` is declared first on purpose: the model fills the fields in
    # order, so it has to state a justification before it commits to a
    # destination. It is not stored — its whole job is to happen before the
    # choice rather than after it.
    reason: str = Field(
        min_length=1,
        description="One sentence: why this worker, and why now.",
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
        _router = build_llm().with_structured_output(Route)
    return _router


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


async def run(state: AgentState) -> dict:
    """Choose the next worker.

    Returns only the routing decision — the orchestrator deliberately adds
    nothing to the conversation, so workers read the user's request and each
    other's findings without routing chatter in between.
    """
    # The cap is enforced here rather than in the router so that hitting it does
    # not cost a model call, and so the decision is recorded like any other.
    if len(state.delegations) >= MAX_DELEGATIONS:
        return {"next_node": RESPONDER, "delegations": [RESPONDER]}

    # The progress note goes in as a user turn rather than a second system
    # message: multiple system messages are handled inconsistently across
    # providers, and this has to be read reliably every turn.
    route: Route = await _get_router().ainvoke(
        [("system", PROMPT), *state.messages, ("human", _progress(state))]
    )
    return {"next_node": route.next_node, "delegations": [route.next_node]}


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
