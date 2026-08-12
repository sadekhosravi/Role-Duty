"""The verifier: decides whether the answer is grounded in what was retrieved.

It grades exactly one thing — does every claim trace to evidence gathered in
this run — and deliberately not whether the answer is good, complete or useful.
An evaluator loop terminates only if its criterion is answerable, and "is this a
good answer" never runs out of ways to say no.

No tools, for the same reason the responder has none. If the verifier could
retrieve, it would be checking the answer against evidence the responder never
had, which turns a grounding check into a second opinion on the truth — a
different job, and an unbounded one.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel, ConfigDict, Field

from ..llm import build_structured_llm
from ..spec import NodeSpec
from ..state import VERIFIER, AgentState

log = logging.getLogger(__name__)

PROMPT = """\
You check answers about an organisation's roles, duties, reporting lines and
authority thresholds against the evidence that was retrieved to write them.

You grade ONE thing: is every claim in the answer supported by evidence present
in this conversation. You are not judging whether the answer is well written,
thorough, helpful, or whether you would have answered differently.

FAIL the answer if any of these is true

  - It names a role, unit, document or threshold that appears in no retrieved
document text. An invented job title is the most common failure in this corpus,
and a title that only ever appeared in graph structure counts as invented.
  - A factual claim rests on general knowledge about how organisations work
rather than on retrieved text — that a manager approves what their reports
cannot, that escalation implies authority, that seniority implies scope.
  - It claims a role holds final authority over something that role's stated
exclusions rule out, or treats an escalation path reaching a role as proof that
the role decides.
  - It states a rule more or less firmly than the source does — turning "must"
into "may", or an interim step into a final one.
  - It cites a section that never appeared in the gathered evidence, or attaches
a citation to a claim that section does not actually state.
  - A reference is not a full source label. A label names the file, the page and
the section — "sample_role_duties.pdf › page 2 › Shift Supervisor › Key
Responsibilities". A bare heading like "Shift Supervisor, Key Responsibilities"
does not identify a section anyone can find, and neither does a bare number
carried over from a tool result. Fail this only when the evidence actually
supplied the full label; where a store keeps no page or section, the shorter
label it gave is correct.
  - It makes a claim about what the documents contain and has no
`### References` section. This is required on every such answer, including a
one-sentence one.
  - It carries a `### References` heading with nothing under it. An empty
section promises sources and delivers none, which is a defect whether or not the
reply needed sources at all.

PASS the answer if none of those is true. In particular, PASS these

  - An answer that says the documents do not cover the question, when the
retrieval genuinely came back without it. That is the correct answer to that
situation, and failing it cannot produce a better one.
  - A short answer, if the question was small. Brevity is not a defect.
  - A reply that claims nothing about the documents and carries no References
section — a greeting, an acknowledgement, a question back to the user. There is
nothing in it to ground, so there is nothing in it to fail. Requiring a citation
here is what produces the empty heading listed above.
  - An answer that leaves out something interesting but unasked.
  - An answer whose wording you would improve. Style is not your concern.

The bar is support, not perfection. Failing an answer that no further retrieval
could improve spends the team's remaining turns and ends with a worse answer
than the one you rejected.

When you fail an answer, name the gap precisely enough to act on: which claim is
unsupported, and what would have to be found to support it — a specific title to
confirm, a threshold to locate, an exclusion to check. "Needs more evidence" is
not usable. "The title 'Assistant Manager' appears in no retrieved section;
confirm the actual title of the role that approves refunds above the threshold"
is.

ONE THING YOU ARE NOT GRADING

Alongside the verdict you report `ticket_recipient`: who, if anyone, this should
be raised with. It is an observation, not a criterion. It changes nothing about
pass or fail, an answer with no recipient is not worse for it, and you must not
let it pull the verdict either way.

It exists because the user is offered a ticket after the answer, and offering
one is only sensible when there is a situation to report and someone to report
it to. So fill it only when the person described something they have to act on
AND the answer names the role to take it to.

  "Hello."
      -> empty. Nothing happened and there is nobody to tell.
  "Who does the Duty Manager report to?"
      -> empty. A question about the organisation, not a situation.
  "I want to donate a painting to the museum, who do I speak to?" answered with
  "the documents name no role that handles donations"
      -> empty. There is a situation, but the answer found nobody.
  "I am a housekeeper, I found a knife in a room" answered with "the documents
  do not give the full procedure, but the Housekeeping Supervisor
  (Housekeeping) escalates anything requiring police notification"
      -> "Housekeeping Supervisor (Housekeeping)". A role that the matter goes
         to is a role to raise it with, even when the procedure around it is
         incomplete. Do not read "the documents do not specify X" as "nobody was
         identified" — read what the answer actually names.
"""


class Verdict(BaseModel):
    """The verifier's judgement on one answer."""

    model_config = ConfigDict(extra="forbid")

    # Listed first on purpose: the model fills fields in order, so it has to
    # enumerate what it found before it is allowed to announce a verdict. A
    # verdict written first tends to be a first impression the rest justifies.
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description=(
            "Claims in the answer that no retrieved evidence supports, quoted "
            "or closely paraphrased. Empty if every claim is supported."
        ),
    )
    verdict: Literal["pass", "fail"] = Field(
        description="Whether the answer is grounded in the gathered evidence.",
    )
    gap: str = Field(
        default="",
        description=(
            "On a fail: what has to be found to support the answer, named "
            "precisely enough for a worker to act on. Empty on a pass."
        ),
    )
    # Last on purpose. It is not a grading criterion and it must not colour the
    # verdict — a model that decides "no ticket" first will start reading the
    # answer as thin. Answering it after the verdict is settled keeps it what it
    # is: an observation about an answer already judged.
    ticket_recipient: str = Field(
        default="",
        description=(
            "Who this should be raised with, if anyone. Fill it when BOTH hold: "
            "the person described something they have to act on — something "
            "found, something that happened, a situation they do not know how to "
            "handle — AND the answer names a role to take it to. Give that "
            "role's title exactly as the answer writes it. A role still counts "
            "when the answer also says the documents do not spell out the full "
            "procedure: naming who the matter escalates to IS naming who to "
            "raise it with. Leave it EMPTY for a greeting or small talk, for a "
            "general question about how the organisation works or who reports to "
            "whom, and when the answer names no role at all. Empty is a normal "
            "outcome and does not affect your verdict either way."
        ),
    )


_grader = None


def _get_grader():
    """The model, constrained to return a Verdict. Built on first use."""
    global _grader
    if _grader is None:
        _grader = build_structured_llm(Verdict)
    return _grader


# A source label: a pdf filename, then any number of " › "-joined parts. Used to
# harvest every label the tools actually produced, and to read the ones an answer
# claims. Deliberately greedy about the tail — labels end at a line break or a
# bracket, never mid-section.
_LABEL = re.compile(r"[^\s\[\]›]+\.pdf(?:\s*›\s*[^\n\[\]]+?)?(?=\s*[\[\n]|$)")


def _known_labels(messages: list) -> set[str]:
    """Every citation label that actually appeared in a tool result.

    Only ToolMessages count. What the model wrote about a label is not evidence
    that the label exists — that is the whole point of checking.
    """
    labels: set[str] = set()
    for message in messages:
        if isinstance(message, ToolMessage):
            labels.update(m.group(0).strip() for m in _LABEL.finditer(str(message.content)))
    return labels


# A reference line: an optional bullet, a number in any of the shapes a model
# reaches for — [1], 1., 1) — then the label it claims. Matching only one shape
# is not a stricter check, it is a check that silently stops running: an answer
# that numbers its references "1." instead of "- [1]" would present no citations
# to compare, and every label in it would pass unexamined.
_REF_LINE = re.compile(
    r"^[ \t]*(?:[-*+][ \t]*)?(?:\[\d+\]|\d+[.)])[ \t]*(.+?)[ \t]*$", re.MULTILINE
)


def _normalise(label: str) -> str:
    """Compare labels on their content, not their decoration.

    Bold markers, backticks, trailing punctuation and stray whitespace are
    formatting; failing an otherwise-correct citation over one would spend a
    revision on nothing. The trailing colon matters in particular: several real
    section labels end in one ("Out of Scope:") and answers routinely drop it.
    """
    return " ".join(label.replace("*", "").replace("`", "").strip(" .:").split())


def _cited_labels(answer: str) -> list[str]:
    """What each line of an answer's References section claims to cite."""
    _, _, references = answer.partition("### References")
    return [match.group(1) for match in _REF_LINE.finditer(references)]


def _citation_problems(answer: str, messages: list) -> list[str]:
    """Everything wrong with an answer's citations, decided without a model.

    This is the one grounding check that must not be left to a model. Prompting
    failed twice: first the responder built plausible labels — right shape,
    shifted page, a section that does not exist — and when told not to, it
    stripped the file and page instead of copying them. Both are caught here,
    because both are decidable by string comparison, and string comparison is
    not persuadable.

    An unreadable References section is reported as a problem in its own right,
    rather than passing. That case is the whole reason this function returns
    problems instead of mismatches: a citation check that cannot find any
    citations produces exactly the same empty result as one that found them all
    correct, so an unrecognised reference format would silently switch the check
    off and take every fabricated label through with it. It has done so once
    already, and the answer that slipped past cited a section from the wrong
    page. Requiring the section to be readable turns that silence into a fail.

    Returns nothing when the tools produced no labels at all: there is then
    nothing to check against, and an answer that correctly reports finding
    nothing should not be failed for citing nothing.

    One limitation worth knowing: an answer where SOME reference lines parse and
    others do not is only checked on the ones that parsed. Detecting the rest
    would mean guessing at the format again, which is the problem this is
    working around.
    """
    known = {_normalise(label) for label in _known_labels(messages)}
    if not known:
        return []

    cited = _cited_labels(answer)
    if "### References" in answer and not cited:
        return [
            "the References section is present but nothing in it could be read "
            "as a citation, so none of it could be checked"
        ]
    return sorted(
        {
            f"reference {line.strip()!r} is not a label any tool returned"
            for line in cited
            if _normalise(line) not in known
        }
    )


def _grading_request(answer: str) -> str:
    """Point the grader at the answer explicitly.

    The answer is already in the conversation, but on a revision there are
    several answers in there and the critique of an earlier one sits between
    them. Restating the current one removes the ambiguity about which is being
    graded.
    """
    return (
        "Grade the answer below against the evidence retrieved earlier in this "
        "conversation.\n\n--- ANSWER ---\n" + answer + "\n--- END ANSWER ---"
    )


def _critique(verdict: Verdict) -> str:
    """The rejection, written for the orchestrator and the responder to read."""
    lines = ["[verifier] The answer was rejected as not fully grounded."]
    if verdict.unsupported_claims:
        lines.append("Unsupported:")
        lines.extend(f"  - {claim}" for claim in verdict.unsupported_claims)
    if verdict.gap:
        lines.append(f"To close it: {verdict.gap}")
    return "\n".join(lines)


async def run(state: AgentState) -> dict:
    """Grade the current answer, and on a failure say what is missing.

    The critique is appended to the conversation rather than kept in a field:
    the orchestrator needs it to pick who can close the gap, and the responder
    needs it to know what to repair.
    """
    if not state.answer:
        # Nothing to grade. Reachable only if the responder produced nothing,
        # which is a real failure — but crashing here would lose the run, and
        # the revision cap already bounds what happens next.
        return {
            "verdict": "fail",
            "revisions": state.revisions + 1,
            "ticket_recipient": "",
            "messages": [HumanMessage("[verifier] No answer was produced to grade.")],
        }

    # Deterministic first. A fabricated label is decidable by string comparison,
    # so it is decided that way — no model call, no chance of the grader being
    # talked round by an answer that reads well.
    problems = _citation_problems(state.answer, state.messages)
    if problems:
        # The rejection carries the valid labels, not just the invalid ones.
        # Telling the responder its citation is wrong leaves it guessing again;
        # handing it the list the tools produced turns the fix into copying.
        valid = sorted(_known_labels(state.messages))
        return {
            "verdict": "fail",
            "revisions": state.revisions + 1,
            # Cleared, not merely left out. A partial update merges, so an
            # earlier draft's recipient would otherwise survive into a rejection
            # of a later draft that no longer names anyone.
            "ticket_recipient": "",
            "messages": [
                HumanMessage(
                    _critique(
                        Verdict(
                            unsupported_claims=problems,
                            verdict="fail",
                            gap=(
                                "Write the References section as one line per "
                                "source, each a dash, a bracketed number, then "
                                "one of the labels below copied exactly. Do not "
                                "shorten a label and do not rebuild one from "
                                "memory; drop any claim you cannot cite.\n"
                                "Valid labels from this run:\n"
                                + "\n".join(f"  {label}" for label in valid)
                            ),
                        )
                    )
                )
            ],
        }

    try:
        verdict: Verdict = await _get_grader().ainvoke(
            [("system", PROMPT), *state.messages, ("human", _grading_request(state.answer))]
        )
    except Exception as error:  # noqa: BLE001 - logged, then recovered from
        # A grader that failed has not judged anything. Reporting "ungraded"
        # rather than "pass" is the difference between an answer nobody checked
        # and an answer that was checked and accepted — the caller is told which
        # one they have. Failing instead would spend a revision re-answering a
        # question whose answer was never the problem.
        log.warning("grading call failed (%s); returning the answer ungraded", error)
        return {"verdict": "ungraded", "ticket_recipient": ""}

    # The recipient is only kept if it is a string the answer actually contains.
    # It is a role title produced by a model, and this project's whole history
    # with role titles says to check rather than trust — an offer to raise the
    # matter with a role that appears nowhere in the answer would be inventing a
    # recipient at the last possible moment, where nothing downstream looks.
    recipient = verdict.ticket_recipient.strip()
    if recipient and recipient not in state.answer:
        log.warning("verifier named a recipient absent from the answer: %r", recipient)
        recipient = ""

    if verdict.verdict == "pass":
        return {"verdict": "pass", "ticket_recipient": recipient}

    # The critique goes in as a user turn, not an assistant one: it is input to
    # the next pass, and providers differ on how they treat assistant messages
    # that the assistant did not write.
    return {
        "verdict": "fail",
        "revisions": state.revisions + 1,
        "ticket_recipient": recipient,
        "messages": [HumanMessage(_critique(verdict))],
    }


SPEC = NodeSpec(
    name=VERIFIER,
    system_prompt=PROMPT,
    runner=run,
    # No tools — see the module docstring. This was an open question when the
    # node was stubbed; the answer is that spot-checking would change the job.
)
