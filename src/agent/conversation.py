"""What a caller does with a finished run, before it shows the user anything.

The workflow returns an `AgentState`. Turning that into a reply involves one
decision that is not presentation — whether to follow the answer with the offer
to file a ticket — and that decision belongs to the agent, not to whichever
front end happens to be asking. It lived in scripts/agent.py while the CLI was
the only caller; it moved here when the HTTP API became the second, because the
alternative was two copies of a rule that has already been wrong once.

Everything here is pure: it reads a state and returns a string or a bool, with
no model call and no I/O, so the rules can be checked without running anything
(see scripts/check_agent.py).
"""

from __future__ import annotations

from .state import FILER, RESPONDER, AgentState

# Where the filer writes, as the offer describes it to the user. The filer's own
# tool reports the real path back after writing; this is only the promise.
TICKETS_LOCATION = "data/tickets/"


def ticket_offer(recipient: str) -> str:
    """The offer, naming who the ticket would go to.

    Naming them is not decoration. "a ticket for the responsible person" is a
    question the user cannot answer without re-reading the answer, and it reads
    as boilerplate — which is exactly what it was when this offer went out after
    every reply, including "Hello".
    """
    return (
        f"Would you like me to file a ticket for the {recipient}? "
        f"Say yes and I will write it to {TICKETS_LOCATION}."
    )


def should_offer_ticket(state: AgentState) -> bool:
    """Whether to follow this reply with the ticket offer.

    Its own function so the rule can be checked without running a model, and it
    has needed the checking. Offering unconditionally was wrong in two ways at
    once: after "Hello" there is nothing to file, and after an answer that
    could not find a responsible role there is nobody to file it with — the
    filer would have refused both, so the offer was writing a cheque the rest
    of the workflow would not honour.

    The condition is now the filer's own precondition, asked before the offer
    instead of after the yes: is there a situation to report, and did the
    answer establish who to report it to. The verifier decides that (it has the
    answer and the evidence in front of it already, so it costs no extra call)
    and `state.ticket_recipient` is what it decided.

    A failed or ungraded verdict blocks the offer too. An answer we have just
    warned the user is not fully grounded is not one to raise a ticket from.
    """
    return (
        bool(state.answer)
        and RESPONDER in state.delegations
        and FILER not in state.delegations
        and state.verdict == "pass"
        and bool(state.ticket_recipient)
    )


def reply_for(state: AgentState) -> str:
    """The whole user-facing reply for one turn: the answer, plus the offer.

    This is also what a caller should keep as the turn's transcript entry.
    Dropping the offer from what is remembered would leave the next turn's
    "yes" agreeing to nothing on record, which is the difference between the
    filer having something to file and it asking the user to start over.
    """
    answer = state.answer or "(no answer was produced)"
    if should_offer_ticket(state):
        return f"{answer}\n\n{ticket_offer(state.ticket_recipient)}"
    return answer
