"""The agent, as a conversation.

This is the endpoint the other six exist around. `/query/graph` answers a
question; this runs the whole workflow — an orchestrator delegating to a
researcher, a responder and a verifier, and to a filer that can write an
incident ticket — and it does so across turns, which is what makes the filer
reachable at all. "Yes, file it" is an answer to something the agent said last
turn, and there is no last turn without a conversation.

Three things about the shape of this endpoint are decisions, not defaults:

* **One request, one turn, one reply.** Not a stream. Server-sent events would
  show the run progressing, and they would also make this untestable from the
  Swagger page the API is being built for. Streaming is a second endpoint when
  it is wanted, not a different shape for this one.
* **The server keeps the transcript, the client keeps an id.** See
  api/sessions.py for why, and for what that costs.
* **The verdict comes back with the answer.** The workflow returns rejected
  answers rather than nothing — the revision cap ends the loop whether or not
  the answer got better — so a client that ignores `verdict` will eventually
  show someone an answer the agent itself flagged as not fully grounded. The
  CLI prints a warning to stderr at this exact point; over HTTP the field is
  the warning.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from langchain_core.messages import AIMessage, HumanMessage

from agent import observability
from agent.conversation import final_reply, should_offer_ticket, ticket_offer
from agent.graph import run_workflow
from agent.state import AgentState

from ..deps import SessionsDep
from ..schemas import (
    AgentChatRequest,
    AgentChatResponse,
    ErrorResponse,
    SessionDetail,
    SessionList,
    SessionSummary,
    ToolCall,
    TranscriptTurn,
)
from ..sessions import Session

router = APIRouter(prefix="/agent", tags=["agent"])


def _tool_calls(state: AgentState) -> list[ToolCall]:
    """Every tool the run actually called, in order — the CLI's --trace."""
    return [
        ToolCall(name=call["name"], args=call["args"])
        for message in state.messages
        for call in getattr(message, "tool_calls", None) or []
    ]


def _summary(session: Session) -> SessionSummary:
    return SessionSummary(
        id=session.id,
        created_at=session.created_at,
        updated_at=session.updated_at,
        turns=session.turns,
    )


@router.post(
    "/chat",
    response_model=AgentChatResponse,
    summary="Start the agent — one turn of a conversation",
    description=(
        "Runs the full LangGraph workflow: the orchestrator delegates to the "
        "researcher (which searches the graph, the vector store and BM25), the "
        "responder writes an answer, and the verifier grades it against the "
        "evidence before it can leave.\n\n"
        "**Multi-turn.** Omit `session_id` on the first call and send back the "
        "one you get for every call after. The agent may offer to file an "
        "incident ticket; replying 'yes' on the next turn is what makes the "
        "filer write it, and that only works within a session.\n\n"
        "**Check `verdict`.** `pass` means the verifier found the answer "
        "grounded. `fail` means it did not and the answer is returned anyway. "
        "`ungraded` means the verifier never ran, so nothing checked it.\n\n"
        "Takes tens of seconds and makes many LLM calls — it is several models "
        "in a loop, not one completion."
    ),
)
async def chat(request: AgentChatRequest, sessions: SessionsDep) -> AgentChatResponse:
    session = sessions.open(request.session_id)

    # One turn at a time per conversation. Two requests on one session id would
    # otherwise both read the same history and both append to it, leaving a
    # transcript in an order that matches neither. Different sessions are
    # unaffected — the lock is the session's, not the store's.
    async with session.lock:
        state, trace_url = await run_workflow(
            request.message,
            history=list(session.history),
            # The API's session id is also the Langfuse session id, so a
            # conversation is one grouped thread in the dashboard rather than a
            # scattering of unrelated traces.
            session_id=session.id,
            user_id=request.user_id,
            tags=request.tags,
        )
        # Read off the state rather than recomposed here: the node that ended
        # the run built it, so this response and the CLI's differ in packaging
        # only, never in wording.
        reply = final_reply(state)
        # What is remembered is what the user was shown, offer included. Storing
        # the bare answer would leave the next turn's "yes" agreeing to nothing.
        session.record(request.message, reply, sessions.max_turns)
        turn = session.turns

    if trace_url:
        # Langfuse batches in a background thread; a URL handed out before the
        # trace has been sent is a link to a 404 until the queue drains.
        await run_in_threadpool(observability.flush)

    return AgentChatResponse(
        session_id=session.id,
        turn=turn,
        answer=state.answer or "(no answer was produced)",
        reply=reply,
        ticket_offer=(
            ticket_offer(state.ticket_recipient) if should_offer_ticket(state) else None
        ),
        ticket_recipient=state.ticket_recipient,
        verdict=state.verdict,
        revisions=state.revisions,
        trace_url=trace_url,
        route=list(state.delegations) if request.include_trace else None,
        tool_calls=_tool_calls(state) if request.include_trace else None,
    )


@router.get(
    "/sessions",
    response_model=SessionList,
    summary="Live conversations, most recently used first",
    description=(
        "In memory and bounded: the coldest conversation is dropped once "
        "API_MAX_SESSIONS is reached, and all of them are gone on restart."
    ),
)
async def list_sessions(sessions: SessionsDep) -> SessionList:
    return SessionList(sessions=[_summary(session) for session in sessions.list()])


@router.get(
    "/sessions/{session_id}",
    response_model=SessionDetail,
    responses={status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}},
    summary="One conversation, with its transcript",
    description=(
        "The transcript is what the agent is given on the next turn: the "
        "questions, and the replies as the user saw them. The evidence behind "
        "an answer is not kept — it runs to tens of thousands of tokens, it is "
        "re-retrievable, and each turn's research starts fresh on purpose."
    ),
)
async def get_session(session_id: str, sessions: SessionsDep) -> SessionDetail:
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no such session: {session_id}")

    transcript = [
        TranscriptTurn(
            role="user" if isinstance(message, HumanMessage) else "assistant",
            content=message.content if isinstance(message.content, str) else str(message.content),
        )
        for message in session.history
        if isinstance(message, (HumanMessage, AIMessage))
    ]
    return SessionDetail(**_summary(session).model_dump(), transcript=transcript)


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}},
    summary="Forget a conversation",
    description=(
        "Drops the transcript. The next call with that id starts a new "
        "conversation rather than failing — an unknown session id is always "
        "treated as a fresh one."
    ),
)
async def delete_session(session_id: str, sessions: SessionsDep) -> None:
    if not sessions.delete(session_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no such session: {session_id}")
