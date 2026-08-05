"""Conversations, kept per session id.

The CLI holds its transcript in a local variable, because a CLI is one
conversation and it ends when the process does. Over HTTP there is no such
variable: each request arrives with no memory of the last one, and the agent's
ticket flow depends on there being one — "yes, file it" is an answer to
something, and without the previous turn it is an answer to nothing.

So the server keeps the transcript and the client keeps only an id. That is a
choice worth naming, because the alternative is real: the client could send the
history back on every request, which makes the server stateless and lets any
process serve any turn. It is rejected here because the history is the agent's
own record of what it offered, and a client that can edit it can put words in
the agent's mouth — and because it would move the token cost of a conversation
into every request body.

What is stored is exactly what the CLI stores: the questions, and the replies as
the user saw them (offer included). Not the evidence behind an answer, which
runs to tens of thousands of tokens, is re-retrievable, and is deliberately
re-gathered per turn — see `agent.graph.run_workflow` on why the working state
starts fresh while the conversation does not.

Same caveats as jobs.py: in memory, bounded, gone on restart. The upgrade is
Redis or a LangGraph checkpointer behind the same interface.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Session:
    """One conversation: its transcript, and the lock serializing its turns."""

    id: str
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)
    turns: int = 0
    history: list[AnyMessage] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    """Held for the whole of a turn. Two requests on one session id are not a
    hypothetical — a Swagger tab and a retry will do it — and without this they
    both read the same history, both run, and both append, leaving a transcript
    where two questions are followed by two answers in an order that matches
    neither. Concurrency between *different* sessions is untouched."""

    def record(self, question: str, reply: str, max_turns: int) -> None:
        """Append both ends of a turn, and forget the oldest ones past the cap.

        The reply stored is the one the user was shown, offer and all. Trimming
        is by turn (a question and its reply together) so the history never
        starts with a dangling answer to a question that is no longer there.
        """
        self.history.append(HumanMessage(question))
        self.history.append(AIMessage(reply))
        self.turns += 1
        self.updated_at = _now()
        if max_turns > 0 and len(self.history) > max_turns * 2:
            del self.history[: len(self.history) - max_turns * 2]


class SessionStore:
    """Every live conversation, oldest evicted once `max_sessions` is reached."""

    def __init__(self, max_sessions: int = 100, max_turns: int = 20) -> None:
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._max_sessions = max_sessions
        self.max_turns = max_turns

    def open(self, session_id: str | None = None) -> Session:
        """The named session, or a new one. Creates on an unknown id.

        An unknown id creates rather than 404s: a client that kept an id across
        a server restart should be able to carry on talking, and the failure
        mode of the alternative — an error where the user expected a reply — is
        worse than the failure mode of this one, which is a fresh start.
        """
        session_id = session_id or f"api-{uuid.uuid4().hex[:8]}"
        session = self._sessions.get(session_id)
        if session is None:
            session = Session(id=session_id)
            self._sessions[session_id] = session
            self._evict()
        # Most-recently used goes last, so eviction takes the coldest.
        self._sessions.move_to_end(session_id)
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def list(self) -> list[Session]:
        """Every remembered conversation, most recently used first."""
        return list(reversed(self._sessions.values()))

    def delete(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def _evict(self) -> None:
        while len(self._sessions) > self._max_sessions:
            self._sessions.popitem(last=False)
