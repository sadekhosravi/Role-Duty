"""CLI: a conversation with the agentic RAG workflow.

Usage:
    python scripts/agent.py                            # start a conversation
    python scripts/agent.py "Who authorizes a UPS bypass?"   # opens with that
    python scripts/agent.py "..." --once               # answer once and exit
    python scripts/agent.py "..." --trace              # show which tools it ran
    python scripts/agent.py "..." --session demo-1     # group runs in Langfuse
    python scripts/agent.py "..." --tag baseline       # label the run (repeatable)
    python scripts/agent.py --check-tracing            # verify Langfuse, then exit

The conversation is the point, not a convenience. When an answer establishes
someone to raise a matter with, the agent offers to file them a ticket, and
"yes" only means something if the previous turn is still there to mean it about.
This module keeps that transcript — the questions and the replies, and nothing
else. The evidence behind an answer is not carried forward: it can run to tens
of thousands of tokens, it is re-retrievable, and each turn's working state is
deliberately fresh (see graph.run_workflow).
"""

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

# Make the src/ packages importable when running as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from agent import observability
from agent.conversation import reply_for, should_offer_ticket, ticket_offer
from agent.graph import run_workflow

PROMPT = "\nyou> "
QUIT = {"exit", "quit", ":q", "bye"}

# Re-exported so the rules stay checkable through this module (check_agent.py
# loads it by path and asks it what it would offer). They live in
# agent/conversation.py because the HTTP API applies the same ones, and a rule
# maintained in two places is a rule maintained in one.
__all__ = ["reply_for", "should_offer_ticket", "ticket_offer"]


async def turn(
    question: str,
    history: list[AnyMessage],
    args: argparse.Namespace,
) -> None:
    """Run one turn, print the reply, and record both ends of it in `history`."""
    state, trace_url = await run_workflow(
        question,
        history=history,
        session_id=args.session,
        user_id=args.user,
        tags=args.tag or None,
    )

    if args.trace:
        print(f"  route: {' -> '.join(state.delegations)}", file=sys.stderr)
        for message in state.messages:
            for call in getattr(message, "tool_calls", None) or []:
                print(f"  -> {call['name']}({call['args']})", file=sys.stderr)
        print(file=sys.stderr)

    # The verifier's judgement belongs on stderr, not in the answer: a rejected
    # answer is still returned — the revision cap ends the loop rather than the
    # answer becoming acceptable — so the caller has to be told which it got.
    if state.verdict == "fail":
        print(
            "warning: the verifier rejected this answer as not fully grounded.",
            file=sys.stderr,
        )
    elif state.verdict == "ungraded":
        print(
            "warning: the verifier failed to run — this answer was not checked.",
            file=sys.stderr,
        )

    reply = reply_for(state)

    print(f"\n{reply}")

    # Both ends go in, and the offer goes in with the reply. Dropping it would
    # leave the next turn's "yes" agreeing to nothing on record.
    history.append(HumanMessage(question))
    history.append(AIMessage(reply))

    # Flushed before the URL is printed, not after: Langfuse batches in a
    # background thread, and a link offered before the trace has been sent is a
    # link to a 404 for however long the queue takes to drain.
    if trace_url:
        observability.flush()
        print(f"\ntrace: {trace_url}", file=sys.stderr)


async def converse(args: argparse.Namespace) -> None:
    """Answer the opening question if there is one, then keep taking turns."""
    history: list[AnyMessage] = []

    if args.question:
        await turn(args.question, history, args)
        if args.once:
            return

    print(f"\n(type {' or '.join(sorted(QUIT))} to leave, or Ctrl-C)", file=sys.stderr)
    while True:
        try:
            question = input(PROMPT).strip()
        except (EOFError, KeyboardInterrupt):
            # A piped stdin that runs out, or Ctrl-C. Both mean "done", and
            # neither is an error worth a traceback.
            print(file=sys.stderr)
            return
        if not question:
            continue
        if question.lower() in QUIT:
            return
        await turn(question, history, args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Talk to the agentic RAG workflow.")
    parser.add_argument(
        "question",
        nargs="?",
        help="An opening question. Without one, the conversation starts empty.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Answer the opening question and exit instead of continuing.",
    )
    parser.add_argument(
        "--trace", action="store_true", help="Print the tool calls it made."
    )
    parser.add_argument(
        "--session",
        help=(
            "Langfuse session id. Every turn of this conversation is filed "
            "under it. Defaults to a new id per conversation."
        ),
    )
    parser.add_argument("--user", help="Langfuse user id to attribute the run to.")
    parser.add_argument(
        "--tag",
        action="append",
        help="Langfuse tag for this run. Repeatable — use it to mark experiments.",
    )
    parser.add_argument(
        "--check-tracing",
        action="store_true",
        help="Check the Langfuse connection and exit without asking anything.",
    )
    args = parser.parse_args()

    if args.check_tracing:
        ok, message = observability.auth_check()
        print(("ok: " if ok else "not tracing: ") + message)
        raise SystemExit(0 if ok else 1)

    if args.once and not args.question:
        parser.error("--once needs a question to answer")

    # One session id for the whole conversation, so its turns are grouped in
    # Langfuse rather than arriving as unrelated traces. Generated here because
    # this is what knows where a conversation begins and ends.
    args.session = args.session or f"cli-{uuid.uuid4().hex[:8]}"

    asyncio.run(converse(args))


if __name__ == "__main__":
    main()
