"""CLI: ask the agentic RAG workflow a question.

Usage:
    python scripts/agent.py "Who authorizes a UPS bypass?"
    python scripts/agent.py "..." --trace              # show which tools it called
    python scripts/agent.py "..." --session demo-1     # group runs in Langfuse
    python scripts/agent.py "..." --tag baseline       # label the run (repeatable)
    python scripts/agent.py --check-tracing            # verify Langfuse, then exit
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Make the src/ packages importable when running as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent import observability
from agent.graph import run_workflow


async def run(args: argparse.Namespace) -> None:
    state, trace_url = await run_workflow(
        args.question,
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

    print(state.answer or "(no answer was produced)")

    # Flushed before the URL is printed, not after: Langfuse batches in a
    # background thread, and a link offered before the trace has been sent is a
    # link to a 404 for however long the queue takes to drain.
    if trace_url:
        observability.flush()
        print(f"\ntrace: {trace_url}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the agentic RAG workflow.")
    parser.add_argument("question", nargs="?", help="Your natural-language question.")
    parser.add_argument(
        "--trace", action="store_true", help="Print the tool calls it made."
    )
    parser.add_argument(
        "--session",
        help="Langfuse session id. Runs sharing one are grouped as a conversation.",
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

    if not args.question:
        parser.error("a question is required (or use --check-tracing)")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
