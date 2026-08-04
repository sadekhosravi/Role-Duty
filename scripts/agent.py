"""CLI: ask the agentic RAG workflow a question.

Usage:
    python scripts/agent.py "Who authorizes a UPS bypass?"
    python scripts/agent.py "..." --trace     # show which tools it called
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Make the src/ packages importable when running as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.messages import HumanMessage

from agent.graph import build_graph
from agent.state import AgentState


async def run(question: str, trace: bool) -> None:
    result = await build_graph().ainvoke(AgentState(messages=[HumanMessage(question)]))
    # ainvoke hands back a plain dict, not an AgentState — see state.py.
    state = AgentState(**result)

    if trace:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the agentic RAG workflow.")
    parser.add_argument("question", help="Your natural-language question.")
    parser.add_argument(
        "--trace", action="store_true", help="Print the tool calls it made."
    )
    args = parser.parse_args()
    asyncio.run(run(args.question, args.trace))


if __name__ == "__main__":
    main()
