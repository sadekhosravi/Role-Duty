"""Check the agent's wiring without needing an API key.

Everything here runs offline except the last check. That is deliberate: the
graph shape, the tool boundaries, the routers and both loop bounds are all
decidable without a model, so they should be checkable without one too.

    python scripts/check_agent.py           # offline only
    python scripts/check_agent.py --live    # plus one real orchestrator call

The --live check is the only one that spends money, and it exercises the one
node that is actually implemented: it asks the orchestrator to route a real
question and prints what it chose and why.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from agent import WORKFLOW, AgentState, NodeSpec, Workflow, build_graph
from agent.nodes import orchestrator as orch
from agent.state import (
    MAX_DELEGATIONS,
    MAX_REVISIONS,
    ORCHESTRATOR,
    RESPONDER,
    WORKERS,
)
from agent.tools import graph_rag_search, keyword_search

PASSED, FAILED = [], []


def check(label: str, ok: bool, detail: str = "") -> None:
    (PASSED if ok else FAILED).append(label)
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{f' — {detail}' if detail else ''}")


def rejects(thunk) -> tuple[bool, str]:
    """True if the call raises a validation error, plus the message."""
    try:
        thunk()
    except (ValidationError, ValueError) as error:
        message = error.errors()[0]["msg"] if isinstance(error, ValidationError) else str(error)
        return True, message
    return False, "accepted, should not have"


# ---------------------------------------------------------------- structure


def check_structure() -> None:
    print("\nGraph structure")
    graph = build_graph().get_graph()
    names = set(graph.nodes)
    expected = {spec.name for spec in WORKFLOW.nodes}
    check("every declared node is in the compiled graph", expected <= names)
    check(
        "nodes with tools got their own executor",
        all(
            spec.tools_node in names
            for spec in WORKFLOW.nodes
            if spec.tools
        ),
    )

    edges = {(edge.source, edge.target) for edge in graph.edges}
    check("START enters at the orchestrator", ("__start__", ORCHESTRATOR) in edges)
    check(
        "orchestrator can reach every worker",
        all((ORCHESTRATOR, worker) in edges for worker in WORKERS),
    )
    check(
        "orchestrator cannot reach the verifier directly",
        (ORCHESTRATOR, "verifier") not in edges,
        "the answer is only ever graded after the responder writes it",
    )
    check("responder always hands to the verifier", (RESPONDER, "verifier") in edges)
    check("the verifier can end the run", ("verifier", "__end__") in edges)
    check("the verifier can send work back", ("verifier", ORCHESTRATOR) in edges)


def check_tool_boundaries() -> None:
    print("\nPer-node tool access")
    by_name = WORKFLOW.by_name
    check(
        "researcher has all three retrievers",
        len(by_name["researcher"].tools) == 3,
    )
    check(
        "gap_auditor is graph-only",
        [tool.name for tool in by_name["gap_auditor"].tools] == ["graph_rag_search"],
    )
    check(
        "orchestrator, responder and verifier have no tools",
        not any(by_name[name].tools for name in (ORCHESTRATOR, RESPONDER, "verifier")),
    )


# ------------------------------------------------------------------- routing


def check_routers() -> None:
    print("\nRouters")
    check(
        "no choice falls through to the responder",
        orch_route(AgentState()) == RESPONDER,
    )
    check(
        "a choice is honoured",
        orch_route(AgentState(next_node="researcher")) == "researcher",
    )
    check("a passing answer ends the run", verify_route(AgentState(verdict="pass")) == "__end__")
    check(
        "a failing answer goes back",
        verify_route(AgentState(verdict="fail", revisions=0)) == ORCHESTRATOR,
    )
    check(
        "a failing answer stops at the revision cap",
        verify_route(AgentState(verdict="fail", revisions=MAX_REVISIONS)) == "__end__",
        f"MAX_REVISIONS={MAX_REVISIONS}",
    )


def orch_route(state: AgentState) -> str:
    from agent.graph import _route_to_worker

    return _route_to_worker(state)


def verify_route(state: AgentState) -> str:
    from agent.graph import _route_after_verify

    return _route_after_verify(state)


def check_orchestrator() -> None:
    print("\nOrchestrator")
    check(
        "its prompt describes every worker it can pick",
        all(worker in orch.PROMPT for worker in WORKERS),
    )
    check(
        "its decision is constrained to an enum",
        orch.Route.model_json_schema()["properties"]["next_node"]["enum"] == list(WORKERS),
    )
    check(
        "it has to justify before it picks",
        list(orch.Route.model_fields) == ["reason", "next_node"],
        "field order is what forces reasoning first",
    )

    capped = AgentState(delegations=["researcher"] * MAX_DELEGATIONS)
    result = asyncio.run(orch.run(capped))
    check(
        "the delegation cap short-circuits with no model call",
        result["next_node"] == RESPONDER,
        f"MAX_DELEGATIONS={MAX_DELEGATIONS}",
    )
    check(
        "it is told what it has already run",
        "researcher" in orch._progress(AgentState(delegations=["researcher"])),
    )
    check(
        "it is told when the verifier rejected the answer",
        "rejected" in orch._progress(AgentState(verdict="fail", revisions=1)),
    )


# ---------------------------------------------------------------- validation


def check_declarations() -> None:
    print("\nDeclarations that should be refused")
    for label, thunk in [
        (
            "a node name that is not an identifier",
            lambda: NodeSpec(name="gap auditor", system_prompt="x", runner=orch.run),
        ),
        (
            "a reserved node name",
            lambda: NodeSpec(name="__start__", system_prompt="x", runner=orch.run),
        ),
        (
            "the same tool listed twice",
            lambda: NodeSpec(
                name="x", system_prompt="x", runner=orch.run,
                tools=[keyword_search, keyword_search],
            ),
        ),
        (
            "a runner that is not callable",
            lambda: NodeSpec(name="x", system_prompt="x", runner="nope"),
        ),
        (
            "a workflow missing a node the wiring needs",
            lambda: Workflow(nodes=[orch.SPEC]),
        ),
        (
            "an unknown state field",
            lambda: AgentState(mesages=[]),
        ),
    ]:
        refused, message = rejects(thunk)
        check(label, refused, message)


def check_tool_arguments() -> None:
    print("\nTool arguments that should be refused")
    for label, tool, args in [
        ("a graph mode that would smuggle in vector search", graph_rag_search, {"question": "q", "mode": "mix"}),
        ("top_k below the floor", keyword_search, {"question": "q", "top_k": 0}),
        ("top_k above the ceiling", keyword_search, {"question": "q", "top_k": 99}),
        ("an empty question", keyword_search, {"question": "", "top_k": 3}),
        ("an argument the tool does not have", keyword_search, {"question": "q", "limit": 3}),
    ]:
        refused, _ = rejects(lambda: asyncio.run(tool.ainvoke(args)))
        check(label, refused)


# -------------------------------------------------------------- behavioural


def check_retrieval() -> None:
    print("\nRetrieval (BM25 only — the one retriever that needs no key)")
    try:
        hits = asyncio.run(keyword_search.ainvoke({"question": "refund approval", "top_k": 2}))
    except Exception as error:  # noqa: BLE001 - reported, not swallowed
        check("keyword_search returns hits", False, f"{type(error).__name__}: {error}")
        return
    check("keyword_search returns hits", "[no keyword matches]" not in hits)
    print("\n".join(f"      {line}" for line in hits.splitlines()[:4]))


def check_full_lap() -> None:
    """Drive the whole topology with fake bodies, so no model is involved.

    This is the check that proves the edges work: a delegation, a hand-back, an
    answer, a rejection, and the revision cap ending it.
    """
    print("\nA full lap with substituted node bodies")
    trace = []

    def recorder(name: str, update: dict):
        async def run(state: AgentState) -> dict:
            trace.append(name)
            return update
        return run

    async def orchestrator(state: AgentState) -> dict:
        """Delegate to the researcher once, then write the answer.

        The first pass has to be a real delegation, otherwise the run goes
        straight to the responder and the worker hand-back edge — the one thing
        a lap is meant to prove — is never crossed.
        """
        trace.append(ORCHESTRATOR)
        first_pass = "researcher" not in state.delegations
        target = "researcher" if first_pass else RESPONDER
        return {"next_node": target, "delegations": [target]}

    async def verifier(state: AgentState) -> dict:
        trace.append("verifier")
        return {"verdict": "fail", "revisions": state.revisions + 1}

    swapped = {
        ORCHESTRATOR: orchestrator,
        "researcher": recorder("researcher", {}),
        RESPONDER: recorder(RESPONDER, {}),
        "verifier": verifier,
    }

    workflow = Workflow(
        nodes=[
            NodeSpec(
                name=spec.name,
                system_prompt=spec.system_prompt,
                runner=swapped.get(spec.name, recorder(spec.name, {})),
                tools=[],  # no tools: nothing here should reach a real retriever
            )
            for spec in WORKFLOW.nodes
        ]
    )
    result = asyncio.run(workflow.compile().ainvoke(AgentState()))

    print(f"      path: {' -> '.join(trace)}")
    check("the run terminated", True)
    check(
        "a delegated worker handed control back",
        trace[:3] == [ORCHESTRATOR, "researcher", ORCHESTRATOR],
    )
    check(
        "the verify loop is bounded",
        trace.count("verifier") == MAX_REVISIONS,
        f"verifier ran {trace.count('verifier')}x, cap is {MAX_REVISIONS}",
    )
    check("the answer was always graded", trace.count(RESPONDER) == trace.count("verifier"))
    check("revisions were counted", result["revisions"] == MAX_REVISIONS)


def check_live(question: str) -> None:
    """One real call to the only implemented node."""
    print("\nLive orchestrator call (needs a working API key)")
    state = AgentState(messages=[HumanMessage(question)])
    try:
        result = asyncio.run(orch.run(state))
    except Exception as error:  # noqa: BLE001 - the point is to report it
        check("the orchestrator routed a real question", False, f"{type(error).__name__}: {error}")
        return
    check("the orchestrator routed a real question", result["next_node"] in WORKERS)
    print(f"      question: {question}")
    print(f"      routed to: {result['next_node']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the agent's wiring.")
    parser.add_argument("--live", action="store_true", help="Also make one real model call.")
    parser.add_argument(
        "--question",
        default="Who has to approve a 700 dollar refund?",
        help="The question used for --live.",
    )
    args = parser.parse_args()

    check_structure()
    check_tool_boundaries()
    check_routers()
    check_orchestrator()
    check_declarations()
    check_tool_arguments()
    check_retrieval()
    check_full_lap()
    if args.live:
        check_live(args.question)

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    for label in FAILED:
        print(f"  FAILED: {label}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
