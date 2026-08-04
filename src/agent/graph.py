"""The LangGraph workflow: one reasoning node with three retrieval tools.

    START -> rag -> (tools -> rag)* -> END

`rag` is the only node that thinks. It reads the question, decides which of the
three retrievers to call, reads what comes back, and writes the final answer —
reconciling the graph, vector and keyword views itself rather than trusting any
one of them. RAG_SYSTEM_PROMPT (prompts.py) governs both halves of that: which
tool to reach for, and how to reason over the results.

`tools` is not a second reasoning step; it is LangGraph's mechanical executor for
whatever `rag` asked for, and it always hands control straight back. The loop
runs until `rag` returns a message with no tool calls, which is the answer.

Usage:
    python scripts/agent.py "Who authorizes a UPS bypass?"
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.prompts import RAG_SYSTEM_PROMPT
from agent.tools import TOOLS
from graph_rag.graph_rag import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


def build_llm() -> ChatOpenAI:
    """The chat model, pointed at OpenRouter like the rest of the project.

    temperature=0 because every rule in the system prompt is about being exact:
    the same question over the same documents should not produce a different
    role title or a different threshold on a second run.
    """
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=0,
    )


def build_graph():
    """Compile the workflow. Call once and reuse — it holds the open stores."""
    llm = build_llm().bind_tools(TOOLS)

    async def rag(state: MessagesState) -> dict:
        """Decide what to retrieve, or write the answer once enough is known.

        The system prompt is prepended per call rather than stored in state, so
        it never accumulates as the tool-call loop appends messages.
        """
        messages = [("system", RAG_SYSTEM_PROMPT), *state["messages"]]
        return {"messages": [await llm.ainvoke(messages)]}

    builder = StateGraph(MessagesState)
    builder.add_node("rag", rag)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "rag")
    # tools_condition routes to "tools" when the model asked for a tool, and to
    # END when it answered instead.
    builder.add_conditional_edges("rag", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "rag")
    return builder.compile()


async def ask(question: str) -> str:
    """Run one question through the workflow and return the final answer."""
    result = await build_graph().ainvoke({"messages": [("user", question)]})
    return result["messages"][-1].content
