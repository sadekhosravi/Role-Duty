"""The chat model, in one place.

Every node builds its model from here rather than constructing its own, so a
change of provider or of temperature is a single edit. Nodes still differ in
what they bind to it — their own tools, or a structured-output schema.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from graph_rag.graph_rag import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

from .spec import Runner
from .state import AgentState


def build_llm() -> ChatOpenAI:
    """The chat model, pointed at OpenRouter like the rest of the project.

    temperature=0 because every rule in these prompts is about being exact: the
    same question over the same documents should not produce a different role
    title or a different threshold on a second run.
    """
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=0,
    )


def build_structured_llm(schema: type):
    """A model constrained to return `schema`, over the robust transport.

    `method` is pinned rather than left to default. langchain-openai defaults to
    "json_schema", which asks the model to emit the JSON as ordinary text and
    parse it afterwards; through OpenRouter that truncates intermittently, and a
    half-written object — `{"reason":` and nothing more — fails validation and
    takes the whole run down with it.

    "function_calling" routes the same schema through the provider's tool-call
    machinery, where the structure is the provider's problem rather than
    something the model has to spell correctly under a token budget. Observed
    stable where the default was not.
    """
    return build_llm().with_structured_output(schema, method="function_calling")


def chat_runner(system_prompt: str, tools: Sequence[BaseTool] = ()) -> Runner:
    """A plain "think, and call tools until done" node body.

    The default shape for a worker that has nothing special to do: it reads the
    conversation, calls its tools as often as it needs, and appends whatever it
    produces. Nodes that need more than that write their own body instead.

    The model is built on first use rather than at import, so declaring a node
    does not require credentials.
    """
    llm: ChatOpenAI | None = None

    async def run(state: AgentState) -> dict:
        nonlocal llm
        if llm is None:
            llm = build_llm()
            if tools:
                llm = llm.bind_tools(list(tools))
        # The system prompt is prepended per call rather than stored in state,
        # so it never accumulates as the tool-call loop appends messages — and
        # so each node reads its own prompt rather than the previous node's.
        messages = [("system", system_prompt), *state.messages]
        return {"messages": [await llm.ainvoke(messages)]}

    return run
