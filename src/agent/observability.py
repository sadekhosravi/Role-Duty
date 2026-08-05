"""Langfuse tracing, in one place.

Every LLM call, tool call and node transition in this workflow already happens
inside LangChain, so almost all of the tracing is free: Langfuse ships a
LangChain callback handler, and LangGraph runs its nodes through the same
callback machinery. Handing that handler to `ainvoke` produces a trace with a
span per node, a generation per model call — with model, token counts, cost and
latency — and a span per tool call, without any node knowing about it.

What is *not* free is the part worth having. A trace of "which functions ran" is
a profiler; a trace you can evaluate on needs the run's own judgement attached to
it. So this module also wraps each run in a root span and, when the run finishes,
posts the verifier's verdict and the three loop counters as Langfuse scores. That
is what makes the dashboard answer questions like "how often does the verifier
reject the first answer" and "do runs that use more retrievals score better",
which is the reason to have observability at all.

Tracing is optional and off by default. With no keys set, every entry point here
returns an inert object: `callbacks` is an empty list, `record` does nothing, and
the workflow runs exactly as it did before. This is deliberate — an observability
layer that can take the agent down when the collector is unreachable is a worse
trade than no observability, and a pet project should not need a running Docker
stack to answer a question.

Configuration (all read from .env, like the rest of the project):

    LANGFUSE_PUBLIC_KEY     pk-lf-...   from Project Settings in the Langfuse UI
    LANGFUSE_SECRET_KEY     sk-lf-...   likewise
    LANGFUSE_BASE_URL       http://localhost:3000 by default — the self-hosted
                            docker-compose instance
    LANGFUSE_TRACING_ENVIRONMENT   tags traces as dev/staging/prod (default
                            "development"), so experiments do not pollute the
                            numbers you actually care about
    LANGFUSE_TRACING_ENABLED       set false to keep the keys but stop sending
    LANGFUSE_TRACE_OPENAI_SDK      see "The double-counting problem" below

Both keys must be present. One key alone is a misconfiguration rather than a
choice, and it is reported as a warning instead of being silently ignored.

The double-counting problem
---------------------------
LightRAG ships its own Langfuse support: when it sees both keys in the
environment it swaps `langfuse.openai.AsyncOpenAI` in for the plain client
(lightrag/llm/openai.py). That import globally monkey-patches the OpenAI SDK
with wrapt, so it does not only instrument LightRAG — it instruments every
OpenAI call in the process, including the ones langchain-openai makes for our
own nodes.

Measured, not assumed: one `ChatOpenAI` call with both integrations active
produces two sibling generations under the same parent — `ChatOpenAI` from the
callback handler and `OpenAI-generation` from the patch — each reporting the
same tokens. Left alone, every cost and token figure on the agent's traces is
exactly twice the truth, which quietly ruins the numbers you would evaluate on.

So by default the patch's spans are dropped at export (see
`_should_export_span`) and each call is counted once, by the handler that knows
which node it belongs to. The cost is that LightRAG's internal calls — its
keyword extraction, mostly — stop appearing as their own generations; what the
tool returned is still on the tool span, so the retrieval is still visible, just
not its internals.

Set LANGFUSE_TRACE_OPENAI_SDK=true to keep them. That is the right choice when
you are debugging LightRAG itself, and the wrong one when you are reading cost
or token charts, because the agent's own calls go back to counting double.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

from dotenv import load_dotenv

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .state import AgentState

load_dotenv()

log = logging.getLogger(__name__)

# The port docker-compose publishes Langfuse on. Overridable, but defaulting to
# it means a local install needs two keys in .env and nothing else.
DEFAULT_BASE_URL = "http://localhost:3000"

# Which deployment these traces came from. Langfuse filters on this, so a run
# fired off while debugging does not land in the same bucket as a real one.
DEFAULT_ENVIRONMENT = "development"


def _falsey(value: str | None) -> bool:
    return (value or "").strip().lower() in {"0", "false", "no", "off"}


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def is_enabled() -> bool:
    """Whether traces will actually be sent.

    Kept separate from `_get_client` so callers — the CLI, the check script —
    can report the configuration without paying for a client and a network
    handshake to find out.
    """
    if _falsey(os.getenv("LANGFUSE_TRACING_ENABLED")):
        return False
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


# The name the globally-patched OpenAI SDK gives its generations. LangChain's
# handler names its own after the runnable ("ChatOpenAI"), so the two are
# distinguishable by name — which is the only thing they differ in that is
# visible at export time, the spans being siblings rather than nested.
_PATCHED_SDK_SPAN = "OpenAI-generation"


def _should_export_span(span: Any) -> bool:
    """Drop the duplicate generations the global OpenAI patch produces.

    Composed with Langfuse's own default predicate rather than replacing it:
    passing `should_export_span` overrides the built-in filter entirely, so
    anything this returns True for is exported, including spans Langfuse would
    normally have discarded.

    Only spans the patch created are dropped, and only when
    LANGFUSE_TRACE_OPENAI_SDK is off. See the module docstring for the trade.
    """
    from langfuse import is_default_export_span

    if not is_default_export_span(span):
        return False
    if getattr(span, "name", None) != _PATCHED_SDK_SPAN:
        return True
    return _truthy(os.getenv("LANGFUSE_TRACE_OPENAI_SDK"))


def _resource_manager(public_key: str) -> Any:
    """Langfuse's per-key singleton, or None if there isn't one yet.

    Private API, so it is probed rather than relied on: every caller treats a
    failure here as "cannot tell" and carries on.
    """
    try:
        from langfuse._client.resource_manager import LangfuseResourceManager

        return LangfuseResourceManager._instances.get(public_key)
    except Exception:  # noqa: BLE001 - diagnostics only, never load-bearing
        return None


def filter_in_force() -> bool | None:
    """Whether the span filter above is the one actually applied.

    None when tracing is off or the answer cannot be determined. See
    `_get_client` for why this is worth asking.
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    if not is_enabled() or not public_key:
        return None
    manager = _resource_manager(public_key)
    if manager is None:
        return None
    return getattr(manager, "should_export_span", None) is _should_export_span


_client: Any = None
_client_built = False


def _get_client() -> Any:
    """The Langfuse client, or None when tracing is off. Built on first use.

    Built lazily and cached for the same reason the models are: importing this
    module must not require credentials, a network connection, or a running
    Langfuse. Construction is also where a broken install surfaces — a bad host,
    a missing dependency — so it is wrapped: a failure here disables tracing for
    the process rather than taking the run with it.
    """
    global _client, _client_built
    if _client_built:
        return _client

    _client_built = True
    if not is_enabled():
        # One key without the other is a typo, not a decision. Say so, because
        # the alternative is a user watching an empty dashboard and wondering.
        if bool(os.getenv("LANGFUSE_PUBLIC_KEY")) != bool(os.getenv("LANGFUSE_SECRET_KEY")):
            log.warning(
                "langfuse: only one of LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY "
                "is set, so tracing is off. Both are required."
            )
        return None

    try:
        from langfuse import Langfuse

        # Langfuse keys its resource manager — which owns the span processor and
        # the export filter below — by public key, and a constructor that finds
        # an existing one returns it and DISCARDS every argument passed here.
        # So whoever builds the client first decides whether the filter exists,
        # and if something else got there first ours is dropped in silence.
        # Nothing in this project does that today; LightRAG installs the OpenAI
        # patch at import but only calls get_client() when a request runs, which
        # is after this. Said out loud anyway, because a silent no-op is exactly
        # the failure this codebase keeps having to relearn.
        if _resource_manager(os.environ["LANGFUSE_PUBLIC_KEY"]) is not None:
            log.warning(
                "langfuse: a client for this key already exists, so the span "
                "filter is not being applied — OpenAI SDK generations will be "
                "recorded twice and token counts doubled. Build the Langfuse "
                "client through agent.observability, before anything else."
            )

        _client = Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            base_url=os.getenv("LANGFUSE_BASE_URL", DEFAULT_BASE_URL),
            environment=os.getenv("LANGFUSE_TRACING_ENVIRONMENT", DEFAULT_ENVIRONMENT),
            should_export_span=_should_export_span,
        )
    except Exception as error:  # noqa: BLE001 - logged, then run without tracing
        log.warning("langfuse: could not start tracing (%s); continuing untraced", error)
        _client = None
    return _client


def auth_check() -> tuple[bool, str]:
    """Ask Langfuse whether these credentials work. Returns (ok, message).

    A real network call, so it is never made on the workflow's path — only by
    the CLI's --check-tracing, where waiting for an answer is the point.
    """
    if not is_enabled():
        return False, (
            "tracing is off: set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env"
        )
    client = _get_client()
    if client is None:
        return False, "tracing is configured but the client failed to start (see warnings)"
    base_url = os.getenv("LANGFUSE_BASE_URL", DEFAULT_BASE_URL)
    try:
        if client.auth_check():
            return True, f"connected to Langfuse at {base_url}"
        return False, f"Langfuse at {base_url} rejected these credentials"
    except Exception as error:  # noqa: BLE001 - reported, not raised
        # The SDK raises with the whole HTTP response, headers included, which
        # buries the one useful word in three lines of noise. A 401 here means
        # the server answered — the host is right and the keys are not.
        detail = str(error)
        if "401" in detail or "Invalid credentials" in detail:
            return False, (
                f"Langfuse at {base_url} rejected these credentials (401) — check "
                "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY against Project Settings"
            )
        return False, f"could not reach Langfuse at {base_url}: {type(error).__name__}"


def flush() -> None:
    """Send anything still queued.

    Langfuse batches in a background thread and registers its own atexit hook,
    so this is not required for correctness. It is required for the trace URL
    the CLI prints to be resolvable by the time the user clicks it, which is the
    difference between a link and a 404.
    """
    client = _get_client()
    if client is not None:
        client.flush()


def _retrievals(state: AgentState) -> int:
    """How many tool calls the run actually completed.

    Counted the same way the researcher counts its own budget — ToolMessages,
    not requests — so the number on the trace is the number the agent was
    working against.
    """
    from langchain_core.messages import ToolMessage

    return sum(1 for message in state.messages if isinstance(message, ToolMessage))


@dataclass
class RunTrace:
    """One traced run, and the handle the caller uses to finish it.

    A single class covers both the traced and untraced cases: with `span` None
    every method is a no-op. The alternative — a null object and a real one —
    means two implementations of the same contract, and the untraced one is the
    one nobody ever runs, so it is the one that rots.
    """

    span: Any = None
    client: Any = None
    callbacks: list = field(default_factory=list)
    url: str | None = None

    def record(self, state: AgentState) -> None:
        """Attach the run's outcome and its self-assessment to the trace.

        Called before the span closes, because Langfuse resolves both the trace
        URL and the current-trace scores from the active OTEL context — after
        the context manager exits there is no trace to attach anything to.

        The scores are chosen to be the ones worth grouping by later:

          grounded     the verifier's verdict, categorical. "ungraded" is kept
                       distinct from "pass" here exactly as it is in the state —
                       collapsing them would report unchecked answers as verified,
                       which is the one number that must not be wrong.
          answered     whether anything came back at all.
          delegations  how many workers the orchestrator ran.
          revisions    how many times the answer was sent back.
          retrievals   how many searches it took.

        The three counters are the loop bounds from state.py. Tracking them is
        what turns "it felt slow" into "the median run spends six delegations",
        and it is how a regression in the orchestrator's routing shows up as a
        number rather than as a vibe.
        """
        if self.span is None:
            return
        try:
            self.span.update(
                output=state.answer,
                metadata={
                    "route": " -> ".join(state.delegations),
                    "verdict": state.verdict,
                    "revisions": state.revisions,
                },
            )
            self.span.score_trace(
                name="grounded",
                value=state.verdict or "unverified",
                data_type="CATEGORICAL",
                comment="the verifier's judgement; 'ungraded' means it failed to run",
            )
            self.span.score_trace(
                name="answered", value=bool(state.answer), data_type="BOOLEAN"
            )
            for name, value in (
                ("delegations", len(state.delegations)),
                ("revisions", state.revisions),
                ("retrievals", _retrievals(state)),
            ):
                self.span.score_trace(name=name, value=float(value), data_type="NUMERIC")
        except Exception as error:  # noqa: BLE001 - the answer matters, the score does not
            log.warning("langfuse: could not score the run (%s)", type(error).__name__)

        # Separately, and deliberately. The scores above are queued locally,
        # but resolving the trace URL calls the API — so a failure here means
        # only that the link is unavailable, and reporting it as a scoring
        # failure would send someone looking for data that was recorded fine.
        try:
            self.url = self.client.get_trace_url()
        except Exception as error:  # noqa: BLE001 - a missing link is not a failure
            log.debug("langfuse: could not resolve the trace URL (%s)", error)


@contextmanager
def trace_run(
    question: str,
    *,
    name: str = "role-duty-agent",
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
) -> Iterator[RunTrace]:
    """Wrap one workflow run in a Langfuse trace. A no-op when tracing is off.

        with trace_run(question) as trace:
            result = await graph.ainvoke(state, config={"callbacks": trace.callbacks})
            trace.record(AgentState(**result))

    Two layers, and both are needed. The root span is what the run's scores and
    the question itself hang off — the callback handler alone would produce a
    trace named after LangGraph's internal runnable, with the raw state dict as
    its input. The handler is what fills that span with the per-node detail.

    `session_id` groups several runs into one conversation in the UI, which is
    where the same question asked twice becomes comparable rather than just
    adjacent. It is threaded through from the CLI rather than generated here: a
    session is a property of the caller's intent, and this module cannot know
    whether two runs belong together.
    """
    client = _get_client()
    if client is None:
        yield RunTrace()
        return

    try:
        from langfuse import propagate_attributes
        from langfuse.langchain import CallbackHandler
    except Exception as error:  # noqa: BLE001 - logged, then run without tracing
        log.warning("langfuse: langchain integration unavailable (%s)", error)
        yield RunTrace()
        return

    with client.start_as_current_observation(
        as_type="agent", name=name, input=question
    ) as span:
        # propagate_attributes is the v3+ replacement for passing these to the
        # handler's constructor: they are set on the OTEL context, so everything
        # nested underneath — including the spans LangGraph creates for us —
        # inherits them without being told.
        with propagate_attributes(
            trace_name=name, session_id=session_id, user_id=user_id, tags=tags
        ):
            # A fresh handler per run, not a module-level singleton. The handler
            # keeps per-run state keyed by LangChain run id, and Langfuse's own
            # docs call out reuse across concurrent requests as a source of
            # crossed traces. Constructing one is cheap; a crossed trace is not.
            yield RunTrace(span=span, client=client, callbacks=[CallbackHandler()])
