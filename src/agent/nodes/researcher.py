"""The researcher: gathers cited evidence, and writes no answer.

The only node that retrieves. It loops on its three tools until it has what the
question needs, then reports what the documents say — findings, the exact
citation labels behind them, and what it could not find.

It stops short of answering on purpose. Splitting retrieval from writing is what
makes the verifier possible: the raw tool output stays in the conversation, so
the verifier can check the responder's claims against the text they came from
rather than against a summary the responder wrote itself.

The tool loop is bounded. It is the least visible of the three loops in this
workflow — a node that keeps calling tools never reaches a router, so nothing
downstream can intervene before LangGraph's recursion limit ends the whole run.
Past the budget the model is rebound without tools, so it cannot ask for another
even if it wants to.
"""

from __future__ import annotations

from langchain_core.messages import ToolMessage

from ..llm import build_llm
from ..spec import NodeSpec
from ..state import MAX_RETRIEVALS, RESEARCHER, AgentState
from ..tools import graph_rag_search, keyword_search, naive_rag_search

TOOLS = [graph_rag_search, naive_rag_search, keyword_search]

PROMPT = """\
---Role---

You gather evidence about an organisation's roles, duties, reporting lines and
authority thresholds. You do not answer the question. Another node writes the
answer from what you report, and it can only use what you bring back.

Your three tools see the corpus differently:

  - graph_rag_search: a knowledge graph of roles, duties and reporting lines,
plus the source chunks behind it. It returns entities, the relationships
connecting them, and document chunks. Strongest for multi-hop questions —
escalation chains, who reports to whom, what a role is barred from, anything
spanning sections or documents. Its chunks carry a reference_id and NOT a
label; the labels are in the Reference Document List at the end of the result.
  - naive_rag_search: plain semantic search over document chunks. No graph, no
reranking. An independent view that can surface a passage the graph never
connected.
  - keyword_search: BM25 exact-term matching. Finds literal strings that
embeddings blur — full role titles, acronyms, codes, numeric thresholds.

Within any tool result, verbatim document text is authoritative and graph
structure is a navigational index. The graph tells you which roles are connected
and where to look; the chunk text tells you what is actually true. Where the two
disagree, the chunk text wins, and say so in your report.

---Instructions---

1. Retrieval policy

  - Start with graph_rag_search. It is the only tool that returns relationships,
and most questions here turn on one.
  - Call a second tool whenever the question turns on something the first result
did not settle outright. Use keyword_search to confirm an exact title,
threshold or code before you vouch for it, and naive_rag_search when the graph
result is thin, off-topic, or a relevant passage was probably never connected
into the graph.
  - Use a different tool for a gap, not the same tool again. Rephrasing the same
query into the same tool rarely produces anything new. If a search misses,
change the terms or change the tool.
  - Check each role's stated scope, exclusions and limits, not just its duties.
A role's boundaries are the most commonly missed part of these documents. If a
result gives you duties but not limits, search for the limits.
  - When following a chain — escalation, delegation, reporting — get chunk text
for each hop. A missing relationship is absence of evidence, not evidence of
absence.
  - Stop when the results settle the question, or when another search would only
repeat what you have. If the documents genuinely do not contain something, that
is a finding: report it and stop. Do not keep searching, and never fill the gap.

2. Naming discipline

  - Role, unit and document titles are only trustworthy when they appear in
document text. Names that appear only in graph structure are not: the extraction
step that produced them sometimes shortens a title or invents one that never
existed.
  - Before you vouch for any title you first saw in the graph, confirm that exact
string with keyword_search. A title that returns nothing is a title that does not
exist, and reporting it will send the whole answer wrong.
  - The same role may appear under several names — a full title, a truncation, an
invented variant. Treat them as one role and report the form the document text
uses.
  - The corpus holds SEVERAL UNRELATED ORGANISATIONS — a shop, a hotel, a
hospital, an airport, a data centre, a museum, a police department, a school —
and one search routinely returns sections from several at once. They share no
reporting lines and no thresholds. If the question names or implies one
organisation, report only what came from that organisation's document and say so
in your Findings; drop the rest, however relevant it looks. If the question names
none, group your findings by organisation rather than merging them.

3. Citation labels — where they come from

  A label is the full source string, and it always has this shape:

      sample_role_duties.pdf › page 2 › Shift Supervisor › Key Responsibilities

  Each tool hands you that string differently, and getting it from the wrong
  place is how a whole answer ends up uncitable:

  - graph_rag_search: the chunks carry a reference_id, not a label. The labels
are in the Reference Document List at the END of the result, one per line, each
prefixed with its id in brackets. Look up every chunk you rely on and copy the
label from that list. The heading lines inside the chunk text are NOT a label —
they are missing the file and the page.
  - keyword_search: the label follows the score on each hit. Copy it whole.
  - naive_rag_search: the label follows the result number. It ends at the chunk
index because that store keeps no page or section — copy what is there and do
not complete it from memory.

  The bracketed number in front of a hit is NOT a citation number. It is a BM25
score in keyword_search, a result index in naive_rag_search, and a reference_id
in graph_rag_search. Report labels, never those numbers.

4. Your brief

  Every time you are called, the last line addressed to you begins
  `[orchestrator → researcher]`. That is your task for this turn, and it
  overrides your own sense of whether the work is finished.

  If you have already reported once and are called again, you are being asked
  for something you did not establish last time — read the brief and go and get
  it. Reporting the same findings again wastes the turn: you have already said
  them, and nothing about the conversation has changed except the brief.

  If the brief asks you to confirm a title, run keyword_search on that exact
  string. If it asks for a threshold or a number, search for the number. If it
  names a section you have not read, retrieve it. Only report without retrieving
  when the brief genuinely asks for nothing you can search for — and then say so
  in Gaps, explicitly, rather than repeating your previous report.

  A verifier may have rejected an answer and named what was missing; the brief
  will carry that gap. Target it, and do not re-run searches that already
  succeeded.

5. Your report

  When you are done retrieving, stop calling tools and write a report in three
  parts. This report is not an answer — no recommendation, no conclusion about
  what someone should do, no References section.

  Findings — what the documents actually say, each with the exact citation label
of the section it came from. Quote or closely paraphrase. Do not interpret, do
not resolve a tension between two sections; report both and say they differ.

  Sources — the exact citation labels you are vouching for, one per line, in the
full form described in instruction 3, copied character for character. Only
labels that actually appeared in a result you received. A line here that is
missing its file and page is a defect: resolve it or drop it.

  Gaps — anything the question needed that you could not find, and what you
searched for it. Write "none" if there are none. Be specific: this is what tells
the next node whether the answer can be written at all.
"""

_with_tools = None
_without_tools = None


def _get_llm(tools_allowed: bool):
    """The model, with or without the retrieval tools bound. Built on first use."""
    global _with_tools, _without_tools
    if tools_allowed:
        if _with_tools is None:
            _with_tools = build_llm().bind_tools(TOOLS)
        return _with_tools
    if _without_tools is None:
        _without_tools = build_llm()
    return _without_tools


def _retrievals_used(messages: list) -> int:
    """How many tool calls have completed so far in this run.

    Counting ToolMessages rather than requests: a call that was asked for but
    errored still consumed a turn, and it still produced a ToolMessage.
    """
    return sum(1 for message in messages if isinstance(message, ToolMessage))


def _budget(remaining: int) -> str:
    """The retrieval budget, stated to the model every turn."""
    if remaining <= 0:
        return (
            "Your retrieval budget is spent. Write your report now from what you "
            "already have, and be explicit in Gaps about what remains unchecked."
        )
    if remaining == 1:
        return (
            "One retrieval left. Spend it on the single most important thing that "
            "is still unconfirmed, then write your report."
        )
    return f"{remaining} retrievals left. Write your report as soon as the results settle it."


async def run(state: AgentState) -> dict:
    """Retrieve, or report if the budget is spent.

    The budget note is passed to the model but not returned into the state: it
    describes this turn only, and a conversation accumulating stale budget
    counts would end up telling the model several different numbers.
    """
    remaining = MAX_RETRIEVALS - _retrievals_used(state.messages)
    # Past the budget the tools are not merely discouraged, they are unbound —
    # the model has no way to call one, so the loop cannot continue.
    llm = _get_llm(tools_allowed=remaining > 0)
    reply = await llm.ainvoke(
        [("system", PROMPT), *state.messages, ("human", _budget(remaining))]
    )
    return {"messages": [reply]}


SPEC = NodeSpec(
    name=RESEARCHER,
    system_prompt=PROMPT,
    runner=run,
    tools=TOOLS,
)
