"""One module per node. Each declares its own prompt, body, tools and spec.

    orchestrator.py   picks the next worker                    IMPLEMENTED
    researcher.py     gathers cited evidence                   stub
    gap_auditor.py    finds unowned and double-owned duties    stub
    notifier.py       delivers a finding outside the run       stub
    responder.py      writes the answer                        stub
    verifier.py       grades the answer against the evidence   stub

A node module is self-contained: its system prompt sits next to the body that
uses it and the tool list it is allowed, so reading one file tells you
everything about that node. What none of them decides is where control goes —
that is graph.py's job, and it is why the modules do not import each other.
"""

from .gap_auditor import SPEC as GAP_AUDITOR_SPEC
from .notifier import SPEC as NOTIFIER_SPEC
from .orchestrator import SPEC as ORCHESTRATOR_SPEC
from .researcher import SPEC as RESEARCHER_SPEC
from .responder import SPEC as RESPONDER_SPEC
from .verifier import SPEC as VERIFIER_SPEC

# Every node in the workflow. Order is only presentational — the wiring in
# graph.py refers to nodes by name, not by position.
NODE_SPECS = [
    ORCHESTRATOR_SPEC,
    RESEARCHER_SPEC,
    GAP_AUDITOR_SPEC,
    NOTIFIER_SPEC,
    RESPONDER_SPEC,
    VERIFIER_SPEC,
]

__all__ = [
    "NODE_SPECS",
    "ORCHESTRATOR_SPEC",
    "RESEARCHER_SPEC",
    "GAP_AUDITOR_SPEC",
    "NOTIFIER_SPEC",
    "RESPONDER_SPEC",
    "VERIFIER_SPEC",
]
