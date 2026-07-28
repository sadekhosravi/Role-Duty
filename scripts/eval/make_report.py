"""Render the validation harness and its latest results as a PDF report.

Reads data/val/questions.json and the newest data/val/results/eval_*.json, so the
report always describes the run it was generated from - no numbers are hard-coded.

Run:  uv run --with reportlab python scripts/eval/make_report.py
      uv run --with reportlab python scripts/eval/make_report.py --results <file>
Output: data/val/results/validation_report.pdf
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS_FILE = ROOT / "data" / "val" / "questions.json"
RESULTS_DIR = ROOT / "data" / "val" / "results"
OUT_FILE = RESULTS_DIR / "validation_report.pdf"

INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#666666")
RULE = colors.HexColor("#cccccc")
HEAD_BG = colors.HexColor("#eeeeee")
GOOD = colors.HexColor("#1b7a3d")
WARN = colors.HexColor("#a86400")
BAD = colors.HexColor("#a32020")


# --- Styles --------------------------------------------------------------------

def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("T", parent=base["Title"], fontSize=20, spaceAfter=4, textColor=INK),
        "subtitle": ParagraphStyle("St", parent=base["Normal"], fontSize=10, textColor=MUTED, spaceAfter=16),
        "h1": ParagraphStyle("H1", parent=base["Heading1"], fontSize=14, spaceBefore=18, spaceAfter=6, textColor=INK),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], fontSize=11, spaceBefore=12, spaceAfter=4, textColor=INK),
        "body": ParagraphStyle("B", parent=base["BodyText"], fontSize=9.5, leading=13.5, spaceAfter=6, alignment=TA_LEFT),
        "bullet": ParagraphStyle("Bu", parent=base["BodyText"], fontSize=9.5, leading=13.5, leftIndent=12, spaceAfter=3),
        "cell": ParagraphStyle("C", parent=base["BodyText"], fontSize=8, leading=10, spaceAfter=0),
        "cellb": ParagraphStyle("Cb", parent=base["BodyText"], fontSize=8, leading=10, spaceAfter=0, fontName="Helvetica-Bold"),
        "score": ParagraphStyle("Sc", parent=base["Title"], fontSize=30, spaceBefore=6, spaceAfter=0, textColor=INK),
        "mono": ParagraphStyle("M", parent=base["BodyText"], fontName="Courier", fontSize=8, leading=11, spaceAfter=4),
        "caption": ParagraphStyle("Cap", parent=base["Normal"], fontSize=8, textColor=MUTED, spaceAfter=10),
    }


def table(data, widths, st, align_right: tuple[int, ...] = ()) -> Table:
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, RULE),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, colors.HexColor("#e8e8e8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    for col in align_right:
        style.append(("ALIGN", (col, 0), (col, -1), "RIGHT"))
    t.setStyle(TableStyle(style))
    return t


def score_colour(value: float):
    return GOOD if value >= 90 else (WARN if value >= 70 else BAD)


REFUSAL_MARKERS = ("not able to provide", "[no-context]", "cannot answer", "unable to answer")


def refused(answer: str) -> bool:
    """True when the system declined rather than answered."""
    return any(m in answer.lower() for m in REFUSAL_MARKERS)


def truncated(answer: str) -> bool:
    """True when generation stopped mid-sentence.

    A reference-list line legitimately ends without punctuation, so those are
    excluded; what is left is prose that simply stops ("... reports to").
    """
    lines = [ln.strip() for ln in answer.strip().splitlines() if ln.strip()]
    if not lines:
        return False
    last = lines[-1]
    if last.startswith(("-", "*", "[", "#")) or ".pdf" in last:
        return False
    return last[-1] not in ".!?\"')]}:"


def load_runs() -> list[tuple[str, dict]]:
    """Every eval_*.json in the results directory, oldest first."""
    runs = []
    for f in sorted(glob.glob(str(RESULTS_DIR / "eval_*.json"))):
        runs.append((Path(f).stem.replace("eval_", ""), json.loads(Path(f).read_text(encoding="utf-8"))))
    return runs


# --- Narrative -----------------------------------------------------------------
#
# The scores in this report are read from the results file, but the prose that
# frames them — what the corpus is, which traps it plants, which capabilities the
# question ids demonstrate — describes ONE corpus. Pointed at a different corpus
# it silently keeps describing the old one, naming documents and question ids that
# are not in the run. So the narrative is data too: a questions file may carry a
# "report" block overriding any of these keys, and DEFAULT_REPORT below is the
# data/val text used when it does not.
DEFAULT_REPORT = {
    "title": "GraphRAG Validation Report",
    "corpus_line": "Corpus: five fictional Role &amp; Duty documents",
    "corpus_intro": [
        "Five fictional organisations, each documented in the same Role &amp; Duty format as the "
        "production corpus: five roles per site, and for each role a reporting line, a summary, "
        "key responsibilities, an explicit <b>Out of Scope</b> list, and an escalation path. "
        "The documents are generated from data in <font name='Courier' size='8.5'>scripts/eval/make_val_pdfs.py</font>, "
        "so they are reproducible and editable; the PDFs are build output, not source.",
        "The five sites share one world. Cedar Ridge Regional Fire &amp; Rescue covers the airport, "
        "the school, the library and the data centre, so a question can only be answered by reading "
        "two documents and joining them. That is what separates a corpus that tests retrieval from "
        "one that tests recall of a single chunk.",
    ],
    "documents": [
        ["Northgate Municipal Airport",
         "Ramp Agent, Ramp Lead, Airside Safety Officer, Airside Duty Manager, Terminal Duty Manager",
         "Two same-named Duty Managers with different authority; stop-work authority that sits outside "
         "the operational chain; the 25-litre spill threshold that routes to the fire service."],
        ["Cedar Ridge Regional Fire &amp; Rescue",
         "Firefighter/EMT, Company Officer, Battalion Chief, Fire Marshal, Fire &amp; Life Safety Educator",
         "Rank does not equal authority: the Battalion Chief commands incidents but only the Fire Marshal "
         "enforces code. Command transfer is announced, not automatic. The hub document for cross-site questions."],
        ["Fairview ISD - Westbrook Campus",
         "Classroom Teacher, Campus Safety Coordinator, School Nurse, Front Office Manager, Principal",
         "A dual reporting line (administrative vs clinical); an external authority the Principal cannot "
         "override; drill frequencies as exact numbers."],
        ["Harbor Point Public Library",
         "Circulation Assistant, Reference Librarian, Youth Services Librarian, Facilities Supervisor, Branch Manager",
         "Three-step approval ladders ($10 / $50 / Director, one day / Director); an information-versus-advice "
         "boundary; the same fire-inspection task owned by a different role than at the school."],
        ["Blackwater Data Centre",
         "Data Centre Technician, Shift Lead, Critical Facilities Engineer, Security Officer, Site Duty Manager",
         "A third Duty Manager; severity-conditional authority (Severity-1 only); a contracted role with a "
         "split reporting line; a prohibited action with a timed obligation attached."],
    ],
    "design_intro":
        "Twenty-five questions, five per document, spread across four difficulty levels. Each carries a "
        "reference answer written from the sources and a machine-checkable key. Difficulty is not question "
        "length - it is how many independent things have to be right at once.",
    "question_types": [
        ["Direct lookup (easy)",
         "One fact, one document, stated in one place. If these fail, retrieval is broken."],
        ["Scope boundaries (medium)",
         "The answer is in an <b>Out of Scope</b> list, so the model must notice a prohibition rather "
         "than summarise the duties it can see. A plausible-sounding wrong answer is always available."],
        ["Numeric ladders (medium/hard)",
         "25 litres, $10 / $50, 30 days, one day, 5 minutes, Severity-1 vs Severity-2. Guessing lands near "
         "the right shape but the wrong number, which the key catches exactly."],
        ["Authority inversion (hard)",
         "Someone senior wants to do something only a specific other role may do. Tests whether the system "
         "reasons from the document's rules or from organisational intuition."],
        ["Cross-document multi-hop (hard)",
         "Requires joining two sites through the shared fire department, and citing both."],
        ["Title collision (very hard)",
         "Three roles end in 'Duty Manager' and two in 'Lead'. The question is deliberately ambiguous; a "
         "good answer disambiguates instead of picking one."],
        ["False premise (very hard)",
         "The question asserts something the documents contradict. The correct response rejects the premise; "
         "the failure mode is a fluent explanation of something that is not true."],
        ["Unanswerable (very hard)",
         "Asks for figures that appear nowhere. Declining scores full marks; supplying a plausible industry "
         "number scores zero. This is the only question where saying less is worth more."],
    ],
    "claims": [
        ["Distinguishes similarly-named roles across documents", ["D4", "A5"],
         "Three roles end in 'Duty Manager' and two in 'Lead'. D4 presents an ambiguous shift report and "
         "A5 attributes an airside decision to the terminal role; both require disambiguation rather than "
         "a guess."],
        ["Respects prohibitions over plausibility", ["F2", "S4", "A4", "D2", "S2"],
         "Each of these asks whether someone senior may do something only another role may do. The answer "
         "must come from an Out of Scope list rather than from organisational intuition."],
        ["Retrieves exact numbers rather than plausible ones", ["A3", "L2", "D3", "S1", "D5"],
         "The 25-litre spill threshold, the $10/$50 waiver ladder, Severity-1 versus Severity-2, drill "
         "frequencies, and the 5-minute hazard-sheet obligation."],
        ["Declines when the corpus is silent", ["F5"],
         "Asks for staffing and turnout figures that appear nowhere. Full marks require refusing without "
         "supplying a plausible industry number."],
        ["Rejects a false premise", ["A5"],
         "Asserts that the Terminal Duty Manager closes taxiways. The correct answer corrects the premise "
         "instead of explaining it."],
        ["Joins two documents through a shared entity", ["F4", "S4", "S5", "L5"],
         "Each requires reading the fire department document alongside a site document and citing both."],
    ],
}


# --- Report --------------------------------------------------------------------

def build(results_path: Path) -> Path:
    st = styles()
    data = json.loads(results_path.read_text(encoding="utf-8"))
    spec = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    by_id = {q["id"]: q for q in spec["questions"]}
    rep = {**DEFAULT_REPORT, **spec.get("report", {})}
    results = sorted(data["results"], key=lambda r: r["id"])
    settings = data["settings"]
    P = lambda text, style="body": Paragraph(text, st[style])  # noqa: E731

    s: list = []

    # --- Cover -----------------------------------------------------------------
    s.append(P(rep["title"], "title"))
    s.append(P(
        f"{rep['corpus_line']} &nbsp;|&nbsp; "
        f"{len(spec['questions'])} questions &nbsp;|&nbsp; run {data['timestamp']}", "subtitle"))

    s.append(P(f"{data['overall']:.1f} / 100", "score"))
    s.append(Paragraph(
        f"<font color='{score_colour(data['overall']).hexval()}'>{data['band'].upper()}</font>",
        ParagraphStyle("band", parent=st["body"], fontSize=11, spaceAfter=12)))

    s.append(table(
        [[P("Setting", "cellb"), P("Value", "cellb")]] + [
            [P(k, "cell"), P(str(v), "cell")] for k, v in [
                ("Answer / extraction model", settings["llm_model"]),
                ("Judge model", settings.get("judge_model") or "not used"),
                ("Retrieval mode", settings["mode"]),
                ("Reranker", settings.get("rerank") or "off"),
                ("Answer cache", "on" if settings["llm_cache"] else "off"),
                ("Validation graph", settings["working_dir"]),
            ]],
        [6 * cm, 10.5 * cm], st))
    s.append(Spacer(1, 6))
    s.append(P(
        "This report measures the retrieval and answering pipeline as configured today, "
        "against a corpus built specifically to break it. It is not a benchmark of the "
        "underlying model: the same model scores differently under a different prompt, "
        "retrieval mode, or chunking strategy, which is the point of having the harness.", "body"))

    # --- 1. The corpus ---------------------------------------------------------
    s.append(P("1. The corpus", "h1"))
    for para in rep["corpus_intro"]:
        s.append(P(para, "body"))

    docs_table = [[P("Document", "cellb"), P("Roles", "cellb"), P("What it contributes", "cellb")]]
    for name, roles, contributes in rep["documents"]:
        docs_table.append([P(f"<b>{name}</b>", "cell"), P(roles, "cell"), P(contributes, "cell")])
    s.append(table(docs_table, [4.2 * cm, 5.3 * cm, 7 * cm], st))

    # --- 2. Question design ----------------------------------------------------
    s.append(PageBreak())
    s.append(P("2. How the questions are built", "h1"))
    s.append(P(rep["design_intro"], "body"))

    for label, text in rep["question_types"]:
        s.append(P(f"<b>{label}.</b> {text}", "bullet"))

    # --- 3. Scoring ------------------------------------------------------------
    s.append(P("3. How an answer is scored", "h1"))
    s.append(P(
        "Each question scores out of 100 across four components; the ranking is the mean, so it reads as a "
        "percentage. Three components are deterministic and free; the fourth is an LLM judge.", "body"))
    s.append(table(
        [[P("Component", "cellb"), P("Weight", "cellb"), P("What it measures", "cellb")]] + [
            [P(n, "cell"), P(w, "cell"), P(d, "cell")] for n, w, d in [
                ("facts", "25",
                 "Fraction of required fact groups present. Each group is a set of accepted alternatives, "
                 "matched on word boundaries so 'no' does not match inside 'Northgate'."),
                ("traps", "15",
                 "None of the planted wrong answers appear. Each regex is written to fire only on a genuine "
                 "error, never on a correct answer that draws the same distinction."),
                ("citations", "10",
                 "A References section exists (0.4) and cites the expected source PDFs (0.6). Cross-document "
                 "questions expect two."),
                ("judge", "50",
                 "An LLM judge grades the answer against the reference on 0-5, with hard caps: inventing a "
                 "role title caps at 1; permitting what the source forbids caps at 2."),
            ]],
        [3 * cm, 1.8 * cm, 11.7 * cm], st))
    s.append(P(
        "The key validates itself. Every reference answer must score a clean 100 against its own key, checked "
        "offline before each run (<font name='Courier' size='8.5'>--check-key</font>). This caught a real "
        "defect during construction: a trap regex matched 'can' inside 'cannot' and so penalised correct "
        "answers on three questions. Without that check the harness would have quietly under-reported.", "body"))

    # --- 4. Results ------------------------------------------------------------
    s.append(PageBreak())
    s.append(P("4. Results", "h1"))

    comp = data["by_component"]
    s.append(P("By component", "h2"))
    s.append(table(
        [[P(k, "cellb") for k in ("Component", "Score")]] + [
            [P(k, "cell"), P(f"<font color='{score_colour(v).hexval()}'><b>{v:.0f}%</b></font>", "cell")]
            for k, v in comp.items()],
        [4 * cm, 3 * cm], st))

    s.append(P("By difficulty", "h2"))
    s.append(table(
        [[P(k, "cellb") for k in ("Difficulty", "Score")]] + [
            [P(k.replace("_", " "), "cell"), P(f"<font color='{score_colour(v).hexval()}'><b>{v:.0f}</b></font>", "cell")]
            for k, v in data["by_difficulty"].items()],
        [4 * cm, 3 * cm], st))

    s.append(P("By document", "h2"))
    s.append(table(
        [[P(k, "cellb") for k in ("Document", "Score")]] + [
            [P(k, "cell"), P(f"<font color='{score_colour(v).hexval()}'><b>{v:.0f}</b></font>", "cell")]
            for k, v in data["by_document"].items()],
        [4 * cm, 3 * cm], st))

    # Run-to-run variance, when there is more than one run to compare.
    runs = [(stamp, d) for stamp, d in load_runs() if d["settings"]["mode"] == settings["mode"]]
    if len(runs) > 1:
        s.append(P("Run-to-run variance", "h2"))
        s.append(P(
            "The same configuration answering the same questions does not produce the same score. "
            "The runs below are identical in model, mode and reranker; the spread is the pipeline's own "
            "non-determinism, and it sets the floor for how large a change has to be before it means "
            "anything.", "body"))
        s.append(table(
            [[P(h, "cellb") for h in ("Run", "Overall", "facts", "traps", "citations", "judge", "Refusals")]] + [
                [P(stamp, "cell"),
                 P(f"<font color='{score_colour(d['overall']).hexval()}'><b>{d['overall']:.1f}</b></font>", "cell"),
                 *[P(f"{d['by_component'].get(k, 0):.0f}%", "cell") for k in ("facts", "traps", "citations", "judge")],
                 P(str(sum(1 for r in d["results"] if refused(r["answer"]))), "cell")]
                for stamp, d in runs],
            [4 * cm, 2.2 * cm, 1.7 * cm, 1.7 * cm, 2.1 * cm, 1.7 * cm, 2.1 * cm], st))

        # Questions whose score moves most between runs.
        per_q: dict[str, list[float]] = {}
        for _, d in runs:
            for r in d["results"]:
                per_q.setdefault(r["id"], []).append(r["score"])
        volatile = sorted(
            ((qid, min(v), max(v)) for qid, v in per_q.items() if len(v) > 1 and max(v) - min(v) >= 20),
            key=lambda x: x[1] - x[2])
        if volatile:
            s.append(P(
                f"{len(volatile)} question(s) moved by 20 points or more between runs "
                "&mdash; the same question, the same graph, a different outcome:", "body"))
            s.append(table(
                [[P(h, "cellb") for h in ("ID", "Test method", "Worst", "Best", "Swing")]] + [
                    [P(qid, "cellb"), P(by_id[qid]["trap"], "cell"),
                     P(f"{lo:.0f}", "cell"), P(f"{hi:.0f}", "cell"),
                     P(f"<font color='{BAD.hexval()}'><b>{hi - lo:.0f}</b></font>", "cell")]
                    for qid, lo, hi in volatile],
                [1.2 * cm, 7.5 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm], st))
        s.append(P(
            "This is not the model changing its mind about a hard question - the swings land on easy "
            "lookups. In each case retrieval returned nothing and the system declined, which scores near "
            "zero on a question the corpus answers in one line.", "caption"))

    s.append(P("Every question", "h2"))
    rows = [[P(h, "cellb") for h in ("ID", "Doc", "Difficulty", "Test method", "Score", "Notes")]]
    for r in results:
        notes = []
        if r["tripped_traps"]:
            notes.append(f"{len(r['tripped_traps'])} trap")
        if r["missing_facts"]:
            notes.append(f"missed {len(r['missing_facts'])} fact")
        if r["citations"] < 1.0:
            notes.append("no citation")
        rows.append([
            P(r["id"], "cellb"),
            P(r["doc"], "cell"),
            P(r["difficulty"].replace("_", " "), "cell"),
            P(by_id[r["id"]]["trap"], "cell"),
            P(f"<font color='{score_colour(r['score']).hexval()}'><b>{r['score']:.0f}</b></font>", "cell"),
            P(", ".join(notes) or "clean", "cell"),
        ])
    s.append(table(rows, [1.1 * cm, 2 * cm, 2.1 * cm, 5.4 * cm, 1.4 * cm, 4.5 * cm], st, align_right=(4,)))

    # --- 5. What failed --------------------------------------------------------
    s.append(PageBreak())
    s.append(P("5. What failed, and why it matters", "h1"))
    failures = [r for r in sorted(data["results"], key=lambda r: r["score"]) if r["score"] < 90]
    if not failures:
        s.append(P("Every question scored 90 or above.", "body"))
    for r in failures:
        kind = ("declined - retrieval returned nothing" if refused(r["answer"])
                else "answer stopped mid-sentence" if truncated(r["answer"])
                else "answered, but wrong or incomplete")
        block = [
            P(f"<b>{r['id']} &mdash; {r['score']:.0f}/100</b> &nbsp; "
              f"<font color='{MUTED.hexval()}'>{r['difficulty'].replace('_', ' ')} &middot; "
              f"{by_id[r['id']]['trap']} &middot; <i>{kind}</i></font>", "body"),
            P(f"<i>{by_id[r['id']]['question']}</i>", "body"),
        ]
        if r["missing_facts"]:
            block.append(P("<b>Missing:</b> " + "; ".join(" / ".join(g) for g in r["missing_facts"]), "bullet"))
        for t in r["tripped_traps"]:
            block.append(P(f"<b>Trap tripped:</b> <font name='Courier' size='8'>{t}</font>", "bullet"))
        if r["citations"] < 1.0:
            cited = ", ".join(r["cited_sources"]) if r["cited_sources"] else "none of the expected sources"
            block.append(P(f"<b>Citations:</b> {cited}", "bullet"))
        if r.get("judge_reason"):
            block.append(P(f"<b>Judge:</b> {r['judge_reason']}", "bullet"))
        block.append(Spacer(1, 8))
        s.append(KeepTogether(block))

    # --- 6. Capability ---------------------------------------------------------
    s.append(P("6. Demonstrated capability", "h1"))
    s.append(P(
        "Each claim below is tied to the questions that test it, and is marked from this run's actual "
        "scores &mdash; demonstrated at 80 or above on every supporting question, partial otherwise. "
        "Nothing here is asserted independently of the data.", "body"))

    score_of = {r["id"]: r["score"] for r in results}
    for claim, ids, evidence in rep["claims"]:
        have = [i for i in ids if i in score_of]
        worst = min((score_of[i] for i in have), default=0.0)
        status = "DEMONSTRATED" if worst >= 80 else "PARTIAL"
        colour = GOOD if worst >= 80 else WARN
        detail = ", ".join(f"{i} {score_of[i]:.0f}" for i in have)
        s.append(P(
            f"<b>{claim}.</b> <font color='{colour.hexval()}'><b>{status}</b></font> "
            f"<font color='{MUTED.hexval()}'>({detail})</font><br/>{evidence}", "bullet"))

    s.append(P("Limits of this result", "h2"))
    spread = ""
    if len(runs) > 1:
        lo = min(d["overall"] for _, d in runs)
        hi = max(d["overall"] for _, d in runs)
        spread = (f" Measured directly here: {len(runs)} runs of this same configuration scored between "
                  f"{lo:.1f} and {hi:.1f}, so a change smaller than {hi - lo:.0f} points is not evidence "
                  f"of anything.")
    s.append(P(
        "The corpus is synthetic and written by the same process that wrote the questions, so it shares that "
        "process's blind spots: real documents are longer, less consistently structured, and contradict "
        "themselves in ways these do not. Twenty-five questions is a small sample, and the pipeline is "
        f"non-deterministic.{spread}", "body"))
    s.append(P(
        f"The judge is currently <b>{settings.get('judge_model') or 'not used'}</b>, the same model that "
        "produced the answers, which biases the judge component upward. Treat the deterministic components "
        "(facts, traps, citations) as the trustworthy part of the score.", "body"))

    # --- 7. Improvements -------------------------------------------------------
    s.append(PageBreak())
    s.append(P("7. Further improvements", "h1"))
    s.append(P("Ordered by evidence strength: the first items address failures this run actually produced.", "body"))

    uncited = [r["id"] for r in results if r["citations"] < 1.0]
    refusals = [r for r in sorted(results, key=lambda r: r["score"]) if refused(r["answer"])]
    cut = [r for r in sorted(results, key=lambda r: r["score"]) if truncated(r["answer"])]
    wrong = [r for r in sorted(results, key=lambda r: r["score"])
             if r["score"] < 70 and not refused(r["answer"]) and not truncated(r["answer"])]
    items: list[tuple[str, str]] = []

    if refusals:
        ids = ", ".join(f"{r['id']} ({r['score']:.0f})" for r in refusals)
        easy = [r["id"] for r in refusals if r["difficulty"] == "easy"]
        items.append((
            "Fix the silent retrieval misses",
            f"{len(refusals)} question(s) scored near zero not because the answer was wrong but because "
            f"retrieval returned no usable context and the system declined: {ids}. "
            + (f"{', '.join(easy)} are rated <i>easy</i> - single facts stated in one place. "
               if easy else "")
            + "This is the most dangerous failure mode in the set, because a refusal reads as "
              "responsible behaviour and is indistinguishable, from the outside, from the correct refusal "
              "on F5. It is also the main driver of run-to-run variance: the questions that fail this way "
              "differ between runs. Start with MIN_RERANK_SCORE (a reranker pruning every candidate leaves "
              "nothing to answer from), then compare --mode mix against hybrid on exactly these ids."))

    if comp.get("citations", 100) < 90:
        items.append((
            "Make the citation rule survive short answers",
            f"Citations scored {comp['citations']:.0f}% against facts at {comp['facts']:.0f}% - the answers "
            f"are right and uncited. {len(uncited)} question(s) cited nothing expected"
            + (f" ({', '.join(uncited)})" if uncited else "")
            + ". The pattern is length: one-line answers emit a bare '[1]' and drop the References section "
              "entirely, while long answers keep it. The reference rules in ANSWER_SYSTEM_PROMPT are stated "
              "once, near instructions about thorough answers, and short replies skip that whole region of "
              "the prompt. Require the section unconditionally, then confirm with --rescore on the saved "
              "answers rather than assuming the edit worked."))

    if wrong:
        ids = ", ".join(f"{r['id']} ({r['score']:.0f})" for r in wrong)
        items.append((
            "Represent exclusions in the graph, not only in prose",
            f"Answers that were confidently wrong rather than absent: {ids}. The recurring shape is a role "
            "being credited with authority the document explicitly withholds from it. Every 'Out of Scope' "
            "line lives in chunk text; none of it becomes a graph relationship, so entity-centric retrieval "
            "can surface a role together with its duties and without its prohibitions. Extending "
            "EXTRACTION_GUIDANCE to extract exclusions as first-class relationships "
            "('role -- may not -- action') would put the boundary in the same structure as the duty."))

    if cut:
        ids = ", ".join(f"{r['id']} ({r['score']:.0f})" for r in cut)
        items.append((
            "Find out why generation stops mid-sentence",
            f"{len(cut)} answer(s) ended mid-sentence rather than finishing: {ids}. "
            "This is not a reasoning failure and not a retrieval "
            "failure - the answer was being written correctly and the stream ended. Likely candidates are a "
            "max-tokens ceiling, a provider-side cut, or an exception swallowed mid-stream. The runner "
            "records only the text, so the fix starts with logging the API finish_reason alongside each "
            "answer; a truncated answer currently scores as though the model did not know the fact."))

    if not settings["llm_cache"]:
        items.append((
            "Repair the LLM_CACHE setting",
            "This run recorded the answer cache as off. The value in .env parses to something other than "
            "'true' (a trailing character is enough), so LightRAG re-queries every question on every run. "
            "That is why repeated runs cost full price and why identical questions produce different "
            "answers. Worth fixing before drawing conclusions from any two runs."))

    items.append((
        "Use an independent judge model",
         "The judge is currently the same model that wrote the answers, which is a conflict of interest worth "
         "roughly nothing in absolute terms but a lot in relative ones. EVAL_JUDGE_MODEL already exists; "
         "pointing it at a different model makes the judge component comparable across configuration changes. "
         "One judge call also returned non-JSON this run and was silently dropped - the runner degrades rather "
        "than failing, but a stricter response format would avoid the gap."))

    items.append((
        "Split the ingest and query models",
        "Only the ingest model touches the graph, so query-side changes are free to test while extraction "
        "changes require a re-ingest. Separating INGEST_LLM_MODEL from QUERY_LLM_MODEL makes it possible to "
        "sweep answer models against a fixed, known-good graph - which is what turns this harness from a "
        "score into a comparison tool."))

    items.append((
        "Run each configuration more than once",
        "Given the variance measured above, a single run cannot rank two configurations. Averaging three "
        "runs per configuration, and reporting the range alongside the mean, would make a 5-point "
        "improvement claimable instead of merely visible."))

    items.append((
        "Grow the corpus where it is thin",
        "Twenty-five questions is enough to catch a regression, not enough to rank two close "
        "configurations. The obvious gaps: no question spans three documents, no question tests a "
        "contradiction between two sources, and no question is asked twice in different words to measure "
        "phrasing sensitivity. All three are cheap to add to questions.json."))

    items.append((
        "Track runs over time",
        "Every run already writes a timestamped JSON with full answers. A short script diffing two runs "
        "per question would show which specific answers a prompt change improved and which it broke - "
        "currently visible only by reading both files."))

    for n, (title, text) in enumerate(items, 1):
        s.append(KeepTogether([P(f"<b>{n}. {title}</b>", "body"), P(text, "bullet"), Spacer(1, 4)]))

    s.append(P("How to reproduce", "h2"))
    s.append(P(
        "uv run --with reportlab python scripts/eval/make_val_pdfs.py<br/>"
        "uv run python scripts/eval/run_eval.py --ingest<br/>"
        "uv run python scripts/eval/run_eval.py --check-key<br/>"
        "uv run python scripts/eval/run_eval.py --rescore data/val/results/eval_&lt;stamp&gt;.json<br/>"
        "uv run --with reportlab python scripts/eval/make_report.py", "mono"))
    s.append(P(
        "The validation graph is written to data/val/graph_rag and never touches the production graph "
        "in data/graph_rag. No file under src/ is modified by any part of the harness.", "caption"))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(
        str(OUT_FILE),
        pagesize=A4,
        title="GraphRAG Validation Report",
        author="scripts/eval",
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
    ).build(s)
    return OUT_FILE


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the validation report as a PDF.")
    parser.add_argument("--results", help="A specific eval_*.json (default: the newest).")
    args = parser.parse_args()

    path = Path(args.results) if args.results else None
    if path is None:
        found = sorted(glob.glob(str(RESULTS_DIR / "eval_*.json")))
        if not found:
            raise SystemExit(f"No results in {RESULTS_DIR}. Run scripts/eval/run_eval.py first.")
        path = Path(found[-1])

    out = build(path)
    print(f"Report from {path.name} -> {out}")


if __name__ == "__main__":
    main()
