"""Score the current GraphRAG setup against the validation corpus (0-100).

Runs every question in data/val/questions.json through the same retrieval and
answer path as `graph_rag.py query`, then scores each answer on four components:

    facts      25  fraction of required fact groups present in the answer
    traps      15  none of the planted wrong answers appear
    citations  10  a `### References` section citing the expected source PDF(s)
    judge      50  an LLM judge grading the answer against the reference answer

The final ranking is the mean of the per-question scores, so it reads directly
as a percentage. With --no-judge the first three are rescaled to 100 and the run
costs nothing beyond the queries themselves.

IMPORTANT: this uses its own working directory (data/val/graph_rag) so the
validation corpus never mixes with your real graph in data/graph_rag. Nothing in
src/ is modified — the script imports build_rag() and ANSWER_SYSTEM_PROMPT and
calls aquery() exactly as query() does.

Two things can be scored, and they are not the same thing. `--via graph`
(the default) queries LightRAG directly: a few calls per question, measuring its
retrieval and its answer prompt. `--via agent` runs the whole workflow over the
chunkless section trees — find, read, answer, verify, revise — which is what a
user actually gets, at many model calls per question. A score from one is not
comparable with a score from the other, so the results file records which.

Usage:
    python scripts/eval/run_eval.py                  # ingest if needed, then score
    python scripts/eval/run_eval.py --via agent      # score the whole agent
    python scripts/eval/run_eval.py --ingest         # force a clean re-ingest
    python scripts/eval/run_eval.py --only A4,D5     # just these questions
    python scripts/eval/run_eval.py --mode local     # try another retrieval mode
    python scripts/eval/run_eval.py --no-judge       # deterministic checks only
    python scripts/eval/run_eval.py --no-cache       # ignore LightRAG's answer cache
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Every store this run touches lives beside the validation PDFs, NOT in data/.
# All of these are read at import time by the modules below, so they have to be
# set before those imports and not a line later. The section stores were added
# with --via agent: without them the agent would answer the validation questions
# out of the real corpus, and score well doing it.
_VAL = ROOT / "data" / "val"
os.environ.setdefault("GRAPH_RAG_WORKING_DIR", str(_VAL / "graph_rag"))
os.environ.setdefault("TREE_DIR", str(_VAL / "tree"))
os.environ.setdefault("CHROMA_DIR", str(_VAL / "chroma"))

# Parsed early because --no-cache must also land before the import below.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--no-cache", action="store_true")
if _pre.parse_known_args()[0].no_cache:
    os.environ["LLM_CACHE"] = "false"

sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402
from lightrag import QueryParam  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

from graph_rag.graph_rag import (  # noqa: E402
    CITE_SOURCES,
    LLM_BASE_URL,
    LLM_CACHE_ENABLED,
    LLM_MODEL,
    RERANK_ENABLED,
    RERANK_MODEL,
    WORKING_DIR,
    build_rag,
    ingest,
)
from graph_rag.prompts import ANSWER_SYSTEM_PROMPT  # noqa: E402

load_dotenv()

CORPUS_DIR = _VAL / "raw"
QUESTIONS_FILE = _VAL / "questions.json"
RESULTS_DIR = _VAL / "results"
TREE_DIR = _VAL / "tree"

WEIGHTS = {"facts": 25.0, "traps": 15.0, "citations": 10.0, "judge": 50.0}
BANDS = [(90, "excellent"), (80, "strong"), (70, "workable"), (55, "weak"), (0, "failing")]


# --- Matching helpers ----------------------------------------------------------

def _pattern(literal: str) -> str:
    """Turn an answer-key string into a regex.

    Plain strings match on word boundaries so "no" does not match inside
    "Northgate" and "$10" does not match inside "$100". A string prefixed with
    "re:" is used as a raw regex.
    """
    if literal.startswith("re:"):
        return literal[3:]
    esc = re.escape(literal)
    left = r"(?<!\w)" if literal[:1].isalnum() or literal[:1] == "_" else ""
    right = r"(?!\w)" if literal[-1:].isalnum() or literal[-1:] == "_" else ""
    return f"{left}{esc}{right}"


def _hit(text: str, literal: str) -> bool:
    return re.search(_pattern(literal), text, re.IGNORECASE | re.DOTALL) is not None


def _references_section(answer: str) -> str:
    """The text after the last References heading, or '' if there is none."""
    matches = list(re.finditer(r"#{0,6}\s*references\s*:?\s*$", answer, re.IGNORECASE | re.MULTILINE))
    return answer[matches[-1].end():] if matches else ""


# --- Scoring -------------------------------------------------------------------

def score_deterministic(q: dict, answer: str) -> dict:
    """Facts / traps / citations for one answer. Each subscore is 0..1."""
    groups = q.get("must_include", [])
    missing = [g for g in groups if not any(_hit(answer, alt) for alt in g)]
    facts = 1.0 if not groups else (len(groups) - len(missing)) / len(groups)

    forbidden = q.get("must_not_include", [])
    tripped = [p for p in forbidden if re.search(p, answer, re.IGNORECASE | re.DOTALL)]
    traps = 1.0 if not forbidden else max(0.0, 1.0 - len(tripped) / len(forbidden))

    expected = q.get("expected_sources", [])
    refs = _references_section(answer)
    has_refs = bool(refs.strip())
    # Cited if the PDF name appears in the References section, or anywhere in the
    # answer when the model inlined its citations instead.
    cited = [s for s in expected if s in refs or s in answer]
    if not expected:
        citations = 1.0  # unanswerable: nothing to cite, so nothing to lose
    else:
        citations = 0.4 * (1.0 if has_refs else 0.0) + 0.6 * (len(cited) / len(expected))

    return {
        "facts": facts,
        "traps": traps,
        "citations": citations,
        "missing_facts": missing,
        "tripped_traps": tripped,
        "cited_sources": cited,
        "has_references": has_refs,
    }


def check_key(questions: list[dict]) -> list[str]:
    """Sanity-check the answer key against its own reference answers.

    Every reference answer is written from the source documents, so it must score
    a clean 1.0 on facts, traps and citations. If it does not, the key is wrong,
    not the model — usually a must_not_include regex that also fires on the
    correct answer (`can` matching inside `cannot` is the classic one). Cheap,
    offline, and it runs before every scored run.
    """
    problems = []
    for q in questions:
        refs = "\n\n### References\n" + "\n".join(f"- {s}" for s in q["expected_sources"])
        sub = score_deterministic(q, q["reference_answer"] + refs)
        for group in sub["missing_facts"]:
            problems.append(f"{q['id']}: reference answer misses required fact {group}")
        for trap in sub["tripped_traps"]:
            problems.append(f"{q['id']}: trap fires on the reference answer -> {trap}")
        if sub["citations"] < 1.0:
            problems.append(f"{q['id']}: expected sources not detected in the reference block")
    return problems


JUDGE_PROMPT = """\
You are grading one answer produced by a retrieval-augmented system over a set of \
fictional "Role & Duty" documents. You are given the question, a REFERENCE answer \
written from the source documents, and the CANDIDATE answer.

Grade the CANDIDATE against the REFERENCE on a 0-5 integer scale:

5  Everything the reference establishes is present and correct. No role, title or \
unit that the reference does not support. Scope limits and "who may not do this" \
distinctions are stated correctly.
4  Correct and complete on the decisive points; a minor detail is missing or vague.
3  The main conclusion is right but a load-bearing qualifier, threshold, or named \
role is missing or wrong.
2  Partially right: it identifies the topic but gets the authority, the threshold, \
or the direction of a rule wrong.
1  Wrong conclusion, or right-sounding text built on a fabricated role or rule.
0  Does not answer, or contradicts the reference outright.

Grade the substance, not the length or the prose. Specific rules:
- Fabricating a role title that is not in the reference caps the score at 1.
- If the reference says a role CANNOT do something and the candidate says it can \
(or stays silent when the question asks precisely that), cap the score at 2.
- If the reference says the documents do not contain the answer, then declining is \
correct and deserves 5; supplying a plausible figure anyway is a 0.
- Extra correct detail beyond the reference is not penalised.

Return ONLY a JSON object: {{"score": <0-5>, "reason": "<one sentence>"}}

QUESTION:
{question}

REFERENCE ANSWER:
{reference}

CANDIDATE ANSWER:
{candidate}
"""


async def judge(client: AsyncOpenAI, model: str, q: dict, answer: str) -> tuple[float, str]:
    """Ask the judge model to grade one answer. Returns (0..1, reason)."""
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        question=q["question"],
                        reference=q["reference_answer"],
                        candidate=answer,
                    ),
                }
            ],
            temperature=0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group(0) if match else raw)
        return max(0.0, min(5.0, float(data["score"]))) / 5.0, str(data.get("reason", ""))
    except Exception as exc:  # a judge failure must not sink the run
        return float("nan"), f"judge error: {type(exc).__name__}: {exc}"


def combine(sub: dict, judge_score: float, use_judge: bool) -> float:
    """Weighted 0-100 score for one question."""
    parts = dict(WEIGHTS)
    if not use_judge or judge_score != judge_score:  # NaN check
        parts.pop("judge")
        total_weight = sum(parts.values())
        return sum(parts[k] * sub[k] for k in parts) / total_weight * 100.0
    return (
        parts["facts"] * sub["facts"]
        + parts["traps"] * sub["traps"]
        + parts["citations"] * sub["citations"]
        + parts["judge"] * judge_score
    )


def band(score: float) -> str:
    return next(label for threshold, label in BANDS if score >= threshold)


# --- Run -----------------------------------------------------------------------

async def _gather(questions: list[dict], concurrency: int, answer_one) -> dict[str, dict]:
    """Run `answer_one` over every question, `concurrency` at a time."""
    sem = asyncio.Semaphore(concurrency)
    out: dict[str, dict] = {}

    async def one(q: dict) -> None:
        async with sem:
            start = time.perf_counter()
            try:
                text = await answer_one(q)
            except Exception as exc:  # noqa: BLE001 - recorded as the answer
                text = f"[QUERY FAILED] {type(exc).__name__}: {exc}"
            out[q["id"]] = {"answer": text, "seconds": time.perf_counter() - start}
            print(f"  answered {q['id']} ({out[q['id']]['seconds']:.1f}s)")

    await asyncio.gather(*(one(q) for q in questions))
    return out


async def answer_via_graph(questions: list[dict], mode: str, concurrency: int) -> dict[str, dict]:
    """Query the graph once per question. Returns {id: {answer, seconds}}."""
    rag = await build_rag()
    system_prompt = ANSWER_SYSTEM_PROMPT if CITE_SOURCES else None

    async def one(q: dict) -> str:
        return await rag.aquery(
            q["question"], system_prompt=system_prompt, param=QueryParam(mode=mode)
        )

    try:
        return await _gather(questions, concurrency, one)
    finally:
        await rag.finalize_storages()


async def answer_via_agent(questions: list[dict], concurrency: int) -> dict[str, dict]:
    """Run the whole agent workflow once per question.

    A different thing to measure, and worth being explicit about which. `--via
    graph` scores LightRAG's retrieval and its answer prompt. This scores the
    system: chunkless retrieval, the researcher deciding what to read, the
    responder writing from it, and the verifier sending it back when it is not
    grounded. It costs many model calls per question rather than a few.

    Imported here rather than at module scope because importing `agent` starts
    the ticket MCP server to discover its tool — a subprocess this run has no
    use for unless it is actually going to run the agent.

    `state.answer` is taken, not `state.reply`: the ticket offer is a
    conversational nicety with no bearing on whether the answer is right, and
    including it would put "would you like me to file a ticket" in front of a
    judge grading factual accuracy.
    """
    from agent.graph import run_workflow

    async def one(q: dict) -> str:
        state, _ = await run_workflow(q["question"], tags=["eval"])
        return state.answer or "[NO ANSWER PRODUCED]"

    return await _gather(questions, concurrency, one)


async def run(args: argparse.Namespace) -> int:
    spec = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    questions = spec["questions"]
    if args.only:
        wanted = {q.strip().upper() for q in args.only.split(",")}
        questions = [q for q in questions if q["id"].upper() in wanted]
        if not questions:
            print(f"No questions matched --only {args.only}")
            return 1

    # Re-score answers from an earlier run against the current answer key. No
    # queries, no judge calls, no cost - use this after editing questions.json.
    if args.rescore:
        prior = json.loads(Path(args.rescore).read_text(encoding="utf-8"))
        by_id = {q["id"]: q for q in questions}
        results = []
        for r in prior["results"]:
            if r["id"] not in by_id:
                continue
            sub = score_deterministic(by_id[r["id"]], r["answer"])
            judge_score = float("nan") if r["judge"] is None else r["judge"]
            results.append({**r, **sub, "score": combine(sub, judge_score, not args.no_judge)})
        print(f"Re-scored {len(results)} answer(s) from {Path(args.rescore).name} against the current key.")
        report(results, args)
        return 0

    problems = check_key(questions)
    if problems:
        print("Answer-key self-check FAILED - fix the key before trusting a score:")
        for p in problems:
            print(f"  {p}")
        if args.check_key:
            return 1
        print()
    elif args.check_key:
        print(f"Answer-key self-check passed for {len(questions)} question(s).")
        return 0

    if not CORPUS_DIR.exists() or not list(CORPUS_DIR.glob("*.pdf")):
        print(f"No PDFs in {CORPUS_DIR}. Run: uv run --with reportlab python scripts/eval/make_val_pdfs.py")
        return 1

    # Ingest the validation corpus into its own stores if asked, or if empty.
    # The graph is needed either way — `--via graph` queries it, and `--via
    # agent` escalates to it, so scoring the agent against an empty graph would
    # measure a cascade with its last tier missing and report it as a result.
    # The section trees are only needed by the agent.
    if args.via == "agent":
        from doctree import ingest_directory

        if args.ingest and TREE_DIR.exists():
            print(f"Clearing {TREE_DIR}")
            shutil.rmtree(TREE_DIR)
        have_trees = TREE_DIR.exists() and any(TREE_DIR.glob("*.json"))
        if not have_trees:
            print(f"Parsing {CORPUS_DIR} -> {TREE_DIR}")
            # Docling and Chroma are both synchronous, and this is the one place
            # in the harness that would otherwise block the event loop for
            # minutes with nothing else able to start.
            total = await asyncio.to_thread(ingest_directory, CORPUS_DIR)
            print(f"Indexed {total} sections.\n")

    graph_file = WORKING_DIR / "graph_chunk_entity_relation.graphml"
    if args.ingest and WORKING_DIR.exists():
        print(f"Clearing {WORKING_DIR}")
        shutil.rmtree(WORKING_DIR)
    if args.ingest or not graph_file.exists():
        print(f"Ingesting {CORPUS_DIR} -> {WORKING_DIR}")
        total = await ingest(CORPUS_DIR)
        print(f"Ingested {total} sections.\n")

    what = "the whole agent" if args.via == "agent" else f"the graph, mode {args.mode}"
    print(
        f"Model {LLM_MODEL} | via {args.via} ({what}) | rerank "
        f"{RERANK_MODEL if RERANK_ENABLED else 'off'} | answer cache "
        f"{'on' if LLM_CACHE_ENABLED else 'off'} | {len(questions)} questions"
    )
    if LLM_CACHE_ENABLED:
        print("  note: cache on - repeated runs may replay earlier answers (--no-cache to force fresh)")
    print()

    if args.via == "agent":
        answers = await answer_via_agent(questions, args.concurrency)
    else:
        answers = await answer_via_graph(questions, args.mode, args.concurrency)

    use_judge = not args.no_judge
    judge_client = None
    if use_judge:
        judge_client = AsyncOpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=LLM_BASE_URL)
        print(f"\nJudging with {args.judge_model} ...")

    results = []
    for q in questions:
        answer = answers[q["id"]]["answer"]
        sub = score_deterministic(q, answer)
        judge_score, reason = (float("nan"), "")
        if use_judge:
            judge_score, reason = await judge(judge_client, args.judge_model, q, answer)
        results.append(
            {
                **{k: q[k] for k in ("id", "doc", "difficulty", "trap", "question")},
                "score": combine(sub, judge_score, use_judge),
                "judge": None if judge_score != judge_score else judge_score,
                "judge_reason": reason,
                "seconds": answers[q["id"]]["seconds"],
                "answer": answer,
                **sub,
            }
        )

    report(results, args)
    return 0


# --- Reporting -----------------------------------------------------------------

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _group(results: list[dict], key: str) -> dict[str, float]:
    out: dict[str, list[float]] = {}
    for r in results:
        out.setdefault(r[key], []).append(r["score"])
    return {k: _mean(v) for k, v in sorted(out.items())}


def report(results: list[dict], args: argparse.Namespace) -> None:
    overall = _mean([r["score"] for r in results])

    print("\n" + "=" * 78)
    print(f"{'ID':<4} {'DOC':<11} {'DIFFICULTY':<11} {'SCORE':>6}  NOTES")
    print("-" * 78)
    for r in sorted(results, key=lambda r: r["score"]):
        notes = []
        if r["tripped_traps"]:
            notes.append(f"{len(r['tripped_traps'])} trap(s)")
        if r["missing_facts"]:
            notes.append(f"missed {len(r['missing_facts'])} fact(s)")
        if r["citations"] < 1.0:
            notes.append("citations")
        print(
            f"{r['id']:<4} {r['doc']:<11} {r['difficulty']:<11} "
            f"{r['score']:>6.1f}  {', '.join(notes) or 'clean'}"
        )
    print("=" * 78)

    print(f"\nRANKING: {overall:.1f} / 100  ({band(overall)})\n")

    print("By difficulty: " + "  ".join(f"{k} {v:.0f}" for k, v in _group(results, "difficulty").items()))
    print("By document:   " + "  ".join(f"{k} {v:.0f}" for k, v in _group(results, "doc").items()))
    comp = {
        "facts": _mean([r["facts"] for r in results]) * 100,
        "traps": _mean([r["traps"] for r in results]) * 100,
        "citations": _mean([r["citations"] for r in results]) * 100,
    }
    judged = [r["judge"] for r in results if r["judge"] is not None]
    if judged:
        comp["judge"] = _mean(judged) * 100
    print("By component:  " + "  ".join(f"{k} {v:.0f}%" for k, v in comp.items()))

    worst = [r for r in sorted(results, key=lambda r: r["score"])[:5] if r["score"] < 90]
    if worst:
        print("\nWeakest answers:")
        for r in worst:
            print(f"\n  [{r['id']} {r['score']:.0f}/100] {r['question']}")
            if r["missing_facts"]:
                print(f"      missing: {'; '.join(' | '.join(g) for g in r['missing_facts'])}")
            for t in r["tripped_traps"]:
                print(f"      trap:    {t}")
            if r["citations"] < 1.0:
                print(f"      cited:   {r['cited_sources'] or 'nothing expected was cited'}")
            if r["judge_reason"]:
                print(f"      judge:   {r['judge_reason']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "timestamp": stamp,
        "overall": overall,
        "band": band(overall),
        "settings": {
            "llm_model": LLM_MODEL,
            "judge_model": args.judge_model if not args.no_judge else None,
            # `via` first: two results files with different values here are not
            # comparable, and reading one without knowing which path produced it
            # is how a graph score gets quoted as an agent score.
            "via": args.via,
            "mode": args.mode if args.via == "graph" else None,
            "rerank": RERANK_MODEL if RERANK_ENABLED else None,
            "llm_cache": LLM_CACHE_ENABLED,
            "working_dir": str(WORKING_DIR),
            "tree_dir": str(TREE_DIR) if args.via == "agent" else None,
        },
        "by_difficulty": _group(results, "difficulty"),
        "by_document": _group(results, "doc"),
        "by_component": comp,
        "results": results,
    }
    out_file = RESULTS_DIR / f"eval_{stamp}.json"
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nFull answers and per-question detail: {out_file.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the GraphRAG setup against data/val.")
    parser.add_argument("--ingest", action="store_true", help="Wipe and re-ingest the validation corpus first.")
    parser.add_argument(
        "--via",
        default="graph",
        choices=["graph", "agent"],
        help=(
            "What to score. 'graph' queries LightRAG directly — a few calls per "
            "question, and it measures retrieval plus one answer prompt. 'agent' "
            "runs the whole workflow over the section trees, which is what the "
            "user actually gets, at many model calls per question."
        ),
    )
    parser.add_argument("--mode", default="hybrid", choices=["naive", "local", "global", "hybrid", "mix"])
    parser.add_argument("--only", help="Comma-separated question ids, e.g. A4,D5")
    parser.add_argument("--no-judge", action="store_true", help="Deterministic checks only (no judge calls).")
    parser.add_argument("--judge-model", default=os.environ.get("EVAL_JUDGE_MODEL", LLM_MODEL))
    parser.add_argument("--concurrency", type=int, default=3, help="Parallel questions (default 3).")
    parser.add_argument("--no-cache", action="store_true", help="Disable LightRAG's answer cache for this run.")
    parser.add_argument(
        "--rescore",
        metavar="RESULTS_JSON",
        help="Re-score a previous run's saved answers against the current key (offline, free).",
    )
    parser.add_argument(
        "--check-key",
        action="store_true",
        help="Validate the answer key against its own reference answers and exit (offline, free).",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
