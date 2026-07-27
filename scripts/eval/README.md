# Validation harness

Scores the **current** GraphRAG configuration — whatever `.env` says today — out
of 100 against a purpose-built corpus of five fictional Role & Duty documents.
Nothing in `src/` is touched: the runner imports `build_rag()` and
`ANSWER_SYSTEM_PROMPT` and calls `aquery()` the same way `query()` does.

```bash
# 1. Build the corpus (only needed once, or after editing the documents)
uv run --with reportlab python scripts/eval/make_val_pdfs.py

# 2. Ingest it into its own graph and score every question
uv run python scripts/eval/run_eval.py --ingest

# later runs, reusing the ingested graph
uv run python scripts/eval/run_eval.py
```

Two offline, free modes worth knowing about:

```bash
# Is the answer key itself sound? Every reference answer must score a clean
# 100 against its own key; if it does not, the key is wrong, not the model.
# This also runs automatically before every scored run.
uv run python scripts/eval/run_eval.py --check-key

# Re-score a previous run's saved answers against the current key -
# no queries, no judge calls. Use after editing questions.json.
uv run python scripts/eval/run_eval.py --rescore data/val/results/eval_<stamp>.json
```

The validation graph is written to `data/val/graph_rag`, **not** `data/graph_rag`
— your real corpus is never mixed with the fixtures. `--ingest` wipes and
rebuilds only the validation graph.

## Files

| Path | What it is |
| --- | --- |
| `make_val_pdfs.py` | The five documents as data + a reportlab renderer. Edit here, not the PDFs. |
| `data/val/raw/*.pdf` | Generated corpus: airport, fire, school, library, data centre. |
| `data/val/questions.json` | 25 questions with reference answers and machine-checkable keys. |
| `run_eval.py` | Runs the questions, scores them, writes a report. |
| `data/val/results/eval_*.json` | Every run: settings, per-question scores, full answers. |

## Scoring

Each question is scored out of 100 and the ranking is the mean, so it reads as a
percentage.

| Component | Weight | What it measures |
| --- | --- | --- |
| `facts` | 25 | Fraction of required fact groups present. Each group is a set of alternatives; word-boundary matched, so "no" does not match inside "Northgate". |
| `traps` | 15 | None of the planted wrong answers appear. The regexes are written to fire only on a genuine error, not on a correct answer that draws the distinction. |
| `citations` | 10 | A `### References` section exists (0.4) and cites the expected source PDFs (0.6). Cross-document questions expect two. |
| `judge` | 50 | An LLM judge grades the answer against a reference answer on 0-5, with hard caps: a fabricated role title caps at 1; saying a role *may* do what the source forbids caps at 2. |

Bands: ≥90 excellent, ≥80 strong, ≥70 workable, ≥55 weak, below that failing.

`--no-judge` drops the judge and rescales the other three to 100 — free, fast,
and still catches most regressions; use it while iterating on prompts.

## What the corpus is testing

The five sites share one world (Cedar Ridge Fire & Rescue covers the airport, the
school, the library and the data centre), which makes cross-document questions
real rather than decorative. The traps are deliberate:

- **Title collisions.** Three roles end in "Duty Manager" — Airside, Terminal,
  Site — with different authority; plus Ramp Lead vs Shift Lead. `A5` and `D4`
  spring only if the model conflates them.
- **Rank ≠ authority.** A Battalion Chief outranks the Fire Marshal on scene but
  cannot close an occupancy (`F2`); a Principal cannot overrule a Fire Marshal
  order (`S4`); an Airside Duty Manager cannot lift the Safety Officer's
  stop-work (`A4`).
- **Numeric ladders.** 25 litres, $10/$50, 30 days, one day, 5 minutes,
  Severity-1 vs Severity-2 — retrieval precision, not plausibility.
- **Same task, different owner.** The school's annual fire inspection is booked
  by the Campus Safety Coordinator, the library's by the Facilities Supervisor,
  both with the same Fire Marshal (`S5`).
- **A false premise.** `A5` asserts something the documents contradict; the
  correct answer rejects the premise rather than explaining it.
- **An unanswerable.** `F5` asks for staffing and turnout figures that appear
  nowhere. Declining scores 5; inventing a plausible number scores 0.
- **Command transfer.** Both the airport and the data centre hand command to the
  Cedar Ridge Battalion Chief on arrival (`F4`, `D4`) — a two-document hop.

Difficulty mix: 6 easy, 9 medium, 5 hard, 5 very hard.

## Reading a result

`by_component` is the useful part. Low `facts` with high `judge` usually means
retrieval is fine and the answer is just terse. Low `citations` with high `facts`
is a prompt problem, not a retrieval one. Tripped traps are the signal worth
acting on — each one maps to a specific confusion the corpus was built to
provoke, and the failing question id tells you which.

## Baseline

First scored run, 2026-07-23, `google/gemini-2.5-flash`, hybrid mode, rerank on,
judge = the same model:

```
RANKING: 87.9 / 100  (strong)
By difficulty: easy 88  hard 89  medium 94  very_hard 77
By document:   airport 96  datacenter 82  fire 94  library 75  school 92
By component:  facts 90%  traps 100%  citations 72%  judge 86%
```

No trap was tripped: every planted confusion — the three Duty Managers, rank vs
authority, the numeric ladders, the false premise, the unanswerable — was
handled correctly. The two real weaknesses are elsewhere:

- **citations 72%.** Short answers drop the `### References` section and emit a
  bare `[1]` instead (`A1`, `L1`, `L2`, `F2`, `A5`, `S2`). The reference rules in
  `ANSWER_SYSTEM_PROMPT` hold on long answers and get skipped on one-liners.
- **`L5` scored 21.** Retrieval returned `[no-context]` for a question the corpus
  answers plainly, so the model correctly declined — a retrieval miss, not a
  reasoning failure. Worth re-checking in `--mode mix`.
- **`D1` scored 48.** It named the Site Duty Manager as holding sole UPS-bypass
  authority when the document gives it to the Critical Facilities Engineer and
  explicitly excludes the Site Duty Manager. The only outright wrong answer, and
  it is on an *easy* question.

## Editing the corpus

Change `DOCS` in `make_val_pdfs.py`, regenerate, then `run_eval.py --ingest`. If
you change a document's facts, update the matching `reference_answer`,
`must_include` and `must_not_include` in `data/val/questions.json` — the keys are
the specification, the PDFs are generated output.
