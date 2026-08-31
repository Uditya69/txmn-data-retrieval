# Eval dataset expansion + headless/resumable runner

## Context

We have four eval scripts grading two things: SLM-level `extract_intent()` output
(category routing, query rewrite, filter extraction) and the full AI Mode pipeline
including synthesis (citation validity, whether the gold doc got cited). Datasets are
small (18-100 cases) and cover mostly caselaw + acts. The user wants ~300-400 queries
total across all four, "various kinds," run as hard testing to find where the system
is weak — executed unattended on a separate server against the local (slow) LLM, over
5-10 hours, with results copied back as JSON afterward. None of the four scripts
persist incrementally today, so a crash mid-run loses everything computed so far.

Scripts and datasets in scope:

| Script | Dataset (current size) | Grades |
|---|---|---|
| `slm_intent_eval.py` | `evals/slm_intent_cases.json` (100) | category + rewrite + filters |
| `intent_eval.py` | `evals/intent_filter_cases.json` (18) | filter extraction |
| `collection_routing_eval.py` | `evals/collection_routing_cases.json` (18) | category routing only |
| `retrieval_eval.py` | `evals/retrieval_cases.json` (53, caselaw) + `evals/statutory_cases.json` (40, acts) | retrieval rank + synthesis citation validity/gold-cited |

## Goals

1. Expand all four datasets with diverse query types, total ≈365 cases across the
   set (exact split below).
2. `retrieval_eval.py`'s coverage deepens across caselaw, acts, rules, articles, and
   commentary (tariff excluded — see below), using **real `gold_doc_id`s mined from the
   live ES/Milvus corpus** (not fabricated, not sourced from the sibling repo's raw
   files).
3. All four scripts get incremental, crash-safe output (`--output`, `--resume`,
   `--summarize`), sharing one small helper module instead of four bespoke
   implementations.
4. A thin headless wrapper (background + log file + PID file) so a run survives an SSH
   disconnect and can be checked on / stopped without a live shell.

## Non-goals

- No new LLM-judge rubric for synthesis quality. `retrieval_eval.py` already grades
  `citation_valid` / `gold_cited`; this round scales that up, it doesn't add a new
  grading dimension.
- No unified cross-eval mega-runner or web UI integration (there's already
  `admin_eval/` for that path if it's ever wanted) — CLI scripts + headless wrapper
  only.
- No changes to gateway/model config or `gateway_client.py` retry behavior.

## Dataset expansion

Target sizes (existing → new):

- `slm_intent_cases.json`: 100 → 160
- `intent_filter_cases.json`: 18 → 40
- `collection_routing_cases.json`: 18 → 80
- `retrieval_cases.json` (caselaw): 53 → 70
- `statutory_cases.json` — **correction**: already covers acts/rules/articles/commentary
  (10 cases each, 40 total, `category` field), not acts-only as first assumed. Expand
  within those same 4 categories instead of creating new per-category files: 40 → 80
  (20 cases each).
- **Tariff dropped from the retrieval/synthesis eval entirely** — `tariff_section` is
  not a live Milvus collection (parked in the ingestion pipeline's
  `_disabled_collections`, not indexed; see `common/schemas.py`'s routing-table
  comment and CLAUDE.md hard rule 4). There is no real corpus to mine gold docs from.
  Tariff stays represented only in the three SLM-level datasets (as an
  `expected_categories` tag) since intent classification doesn't require the target
  collection to actually be searchable — the moment tariff is enabled here, it fits
  the same `statutory_cases.json` pattern.

Combined total ≈ 365 (down from the earlier ≈425 estimate, since the 4 new per-category
files are no longer needed). New cases for the three SLM datasets cover, deliberately mixed
in: each of the six intent categories solo and in multi-label combination; filter-heavy
queries (dates, courts, party names, section numbers); and sibling-section confusion
probes (e.g. 54F vs 54B — tracking the known SLM section-conflation risk over a larger
sample instead of a handful of cases). Two distinct "vague" subtypes get separate
coverage, since they're different failure modes:

- **Vague-but-answerable** (`expect: "confident"`) — plain, colloquial, non-jargon
  phrasing that still has real legal meaning underneath (e.g. "boss won't give me
  credit for the tax I already paid abroad" instead of "foreign tax credit under
  section 91"). The classifier has to extract the legal concept from lay language and
  route correctly, not just pattern-match on statutory terminology. This is the
  regression-relevant slice: it stress-tests whether extract_intent() actually
  understands the query versus keyword-matching known legal terms.
- **Genuinely ambiguous** (`expect: "vague"`) — no recoverable legal signal at all;
  the safe-empty fallback is the correct answer, not a cop-out.

New retrieval-eval cases keep the
existing direct/indirect/adversarial structure per gold doc (see
`evals/retrieval-eval-queries.md`'s protocol) so pass/fail bands stay comparable to the
existing 53/40-case runs.

### Sourcing real gold_doc_ids

For the four new categories (and to grow caselaw/acts), gold docs are found by
querying the **running** ES/Milvus services directly (`common/es_client.py`,
`common/milvus_client.py`) rather than reading the sibling `data-extraction-pipeline`
repo's raw files — this exercises the actual indexed corpus, not the pre-ingestion
source. A one-off mining script (throwaway, not committed as permanent eval tooling)
pulls candidate docs per target collection, printing `doc_id` + citation/snippet so
queries can be hand-phrased against them the way the existing datasets were built.
Each new case is validated by running it once through `retrieval_eval.py` before being
committed, confirming the gold doc actually surfaces (not necessarily top-1 — that's
what the eval measures — but present in the corpus and reachable).

`gold_doc_ids` stays exact-match grading (no similarity/fuzzy scoring added) — but
mining should curate it liberally where it's genuinely warranted: where more than one
real corpus doc equally answers a query (e.g. several rulings on the same point of
law, several sections that all govern the same fact pattern), list all of them in that
case's `gold_doc_ids` rather than picking one arbitrarily. This is opportunistic, not a
quota — most cases will still have exactly one gold doc, same as today's datasets; only
add a second/third id when the corpus actually contains an equally-valid alternate
answer, never pad a case with a weaker doc just to inflate the list.

## Shared incremental/resumable output — `retrieval_api/eval_io.py`

New helper module used by all four scripts:

- `append_result(path: Path, record: dict) -> None` — appends one JSON object as a
  line, flushes + fsyncs immediately. This is JSONL, not a growing JSON array,
  specifically so a kill mid-write corrupts at most the one in-flight line rather than
  the whole file.
- `load_completed_ids(path: Path) -> set[str]` — reads an existing JSONL file, returns
  the `id`s already recorded; a trailing unparseable line (crash mid-write) is dropped
  rather than raising.
- `read_records(path: Path) -> list[dict]` — full read-back, used by `--summarize`.

CLI additions, consistent across all four scripts:

- `--output <path.jsonl>` — append each case's result as it completes (in addition to
  whatever final summary each script already prints).
- `--resume` — requires `--output` pointing at an existing file; skips case IDs already
  present in it, appends new ones.
- `--summarize <path.jsonl>` — standalone mode: no gateway/ES/Milvus calls, just reads
  the file back and reprints that script's existing tally format. Lets the user run
  this locally after copying a JSONL file back from the server.

`retrieval_eval.py` keeps its existing end-of-run full-payload JSON (git revision,
dataset hash, per-case results) — that format has metadata that doesn't belong on a
per-line record — and additionally appends to `--output` per case as it completes, so
a crash before the final write still leaves usable per-case results. Its existing
`--cache-dir` stage cache is unaffected (already resumes retrieval stages per case);
JSONL resume additionally skips already-graded cases entirely, including synthesis.

## Headless wrapper — `evals/run_headless.sh`

Thin bash script, not a new framework:

```
evals/run_headless.sh start <name> -- python -m retrieval_api.slm_intent_eval --dataset ... --output .eval-results/<name>.jsonl --resume
evals/run_headless.sh status <name>
evals/run_headless.sh stop <name>
```

`start` launches the given command via `nohup ... > .eval-results/headless/<name>.log 2>&1 &`
(portable across macOS and Linux, unlike `setsid` which is Linux-only), writes
`.eval-results/headless/<name>.pid`. `status` checks whether that PID is alive. `stop`
sends `SIGTERM`. Nothing more — tmux/screen remains an equally valid choice on the
user's server; this just gives a scriptable non-interactive option.

## Testing

- New tests for `eval_io.py`: append/read-back, resume skips known ids, truncated
  trailing line is tolerated not fatal.
- Extend each eval script's existing test file for `--output`/`--resume`/`--summarize`
  wiring.
- Existing `load_cases`-style schema validation (required keys, no duplicate ids)
  already runs on load — expanded dataset files must pass it as-is, no new validation
  needed.
- No test asserts on live-corpus content; a one-off (uncommitted) sanity pass resolving
  every new `gold_doc_id` against the live corpus happens during dataset authoring, not
  as a checked-in test.
