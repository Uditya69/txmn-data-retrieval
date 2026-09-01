# eval-results/

Committed, date-organized eval run outputs — for comparing "did this change make
things better or worse" over time.

Layout: `eval-results/YYYY-MM-DD/pN/`, one `pN` (`p1`, `p2`, ...) per run started that
day — multiple runs per day are normal (e.g. a full sweep in the morning, then a
retrieval-only rerun after a config fix in the afternoon). Each `pN` folder is one
self-contained run and contains whichever of these apply (not every run touches all
four evals):

- `slm-intent.jsonl`, `collection-routing.jsonl`, `intent-filter.jsonl`,
  `retrieval.jsonl` — the raw per-case JSONL output of each eval script
  (`--output`/`--jsonl-output`, see each script's `--help`), one line per case: query,
  expected vs. actual, pass/fail, and the SLM's reasoning trace.
- `SUMMARY.md` — the pass/fail tallies for that run, so you can compare `pN` folders
  (same day or across days) without opening the raw files. Worth noting what changed
  since the previous `pN` at the top (e.g. "sparse now off, matches MILVUS_SPARSE_ENABLED").
- `dashboard.html` — a self-contained, offline-browsable dashboard over that run's
  `.jsonl` files (search/filter by pass-fail-error, expand any case for its full
  query/rewrite/reasoning detail). Just open it in a browser, no server needed.

Not to be confused with `.eval-results/` (gitignored) — that's `retrieval_eval.py`'s
own local scratch output (timestamped full-payload JSON with git revision/dataset
hash, `latest.json` alias) for quick ad-hoc local runs. This folder is for runs worth
keeping and comparing against later — multi-hundred-case sweeps, before/after a model
or prompt change, etc.

To start a new run, pick the next `pN` under today's date folder:

```bash
mkdir -p eval-results/YYYY-MM-DD/pN
uv run python -m retrieval_api.retrieval_eval --jsonl-output eval-results/YYYY-MM-DD/pN/retrieval.jsonl --resume
```

then summarize each file it produced (e.g.
`uv run python -m retrieval_api.retrieval_eval --summarize eval-results/YYYY-MM-DD/pN/retrieval.jsonl`)
into that folder's `SUMMARY.md`.

## Runs so far

- `2026-09-01/p1/` — first full sweep (all 4 evals, 351 cases). Note: Milvus sparse ran
  unconditionally in this run (the `retrieval_eval.py` bug that always ran sparse
  regardless of `MILVUS_SPARSE_ENABLED` wasn't fixed yet) — its `retrieval.jsonl`
  sparse-stage ranks don't reflect production behavior; treat es/raw_dense/reranker
  numbers as the reliable ones from this run.
