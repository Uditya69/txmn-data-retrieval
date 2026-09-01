# eval-results/

Committed, date-organized eval run outputs — for comparing "did this change make
things better or worse" over time. One folder per run date (`YYYY-MM-DD`), containing:

- `slm-intent.jsonl`, `collection-routing.jsonl`, `intent-filter.jsonl`,
  `retrieval.jsonl` — the raw per-case JSONL output of each of the four eval scripts
  (`--output`/`--jsonl-output`, see each script's `--help`), one line per case: query,
  expected vs. actual, pass/fail, and the SLM's reasoning trace.
- `SUMMARY.md` — the pass/fail tallies for that run, so you can compare dates without
  opening the raw files.
- `dashboard.html` — a self-contained, offline-browsable dashboard over that run's 4
  `.jsonl` files (search/filter by pass-fail-error, expand any case for its full
  query/rewrite/reasoning detail). Just open it in a browser, no server needed.

Not to be confused with `.eval-results/` (gitignored) — that's `retrieval_eval.py`'s
own local scratch output (timestamped full-payload JSON with git revision/dataset
hash, `latest.json` alias) for quick ad-hoc local runs. This folder is for runs worth
keeping and comparing against later — multi-hundred-case sweeps, before/after a model
or prompt change, etc.

To add a new dated run, re-run the eval scripts with `--output`/`--jsonl-output`
pointed at a new `eval-results/YYYY-MM-DD/` folder, then summarize each file (e.g.
`uv run python -m retrieval_api.slm_intent_eval --summarize eval-results/YYYY-MM-DD/slm-intent.jsonl`)
into that folder's `SUMMARY.md`.
