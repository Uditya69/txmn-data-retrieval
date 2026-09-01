# eval-results/

Committed, date-organized eval run outputs — for comparing "did this change make
things better or worse" over time.

Layout: `eval-results/YYYY-MM-DD/NN-slug/`, one folder per run started that day —
multiple runs per day are normal (e.g. a full sweep in the morning, then a
retrieval-only rerun after a config fix in the afternoon). `NN` is a zero-padded
sequence number (`01`, `02`, ...) so runs sort chronologically; `slug` is a short
kebab-case description of what that run actually was or what changed since the last
one (e.g. `01-full-sweep`, `02-retrieval-sparse-fixed-skip-synthesis`) — the folder
name alone should tell you what you're looking at, without opening `SUMMARY.md`.

Each run folder contains whichever of these apply (not every run touches all four
evals):

- `slm-intent.jsonl`, `collection-routing.jsonl`, `intent-filter.jsonl`,
  `retrieval.jsonl` — the raw per-case JSONL output of each eval script, one line per
  case: query, expected vs. actual, pass/fail, and the SLM's reasoning trace.
- `SUMMARY.md` — the pass/fail tallies for that run, so you can compare runs without
  opening the raw files.
- `dashboard.html` — a self-contained, offline-browsable dashboard over that run's
  `.jsonl` files (search/filter by pass/fail/error, expand any case for its full
  query/rewrite/reasoning detail). Just open it in a browser, no server needed.

Not to be confused with `.eval-results/` (gitignored) — that's `retrieval_eval.py`'s
own local scratch output (timestamped full-payload JSON with git revision/dataset
hash, `latest.json` alias) for quick ad-hoc local runs. This folder is for runs worth
keeping and comparing against later.

## Prerequisites (once per server)

`model-gateway` must be running and reachable (retrieval-api itself is NOT required —
none of the eval scripts call it):

```bash
tmux new -d -s gateway 'uv run --package model-gateway uvicorn model_gateway.main:app --host 0.0.0.0 --port 8001'
uv run python evals/preflight.py --gateway-url http://localhost:8001
```

All commands below assume `--gateway-url http://localhost:8001` and that you're at the
repo root.

## Running one eval, headless

Pick a run folder name first (see naming convention above), then:

```bash
RUN="eval-results/YYYY-MM-DD/NN-slug"
mkdir -p "$RUN"
```

**SLM intent** (category + rewrite + filters, graded together):
```bash
tmux new -d -s slm-intent "uv run python -m retrieval_api.slm_intent_eval --gateway-url http://localhost:8001 --output $RUN/slm-intent.jsonl --resume"
```

**Collection routing** (category tag routing only):
```bash
tmux new -d -s collection-routing "uv run python -m retrieval_api.collection_routing_eval --gateway-url http://localhost:8001 --output $RUN/collection-routing.jsonl --resume"
```

**Intent filters** (exact-match filter + category extraction):
```bash
tmux new -d -s intent-filter "uv run python -m retrieval_api.intent_eval --gateway-url http://localhost:8001 --output $RUN/intent-filter.jsonl --resume"
```

**Retrieval + synthesis** (ES/Milvus/RRF/reranker rank, citation validity):
```bash
tmux new -d -s retrieval "uv run python -m retrieval_api.retrieval_eval --gateway-url http://localhost:8001 --jsonl-output $RUN/retrieval.jsonl --resume"
```
Add `--skip-synthesis` to that last command to skip the (slowest) synthesis LLM call
and just grade ES/Milvus/RRF/reranker rank.

## Full sweep (all 4 evals in one command)

The `mkdir` matters here — each of these 4 scripts writes directly into `$RUN`, so
nothing needs copying/renaming afterward:

```bash
RUN="eval-results/YYYY-MM-DD/NN-full-sweep"
mkdir -p "$RUN"

tmux new -d -s full-sweep "
  uv run python -m retrieval_api.slm_intent_eval --gateway-url http://localhost:8001 --output $RUN/slm-intent.jsonl --resume &&
  uv run python -m retrieval_api.collection_routing_eval --gateway-url http://localhost:8001 --output $RUN/collection-routing.jsonl --resume &&
  uv run python -m retrieval_api.intent_eval --gateway-url http://localhost:8001 --output $RUN/intent-filter.jsonl --resume &&
  uv run python -m retrieval_api.retrieval_eval --gateway-url http://localhost:8001 --jsonl-output $RUN/retrieval.jsonl --resume
"
```

Alternative: `evals/run_all.sh` runs the same 4 scripts but **ignores `$RUN`
entirely** — it always writes to fixed, gitignored paths
(`.eval-results/slm-intent.jsonl` etc., not this folder) regardless of anything you
set beforehand, so don't bother `mkdir`ing a dated folder first if you use it:
```bash
tmux new -d -s full-sweep 'bash evals/run_all.sh'
```
Move/rename its output into a dated `eval-results/` folder afterward if you decide you
want that run kept.

## Checking on / stopping a headless run

```bash
tmux ls                              # see what's running
tmux capture-pane -t <name> -p | tail -30   # peek without attaching
tmux attach -t <name>                # attach live (Ctrl-b then d to detach)
tmux kill-session -t <name>          # stop it
```

Every eval supports `--resume`: if a run dies or you kill it, just re-run the exact
same command — it skips case IDs already in the output file and continues.

## After a run finishes

Summarize each `.jsonl` (no live calls, reads the file back):
```bash
uv run python -m retrieval_api.slm_intent_eval --summarize $RUN/slm-intent.jsonl
uv run python -m retrieval_api.collection_routing_eval --summarize $RUN/collection-routing.jsonl
uv run python -m retrieval_api.intent_eval --summarize $RUN/intent-filter.jsonl
uv run python -m retrieval_api.retrieval_eval --summarize $RUN/retrieval.jsonl
```
Write the results into `$RUN/SUMMARY.md`, then `git add`/`commit`/`push` the run
folder so it's kept for comparison against future runs.

## Runs so far

- `2026-09-01/01-full-sweep-sparse-unfixed/` — first full sweep (all 4 evals, 351
  cases). Milvus sparse ran unconditionally in this run (the `retrieval_eval.py` bug
  that always ran sparse regardless of `MILVUS_SPARSE_ENABLED` wasn't fixed yet) — its
  `retrieval.jsonl` sparse-stage ranks don't reflect production behavior; treat
  es/raw_dense/reranker numbers as the reliable ones from this run.
- `2026-09-01/02-retrieval-sparse-fixed-skip-synthesis/` — retrieval-only rerun after
  the sparse-gating fix, `--skip-synthesis`. Sparse correctly at 0/71 now; reranker
  recall improved slightly (56→60/71) vs. run 01, though 9 cases hit ES connection
  timeouts and should be rerun before treating that as a firm conclusion. See its
  `SUMMARY.md` for the full comparison table.
