# Evals

Every eval/benchmark for this repo lives here, split into subfolders:

- `datasets/` — gold-case JSON files (machine-readable source of truth)
- `scripts/` — runner scripts (`.py`/`.sh`) that execute against a dataset
- `results/` — raw run output committed to git (one-off artifacts, not the timestamped
  per-run archive — that's `.eval-results/`, gitignored)
- `docs/` — prose write-ups mirroring a dataset/result for humans skimming a PR

Run any script from the repo root, e.g. `uv run python evals/scripts/instant_eval.py ...`.

## Case law retrieval

- `datasets/retrieval_cases.json` / [`docs/retrieval-eval-queries.md`](docs/retrieval-eval-queries.md)
  — 53 queries / 21 matched pairs against real case-law gold `doc_id`s (direct lexical,
  indirect paraphrase, adversarial-noise variants). Runs via `retrieval-eval` (see
  `packages/retrieval-api/src/retrieval_api/retrieval_eval.py`).
- `datasets/instant_rerank_sample.json` / [`docs/instant-rerank-sample.md`](docs/instant-rerank-sample.md)
  — 10-query hand-picked sample for manually A/B-testing Instant mode's `rerank` toggle.
- `scripts/instant_eval.py` / [`docs/instant-mode-fusion-strategy-eval-results.md`](docs/instant-mode-fusion-strategy-eval-results.md)
  — full 71-case sweep across Instant mode's `off`/`rrf`/`rerank`/`rrf_then_rerank` configs.
  Verdict: `rrf` and `rrf_then_rerank` are tied; the reranker never beat RRF-only recall in
  Instant mode (unlike AI Mode, where it's a modest net positive).

## Milvus dense-only diagnostics

- `scripts/milvus_dense_only_eval.py` / `results/milvus_dense_only_results.{json,csv}` —
  straight query_embed → Voyage → Milvus `dense_vector` search (no ES, sparse, rerank, or
  RRF) over a mixed `retrieval_cases.json` + `statutory_cases.json` sample.
- `scripts/keyword_only_probe.py` / `datasets/keyword_only_cases.json` /
  [`docs/keyword-only-queries.md`](docs/keyword-only-queries.md) /
  `results/keyword_only_results.{json,csv}` — bare 1-3 word statutory-reference queries;
  ES BM25 vs Milvus dense. Gold is regenerated live from ES each run, not hand-curated.

## Statutory retrieval (acts / rules / articles / commentary)

- `datasets/statutory_cases.json` / [`docs/statutory-eval-queries.md`](docs/statutory-eval-queries.md)
  — 40 queries / 20 matched pairs, Milvus-only.

## Intent classification & collection routing

- `datasets/intent_filter_cases.json` — intent-category classifier eval cases.
- `datasets/slm_intent_cases.json` — SLM intent-classifier eval cases.
- `datasets/collection_routing_cases.json` — `collections_for_intent()` routing eval cases.
  (No standalone doc — see `docs/superpowers/specs/`/`docs/superpowers/plans/` at the repo root.)

## Persona / keyword-expansion eval infra

- `scripts/build_persona_test_snapshots.py` → `results/persona_test_snapshots.json` —
  real pipeline-derived persona snapshots (input for the rigorous eval below).
- `scripts/keyword_expansion_probe.py` → `results/keyword_expansion_probe_{before,after}.json`
  — quick before/after probe.
- `scripts/keyword_expansion_rigorous_eval.py` → `results/keyword_expansion_rigorous_results.json`
  — rigorous version (adversarial mismatched-persona coverage, N-run variance).

## Model comparison results

- [`docs/small-model-eval-results.md`](docs/small-model-eval-results.md) — A/B results for
  self-hostable model candidates against DeepInfra defaults for AI Mode's `slm`, `reranker`,
  `synthesis` roles.
- `results/statutory_eval_results.xlsx` — raw statutory-eval run output.

## Other scripts

- `scripts/rerank_cap_sweep.py` + `scripts/run_rerank_cap_sweep.sh` — AI Mode reranker
  candidate-cap sweep against a `retrieval-eval --cache-dir` stage cache.
- `scripts/preflight.py`, `scripts/run_all.sh`, `scripts/run_headless.sh` — CLI harness
  utilities (see each script's own docstring/header comment for usage).

## Conventions

- `doc_id` gold answers are always verified against the live ES/Milvus corpus before being
  written into a dataset — never invented.
- `pass_at` conventions: direct ≤5, indirect ≤10, adversarial ≤20 (see
  `docs/retrieval-eval-queries.md` for the rationale).
- Timestamped per-run output lands in `.eval-results/` (gitignored) — `results/` here is only
  for a script's own committed/curated output file, not every run's archive.
