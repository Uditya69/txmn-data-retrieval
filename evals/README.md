# Evals

Every eval/benchmark dataset for this repo lives here — the JSON is always the
machine-readable source of truth; a same-topic `.md` file (where one exists) mirrors it
in prose for humans skimming a PR or deciding what to test manually.

## Case law retrieval

- `retrieval_cases.json` / [`retrieval-eval-queries.md`](retrieval-eval-queries.md) — 53
  queries / 21 matched pairs against real case-law gold `doc_id`s (direct lexical,
  indirect paraphrase, adversarial-noise variants). Runs via `retrieval-eval` (see
  `packages/retrieval-api/src/retrieval_api/retrieval_eval.py`).
- `instant_rerank_sample.json` / [`instant-rerank-sample.md`](instant-rerank-sample.md) —
  10-query hand-picked sample (5 fresh cases, direct+indirect+mixed variants of each) for
  manually A/B-testing Instant mode's `rerank` on/off toggle.

## Milvus dense-only diagnostics

- `milvus_dense_only_eval.py` / `milvus_dense_only_results.{json,csv}` — runs a
  straight query_embed → Voyage → Milvus `dense_vector` search (no ES, no
  sparse, no rerank, no RRF) over a ~66%-direct/34%-indirect mix of
  `retrieval_cases.json` + `statutory_cases.json`. Diagnostic for how dense
  alone performs in isolation.
- `keyword_only_probe.py` / `keyword_only_cases.json` /
  [`keyword-only-queries.md`](keyword-only-queries.md) /
  `keyword_only_results.{json,csv}` — bare 1-3 word statutory-reference queries
  (`section 55`, `rule 6`, `80HH`) with no other context, comparing ES BM25
  (which nails literal heading matches) against Milvus dense (which had 0/8
  hits at `pass_at=5` on the 2026-08-25 run — see the `.md` for detail and a
  caveat on the `article 14` case). Gold here is regenerated live from ES on
  each run rather than hand-curated, since these queries are inherently
  multi-way ambiguous.

## Statutory retrieval (acts / rules / articles / commentary)

- `statutory_cases.json` / [`statutory-eval-queries.md`](statutory-eval-queries.md) — 40
  queries / 20 matched pairs, Milvus-only (no ES leg — case law and statutory content
  live in different indexes).

## Intent classification & collection routing

- `intent_filter_cases.json` — intent-category classifier eval cases.
- `slm_intent_cases.json` — SLM intent-classifier eval cases.
- `collection_routing_cases.json` — `collections_for_intent()` routing eval cases.
  (No standalone `.md` yet — see the intent/routing design docs under
  `docs/superpowers/specs/` and `docs/superpowers/plans/` for context on how these were
  built.)

## Model comparison results

- [`small-model-eval-results.md`](small-model-eval-results.md) — A/B results for
  self-hostable model candidates against DeepInfra defaults for AI Mode's `slm`,
  `reranker`, and `synthesis` roles.
- `statutory_eval_results.xlsx` — raw statutory-eval run output.

## Conventions

- `doc_id` gold answers are always verified against the live ES/Milvus corpus before
  being written into a dataset — never invented.
- `pass_at` conventions: direct ≤5, indirect ≤10, adversarial ≤20 (see
  `retrieval-eval-queries.md` for the rationale).
- Raw run output lands in `.eval-results/` (gitignored), not here.
