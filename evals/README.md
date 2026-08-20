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
- `docs/agent-model-comparison.md` (not in this folder — tracks the `agent_chat` role,
  which sits outside the retrieval eval sets above) also references
  `retrieval-eval-queries.md`'s pass criteria.

## Conventions

- `doc_id` gold answers are always verified against the live ES/Milvus corpus before
  being written into a dataset — never invented.
- `pass_at` conventions: direct ≤5, indirect ≤10, adversarial ≤20 (see
  `retrieval-eval-queries.md` for the rationale).
- Raw run output lands in `.eval-results/` (gitignored), not here.
