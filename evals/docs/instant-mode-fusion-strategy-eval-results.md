# Instant mode fusion-strategy eval (2026-09-07)

Question: which of Instant mode's candidate-fusion configs (`off` / `rrf` / `rerank` /
`rrf_then_rerank`) gives the best gold-`doc_id` recall? Script: `evals/scripts/instant_eval.py`. Dataset:
`evals/datasets/retrieval_cases.json` (71 queries). Full run outputs: `.eval-results/instant-eval-*.json`
(gitignored).

## Verdict

**Default Instant mode to `rrf=True`, `rerank=False`.** `rrf` and `rrf_then_rerank` are tied at
~24-25/71 in every variant tested; the reranker never beat RRF-only recall here. If a reranker step
is kept for architectural parity with AI Mode, always compose it with RRF (`rrf_then_rerank`) —
never plain `rerank` (union-based candidate gathering, strictly worse, no upside).

## Config definitions

- `off`: no fusion, single-source ranking (today's default).
- `rrf`: ES + Milvus dense fused by rank position, no cross-encoder.
- `rerank`: cross-encoder over a plain ES∪Milvus union (no RRF in candidate selection).
- `rrf_then_rerank`: RRF selects the top-20 candidates, cross-encoder reorders them. (Added
  mid-session — `rerank_instant_results` previously had `rrf=True, rerank=True` silently collapse
  to plain `rerank`; fixed to a real 3-way branch.)

## Results (all runs against `retrieval_cases.json`, 71 queries)

| Run | off | rrf | rerank | rrf_then_rerank |
|---|---|---|---|---|
| Baseline (pre-fix, full-doc-head reranker text) | 9 | 34 | 21 | n/a (not composable yet) |
| 4-config, no instruction | 5 | 25 | 11 | 25 |
| 4-config, with reranker instruction, 4B | 5 | 24 | 12 | 25 |
| 4-config, with reranker instruction, 0.6B | 4 | 23 | 9 | 25 |

`off` varies run-to-run from live ES index drift (unrelated to code) — only compare configs
*within* the same run. `rrf_then_rerank` == `rrf` exactly in the 4-config runs (same queries
pass/fail); `rerank` alone is worse in every run, mostly on adversarial (code-mixed) queries.

## Experiments on reranker candidate text

1. **ES highlight snippet** (raw_search highlight → strip tags → center-trim, same pattern as
   `keyword_mode_search`). Result: regression (`rerank` 21→15/71), zero gains, 6 `direct`-query
   losses — ES's highlight centers on wherever a keyword literally lands, not the identifying
   summary content. **Reverted.**
2. **Milvus-chunk enrichment** (kept): for any ES-only union candidate, do a targeted
   Milvus `doc_id_allowlist` lookup for that same `doc_id` and prefer its real chunk text over ES's
   highlight; falls back to ES text only if Milvus has nothing. Dense-only, doesn't touch
   `sparse_vector`. See `instant/rerank.py::_enrich_es_only_candidates_with_milvus_text`.

## Reranker instruction fix (kept, small effect)

Traced the actual DeepInfra request — no `instruction` field was ever sent (Qwen3-Reranker is
instruction-tuned; DeepInfra's docs cite a 1-5% relevance drop without one). Wired `instruction`
through `DeepInfraAdapter` → `model-gateway` `/v1/rerank` → `GatewayClient` → `instant/rerank.py`
(`_RERANK_INSTRUCTION` constant, covers all content types — statutory text, case law, commentary,
articles — matching `ai_mode/synthesize.py`'s own domain framing, not case-law-only).

4B, same-environment (71/71 stable) comparison, no-instruction vs with-instruction: `rrf` 25→24
(-1), `rerank` 11→12 (+1), `rrf_then_rerank` 25→25 (net 0). Real but small effect; doesn't change
the RRF-vs-reranker conclusion.

## Reranker model size: 4B vs 0.6B

Both with the instruction fix. `rrf_then_rerank` ties exactly (25/71 both, 24/70 on the
env-stable subset). `rerank` alone: 4B wins (12 vs 9/71). Since `rrf_then_rerank` is the only
reranker config worth using, **model size doesn't matter here** — 0.6B is the cheaper/faster
choice with no recall cost.

## AI Mode comparison (for context)

AI Mode has no plain-union reranker path — the reranker always reorders RRF-selected real Milvus
chunks. Pulled from existing `.rerank-cache/` stage cache, no new calls:

| | RRF-only | RRF+reranker |
|---|---|---|
| 71 queries | 61 | 62 |

Net +1, genuinely positive (unlike Instant). Why: stronger baseline (SLM query rewrite + intent
routing), reranker candidates are always real chunks, no worse-than-RRF fallback path exists.

## Unrelated infra finding

Milvus's `ruling` collection (2.35M rows, `num_shards=1`) takes 7+s per dense search alone;
11-collection concurrent search barely beats that one collection's latency, suggesting
server-side queuing rather than real parallelism. Not fixable from this repo — flag to whoever
owns the Milvus deployment (re-shard `ruling`, more query-node capacity).

## Key files

- `evals/scripts/instant_eval.py` — the eval script (`CONFIGS`, `--reranker-model`)
- `packages/retrieval-api/src/retrieval_api/instant/rerank.py` — `_enrich_es_only_candidates_with_milvus_text`, `_RERANK_INSTRUCTION`, the rrf/rerank composability fix
- `packages/common/src/common/es_client.py::raw_search` — highlight field added (kept, though Instant no longer relies on it as primary reranker text)
- `.rerank-cache/` — AI Mode stage cache used for the AI Mode comparison
