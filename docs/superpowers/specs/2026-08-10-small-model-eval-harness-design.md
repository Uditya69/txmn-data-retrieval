# Small-model eval harness for AI Mode — design

## Goal

Evaluate whether 3-6B-class open-weight models can replace the current DeepInfra-hosted
models for AI Mode's `slm`, `reranker`, and `synthesis` roles, as a step toward
self-hosting. `agent_chat` is out of scope this round (P0 is AI Mode).

## Candidates (this round)

| Role | Baseline (current) | Candidate(s) |
|---|---|---|
| slm | Meta-Llama-3.1-8B-Instruct | Qwen3-4B-Instruct-2507 |
| reranker | Qwen3-Reranker-4B | Qwen3-Reranker-0.6B, bge-reranker-v2-m3 |
| synthesis | Meta-Llama-3.1-70B-Instruct | Qwen3-4B-Thinking-2507 |

All candidates run via DeepInfra API (no self-host infra stood up yet) — this round
answers "is it worth self-hosting," not "here is the self-hosted deployment."
`query_embed` role is untouched (Voyage-only, hard rule, not part of this eval).

## Changes

### 1. Gateway per-request model override

`packages/model-gateway/src/model_gateway/routes.py`: add optional `model: str | None`
to `ChatRequest`, `EmbedRequest`, `RerankRequest`. Each endpoint resolves
`req.model or ROLE_MODEL_MAP[req.role]` instead of the current strict role-only lookup.
Provider/adapter selection (`ROLE_PROVIDER_MAP[req.role]`) is unchanged — override only
picks the model string, not the provider. `DeepInfraAdapter.chat/embed/rerank` already
take `model` as a per-call parameter, so no adapter change needed.

`packages/retrieval-api/src/retrieval_api/gateway_client.py`: `chat`, `embed`, `rerank`
gain an optional `model: str | None = None` kwarg, passed through in the POST body when set.

### 2. `retrieval_eval.py` CLI flags

Add `--slm-model`, `--reranker-model`, `--synthesis-model` (each optional, default None
= current `.env` role default). Threaded into `extract_intent()`, `rerank_top_chunks()`,
and the new synthesis call via the gateway_client `model` kwarg. Existing `--run-name`
is reused for labeling candidate runs (e.g. `qwen3-4b-slm-candidate`).

### 3. Synthesis stage added to the eval

`evaluate_case()` currently stops after the reranker stage. Add a `synthesize()` call
using the rewritten query + reranked top chunks (mirrors production AI Mode). Record:

- `citation_valid`: every `[doc_id]` token in the synthesis answer is in the
  reranked/retrieved chunk set (rule-based, same pattern as
  `agents/citations.py: validate_citations()` — no LLM judge this round).
- `citation_invalid_ids`: any cited doc_id not in the retrieved set.
- `gold_cited`: whether the case's gold `doc_id` appears among the answer's citations.

### 4. Query sample

Reuse the 12-query stratified sample (per prior work, S598) so every candidate run is
fast and directly comparable. Materialize as a fixed list of query IDs (a subset of
`evals/retrieval_cases.json`) rather than re-deriving the sample each time.

### 5. Compare script

New script (`packages/retrieval-api/src/retrieval_api/compare_eval_runs.py`) takes N
`.eval-results/*.json` paths, first argument is baseline. Prints, per candidate vs
baseline: per-stage recall@pass_at delta (es/raw_dense/raw_sparse/rewritten_dense/
rewritten_sparse/rrf/reranker), citation-pass-rate delta, and mean per-stage timing delta.

## Out of scope this round

- `agent_chat` role / agentic path metrics (round-count, tool-call validity) — explicitly
  deferred.
- LLM-as-judge faithfulness scoring — rule-based citation check only.
- Actual self-hosted inference (vLLM/TGI) — DeepInfra hosts all candidates for now.
- Full 53-query set — only the 12-query sample, for this round.

## Execution plan (runs)

1 baseline (all defaults) + 1 slm candidate + 2 reranker candidates + 1 synthesis
candidate = 5 runs against the 12-query sample, each producing a timestamped
`.eval-results/*.json`, then one `compare_eval_runs.py` invocation against all 5.
