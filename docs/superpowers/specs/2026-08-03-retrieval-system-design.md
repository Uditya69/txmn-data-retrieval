# retrieval-system — Design v1

## Purpose

New standalone retrieval service for Taxmann legal/tax caselaw. Reads from
the Milvus (`aic` DB, 7 collections) and Elasticsearch stores populated by
the separate `data-extraction-pipeline` repo. Implements the two-loop "AI
Mode" query flow: an instant raw-search preview (Instant) and a
background SLM→rerank→LLM-synthesis pipeline (AI Mode), both fired from the
same query.

Fully separate repo/deployment from `data-extraction-pipeline` — no code
dependency on it. Owns its own ES/Milvus client code, config, and schema
constants (kept in sync by hand with `schemas/Milvus.json`/`schemas/ES.json`
in that repo).

## Confirmed source-of-truth facts (verified against data-extraction-pipeline code, not docs)

- Milvus collections: `case_summary`, `digest`, `headnotes`, `facts`,
  `held`, `ruling`, `metadata`.
- Chunked (token-based, `CHUNK_SIZE_TOKENS=1024`/`CHUNK_OVERLAP_TOKENS=100`,
  `tiktoken cl100k_base`): `digest`, `facts`, `held`, `ruling`. Every row —
  even one that fits in a single chunk — carries `chunk_part`/`total_chunks`
  for these four.
- Not chunked (`_single_row`, one row per section instance regardless of
  length, no `chunk_part`/`total_chunks` field): `case_summary`,
  `headnotes`, `metadata`.
  (`data-extraction-pipeline/docs/{EXTRACTION_PIPELINE.md,milvus.md}` are
  stale on this point — they describe an older word-based, ruling-only
  chunking scheme. Verified directly against
  `packages/data-pipeline/src/data_pipeline/{build_collections.py,chunking.py}`.)
- Two independent dense vector fields per row: `dense_vector` (Voyage,
  1024-dim) and `dense_vector_2` (pluggable local/DeepInfra embedder,
  1024-dim). `sparse_vector` is server-computed BM25, never client-set.
- ES holds the rich faceted metadata (`masterinfo.*`, `otherinfo.*`) that
  Milvus's `metadata` collection deliberately excludes; Milvus and ES are
  populated by separate, unrelated pipelines with no shared output file —
  join key is `doc_id` only, no ranking fusion between the two stores.

## Architecture

Docker-compose, two services:

- **`model-gateway`** — thin FastAPI service, the single seam between
  business logic and LLM/embedding/rerank providers. Routes:
  `POST /v1/chat`, `POST /v1/embed`, `POST /v1/rerank`, each taking a
  `role` (`slm`, `synthesis`, `query_embed`, `reranker`) that resolves to a
  concrete model/provider via config. DeepInfra adapter for `slm`/
  `synthesis`/`reranker`. `query_embed` is Voyage-only, not DeepInfra or any
  other provider — the Milvus corpus's `dense_vector` field was embedded
  with Voyage by `data-extraction-pipeline`, so a query embedded with a
  different model would land in a different vector space and cosine
  similarity against it would be meaningless. `retrieval-system` reuses the
  same Voyage account/key `data-extraction-pipeline` already uses for
  ingestion. Adding a second provider for the DeepInfra-backed roles later
  means a new adapter behind the same `chat`/`embed`/`rerank` Protocol — no
  caller-side change; swapping `query_embed`'s provider away from Voyage is
  not a drop-in config change, since it would require re-embedding the
  entire corpus.
- **`retrieval-api`** — FastAPI + WebSocket app implementing Instant/AI Mode,
  using LangChain for prompt/chain orchestration against `model-gateway`.

uv workspace, three packages:

```
retrieval-system/
  pyproject.toml
  docker-compose.yml
  packages/
    common/
      src/common/
        config.py          # pydantic-settings, env-driven
        milvus_client.py    # pymilvus connection, 7-collection search helpers
        es_client.py        # ES client, masterinfo/otherinfo query helpers
        schemas.py           # collection/field name constants, kept in sync
                              # by hand with schemas/Milvus.json + schemas/ES.json
    model-gateway/
      src/model_gateway/
        adapters/
          base.py            # Protocol: chat(), embed(), rerank()
          deepinfra.py
        routes.py            # /v1/chat /v1/embed /v1/rerank
        config.py            # role -> model name mapping
        main.py
    retrieval-api/
      src/retrieval_api/
        instant/
          search.py           # parallel ES + Milvus raw fetch
        ai_mode/
          intent.py           # SLM rewrite/intent/filter extraction
          filter_resolve.py   # ES filter -> doc_id allowlist
          retrieve.py         # rewritten-query Milvus fetch, RRF merge
          rerank.py            # gateway rerank call
          citations.py         # ES citation prefetch
          synthesize.py        # gateway chat (synthesis), streamed
        gateway_client.py      # thin httpx client -> model-gateway
        ws.py                  # /ws/search websocket endpoint
        main.py
```

## Data flow

One WebSocket connection per query (`/ws/search`). At `t=0` the query is
dispatched to Instant and AI Mode concurrently (`asyncio.gather` style, not
sequential).

**Instant — instant preview (~300-800ms)**
- Fork: `es_search(raw_query)` (BM25 + Snowball stemming + slop + fuzzy,
  whole-document fields — `facts_text`/`held_text`/`headnotes_text`/etc.,
  doc-level scoring) **and** `milvus_search(raw_query)` (dense ANN +
  native sparse BM25, fired across all 7 collections at once, chunk-level
  scoring, no query rewriting) — run concurrently.
- Join: emit one `instant_result` WebSocket event with both result sets
  side by side, explicitly labeled as an unranked preview. This event is
  sent exactly once and never revised later.

**AI Mode — background pipeline**
1. `gateway.chat(role="slm")` — produces rewritten query (crosswalk
   expansion, e.g. IPC→BNS), intent category, and structured filters
   (court/act/section/date/party). Must complete before step 3 — the
   rewritten query, not the raw one, gets embedded.
2. If filters were extracted: query ES `masterinfo.*` for a `doc_id`
   allowlist (pure filter lookup, not scored full-text). Skipped entirely
   if no filters were detected.
3. `gateway.embed(role="query_embed")` on the rewritten query, then Milvus
   dense+sparse fetch (own top-50/top-50, independent from Instant's raw
   fetch), scoped to the allowlist if one exists.
4. RRF (Reciprocal Rank Fusion) merges the dense-50 and sparse-50 lists
   into ~100 candidate chunks.
5. Fork: `gateway.rerank(role="reranker")` on the ~100 candidates against
   the actual query text, keeping only the top 2-3 **and**, concurrently,
   an ES citation prefetch for the top 15-20 unique `doc_id`s (by merged
   score) pulling `masterinfo` citation/court/bench/judge/party fields.
6. Join: `gateway.chat(role="synthesis")` receives the query + top 2-3
   reranked chunks + prefetched citation metadata, streams the answer back
   as `ai_mode_token` events, then a final `ai_mode_done` event with full
   citations. If the winning chunk's `doc_id` wasn't in the prefetched set,
   one on-demand ES lookup covers just that doc before `ai_mode_done`.

Confirmed invariants carried over from the artifact (not to be violated by
implementation): AI Mode searches all 7 Milvus collections every query, no
intent-based collection routing; no ranking fusion between ES and Milvus,
`doc_id` is join-only; this is a deterministic pipeline with an
intent-routing layer, not an agentic loop.

## Error handling

- Instant: if either ES or Milvus fails, still emit `instant_result` with the
  other branch's data, explicitly flagged partial (e.g.
  `{"es": null, "es_error": "...", "milvus": [...]}`) — never blocks on one
  slow/failed branch waiting for a retry.
- AI Mode: any stage failure (gateway unreachable, ES filter lookup error,
  etc.) emits a `ai_mode_error` event with the failure reason. Instant's
  result, already delivered, is untouched — AI Mode failing never retracts
  or blocks Instant.
- `model-gateway` adapter failures (provider timeout/5xx) surface as a
  typed error response (not a raw 500 passthrough) so `retrieval-api` can
  distinguish "gateway down" from "provider rejected the request".

## Testing

- `common`: unit tests for Milvus/ES client helpers against mocked
  clients (query shape, field mapping correctness).
- `model-gateway`: unit tests per adapter (request/response shape) with
  DeepInfra calls mocked at the HTTP layer; route-level tests for role→model
  resolution.
- `retrieval-api`: unit tests per AI Mode stage (intent parsing, RRF merge
  math, prefetch/rerank fork-join) with `common`/gateway_client mocked;
  one integration test running the full `/ws/search` flow against a
  docker-compose test stack with fixture Milvus/ES data, asserting both
  `instant_result` and `ai_mode_done` events arrive with expected shape.

## Open items (explicitly deferred, not blocking v1)

- Second embedder / `dense_vector_2` parity in AI Mode's Milvus fetch — v1
  queries `dense_vector` (Voyage) only, matching the existing
  `retrieval.milvus.rag` CLI's current scope; wiring in `dense_vector_2`
  is a follow-up once its corpus coverage is fuller.
- `model-gateway` provider #2 (beyond DeepInfra) — adapter interface is
  ready for it, no adapter written yet.
