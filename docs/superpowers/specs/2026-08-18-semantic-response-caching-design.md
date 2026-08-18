# Semantic response caching for AI Mode and Instant mode

Date: 2026-08-18
Status: Approved for planning

## Problem

Every query re-runs the full pipeline even when a semantically identical (or
near-identical) question has already been answered:

- **AI Mode** (`run_ai_mode()`, `packages/retrieval-api/src/retrieval_api/ai_mode/pipeline.py:11`):
  SLM intent extraction → allowlist resolution → ES+Milvus retrieval → rerank
  → LLM synthesis, on every request. Two LLM calls (SLM + synthesis) plus a
  hybrid retrieval round trip, even for a rephrasing of a question already
  answered minutes earlier.
- **Instant mode**: a raw ES+Milvus preview with no LLM synthesis, but still a
  full ES+Milvus round trip per query.

There is no caching of any kind in the repo today (`lru_cache` is only used
for Mongo client/settings singletons, e.g. `packages/chat/src/chat/db.py:8`).
Callers routinely ask near-duplicate questions (different phrasing, same
intent), and a semantic — not exact-string — cache can serve these without
re-running retrieval or synthesis.

This spec was explicitly called out as out-of-scope in
`docs/superpowers/specs/2026-08-18-server-side-chat-storage-design.md` and is
designed here on its own.

## Current flow (verified against code)

- **AI Mode entry point**: `run_ai_mode()`
  (`ai_mode/pipeline.py:11-67`), called from `ws.py:146-155`. Sequence:
  `extract_intent` (SLM rewrite, line 21) → `resolve_allowlist` (27) →
  `retrieve` (RRF/hybrid search, 33-36) → `rerank_and_prefetch` (42-45) →
  `synthesize` (49-52). The final answer is sent to the client at
  `ws.py:175-180`.
- **`resolve_allowlist`'s filters (court/act/date) are derived entirely from
  the query itself** (SLM-extracted), not from user identity, subscription
  tier, or org — confirmed by reading the allowlist resolution code. This
  means a cached answer is not scoped to any particular user's document
  access, so a **global cache with no per-user scoping is safe** from a
  permission-leak standpoint.
- **`persona_context`** (derived signals from `packages/persona`) also feeds
  intent extraction and synthesis, affecting answer tone/style but not
  content correctness. A global cache means a cache hit can occasionally
  return an answer synthesized in a different user's persona style. Accepted
  tradeoff for this design — content stays correct, tone may not perfectly
  match the requesting user.
- **Embedding**: exactly one embed role exists,
  `query_embed` → Voyage (`packages/model-gateway/src/model_gateway/config.py:30-51`
  `ROLE_PROVIDER_MAP`). Reusable via `GatewayClient.embed(role, text)`
  (`packages/retrieval-api/src/retrieval_api/gateway_client.py:72-78`), already
  used the same way in `ai_mode/retrieve.py:65`.
- **Mongo**: each package (`chat`, `persona`, `auth`) defines its own
  `db.py` with `AsyncIOMotorClient` + `lru_cache`'d getters
  (e.g. `packages/chat/src/chat/db.py:8-14`). No vector index of any kind
  exists in Mongo today — all existing vector storage/search is Milvus. This
  is a first-of-its-kind Mongo vector index in the codebase, on Atlas (native
  `$vectorSearch` support confirmed available in the deployment).
- **Agentic search** (`packages/agents`) has no single query→answer
  short-circuit point — `run_agentic_search` wraps a variable-length
  tool-calling loop with citation-retry re-attempts. Out of scope here; would
  need its own design.
- **Background-task pattern**: `ws.py` already fires-and-forgets background
  work after a response completes (persona signal write, chat turn
  persistence, `ws.py:183-211`) — the cache write reuses this same pattern.

## Design

### Scope

- **In scope**: AI Mode and Instant mode.
- **Out of scope**: Agentic search (no clean entry point — see above).

### New package: `packages/semantic_cache`

Structured like `packages/chat`/`packages/persona`:

- `config.py` — pydantic-settings: Mongo URI/db (reuse the same Mongo
  deployment as `chat`/`persona`/`auth`, new collection), similarity
  threshold (`SEMANTIC_CACHE_THRESHOLD`, default `0.95`).
- `db.py` — `AsyncIOMotorClient` + `lru_cache`'d getters, same pattern as
  `packages/persona/src/persona/db.py`.
- `repository.py` — `lookup(mode, query_embedding) -> CachedResult | None`
  and `write(mode, query_text, query_embedding, result) -> None`.

**Collection: `semantic_cache`** — one document per cached query:

```jsonc
{
  "_id": "<ObjectId>",
  "mode": "ai_mode" | "instant",
  "query_text": "<str>",       // original text, for observability/debugging only
  "query_embedding": [<float>, ...],  // Voyage query_embed vector
  "result": { /* mode-specific payload, see below */ },
  "created_at": "<iso8601>"
}
```

An Atlas Vector Search index on `query_embedding`, with `mode` as a filter
field (so lookups only match within the same mode). No TTL for now —
entries are kept indefinitely; an `expireAfterSeconds` index can be added
later without any schema migration, once real staleness behavior against
corpus updates is observed.

`result` shape is mode-specific:
- **AI Mode**: the same payload shape sent to the client as `ai_mode_message`
  at `ws.py:175-180` (answer, citations, intent, etc.) — cached verbatim so a
  hit can be replayed byte-for-byte.
- **Instant mode**: the raw ES+Milvus preview payload sent to the client for
  that mode.

### Lookup/write flow (both modes, hooked in at `ws.py`)

1. On receiving a query for AI Mode or Instant mode, `ws.py` computes
   `query_embedding = await gateway.embed(role="query_embed", text=query)`
   once, before dispatching to either pipeline.
2. Calls `semantic_cache.lookup(mode, query_embedding)`:
   - Runs a Mongo `$vectorSearch` aggregation against the `semantic_cache`
     collection, filtered to the current `mode`, `limit=1`.
   - If the top result's cosine similarity is `>= SEMANTIC_CACHE_THRESHOLD`,
     it's a **hit**: return the cached `result` directly, send it to the
     client exactly as the normal pipeline would (skipping `run_ai_mode()` or
     the Instant ES/Milvus dispatch entirely).
   - Otherwise, a **miss**: proceed to run the mode's normal pipeline.
3. On a miss, after the pipeline produces its result and the response has
   been sent to the client, fire a background task (same
   fire-and-forget pattern as `record_persona_signal`/chat persistence) that
   calls `semantic_cache.write(mode, query_text, query_embedding, result)` —
   reusing the embedding already computed in step 1, no recomputation.
4. A hit still goes through the existing chat-persistence background task
   (`record_conversation_turn`) unchanged, so conversation history looks
   identical to the caller regardless of cache hit/miss.

This keeps `ai_mode/pipeline.py` and Instant's search module completely
untouched — all caching logic lives in `ws.py` (orchestration, mirroring the
existing background-task dispatch there) and `packages/semantic_cache`
(storage/lookup logic).

### Error handling

- Cache lookup failures (Mongo unreachable, `$vectorSearch` error) are caught
  and treated as a miss — the normal pipeline always runs as a fallback, and
  the failure is logged server-side only. Caching must never block or break
  a live response.
- Cache write failures are logged server-side only, same as existing
  background-task persistence failures — never surfaced to the client.

### Testing

- `packages/semantic_cache/tests` — unit tests for `repository.py`
  (`lookup`/`write`) against a test Mongo/fixture, matching however
  `packages/persona/tests` fakes Mongo access (Atlas `$vectorSearch` itself
  isn't fakeable via mongomock — use a fake collection abstraction that
  returns a configured "nearest neighbor" result, so similarity-threshold
  branching can be tested without a live Atlas cluster).
- `packages/retrieval-api/tests` — tests that a cache hit skips
  `run_ai_mode()`/Instant search and returns the cached payload; a cache miss
  runs the pipeline and triggers a background write; a lookup/write error
  degrades to a normal pipeline run without failing the response.

## Out of scope

- Agentic search caching (no single query→answer entry point today).
- TTL/expiry (deferred — can be added later as a Mongo TTL index with no
  schema change).
- Per-user or per-persona cache scoping (accepted tradeoff: content is safe
  to share globally since allowlist filters come from the query, not user
  identity; persona-driven tone mismatch on a hit is accepted).
- Two-tier/loose-match thresholds — single fixed cosine threshold only, for
  now.
- Cache invalidation tied to corpus/ingestion updates.
