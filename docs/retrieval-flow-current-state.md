# Current retrieval flow: what's actually implemented

Written 2026-08-04 against `master` (`e7a7572`). Source of truth is the code
below, not the design spec (`docs/superpowers/specs/2026-08-03-retrieval-system-design.md`)
— this doc calls out anywhere the two diverge.

There are two independent paths per query, run in parallel, never mixed:
**Instant** (raw preview, no LLM) and **AI Mode** (SLM rewrite → hybrid
retrieve → RRF → rerank → LLM synthesis).

---

## 1. Instant path — raw query, no fusion

`packages/retrieval-api/src/retrieval_api/instant/search.py`

```
run_instant(query)
 ├── ES:     raw_search(query)      — plain multi_match, fuzziness AUTO
 └── Milvus: hybrid_search(query)   — dense + sparse ANN, per collection
```

- **ES side is a raw query.** `raw_search()` (`common/es_client.py:32`) runs
  `multi_match` across `facts_text`, `held_text`, `headnotes_text` with
  `fuzziness: AUTO`. No embedding, no rewrite, no RRF — a single BM25-style
  ES query, exactly the "raw ES preview" the design calls for.
- **Milvus side already uses hybrid dense+sparse ANN**, but only within
  Milvus itself — see §3 below for what that means concretely. It embeds
  the query via `gateway.embed(role="query_embed", ...)` (Voyage, per the
  hard rule) and searches all 7 collections.
- ES and Milvus results are returned **side by side, unmerged**
  (`{"es": ..., "milvus": ...}`). No RRF, no score blending between the two
  engines — matches hard rule #3 in `CLAUDE.md` ("No ranking fusion between
  ES and Milvus. `doc_id` is join-only").
- Either branch can fail independently (`_run_es` / `_run_milvus` each
  swallow exceptions into an `_error` field) — one engine being down doesn't
  kill the other's preview.

**Verdict: Instant is a raw query on both sides**, fused only inside Milvus
(dense+sparse), never across engines.

---

## 2. AI Mode path — every technique from the design spec, wired in order

`packages/retrieval-api/src/retrieval_api/ai_mode/pipeline.py`

```
run_ai_mode(query)
 1. extract_intent      — SLM rewrites query + pulls out filters
 2. resolve_allowlist   — ES lookup → doc_id allowlist (citation-index use only)
 3. retrieve            — dense search + sparse search (7 collections each) → RRF merge
 4. rerank_and_prefetch — cross-encoder rerank top-N + ES citation metadata prefetch (parallel)
 5. synthesize          — LLM answer generation with inline [doc_id] citations
```

Any exception anywhere in this chain is caught at the top and turned into
`{"ok": False, "error": ...}` — AI Mode failing never takes down Instant's
result (`pipeline.py:16`).

### Step-by-step, what's real vs. what's a stub

| Step | File | What it does | Real or raw? |
|---|---|---|---|
| Intent + rewrite | `ai_mode/intent.py` | SLM call (`role="slm"`) returns JSON: `rewritten_query`, `intent`, `filters`. Prompt explicitly rewrites old-law → new-law refs (IPC→BNS, CrPC→BNSS, Evidence Act→BSA). | Real LLM call using DeepInfra's native structured-output mode (`response_format: {"type": "json_object"}`) — no regex/brace-extraction fallback; a non-compliant response is treated as a hard failure and degrades to `_fallback_intent`. |
| Filter → allowlist | `ai_mode/filter_resolve.py` → `common/es_client.py:43` | Turns `filters` (court/act/section/party/date_range) into an ES `bool.must` query, returns matching `doc_id`s. | Real ES query. Purely a **filter allowlist** — feeds into Milvus as `doc_id in [...]`, never scored or fused. Matches hard rule #3. |
| Retrieve | `ai_mode/retrieve.py` | Embeds `rewritten_query` once (Voyage), then runs **two separate `hybrid_search` calls** across all 7 collections: one dense-only, one sparse-only (`dense_vector=None` forces the `sparse_vector` ANN branch in `milvus_client._search_one`). Flattens each into a single ranked list, then RRF-merges the two lists. | Real hybrid dense+sparse retrieval + real RRF (`rrf_merge`, k=60, standard `1/(k+rank)` formula). |
| Rerank + citation prefetch | `ai_mode/citations.py` + `ai_mode/rerank.py` | Runs concurrently: (a) cross-encoder rerank of RRF-merged candidates via `gateway.rerank(role="reranker", ...)`, keep top 3; (b) ES `mget` prefetch of citation metadata for the top 20 unique `doc_id`s by RRF score (speculative — done before rerank finishes, to save a round trip). | Real reranker model call (Qwen3-Reranker via DeepInfra, per `.env`). Prefetch is a genuine optimization, not a stub. |
| Synthesize | `ai_mode/synthesize.py` | Backfills any citation metadata missed by the speculative prefetch (chunks the reranker kept that weren't in the top-20-by-RRF set), builds a `[doc_id] excerpt` block, one LLM call (`role="synthesis"`) told to cite `doc_id` per claim. | Real LLM call. Citation correctness depends entirely on the model actually citing the bracketed IDs it was given — no post-hoc validation that a cited ID appears in `top_chunks`. |

**Verdict: AI Mode uses every technique discussed in the design — SLM query
rewrite, dense+sparse hybrid retrieval, RRF fusion, cross-encoder rerank, and
LLM synthesis with citations — nothing here is a raw/stub query.** The one
simplification vs. a "full" system: no ranking fusion between ES and Milvus
scores anywhere (by design, hard rule #3) — ES is used only for
metadata/filtering/citations, never for scored search inside AI Mode.

---

## 3. Milvus flow in detail

`packages/common/src/common/milvus_client.py`

### 3.1 Collections searched — always all 7, no routing

```python
MILVUS_COLLECTIONS = ["case_summary", "digest", "headnotes", "facts", "held", "ruling", "metadata"]
```

Every query (Instant and AI Mode) searches **all 7** in parallel
(`asyncio.gather` + `asyncio.to_thread` per collection in `hybrid_search`).
No intent-based collection routing — matches hard rule #4. `metadata` is the
one doc-level collection (row per `doc_id`); the other 6 are chunked
(`digest`, `facts`, `held`, `ruling` are explicitly chunked per
`CHUNKED_COLLECTIONS`; `case_summary`/`headnotes` are doc-level text but
still queried the same way).

### 3.2 Per-collection search — one call, two possible ANN fields

`_search_one()` picks the field based on what's passed in, it never sets both
in one call:

```
dense_vector given?
 ├── yes → anns_field="dense_vector", data=[<embedding floats>]
 └── no  → anns_field="sparse_vector", data=[<raw query text string>]
```

- **Dense branch**: `dense_vector` comes from `gateway.embed(role="query_embed", ...)`
  upstream — always Voyage (hard rule #1), because the corpus's
  `dense_vector` field was embedded with Voyage at ingestion time; any other
  embedder would put the query in a different vector space and cosine
  similarity would silently return garbage.
- **Sparse branch**: the **raw query text string** is passed as `data`, not
  a precomputed sparse vector. Milvus's server-side BM25 `Function`
  (configured at collection-creation time in the ingestion pipeline)
  converts it internally. Client code never touches `sparse_vector`
  directly for either writing or reading — matches hard rule #2 exactly.
- Both branches filter with the same `filter_expr` (the `doc_id in [...]`
  allowlist string built from ES filter resolution, when present).
- Output is normalized: doc-level `metadata` collection maps its ES `doc_id`
  field to `chunk_id` too (since there's no separate chunk key); others
  return their real `chunk_id`.

### 3.3 How dense and sparse are combined — client-side RRF, not Milvus's built-in hybrid search

Important nuance: Milvus SDK has a native `hybrid_search()` API that fuses
multiple ANN fields **inside a single Milvus call** using its own
RRFRanker/WeightedRanker. **This codebase does not use that API.** Despite
the function being named `hybrid_search()` here, it issues **one single-field
search per call** — callers get dense-only or sparse-only results back, and
fusion happens **outside Milvus, in Python**:

- Instant path: only ever calls `hybrid_search()` once with a dense vector
  (see `instant/search.py:16-21`) — no sparse call, no RRF. So "Instant
  Milvus" is dense-ANN-only, not actually hybrid despite the function name.
- AI Mode path (`ai_mode/retrieve.py`): calls `hybrid_search()` **twice** —
  once dense (`dense_vector=<embedding>`), once sparse
  (`dense_vector=None`) — flattens each of the 7-collection result sets by
  raw `distance` score, then RRF-merges the two flattened lists client-side
  via `rrf_merge()` (k=60).

So: **AI Mode is genuinely dense+sparse hybrid with RRF fusion; Instant's
"hybrid_search" call is dense-vector search only** — the name is shared
infrastructure, not a claim that Instant fuses anything.

### 3.4 Full AI Mode Milvus flow, end to end

```
rewritten_query (from SLM)
  │
  ├─► gateway.embed(role="query_embed") ──► dense_vector (Voyage)
  │
  ├─► hybrid_search(dense_vector, sparse_query_text=rewritten_query)
  │     for each of 7 collections, in parallel:
  │       client.search(anns_field="dense_vector", data=[dense_vector],
  │                      filter=doc_id allowlist, limit=50)
  │   ──► dense_by_collection: {collection: [ {chunk_id, doc_id, text, score} ]}
  │
  ├─► hybrid_search(dense_vector=None, sparse_query_text=rewritten_query)
  │     for each of 7 collections, in parallel:
  │       client.search(anns_field="sparse_vector", data=[rewritten_query],
  │                      filter=doc_id allowlist, limit=50)
  │   ──► sparse_by_collection: {collection: [ {chunk_id, doc_id, text, score} ]}
  │
  ├─► flatten each dict of lists → single list sorted by raw score, per side
  │
  └─► rrf_merge(dense_flat, sparse_flat, k=60)
        score[chunk_id] += 1/(k + rank)  summed across both ranked lists
      ──► single ranked candidate list, rrf_score attached
            ↓
      (feeds into rerank_and_prefetch → synthesize, §2 above)
```

Two full round trips to Milvus per AI Mode query (dense pass + sparse pass),
each fanned out to all 7 collections concurrently — 14 Milvus `search()`
calls total per query, before rerank/synthesis.
