# Current retrieval flow: what's actually implemented

Written 2026-08-04 against `master` (`e7a7572`); updated 2026-08-11 after the
Instant mode ES redesign and AI Mode filter fix below. Source of truth is the
code, not the design spec (`docs/superpowers/specs/2026-08-03-retrieval-system-design.md`)
— this doc calls out anywhere the two diverge.

**Read `docs/SEARCH_FLOW_ES.md` first if you haven't** — notes on the sibling
`centax-node` Node service's ES query architecture (same corpus, older Node
codebase). Several fixes below are directly informed by comparing this
codebase against that one, then verifying against the *actual* live ES index
rather than trusting either codebase's assumptions. See
`docs/superpowers/specs/2026-08-11-instant-mode-es-retrieval-redesign-design.md`
for the full audit and design writeup.

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

- **ES side is a raw query, but no longer a naive one (fixed 2026-08-11).**
  `raw_search()` (`common/es_client.py`) used to run a flat `multi_match`
  across only `facts_text`/`held_text`/`headnotes_text` — fields populated on
  just 25.7% / 26.6% / 58.1% of the live index, while `heading`/`subheading`/
  `fullcontent` (100% populated) were never searched. Now it searches all six,
  with per-field boosts chosen by a **no-LLM query-shape classifier**
  (`common/query_tokenizer.classify_query_shape` → `"citation"` /
  `"provision"` / `"plain"`, built on `common/legal_lexicon` — a cleaned-up,
  typed port of `centax-node`'s `constants/token.js` lexicon, extracted via
  `scripts/extract_token_lexicon.py`), and wraps the query in a
  `function_score` using `documenttypeboost`/`court_boost`/`landmarkruling` —
  three real, populated boost fields the index already carried that nothing
  in this codebase used before. Still no embedding, no rewrite, no RRF —
  still a single ES round trip, still well under the 1s Instant-mode budget
  (no model calls anywhere in this path).
- **Milvus side already uses hybrid dense+sparse ANN**, but only within
  Milvus itself — see §4 below for what that means concretely. It embeds
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
| Filter → allowlist | `ai_mode/filter_resolve.py` → `common/es_client.py` | Turns `filters` (court/act/section/bench/judge/party/date_range) into an ES `bool.must` query, returns matching `doc_id`s. | Real ES query. Purely a **filter allowlist** — feeds into Milvus as `doc_id in [...]`, never scored or fused. Matches hard rule #3. `court`/`bench`/`section` were fixed 2026-08-11 (see below) to target real data instead of an always-empty field; `act` remains best-effort (no reliable field exists for it — see below). |
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

## 3. Live ES index field population — read before touching any ES query

Audited 2026-08-11 against the real index (`researchindex_aic_test` —
confirmed by the product owner to be the actual collection used, not a stale
test snapshot; 410,427 docs). Full raw numbers in the design spec's Motivation
section. The short version, so nobody re-guesses this from scratch:

**Reliably populated, safe to query directly:**
- `heading`, `subheading`, `fullcontent` — 100% populated, every doc.
- `groups.group.name` — real content-type taxonomy: `CASELAWS` (241,694),
  `ACT` (83,309), `RULE` (48,014), `COMMENTARY` (27,291),
  `Experts Opinion` (5,975), `Tariff` (4,144).
- `categories.name` — 16 real subject areas (DIRECT TAX LAWS, GST, Transfer
  Pricing, Bare Act, Labour Laws, IBC, Criminal Laws, Competition Law, etc.).
- `documenttypeboost` (100%, range 0–10,000), `court_boost` (99.9%, range
  0–294) — precomputed ranking signals, now used in `raw_search`'s
  `function_score` (§1 above).
- `landmarkruling` (2.1% populated, range -10 to 20; `-10` marks an excluded/
  blacklisted doc) — sparse but real, also wired into `function_score`.
- `otherinfo.judge.name` (99.4% on `CASELAWS`), `otherinfo.partyname.name`
  (100% on `CASELAWS`), `formatteddocumentdate` (100% everywhere).
- ACT/RULE-group docs: `heading` **is** the section/rule identifier, verbatim
  (`"Section - 184"`, `"Rule - 37CA"`).
- Caselaw docs: the court/bench appears as an abbreviation inside `heading`
  (`"(SC)"`, `"(Bombay)"`, `"(Chennai - Trib.)"`) — not in a separate field.

**Confirmed 0% populated — do not build a filter or ranking signal on these
without re-auditing first:**
- `masterinfo.info.{court,act,section,bench}.name` — 0% across all 410,427
  docs, every content group. This is what AI Mode's filters used to target;
  fixed 2026-08-11 (see §2 table above) by redirecting to `heading`
  (court/bench/section) or `fullcontent` (act, best-effort — no reliable
  field exists anywhere in the index for act↔document linkage).
- `searchboosttext` (centax's composite metadata boost field), `boostpopularity`,
  `incometaxactinfo`/`companyactinfo`/`incometaxruleinfo`/`tariffinfo`
  structured sub-fields — all 0%.

If you're about to add a new ES filter or ranking signal, check the real
population rate first (`client.count(query={"bool": {"filter": {"exists":
{"field": "..."}}}})` against the live index) rather than trusting the field
name or the mapping alone — this whole audit started because a plausible
field name (`masterinfo.info.court.name`) had zero actual data behind it.

---

## 4. Milvus flow in detail

`packages/common/src/common/milvus_client.py`

### 4.1 Collections searched — always all 7, no routing

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

### 4.2 Per-collection search — one call, two possible ANN fields

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

### 4.3 How dense and sparse are combined — client-side RRF, not Milvus's built-in hybrid search

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

### 4.4 Full AI Mode Milvus flow, end to end

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
