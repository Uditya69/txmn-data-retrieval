# ES fallback for Milvus's sparse-vector-missing collections — design

Builds on `docs/superpowers/specs/2026-08-14-category-collection-routing-design.md` (intent
category routing via `collections_for_intent()`). That spec's "Known consequence" section
flagged that `act_section`/`rule_section`/`article_section`/`commentary_section` (and,
separately, `ruling`) have no `sparse_vector` field — routing to any of them degrades RRF to
pure-dense ranking, losing lexical/keyword signal entirely for that portion of the search. This
spec closes that gap by falling back to Elasticsearch for lexical search on exactly those
collections, leaving every collection that still has a real Milvus `sparse_vector` untouched.

## Motivation

The ingestion pipeline dropped the BM25 `Function`-backed `sparse_vector` field for 5
collections to save space: `ruling`, `act_section`, `rule_section`, `article_section`,
`commentary_section` (tracked today as `MILVUS_COLLECTIONS - SPARSE_VECTOR_COLLECTIONS` in
`common/schemas.py`). `common/milvus_client.py::hybrid_search` already silently drops these
collections from the sparse pass (`if dense_vector is None: collections = [c for c in
collections if c in SPARSE_VECTOR_COLLECTIONS]`) rather than querying a field that doesn't
exist — correct, but it means these 5 collections currently get dense-only search, no keyword
matching at all.

ES holds the same underlying content (confirmed live — same `doc_id`s resolve in both stores)
and already has a well-tuned lexical query builder (`es_client.py::_build_field_query`, with
query-shape classification, phrase chunking, synonym expansion). Falling back to it recovers
lexical search for these 5 collections without touching the working Milvus-native path for the
other 6.

## Scope decision

- **Only the 5 sparse-missing collections** get ES fallback. The 6 collections that still have
  `sparse_vector` (`case_summary`, `digest`, `headnotes`, `facts`, `held`, `metadata`) keep
  native Milvus sparse search exactly as today — considered and rejected unifying all sparse
  search onto ES (see "Rejected: unify all sparse search onto ES" below).
- Trigger is a static, compile-time-known check (`collection not in SPARSE_VECTOR_COLLECTIONS`),
  not a runtime "did sparse come back empty" probe — so the ES call can run in parallel with the
  Milvus dense/sparse calls from the start, no sequencing cost.
- No raw-score fusion between ES's BM25 score and Milvus's native BM25-Function score anywhere
  (CLAUDE.md hard rule 3) — this is the central constraint the whole ranking design below is
  built to satisfy.

## Rejected: unify all sparse search onto ES

Considered replacing Milvus sparse search entirely (all 11 collections) with ES, to sidestep the
score-mixing problem below by construction (a single sparse source is trivially self-consistent).
Rejected:

- The 6 collections that still have `sparse_vector` use it correctly today (CLAUDE.md hard rule
  2's sanctioned path) — nothing broken there to fix.
- Dense and native-Milvus-sparse hits on those 6 collections currently land on the *same*
  `chunk_id` when both signals agree (same chunking, same text) — a real reinforcement signal
  RRF benefits from. Swapping sparse to ES's doc-level snippets for those collections breaks that
  chunk-identity alignment for collections where it currently works.
- Adds an ES round-trip to every AI Mode query, not just ones touching a sparse-missing
  collection.

## Design

### Category filter — `groups.group.name`

ES's `groups.group.name` field is populated, confirmed **never null** (verified live, confirmed
by a domain teammate), and maps 1:1 onto 4 of the 5 gap collections by name. The 5th
(`article_section`) is a naming mismatch, not a data problem — verified against 20 doc_ids
spanning the full id range, 20/20 consistent, and independently confirmed by a teammate familiar
with the ingestion side:

| Milvus collection     | ES `groups.group.name` |
|------------------------|-------------------------|
| `act_section`          | `ACT`                   |
| `rule_section`         | `RULE`                  |
| `commentary_section`   | `COMMENTARY`            |
| `article_section`      | `Experts Opinion`       |
| `ruling`                | `CASELAWS`              |

This becomes a new mapping in `common/schemas.py`, e.g. `ES_GROUP_FOR_COLLECTION`, alongside the
existing `CATEGORY_COLLECTIONS`/`SPARSE_VECTOR_COLLECTIONS` tables. The inverse (group name →
collection) is used to re-partition a single ES response back into per-collection buckets (see
below) — safe because the mapping is bijective.

### One ES call per query, not one per gap-collection

`collections_for_intent()` can route more than one gap-collection into a single query (e.g.
`intent: ["caselaws", "articles"]` routes both `ruling` and `article_section`, both sparse-
missing). Rather than one ES round-trip per gap-collection, issue a **single** ES call filtered
by `groups.group.name IN [<every routed gap-collection's mapped group>]` (an OR/`terms` filter).
Each hit in the response carries its own `groups.group.name`; that field is used to split the
flat ES hit list back into the same `dict[collection, list[row]]` shape
`hybrid_search`/`retrieve()` already work with — `sparse_by_collection["ruling"]` gets every hit
tagged `CASELAWS`, `sparse_by_collection["article_section"]` gets every hit tagged
`Experts Opinion`, etc. Downstream code (`_flatten`, `rrf_merge`) doesn't need to know or care
that 2 of N dict entries came from one shared ES call instead of N independent Milvus calls.

If zero gap-collections are routed for a given query (e.g. `intent: ["commentary"]` alone, once
`commentary_section` regains a sparse field — hypothetically — or more realistically any query
where the routed set happens to avoid every gap collection), no ES fallback call is made at all.

**Per-group starvation cap.** A plain top-20-overall pull risks one group crowding out another —
`groups.group.name: CASELAWS` (241,694 docs) routed alongside `Experts Opinion` (5,975 docs) in
the same call could plausibly return 19-1 or 20-0 in CASELAWS's favor on a query that happens to
skew that way, even though `article_section` was legitimately routed as relevant. Fetch stays
relevancy-ranked (ES's own score, no artificial even-split), but capped so no single group can
claim more than **15 of the 20** total slots: walk the fetched pool in relevance order, keep a hit
only while its group is under 15, stop once 20 total are kept. **Correction from implementation
(the version above, describing a separate backfill-from-excess step, is stale — see below):** an
earlier version of this design described trimming a dominant group's excess back to 15 and then
backfilling the freed slots from the other group's still-untaken hits. That's wrong — every
"excess" hit belongs to a group already at its cap by construction, so backfilling from it just
puts the group back over cap. The single relevance-ordered walk above is both correct and
sufficient: a minority group's hits get included naturally as they're encountered in the same
pass, with no separate backfill step needed. A consequence worth stating plainly: if the minority
group(s) don't have enough hits in the fetched pool to fill out to 20 once the dominant group hits
its cap, the result is simply shorter than 20 — never padded, never reaching back into the capped
group's overflow. With only one gap-group routed (the common case — most `intent` tags route to a
single gap-collection), this cap never engages.

### Snippet extraction

- Fetch 20 docs per ES fallback call (matches `raw_search`'s existing default `limit=20`), one
  snippet per doc — subject to the per-group starvation cap described below when more than one
  gap-group is routed in the same call.
- Use ES's `highlight` API on `fullcontent` with `number_of_fragments: 1` — this is relevance-
  scored fragment selection (ES's `unified` highlighter scores every candidate window by query-
  term match density/proximity and returns the single best-scoring one), not first-occurrence.
  Reuses the same query (`_build_field_query`) already built for `raw_search`, so highlight
  scoring benefits from the same fuzziness/phrase/synonym handling.
- Request an oversized character fragment, then trim/center it with `tiktoken` (`cl100k_base` —
  the same tokenizer `tm-dp/packages/data-pipeline/src/data_pipeline/chunking.py` uses) to
  **~1024 tokens**, approximating that pipeline's `CHUNK_SIZE_TOKENS` splitter cap. This is a
  token budget, not a character count — real Milvus chunks for these collections run up to
  ~1024 tokens (~4000+ characters for English legal prose), so a smaller ES snippet would be
  systematically under-scored by the reranker for carrying less context than what it's competing
  against, not because it's actually less relevant. Note this is an approximation, not an exact
  match: `_stitch_overlap()` prepends up to `CHUNK_OVERLAP_TOKENS` (100) more tokens onto most
  real stored middle-chunks, so an actual Milvus chunk can run slightly above 1024 tokens — close
  enough to target 1024 for the ES snippet, not worth replicating the overlap-stitching logic
  itself for a 100-token difference.
- No overlap-stitching, no recursive splitting, no table-atom handling — that machinery exists
  in `tm-dp` for pre-chunking a whole document at ingestion time into an ordered sequence of
  chunks. This is a single ad-hoc snippet extraction around one query match, a different problem;
  and per CLAUDE.md, this repo has no code dependency on `tm-dp` regardless.

### Row shape

Each ES-sourced row gets a synthetic `chunk_id` (`f"es:{doc_id}:{offset}"`) and a `source:
"es_fallback"` tag. Verified downstream (`citations.py`, `rerank.py`, `trace_utils.py`, the web
`TracePanel`) treats `chunk_id` as an opaque identity/dedup/display key — nothing does a lookup
against a real Milvus chunk by id — so a synthetic id is safe.

### Ranking and fusion — no raw-score mixing across sources

`retrieve.py::_flatten()` currently sorts a collection-dict's rows by raw `score` before handing
the result to `rrf_merge()` as one "ranked list" (RRF itself only consumes list **position**, via
`1/(k + rank)` — never the score value itself). Naively flattening Milvus-native and ES-origin
buckets into one shared sort-by-score list would compare two different, non-comparable BM25
scales — exactly what hard rule 3 forbids.

Fix: rank each source **locally**, then **interleave by rank** (round-robin: source-A rank 1,
source-B rank 1, source-A rank 2, source-B rank 2, ...) to build the single combined
`sparse_ranked` list that feeds the existing, unchanged `rrf_merge()`:

1. Sort the Milvus-native collections' rows together by their own score → local rank list A
   (comparable to each other — same engine, same scoring function — matches existing behavior,
   unchanged).
2. Sort the ES-origin rows together by their own ES score → local rank list B.
3. Interleave A and B by rank position to produce `sparse_ranked`.

Round-robin, not concatenation — concatenating all of A ahead of all of B would silently bury
every ES-origin row behind every Milvus-origin row by construction, regardless of true relevance.
Interleaving gives each source's own top pick equal footing. Raw score values from A and B are
never compared against each other anywhere in this process, only used to rank within their own
source — `rrf_merge()` itself is untouched, still fusing dense and sparse rank lists exactly as
it does today.

Lists A and B will very often be uneven length (e.g. 6 Milvus-native collections' combined rows
vs. 1 gap-collection's ES rows) — once the shorter list is exhausted, append the longer list's
remaining rows in their own existing rank order, don't stop early and don't pad.

**The dense pass is untouched by any of this.** Interleave-by-rank is a sparse-side-only fix.
Dense search runs the same Voyage embedding model across all 11 collections regardless of source
— a cosine-similarity score from `case_summary` and one from `article_section` are directly
comparable (same model, same vector space), so `_flatten()`'s existing single global sort-by-score
stays correct and unchanged for the dense pass. There's no ES-sourced dense signal to mix in in
the first place — ES never produces embeddings — so this section's fix has nothing to do on the
dense side.

### `doc_id_allowlist`

The ES fallback query applies the same `doc_id_allowlist` term filter on `id` that the dense
Milvus call already respects, using the same pattern `resolve_doc_id_allowlist`/`fetch_citations`
use today (`{"terms": {"id": doc_id_allowlist}}` or equivalent, ANDed with the `groups.group.name`
filter and the lexical `_build_field_query`). The existing zero-hit circuit breaker in
`retrieve.py` (retry unfiltered when allowlist zeroes every collection) applies unchanged — an ES
fallback bucket returning zero hits under a non-empty allowlist counts the same as a Milvus
bucket returning zero hits for that check.

### Timing

The ES fallback call is added to the same `asyncio.gather` that already runs the dense Milvus
call, decided upfront by static `SPARSE_VECTOR_COLLECTIONS` membership of the routed set — no
sequential "try Milvus sparse, fall back to ES" step, no latency penalty from sequencing.

## Code changes summary

- `common/schemas.py`: add `ES_GROUP_FOR_COLLECTION` mapping (5 gap collections → ES group
  name).
- `common/es_client.py`: new function (e.g. `sparse_fallback_search`) — builds the
  `groups.group.name`-filtered, `doc_id_allowlist`-filtered query reusing `_build_field_query`,
  requests highlights, returns rows partitioned by collection (via the inverse group mapping) in
  the same `dict[str, list[dict]]` shape `hybrid_search` returns.
- `common/milvus_client.py` or `retrieve.py`: wire the new ES call into the existing
  `asyncio.gather`, alongside dense and native-sparse Milvus calls.
- `retrieve.py::_flatten()` (or a new sibling function used specifically for the sparse pass):
  local-rank-then-interleave instead of a single global sort-by-score, when a sparse-side
  collection dict mixes Milvus-native and ES-origin buckets.
- New dependency: `tiktoken` (already used elsewhere in the org's stack — `tm-dp`'s own
  chunking pipeline — same tokenizer, `cl100k_base`).
- `CLAUDE.md` hard rule 3 **must be rewritten**, not left as-is. Its current text names only
  Instant mode's `rrf_merge_by_doc_id` as the sanctioned rank-fusion exception, and explicitly
  warns "don't extend raw-score blending elsewhere off the back of this exception" — read
  literally, that's a standing objection to adding a second exception without the rule itself
  being updated to name it. Add this interleave-by-rank design as a second named, sanctioned
  exception (same category: rank-position fusion, never raw-score fusion), mirroring how the
  Aug 14 routing spec rewrote rule 4's text outright when that spec changed its behavior — see
  `docs/superpowers/specs/2026-08-14-category-collection-routing-design.md`'s "CLAUDE.md rule 4"
  section for the precedent to follow.

## Testing

- `ES_GROUP_FOR_COLLECTION` / group-to-collection partitioning: unit tests for a single-group
  response, a multi-group response (mixed `ruling` + `article_section` hits in one ES response
  split correctly), and an unrecognized/missing `groups.group.name` value (shouldn't happen per
  Ameti's confirmation, but shouldn't silently drop the row either — decide and test explicit
  behavior).
- `sparse_fallback_search`: test the ES query includes the right `groups.group.name` filter for
  a given routed collection set, respects `doc_id_allowlist`, requests highlights with
  `number_of_fragments: 1`.
- Per-group starvation cap: test that a single-group response is unaffected (no trimming), and
  that a multi-group response where one group holds >15 of the top 20 gets trimmed to 15 with the
  freed slots backfilled from the other group's next-best hits, not simply dropped.
- Snippet trimming: unit test the tiktoken trim/center logic against a fragment longer than 1024
  tokens and one shorter (no-op case).
- `_flatten`/interleave: unit test that Milvus-native and ES-origin rows never get compared by
  raw score — assert output order reflects local-rank interleaving, not a global score sort,
  using two buckets whose raw score scales would produce a different (wrong) order under a naive
  global sort.
- `retrieve.py` end-to-end: test that an `intent` routing only to gap collections still returns
  ranked results (no silent empty-sparse degradation), and that mixed intent (gap + non-gap
  collections routed together) produces both native-Milvus and ES-origin rows in the final
  merged candidates.
- Existing zero-hit circuit breaker tests: extend to cover an ES fallback bucket returning zero
  hits under a non-empty `doc_id_allowlist`.

## Open items (not blockers, flagged for implementation time)

- Exact ES query shape for `sparse_fallback_search` (reusing `_build_field_query`'s boost
  profiles as-is, or a fallback-specific profile) — implementation detail, not a design decision
  requiring sign-off.
- Whether `ES_GROUP_FOR_COLLECTION` belongs in `common/schemas.py` next to
  `CATEGORY_COLLECTIONS`/`SPARSE_VECTOR_COLLECTIONS`, or in `es_client.py` closer to its only
  consumer — implementation detail.
