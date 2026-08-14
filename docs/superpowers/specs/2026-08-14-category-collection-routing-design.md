# Category-based Milvus collection routing — design

Builds on `docs/superpowers/specs/2026-08-13-intent-category-classification-design.md`
(the `intent: list[str]` category tag on `extract_intent`'s output — assumed implemented
first). Adds: (1) routing which Milvus collections get searched based on that tag, (2) a
matching `search_query` phrasing bias. Supersedes the RRF-weighting idea floated during
brainstorming (category-anchored lexical/dense weight mapping) — rejected outright, not
implemented. RRF stays neutral 1.0/1.0 always, exactly as the Aug 13 spec already left it.

## Motivation

`common/schemas.py::MILVUS_COLLECTIONS` now holds 11 collections (7 original case-law
collections + `act_section`/`rule_section`/`article_section`/`commentary_section`, wired in
upstream 2026-08-11). Every AI Mode query currently searches all 11 regardless of what it's
about — a pure case-law query still pays for 4 extra Milvus round-trips against acts/rules/
articles/commentary collections it has no chance of matching well, and vice versa. The new
`intent` category tag gives a real signal to narrow this.

## Scope decision

- **This changes CLAUDE.md hard rule 4** ("AI Mode searches all 11 Milvus collections every
  query. No intent-based collection routing.") — the one intentional hard-rule edit in this
  project. Confirmed with user this session.
- `tariff` has no routing target — `tariff_section` isn't in `MILVUS_COLLECTIONS` (parked in
  `_disabled_collections` upstream, not live). A `tariff`-only intent tag falls through to the
  safe-fallback behavior below, same as an empty/unrecognized tag.
- RRF weighting is explicitly **not** touched by category — stays neutral 1.0/1.0 (the
  category-anchored lexical/conceptual weight mapping considered during brainstorming was
  rejected by the user).
- No change to `doc_id_allowlist` resolution (`filter_resolve.py`/`es_client.py`) — it still
  resolves against the case-law ES index regardless of routed category. See Known edge case.

## Design

### Collection grouping (`common/schemas.py`)

```python
CATEGORY_COLLECTIONS: dict[str, list[str]] = {
    "caselaws": ["case_summary", "digest", "headnotes", "facts", "held", "ruling", "metadata"],
    "acts": ["act_section"],
    "rules": ["rule_section"],
    "articles": ["article_section"],
    "commentary": ["commentary_section"],
}
```

`caselaws` maps to the original 7 collections, including `metadata` — its fields
(`landmark_ruling`, doc-level heading/subheading) are case-doc-specific, not generic across
every document type, so it belongs in the caselaws group, not searched independently. `tariff`
has no key (see Scope decision).

### Routing function (`common/schemas.py`)

```python
def collections_for_intent(intent: list[str]) -> list[str]:
    if not intent:
        return MILVUS_COLLECTIONS
    routed = {c for tag in intent for c in CATEGORY_COLLECTIONS.get(tag, [])}
    return [c for c in MILVUS_COLLECTIONS if c in routed] or MILVUS_COLLECTIONS
```

- Empty `intent` (nothing confidently tagged, or `_fallback_intent`'s `[]`) → search all 11.
  Never worse than today's always-search-everything behavior.
- Non-empty `intent` where every tag is unroutable (`tariff`, or an unrecognized value that
  slipped through `_ALLOWED_CATEGORIES` validation somehow) → the union is empty → falls back
  to all 11 (the `or MILVUS_COLLECTIONS` clause), not zero collections.
- Multi-category tags union their groups (e.g. `["acts", "caselaws"]` → 8 collections).
- Return order follows `MILVUS_COLLECTIONS`'s existing order, not the routed set's insertion
  order — keeps trace/log output collection-order-stable regardless of tag order.

### `retrieve.py`

- Both `hybrid_search` calls (dense pass, sparse pass) use
  `collections=collections_for_intent(intent)` instead of the hardcoded `MILVUS_COLLECTIONS`.
- The existing zero-hit circuit breaker (retries unfiltered when `doc_id_allowlist` zeroed
  every collection) keeps retrying against the **same routed set**, not a widened one. A
  routed-but-genuinely-wrong-category query should surface as zero results, not silently
  fall back to searching everything — that would defeat the purpose of routing. This is a
  deliberate behavior difference from the filter-allowlist retry, which does widen (it drops
  the allowlist, not the collection set).
- `retrieve()`'s `intent: list[str]` parameter (already present from the Aug 13 spec's
  category work, previously slated for RRF-weight lookup) is repurposed: it now drives
  `collections_for_intent()` only, no weighting use.
- `rrf_merge` call keeps hardcoded `dense_weight=1.0, sparse_weight=1.0` — no `_rrf_weights`
  function, no lexical/conceptual mapping. This is the explicit rejection of that idea.

### `search_query` phrasing bias (`intent.py`, prompt only)

Add to `_LLAMA_SYSTEM_PROMPT`, after the category taxonomy block: once `intent` is decided,
phrase `search_query` to match what's being searched — if `acts`/`rules` present, prefer
Act/Rule-name-plus-section/rule-number phrasing already present in the query; if `caselaws`/
`articles` present, prefer party/court/precedent-style phrasing; if `commentary` alone, keep
plain-language phrasing. Governed by the existing `_safe_rewrite` guardrails (number/identifier
preservation, ≥60% token overlap, no invented Act/court names) — nudges word order/framing
only, not content. Add 1-2 forbidden-rewrite examples if the category taxonomy addition creates
a new hallucination surface worth guarding against explicitly (e.g. don't invent a Rule number
just because `rules` was tagged).

### CLAUDE.md rule 4

Replace:
> 4. **AI Mode searches all 11 Milvus collections every query.** No intent-based collection
> routing. (7 original + `act_section`/`rule_section`/`article_section`/`commentary_section`
> added 2026-08-11 upstream; `tariff_section` exists in the pipeline schema but is parked in
> `_disabled_collections` — not live, don't add it here yet.)

With:
> 4. **AI Mode routes which Milvus collections get searched by the `intent` category tag**,
> via `collections_for_intent()` (`common/schemas.py`). Empty/unrecognized-only tags fall back
> to searching all 11. `tariff_section` has no routing entry — not live yet
> (`_disabled_collections` upstream). RRF fusion weight stays neutral (1.0/1.0) regardless of
> category — don't reintroduce category-based dense/sparse weighting off the back of this
> rule; that idea was considered and explicitly rejected (see
> `docs/superpowers/specs/2026-08-14-category-collection-routing-design.md`).

### Known edge case (not addressed, not a blocker)

`doc_id_allowlist` (`filter_resolve.py` → `es_client.resolve_doc_id_allowlist`) resolves
against the case-law ES index regardless of routed category. If a query routes to `acts`-only
and also resolves a non-empty allowlist, those doc_ids likely won't match `act_section` rows —
this hits the existing zero-hit circuit breaker in `retrieve.py` (retries unfiltered, within
the same routed collection set). No new code needed; flagged here so it isn't mistaken for a
routing bug later.

## Code changes summary

- `common/schemas.py`: add `CATEGORY_COLLECTIONS`, `collections_for_intent()`.
- `retrieve.py`: route both `hybrid_search` calls; drop any RRF-weight-by-intent code path
  (there wasn't one live yet — `_INTENT_RRF_WEIGHTS` was already deleted by the Aug 13 spec).
- `pipeline.py`: pass `intent_result["intent"]` into `retrieve()` (already required by the
  Aug 13 spec's signature; confirm it's wired, not dropped).
- `intent.py`: prompt-only addition for `search_query` phrasing bias.
- `retrieval_eval.py`: mirror the routing call so eval runs match production behavior.
- `CLAUDE.md`: rule 4 rewrite as above.

## Testing

- `collections_for_intent()`: unit tests for empty list → all 11, single category → its group,
  multi-category → union, `tariff`-only → all 11, unrecognized-tag-only → all 11, order
  stability.
- `retrieve.py`: test that `hybrid_search` is called with the routed set for a given `intent`,
  not always `MILVUS_COLLECTIONS`. Existing zero-hit circuit-breaker tests updated to assert
  retry uses the same routed set.
- `intent.py`: best-effort eval-dataset cases (via `intent_eval.py`) checking `search_query`
  phrasing leans the right way per category — don't over-assert exact wording, SLM output
  isn't deterministic.
- `pipeline.py`: test `intent` list is passed through to `retrieve()` unchanged.
