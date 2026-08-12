# Intent-driven RRF weighting — design

Phase 2 of the intent-extraction work (`docs/superpowers/specs/2026-08-10-intent-extraction-redesign-design.md`
explicitly deferred this). Phase 1 made `intent` classification schema-correct
and confidence-checked; this phase gives it its first consumer: biasing the
Reciprocal Rank Fusion (RRF) weighting between Milvus's dense (semantic) and
sparse (BM25) result lists in AI Mode's retrieval stage, based on the
classified query intent.

## Problem

`retrieve.py`'s `rrf_merge` currently fuses the dense-50 and sparse-50 ranked
lists with equal weight regardless of query shape. A query anchored on a
citation, party name, or section number (`citation_lookup`/`provision_lookup`)
is better served by lexical/BM25 matching; an open conceptual question
(`conceptual`) is better served by semantic/dense matching. Today's equal
weighting can't express that distinction. `intent` has been classified since
Phase 1 but consumed by nothing — this design gives it exactly one consumer.

## Non-goals / explicit constraints (unchanged from Phase 1)

- **No intent-based Milvus collection routing, ever** (CLAUDE.md hard rule).
  This design only reweights how the *same* full dense-50/sparse-50 candidate
  set gets fused — it never skips a collection or reduces the candidate set.
  Recall is unaffected; only ranking/ordering changes.
- No change to `filter_resolve.py`, `_safe_rewrite`, `_sanitize_filters`, or
  the ES allowlist mechanism.
- No change to how many results are fetched (`limit=50` per side, unchanged).

## Design

### 1. Weighted `rrf_merge`

```python
def rrf_merge(
    dense_ranked: list[dict], sparse_ranked: list[dict], k: int = 60,
    dense_weight: float = 1.0, sparse_weight: float = 1.0,
) -> list[dict]:
    scores: dict[str, float] = {}
    rows: dict[str, dict] = {}
    for ranked_list, weight in ((dense_ranked, dense_weight), (sparse_ranked, sparse_weight)):
        for rank, row in enumerate(ranked_list, start=1):
            chunk_id = row["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)
            rows.setdefault(chunk_id, row)
    ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return [{**rows[chunk_id], "rrf_score": score} for chunk_id, score in ordered]
```

Default `dense_weight=1.0, sparse_weight=1.0` preserves today's exact
behavior for any caller that doesn't pass weights — no regression for
existing callers (there are none outside `retrieve.py`, but the eval harness
and any future direct caller are unaffected either way).

### 2. Intent → weight lookup in `retrieve.py`

```python
_INTENT_RRF_WEIGHTS: dict[str, tuple[float, float]] = {
    "citation_lookup": (0.5, 1.5),
    "provision_lookup": (0.5, 1.5),
    "conceptual": (1.5, 0.5),
    "unknown": (1.0, 1.0),
}
```

`retrieve()` gains an `intent: str` parameter, looks up
`_INTENT_RRF_WEIGHTS.get(intent, (1.0, 1.0))` (the `.get` default is a
defensive fallback — `intent.py`'s `_ALLOWED_INTENTS` enum already
guarantees `intent` is always one of the four keys, but `retrieve()` doesn't
import that enum, so it degrades to neutral weighting rather than raising
`KeyError` if the two modules ever drift), and passes `dense_weight`/
`sparse_weight` through to `rrf_merge`.

### 3. Wiring in `pipeline.py`

`run_ai_mode` already computes `intent_result["intent"]` before calling
`retrieve()`. Thread it through:

```python
candidates = await retrieve(
    gateway, milvus_client, intent_result["rewritten_query"], doc_id_allowlist,
    intent_result["intent"], on_step=on_step,
)
```

### 4. Trace visibility

The existing `rrf_merge` trace step (`on_step("rrf_merge", {...})`) gains the
resolved weights in its payload (`{"dense_weight": ..., "sparse_weight": ...}`
alongside the existing `candidate_count`/`top_candidates`), so the AI Mode
trace panel can show which weighting was applied per query — useful for
debugging why a specific query's ranking looks the way it does.

## Testing

- `rrf_merge`: new tests for weighted fusion (a chunk that only appears in
  the up-weighted list should outrank an equal-rank chunk that only appears
  in the down-weighted list — today's equal-weight tests must still pass
  unchanged, since the new params default to `1.0`/`1.0`).
- `retrieve()`: new tests asserting each of the 4 intent values resolves to
  its documented weight pair and gets passed into `rrf_merge`; an unrecognized
  string falls back to `(1.0, 1.0)`.
- `pipeline.py`: existing test(s) updated so the mocked `retrieve` call
  asserts the `intent` argument is forwarded.

## Validation before merge

Unlike Phase 1 (extraction-only, validated via the cheap prompt-only
gold-filter check), this phase changes retrieval ranking directly, so it
needs a real regression check: one run of
`retrieval_eval.py --sample12` (or the full 53-query set, cache permitting)
comparing the weighted branch against the current `dev` baseline. Any
retrieval-rank regression on the 12-query sample blocks merge; the
`citation_lookup`/`provision_lookup`-weighted queries (direct-class, per the
existing eval set's own class labels) are the ones most likely to show a
*positive* effect, so a flat or negative result there specifically is worth
investigating before accepting the change.
