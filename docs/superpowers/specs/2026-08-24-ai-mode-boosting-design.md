# Boosting AI Mode's ES fallback — design

Extends Instant mode's `boost` toggle (`common/es_client.py::_apply_boost`, additive/sum-mode
ranking boost - documenttypeboost/court_boost/landmarkruling/recency/statutory-group/current-
edition signals) into AI Mode's one and only ES touchpoint: `sparse_fallback_search`, used for
the 5 Milvus collections that shipped without a native `sparse_vector`
(`ruling`/`act_section`/`rule_section`/`article_section`/`commentary_section` -
`SPARSE_VECTOR_COLLECTIONS` gap set, see `docs/superpowers/specs/2026-08-17-milvus-sparse-es-fallback-design.md`).

## Where AI Mode touches ES, and where it doesn't

AI Mode never routes to "ES only" the way Instant mode's KEYWORD lane does. Every routed
collection gets a Milvus dense pass regardless; ES only shows up as a stand-in for the 5
gap collections' sparse search. Instant mode's INTENT lane (Milvus-only, no ES call at all)
has nothing to boost and is out of scope here - boosting an ES call that doesn't exist would
be a different, separate feature (adding a new ES call), not this one.

`retrieve()`'s pipeline: `extract_intent()` → `collections_for_intent()` → three parallel
searches (Milvus dense on every routed collection; Milvus native sparse on non-gap collections;
ES fallback sparse on gap collections only) → `_flatten()` rank-interleaves Milvus-native and
ES-origin sparse rows by local rank position (never raw score - see hard rule 3) → that merged
sparse list, plus the dense list, get fused by `rrf_merge()` (rank-position only, `1/(k+rank)`)
→ rerank → synthesize.

## Why boosting is safe here (the concern this design answers)

The worry: would boosting ES's score let it "win" unfairly once merged with Milvus? No -
by construction, for two independent reasons stacked in the pipeline:

1. **`_flatten()`'s round-robin interleave runs first**, and it never compares ES's score to
   Milvus's score at all - each source is locally rank-sorted on its own, then alternated by
   position (source-A rank1, source-B rank1, source-A rank2, ...). A boosted ES score can only
   change which ES row ranks 1st/2nd/3rd *among ES's own rows*; it cannot cross into a
   raw-score comparison against Milvus, because that comparison never happens anywhere in this
   step. (This step is round-robin, not RRF - RRF's overlap-reward doesn't apply to two
   structurally disjoint row sets that can never share a `chunk_id`.)
2. **`rrf_merge()` runs second**, fusing the already-interleaved sparse list against the dense
   list by rank position only (`1/(k+rank)`) - it never sees or uses any raw score, boosted or
   not.

So every step downstream of the boosted score only ever consumes rank position, never the
value - the same property that made the original `documenttypeboost`/`court_boost`/
`landmarkruling` multiply-mode regression (CLAUDE.md, `_wrap_function_score`) structurally
impossible to repeat here.

## What changes

- `common/es_client.py::sparse_fallback_search` gains `boost: bool = False`, threading straight
  into `build_query_preview(query, boost=boost)` - identical mechanism `raw_search` already uses.
- `retrieve_api/ai_mode/retrieve.py::retrieve()` gains `boost: bool = False`, passed to both
  `_run_es_fallback` call sites (initial + the zero-hit-allowlist retry).
- `retrieve_api/ai_mode/intent.py` (or wherever `run_ai_mode` lives) threads `boost` through to
  `retrieve()`.
- `ws.py` passes the same request-level `boost` flag (already read for Instant mode) into
  `run_ai_mode` too.

## What doesn't change

- `_flatten`, `rrf_merge`, the dense pass, and every non-gap collection - untouched.
- Instant mode - untouched, already shipped and eval-verified separately.

## Open question, not a blocker

The current-edition should-clause boost (`_CURRENT_EDITION_SHOULD_BOOST`,
`_build_field_query`) was tuned and eval-verified against literal `"Section 52"`-style Instant
queries. AI Mode's `search_query` is an LLM-rewritten natural-language query
(`extract_intent()`'s output) - whether the same section-chunk detector fires usefully there
needs its own check against `evals/retrieval_cases.json` before trusting it, the same way the
Instant boost toggle was eval-verified before shipping. Ship the toggle now; verify with eval
before recommending default-on for AI Mode specifically.
