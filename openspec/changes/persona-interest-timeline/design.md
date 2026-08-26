## Context

The live implementation this change replaces is `packages/persona` (merged to `dev` via PR #5):
one Mongo `personas` document per `user_id`, updated by `record_signal` via read-modify-
`replace_one`, with `category_affinity` as a lifetime cumulative mean over six fixed intent tags
and `expertise_level`/`query_style` as majority-vote fields gated on `query_count >= 20`. Wiring
lives in `packages/retrieval-api/src/retrieval_api/ws.py` (persona read ~L120-136, async write
~L280-283) and `ai_mode/persona_signal.py` (`extract_expertise_signal`, `record_persona_signal`).
See proposal.md for why this is insufficient. Constraints carried forward unchanged:
- Python 3.11 (root `CLAUDE.md` hard rule 5).
- Persona never feeds RRF weights or Milvus collection routing (hard rule 4).
- Guest requests (no/invalid token) get byte-identical behavior — no reads, no writes.
- The async write must never block or fail the user-visible response.
- Same shared MongoDB instance/env vars as `packages/auth`, no new infra dependency mandated.

## Goals / Non-Goals

**Goals:**
- Replace the flat lifetime-mean document with an append-only event history plus a
  reproducibly-derived current snapshot (specs: `persona-timeline-storage`).
- Keep the *inference* pipeline deterministic and cheap where possible — one SLM call per query
  for extraction, everything downstream (clustering thresholds, scoring, state transitions) is
  plain arithmetic/heuristics, not further model calls (per proposal §10's explicit rejection of
  "ask an LLM daily what the persona is").
- Make topic identity emerge from query content (semantic + entity + temporal + behavioral
  similarity), not be limited to the existing six coarse intent tags.
- Preserve every existing hard constraint (guest transparency, non-blocking write, no RRF/routing
  leakage, Python 3.11, shared Mongo).

**Non-Goals:**
- Not building a general-purpose event-sourcing framework — this is scoped to persona/interest
  tracking only.
- Not introducing a vector database or graph database as new infra (design reuses whatever
  embedding capability already exists in this repo, e.g. `model-gateway`'s embed role, for topic
  similarity — no new vector store).
- Not real-time/streaming state transitions — derivation can run at read time or on a short-lived
  cache, not via a continuously running stream processor.
- Not solving interaction-signal instrumentation (click/open/save tracking on the frontend) as
  part of this change — `persona-interest-evidence`'s requirements are written to degrade
  gracefully when only "query submitted" is available today; richer signals are additive later.

## Decisions

### 1. Storage shape: two Mongo collections, not one
- `persona_events` (append-only): one document per query event —
  `{user_id, query_text, concepts, legal_entities, research_objective, specificity, confidence,
  evidence_weight, topic_id, episode_id, timestamp}`. Never updated after insert, only appended
  (satisfies `persona-timeline-storage`'s append-only requirement) and indexed on
  `(user_id, timestamp)` and `(user_id, topic_id)`.
- `persona_topics`: one document per (user_id, topic) — `{user_id, topic_id, label,
  representative_embedding, state, state_history: [{state, entered_at}], episodes: [{episode_id,
  started_at, ended_at | null}], score_series: [{t, score}] | score computed on read from events}`.
  This *is* mutable (state, score cache) but is a derived/cache layer, not the source of truth —
  `persona-timeline-storage`'s "current snapshot is derived, reproducible from history" requirement
  is satisfied because `persona_topics` can always be rebuilt from `persona_events` alone.
- Alternative considered: single collection with embedded event arrays per topic document.
  Rejected — unbounded array growth per topic, and loses the clean "recompute snapshot from raw
  events" story since events and derived state would be intermingled in one mutable document.

### 2. Topic clustering: embedding similarity + entity overlap, threshold-based, no LLM call
- Each Query Understanding Record's `concepts`/`legal_entities` are embedded once (reusing the
  existing embed role/provider already wired for `query_embed`-adjacent uses — subject to the
  `query_embed`-must-be-Voyage hard rule only if the *same* Voyage-embedded space is reused for
  cosine comparison against the Milvus corpus; topic-clustering embeddings are a separate space
  for persona-internal similarity only, so this rule does not constrain provider choice here, but
  implementation must not accidentally reuse `query_embed`'s role name for a different purpose).
- New query event's embedding is compared (cosine) against each of the user's existing topics'
  `representative_embedding`; combined with legal-entity Jaccard overlap, intent-tag overlap, and
  a temporal-proximity decay term. A weighted sum above a fixed threshold assigns to the closest
  existing topic (episode continuity decided by a separate, smaller temporal gap threshold within
  the same topic); below threshold creates a new topic (`discovered` state).
- Alternative considered: LLM-driven clustering ("is this query about the same topic as these
  N recent topics?"). Rejected per proposal §10 — expensive per query, non-deterministic, harder
  to test; a threshold-based deterministic similarity function satisfies
  `persona-interest-scoring`'s determinism requirement directly, an LLM-in-the-loop would not.

### 3. Interest scoring: exponential-recency-decayed weighted sum, not a lifetime mean
- `interest_score(topic, t) = Σ_events (evidence_weight_i * exp(-λ * (t - timestamp_i)))`,
  optionally scaled by a coherence factor (fraction of the topic's events that are mutually
  high-similarity, penalizing a topic that clustering placed together too loosely). λ (decay
  rate) is a tunable `PersonaSettings` field, not hardcoded, so it can be adjusted without a code
  change.
- This directly satisfies `persona-interest-scoring`'s recency-discount and determinism
  requirements, and resolves the design question the prior implementation's final review flagged
  but left open ("lifetime mean vs. rolling/EWMA" — this design commits to exponential decay,
  a form of EWMA, rather than a lifetime mean).
- Alternative considered: fixed rolling window (e.g. "last 90 days only"). Rejected — a hard
  cutoff creates a cliff-edge discontinuity (a relevant event from 91 days ago vanishes entirely);
  exponential decay degrades smoothly and is standard for this kind of recency-weighted signal.

### 4. State machine: hysteresis via minimum session/evidence counts, not raw score thresholds alone
- Each transition (discovered→emerging, emerging→active, active→fading, fading→dormant,
  dormant→reactive, reactive→active) is gated on *both* an interest-score threshold *and* a
  minimum count of distinct sessions/days contributing corroborating evidence since the last
  transition — satisfies `persona-interest-state-machine`'s "sustained evidence, not single-
  session spike" and "fading before dormant" requirements structurally (state transitions are a
  strict graph with no active→dormant edge).
- Pivot detection is a derived observation (topic A trending down while topic B independently
  reaches `active`/`reactive` with corroboration), not a separate mechanism — reuses the same
  per-topic state transitions, satisfying the corroboration requirement without new machinery.

### 5. Context rendering: replace the flat sentence with a small ranked list + optional
   query-time hypothesis weighting
- `render_persona_context` (redesigned) takes the current snapshot (topics in `active`/
  `reactive`/`dominant`-equivalent states, most-recent-first) and renders up to N topics with a
  qualitative confidence descriptor, plus the existing "this is a prior, not a fact" instruction
  (kept verbatim from the current implementation — proven-correct language).
- Ambiguous-current-query hypothesis weighting (proposal §11) is implemented as an optional,
  separate small function that, given the current query's embedding and the user's topic set,
  returns top-K candidate topics with normalized confidence — rendered only when more than one
  candidate clears a minimum confidence floor, otherwise omitted (keeps output empty/quiet for
  clear-cut queries, per `persona-context-rendering`'s low-evidence requirement).
- This remains purely additive text into the synthesis (and optionally SLM rewrite) prompt; no
  code path here touches `common/schemas.py::collections_for_intent` or any RRF weight, per hard
  rule 4 — carried forward unchanged from the current implementation's own constraint.

### 6. Extraction call: one SLM call per query, richer schema, same failure-isolation pattern
- `extract_expertise_signal` is renamed/expanded to `extract_query_understanding`, returning the
  full Query Understanding Record shape instead of just `{expertise_level, query_style}`. The
  existing failure-isolation pattern is kept: malformed JSON is swallowed and treated as "no
  record produced"; a transport-level gateway failure propagates to the caller so the whole
  persona write for that query is skipped (mirrors the current, already-reviewed distinction in
  `persona_signal.py` between `(TypeError, json.JSONDecodeError)` and other exceptions).
- Output validation before storage/reuse (per `persona-query-understanding`'s injection-safety
  requirement) follows the existing precedent in `merge.py::merge_expertise_patch`: constrain
  each field to its expected type/enum/shape, drop anything else, never merge raw model output
  verbatim.

## Risks / Trade-offs

- [Clustering threshold miscalibration → topics fragment (too many near-duplicate topics) or
  collapse (unrelated queries merged)] → Mitigation: ship with a conservative threshold validated
  against a hand-labeled sample of real query logs before rollout; expose the threshold as a
  `PersonaSettings` field so it can be retuned without a redeploy.
- [Two Mongo collections + derived cache adds operational complexity vs. today's single document]
  → Mitigation: `persona_topics` is always rebuildable from `persona_events` alone (explicit
  design decision #1), so the cache can be safely dropped/rebuilt if it drifts; this is strictly
  more recoverable than the current single-document model, which has no history to rebuild from
  at all.
- [Existing users have a flat persona document with no event history to derive a snapshot from]
  → Mitigation (cold-start policy, resolving the open question the proposal flagged): on first
  read/write after this change ships, an existing flat document is converted into one synthetic
  seed event per known signal (one seed event per non-zero `category_affinity` entry, carrying
  low evidence weight and a timestamp of "migration time") so the topic-less legacy signal decays
  away naturally under the new recency-weighted scoring rather than being discarded or crashing.
  The legacy `personas` collection is left in place, unread, after migration — not deleted in this
  change.
- [Richer interaction signals (click/open/save) don't exist yet on the frontend] → Mitigation:
  `persona-interest-evidence`'s requirements are written to degrade gracefully to "submitted-only"
  weight; this design does not block on frontend instrumentation work landing first.
- [Per-query embedding call adds latency/cost beyond today's one SLM call] → Mitigation: run
  extraction + embedding + clustering + scoring entirely in the existing async, post-response,
  fire-and-forget path (same pattern as today's `record_persona_signal`) — none of it is on the
  user-visible request latency path.

## Migration Plan

1. Add `persona_events`/`persona_topics` collections and indexes; ship code that can read/derive
   from them behind the existing `user_id`-gated seam in `ws.py` — no schema change to the
   request/response contract.
2. Ship the cold-start conversion (risk section above) as a lazy, on-first-touch migration per
   user, not a bulk offline migration job — avoids a big-bang cutover and only pays the
   conversion cost for users who are actually still active.
3. Cut `record_persona_signal`/`render_persona_context` over to the new event-append +
   derive-snapshot contract in one release; the old `merge_category_affinity`/`record_signal`
   read-modify-`replace_one` path is removed, not kept behind a flag (no long-term dual-write
   planned).
4. Rollback: since `persona_events` is append-only and additive, rolling back the code to the
   prior flat-document implementation is safe (old code simply ignores the new collections); the
   only loss on rollback is the interim event history, which is acceptable for an alpha-stage
   feature.

## Open Questions

None — the one open question the prior implementation's final review surfaced (lifetime-mean vs.
rolling/EWMA affinity) is resolved by design decision #3 above (exponential decay). The cold-start
policy for pre-existing flat persona documents is resolved by the Migration Plan above.
