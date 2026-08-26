## 1. Event storage foundation

- [x] 1.1 Add `persona_events` collection accessor (append-only insert, indexed on
  `(user_id, timestamp)` and `(user_id, topic_id)`) to `packages/persona` and verify with an
  in-memory fake that inserting never mutates or removes a prior event.
- [x] 1.2 Add `persona_topics` collection accessor (topic doc: state, state_history, episodes,
  score cache) and verify a topic document can be fully rebuilt from a given `persona_events`
  slice, matching design.md decision #1.
- [x] 1.3 Add `PersonaSettings` fields for the new tunables (decay rate λ, clustering similarity
  threshold, episode-gap threshold, state-transition minimum session counts) and verify they load
  from env vars with the existing `model_config` pattern (`.env` loading + `lru_cache`-hashable).

## 2. Query Understanding extraction

- [x] 2.1 Implement `extract_query_understanding` (rename/expand of `extract_expertise_signal`)
  returning `{concepts, legal_entities, research_objective, specificity, confidence}` via one SLM
  call, and verify with a fixture query it produces all fields per `persona-query-understanding`.
- [x] 2.2 Preserve the existing failure-isolation split (malformed JSON swallowed vs. gateway
  transport error propagated) and verify both paths with tests mirroring
  `test_record_persona_signal_swallows_gateway_errors`.
- [x] 2.3 Add field validation/sanitization (type/shape constraints, drop unexpected keys) before
  any extracted field is stored, and verify an adversarial/malformed model response cannot inject
  content that later reaches a rendered prompt unsanitized (`persona-query-understanding`'s
  injection-safety requirement).
- [x] 2.4 Verify guest requests (no `user_id`) never trigger extraction or storage — add/extend a
  test asserting no `persona_events` write occurs for a request with no resolvable `user_id`.

## 3. Topic clustering

- [x] 3.1 Implement embedding + legal-entity-overlap + temporal-proximity similarity scoring
  between a new query event and a user's existing topics, and verify against fixture sequences
  that clearly related queries (design.md's "Section 7 IBC" example) land in one topic.
- [x] 3.2 Implement the below-threshold path that creates a new topic in `discovered` state, and
  verify an unrelated query does not attach to an existing topic (`persona-topic-clustering`'s
  "unrelated query starts a new topic" scenario).
- [x] 3.3 Implement episode boundaries within a topic (temporal-gap threshold reopens a new
  episode under the same topic) and verify the two-separated-episodes scenario from
  `persona-topic-clustering` (same topic, two non-adjacent research periods).
- [x] 3.4 Add an explainability accessor that reports, for a given event, its assigned
  topic/episode and which similarity signals contributed, and verify it returns non-empty signal
  attribution for a clustered event.

## 4. Interest evidence weighting

- [x] 4.1 Implement `evidence_weight(event, interaction_signals)` combining submit/click/open/
  save/repeat/return signals into one numeric weight, defaulting to submitted-only when no richer
  signal is available, and verify monotonicity (upgrading a signal never lowers the weight) per
  `persona-interest-evidence`.
- [x] 4.2 Verify graceful degradation: an event with zero interaction signals beyond "submitted"
  still produces a valid, non-blocking weight.

## 5. Interest scoring

- [x] 5.1 Implement `interest_score(topic, t)` as the exponential-recency-decayed weighted sum
  from design.md decision #3, parameterized by the λ setting from Task 1.3, and verify
  determinism (same event history recomputed twice yields the same score) per
  `persona-interest-scoring`.
- [x] 5.2 Verify recency discounting: a topic quiet for months scores lower than its own peak
  score from when it was active, using a fixture event history.
- [x] 5.3 Verify no single event can push a brand-new topic's score to the maximum representable
  value (`persona-interest-scoring`'s single-event-cannot-dominate requirement).

## 6. Interest state machine

- [x] 6.1 Implement the fixed state graph (discovered → emerging → active →
  fading → dormant → reactive → active) with no active→dormant edge, and verify each edge
  transition is reachable and no disallowed edge exists.
- [x] 6.2 Implement hysteresis gating (score threshold AND minimum corroborating-session count
  since last transition) and verify a single-session spike does not promote a topic
  (`persona-interest-state-machine`'s "single-session spike does not promote" scenario).
- [x] 6.3 Implement dormant→reactive transition preserving prior topic identity/history, and
  verify a topic reactivated after months of dormancy keeps its original discovery date and past
  episodes.
- [x] 6.4 Implement pivot detection as a derived read over two topics' state/score histories
  (one declining, one independently reaching active/reactive with corroboration) and verify both
  the false-positive-avoided and genuine-pivot-detected scenarios from
  `persona-interest-state-machine`.

## 7. Context rendering

- [x] 7.1 Reimplement `render_persona_context` to render current active/reactive topics
  ranked by recency/score instead of the flat lifetime-mean sentence, keeping the existing
  "prior, not fact; disregard if conflicting" instruction text verbatim, and verify against the
  pivoted-user scenario in `persona-context-rendering` (shows current focus, not a blend).
- [x] 7.2 Implement the low-evidence empty-context path (no topic clears the minimum confidence/
  evidence floor) and verify a new user gets empty persona context.
- [x] 7.3 Implement the optional ambiguous-current-query topic-hypothesis renderer (top-K
  candidate topics with normalized confidence, per design.md decision #5) and verify it appears
  only when more than one candidate clears the confidence floor, and is omitted otherwise.
- [x] 7.4 Add a test asserting persona context (including any hypothesis weighting) never alters
  which Milvus collections are searched or the RRF fusion weights used, for a fixed request with
  and without persona context (`persona-context-rendering`'s no-routing-side-effect requirement,
  hard rule 4).

## 8. Wiring into retrieval-api

- [x] 8.1 Update `ai_mode/persona_signal.py`'s `record_persona_signal` to append a
  `persona_events` document and trigger topic/state/score derivation instead of the old merge-and-
  replace call, keeping the existing fire-and-forget, exception-swallowing contract in `ws.py`
  intact; verify with the existing `asyncio.create_task` wiring test pattern that no exception
  escapes into the websocket handler.
- [x] 8.2 Update `ws.py`'s persona read path (~L120-136) to derive the current snapshot and call
  the new `render_persona_context`, preserving the existing degrade-to-guest-equivalent behavior
  on a store failure; verify with a test that an unreachable event store still returns
  `persona_context=""` without raising.
- [x] 8.3 Update `ai_mode/pipeline.py`/`synthesize.py` call sites for the new `persona_context`
  shape (if the shape changes beyond a string) and verify existing AI Mode pipeline tests pass
  with the updated signature.
- [x] 8.4 Run the full aggregated suite (`uv run pytest` from repo root) and verify no regression
  in existing auth/AI-mode/instant-mode tests from this rewiring.

## 9. Migration and cold start

- [x] 9.1 Implement the lazy, on-first-touch cold-start conversion of an existing flat `personas`
  document into seed `persona_events` (one low-weight seed event per non-zero
  `category_affinity` entry, timestamped at migration time), and verify a fixture legacy document
  converts without error and without crashing the request path, per `persona-timeline-storage`'s
  "existing flat persona documents are handled on transition" requirement.
- [x] 9.2 Verify the legacy `personas` collection is left untouched (not deleted, not written to)
  after a user's cold-start conversion completes.
- [x] 9.3 Add Mongo index creation (or a migration script) for the new collections to whatever
  deployment/startup mechanism this repo already uses for index setup, and verify indexes exist
  after a fresh local `docker compose up` against an empty Mongo.

## 10. Final verification

- [x] 10.1 Run `openspec validate persona-interest-timeline --strict` and fix any reported issues.
- [x] 10.2 Confirm every scenario in the seven new spec files has at least one corresponding test,
  and record any gap found for follow-up before this change is archived.
