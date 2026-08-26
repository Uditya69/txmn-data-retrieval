## Why

The current persona system (`packages/persona`, live on `dev`) stores one flat, overwritten-in-place
document per user: a lifetime-mean `category_affinity` over six fixed tags plus a majority-vote
`expertise_level`/`query_style`, gated on a raw `query_count >= 20` threshold. It cannot represent that
a user's interests *change over time* — a user who did IBC research for six months and pivoted to GST
last month still reads as "IBC-leaning" for a long time, because the affinity is a cumulative mean, not
a temporal signal, and there is no notion of when an interest started, strengthened, faded, or was
abandoned. One-off or exploratory queries are also counted identically to sustained research focus,
because a raw category tag is the only unit of evidence — there is no concept of interaction strength,
recurrence, or topic-level semantic identity distinct from the six coarse intent tags.

This change replaces the flat, overwritten persona model with an event-sourced temporal interest
system: every query becomes a structured, evidence-weighted event; events accumulate into
per-topic interest scores over time; sustained, corroborated shifts (not single sessions) promote a
topic through an explicit state machine (discovered → emerging → active → fading/reactive →
dormant); and the persona injected into prompts becomes a snapshot derived from current interest
state plus confidence, not a single mutable document. This makes the "why does the system think
this user cares about X" question answerable after the fact, and prevents short rabbit holes from
permanently overwriting a user's inferred profile.

## What Changes

- Introduce a **Query Understanding Record**: per-query structured extraction (concepts, legal
  entities, research objective, specificity, confidence) via one cheap SLM call, replacing the
  current `extract_expertise_signal` call's narrower `{expertise_level, query_style}` output.
  **BREAKING**: `persona.persona_signal.extract_expertise_signal`'s output shape changes.
- Introduce **behavioral evidence weighting**: each query event carries a weight derived from
  available interaction signals (query submitted, result clicked, document opened/saved, repeated
  related query, return-to-topic-later), not a flat "+1 per category tag."
- Introduce **research-topic clustering**: queries group into topics via semantic similarity +
  legal-entity overlap + intent similarity + temporal proximity + behavioral continuity, replacing
  the fixed six-tag `acts`/`rules`/`caselaws`/`articles`/`commentary`/`tariff` bucket as the unit of
  affinity tracking (those tags remain available as one clustering signal, not the whole model).
- Introduce **research episodes**: a topic can have multiple time-bounded episodes (e.g. GST → ITC
  research in January, GST → registration cancellation in March) rather than one lifetime bucket per
  topic.
- Introduce an **interest-scoring function** per (topic, time) combining frequency, semantic
  coherence, interaction strength, recency, and repetition into a time series, replacing the
  cumulative-mean `merge_category_affinity` formula.
- Introduce an **interest state machine** per topic (discovered → emerging → active →
  fading/reactive → dormant) with persistence rules so a state transition requires sustained
  evidence across multiple sessions, not one session's spike.
- Replace the single overwritten `personas` Mongo document with an **append-only event history**
  (query understanding records + derived interest-state transitions) from which the current persona
  is a *derived snapshot*, not the source of truth. **BREAKING**: `persona.repository.record_signal`'s
  read-modify-write-and-replace contract is replaced by an append + derive model;
  `persona.repository.get_persona` becomes "compute current snapshot from event history," not "read
  one document."
- Replace `render_persona_context`'s flat "frequently asks about X, Y; expertise: Z" sentence with a
  **timeline-aware, confidence-weighted context**: active/dominant topics with their confidence, and
  (per §11 of the design brainstorm) topic-hypothesis confidences for ambiguous current-query terms
  (e.g. "limitation period" → IBC 0.81 / Civil 0.12 / GST 0.07) usable as a soft personalization
  signal — never as a hard reinterpretation and never fed into RRF/Milvus routing (unchanged hard
  rule).
- The `query_count >= 20` trust gate is replaced by a per-topic confidence/evidence threshold
  (a topic needs sufficient accumulated evidence, not the account needing 20 total queries).

## Capabilities

### New Capabilities
- `persona/query-understanding`: extracting a structured Query Understanding Record (concepts,
  legal entities, research objective, specificity, confidence) from each query via one SLM call.
- `persona/interest-evidence`: behavioral weighting of query events (submit/click/open/save/repeat/
  return) into an evidence score usable by topic scoring.
- `persona/topic-clustering`: grouping queries into research topics and time-bounded research
  episodes via semantic + entity + intent + temporal + behavioral similarity.
- `persona/interest-scoring`: computing a per-topic, per-time interest_score time series from
  frequency, coherence, interaction strength, recency, and repetition.
- `persona/interest-state-machine`: tracking each topic's state (discovered/emerging/active/
  fading/reactive/dormant) with persistence rules that require sustained evidence before a
  transition, including pivot detection between topics.
- `persona/timeline-storage`: append-only event history (query understanding records + state
  transitions) per user, from which the current persona is derived rather than stored directly.
- `persona/context-rendering`: deriving the prompt-facing persona context (active/dominant topics
  with confidence, ambiguous-query topic-hypothesis weighting) from current interest state.

### Modified Capabilities
(none — no existing `openspec/specs/` capability currently describes the persona system; the
existing implementation in `packages/persona` predates this repo's OpenSpec adoption, so this
change introduces its spec surface fresh rather than delta-ing an existing one.)

## Impact

- `packages/persona`: `merge.py`, `repository.py`, `prompt.py`, `db.py` are substantially
  redesigned (event-history storage and derivation replace the flat merge-and-replace document
  model). `config.py` likely gains settings for scoring/decay parameters.
- `packages/retrieval-api/src/retrieval_api/ai_mode/persona_signal.py`: `extract_expertise_signal`
  is replaced/expanded into the Query Understanding Record extraction call; `record_persona_signal`
  changes from "merge one document" to "append one event + trigger derivation."
  `packages/retrieval-api/src/retrieval_api/ws.py`'s persona read/write wiring (lines ~120-136,
  ~280-283) calls the same seam but against the new repository contract.
  `packages/retrieval-api/src/retrieval_api/ai_mode/pipeline.py`/`synthesize.py`'s `persona_context`
  parameter now carries timeline-aware, confidence-weighted text instead of a flat sentence.
- MongoDB `personas` collection: schema change from one-document-per-user to an event-collection
  (e.g. `persona_events`) plus a derived/cached snapshot; existing `personas` documents need a
  migration or cold-start policy (open question for design.md).
- Hard rule 4 (`CLAUDE.md`) still applies: none of this feeds into RRF fusion weights or Milvus
  collection routing — confidence-weighted topic hypotheses are a synthesis-prompt signal only.
- No new external dependency mandated by this proposal; semantic-similarity clustering may reuse the
  existing embedding/model-gateway roles already in this repo (design.md to confirm) rather than
  introduce a new one.
