# User Persona System — Design

Date: 2026-08-15
Status: approved for planning

## Problem

The retrieval system has three query paths (Instant, AI Mode, Agentic) but no concept of who is
asking. Every request is stateless and identical regardless of the user's expertise level or
habitual query focus (e.g. a litigator who mostly wants caselaw vs. a CA who mostly wants
act/rule sections). This design introduces a minimal **auth gate** and a **persona system**
behind it, so logged-in users get retrieval/synthesis personalized to an inferred profile, while
guests keep today's behavior unchanged.

This repo currently has zero identity or persistence infrastructure: no auth, no database, no
session/cookie handling anywhere (`retrieval-api/main.py` is three routers on bare FastAPI). This
is greenfield.

## Decisions made during brainstorming (with rationale)

- **Guest vs. logged-in, not anonymous-then-merge.** Rejected an anonymous persistent
  `user_id` (cookie/header issued to everyone, merged into a real account on login) — that
  identity-resolution problem (merging guest history into a real account) is a known source of
  complexity not worth taking on for an alpha. Instead: not logged in → no `user_id`, no
  persona, exactly today's behavior. Logged in → real `user_id` from auth, persona persists
  across sessions/devices.
- **Auth is the gate, not a side feature.** Persona cannot exist without a `user_id`, so a
  minimal auth service is a hard prerequisite, not an optional nice-to-have.
- **MongoDB, not Postgres, for storage.** The persona schema is expected to change shape
  repeatedly as this is iterated on (fields added/removed/restructured). Mongo's schemaless
  documents absorb that churn without migrations; a relational schema would fight active
  development. Tradeoff accepted: no FK-enforced cascade delete (user delete → persona delete
  becomes app-level logic, not DB-enforced) — acceptable for alpha.
- **No graph database, for now.** A graph earns its cost when there are multi-hop relationships
  or contradiction/versioning to track ("preferred X, superseded by Y, keep both with
  provenance"). A flat inferred persona (expertise level, category affinity, query style)
  doesn't need traversal — it needs "current value of field X," which Mongo answers directly.
  Revisit only if a concrete case emerges that Mongo can't express cleanly.
- **No vector-store episodic memory, for now.** Same YAGNI reasoning — ship the flat persona
  first, add semantic memory search only if a real need for "recall a specific past exchange"
  shows up.
- **Reuse `extract_intent()` output as a free signal.** AI Mode/Agentic already classifies every
  query into `acts`/`rules`/`caselaws`/`articles`/`commentary`/`tariff` categories (hard rule 4,
  `common/schemas.py::collections_for_intent`). This is a ready-made signal for persona's
  `category_affinity` field — no extra LLM call needed to know what a user's queries skew
  toward.
- **Persona informs prompts, not fusion weights.** Persona must never feed into RRF
  weighting or Milvus collection routing directly — hard rule 4 already forbids
  category-weighted fusion (explicitly rejected during that design), and persona must not become
  a backdoor around it. The one flagged-but-not-committed exception: persona could plausibly
  help break ties in the existing empty/unrecognized-intent fallback (which currently searches
  all 11 collections) — noted as a future candidate, not part of this design.

## Architecture

### Auth

- New workspace package, `packages/auth`, alongside `common`/`model-gateway`/`retrieval-api`/`agents`.
- Storage: MongoDB, `users` collection — `_id`, `email`, `password_hash`, `created_at`. Same
  Mongo instance the persona collection lives in (one new infra dependency, not two). New
  `mongo` service added to `docker-compose*.yml`.
- Password hashing via a standard library (bcrypt/argon2 through `passlib` or equivalent) — no
  custom crypto.
- Token: a single signed JWT carrying a `user_id` claim. No refresh-token flow for this phase
  (one reasonably-long-lived token); revisit if forced logout/rotation becomes a real need.
- Endpoints: `POST /auth/signup`, `POST /auth/login`. `POST /auth/logout` is client-side token
  discard only — no server-side blacklist yet.
- Middleware in `retrieval-api`: reads `Authorization: Bearer <token>` if present, validates it,
  sets `request.state.user_id`. Missing/invalid token is **not** rejected — request falls
  through as a guest, matching today's behavior exactly. Only persona-touching code paths
  branch on `if user_id`.
- No OAuth/social login, no email verification, no password reset in this phase — addable later
  without changing the `user_id`-in-`request.state` contract downstream code depends on.

### Persona storage

- MongoDB, `personas` collection, one document per `user_id`. Schema is intentionally loose and
  expected to evolve; the following is a starting sketch, not a locked contract:

```json
{
  "user_id": "...",
  "category_affinity": {"acts": 0.4, "caselaws": 0.5, "commentary": 0.1},
  "expertise_level": "practitioner",
  "query_style": "precise-citation",
  "query_count": 47,
  "updated_at": "..."
}
```

### Extraction pipeline

1. **Category affinity** — tallied directly from the `acts`/`rules`/`caselaws`/`articles`/
   `commentary`/`tariff` tags `extract_intent()` already produces on every AI Mode/Agentic query.
   No new model call for this signal.
2. **Expertise/style signal** — a separate, cheap/small model call (not the main synthesis
   model), run asynchronously *after* the response has been sent to the user (does not block
   request latency). Takes the query text (and optionally the intent tags) and returns a small
   structured patch, e.g. `{"expertise_level": "...", "query_style": "..."}`.
3. **Merge, never overwrite.** Both signals merge into the existing persona document — e.g. a
   rolling average for `category_affinity`, a simple recency/majority heuristic for
   `expertise_level` — so a single atypical query doesn't reset accumulated signal.
4. **Read path.** At the top of AI Mode/Agentic request handling: if `request.state.user_id` is
   set, load the persona document; if not, skip entirely (guest path is untouched). When
   present, persona is injected into the SLM rewrite prompt and the LLM synthesis system prompt
   (tone, depth, framing) — never into RRF weights or collection routing (see decision above).

## Explicitly out of scope for this phase

- Anonymous/guest persona persistence or guest→account merge on login.
- Vector-store episodic memory, graph-store relationship/versioning memory.
- Category-weighted RRF fusion or persona-driven collection routing (would violate hard rule 4).
- OAuth/social login, email verification, password reset, refresh tokens, server-side logout
  blacklist.
- FK-enforced cascade delete between `users` and `personas` (Mongo has no FKs; handle at app
  level if/when needed).

## Open question for implementation planning

Where does the async extraction step run — inline background task in `retrieval-api`'s process,
or a separate worker/queue? Not decided here; the writing-plans pass should pick based on
whatever's simplest to stand up first (likely: FastAPI `BackgroundTasks` for alpha, revisit if
volume demands a real queue).
