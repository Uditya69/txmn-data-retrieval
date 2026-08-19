# Server-side chat storage, no localStorage

Date: 2026-08-18
Status: Approved for planning

## Problem

Chat conversation history currently lives only in the browser's `localStorage`
(`packages/web/src/App.tsx`), scoped per email (`taxmann-retrieval-conversations:<email>`)
or a shared `...:anon` bucket for guests. This means:

- History doesn't sync across devices for logged-in users.
- Clearing browser storage silently destroys history.
- Guest and logged-in users share the same (client-only) storage mechanism,
  which is not what either should have going forward.

Desired end state:

- **Guest users**: no persistence at all, anywhere. A guest's chat is a single,
  ephemeral session — it exists only in memory for the lifetime of the browser
  tab. Once the tab closes or is refreshed, it's gone. Nothing is written to
  `localStorage` and nothing is written to any database for guests.
- **Logged-in users**: conversations are persisted server-side in the database,
  keyed to the user's identity, so history is available across devices/sessions.
- **No `localStorage` usage anywhere in the web app** — including non-chat UI
  preferences (e.g. sidebar collapsed/expanded state), which move to plain
  in-memory React state.

Semantic response caching (a separate, related idea — reusing prior answers
for semantically similar queries via Mongo vector search) is **explicitly out
of scope** for this spec and will be designed separately.

## Current flow (verified against code)

- `packages/retrieval-api/src/retrieval_api/ws.py` — `/ws/search` and
  `/ws/agent` each handle exactly one `receive_json()` per connection: one
  query in, streamed events out, connection ends. There is no existing
  `conversation_id` concept anywhere server-side.
- When a valid access token resolves a `user_id`, the handler reads
  (never writes) the user's persona document to build `persona_context`,
  via a fire-and-forget background task pattern (`record_persona_signal`,
  fired after the response completes) for updating persona signals.
  This same background-task-after-response pattern is what we'll reuse for
  persisting a conversation turn.
- `packages/persona` is the template for a small Mongo-backed package in this
  repo: `db.py` (Motor `AsyncIOMotorClient`, `lru_cache`'d settings/client
  getters), `repository.py` (plain async functions over one collection),
  `config.py` (pydantic-settings). We'll mirror this shape for chat.
- `packages/web/src/App.tsx` owns `conversations` state, `loadConversations`/
  `persistConversations` (localStorage read/write, including quota-exceeded
  eviction of oldest conversations), `conversationsKey(email)` for bucket
  selection, and swaps buckets on login/logout.
- `packages/web/src/lib/auth.ts` stores only the access token, refresh token,
  and email in `localStorage` — untouched by this spec except that the token
  is still what authenticates the new REST/WS calls.

## Design

### Guests: no change in storage, only removal

Guests already never touch the database (no `user_id` resolved from an
access token → no persona read, no future chat write). The only change for
guests is on the frontend: conversation state and UI-preference state
(sidebar collapsed, etc.) move from `localStorage`-backed to plain
in-memory React state (`useState`), with no read/write to `localStorage`
at all. Closing or refreshing the tab clears everything, by design.

### Logged-in users: new `packages/chat` package

New workspace package `packages/chat`, structured like `packages/persona`:

- `config.py` — pydantic-settings, Mongo URI/db name (reuse the same Mongo
  deployment/db as `persona`/`auth`, new collection).
- `db.py` — `AsyncIOMotorClient` + `lru_cache`'d getters, same pattern as
  `packages/persona/src/persona/db.py`.
- `repository.py` — plain async functions: `create_conversation`,
  `append_turn`, `list_conversations`, `get_conversation`,
  `delete_conversation`.

**Collection: `conversations`** — one document per conversation:

```jsonc
{
  "_id": "<uuid string, client-generated>",
  "user_id": "<str>",
  "title": "<str>",              // derived from first query, truncated
  "messages": [ /* same per-turn shape the frontend renders today */ ],
  "created_at": "<iso8601>",
  "updated_at": "<iso8601>"
}
```

No retention cap or TTL — conversations are kept indefinitely for now.

### Write path

1. Frontend generates a `conversation_id` (`crypto.randomUUID()`) the first
   time a user starts a new conversation, and includes it in every
   `/ws/search` (and `/ws/agent`) request message for that conversation.
2. After `ws.py` finishes streaming the response for a request that resolved
   a `user_id`, it fires a background task (same fire-and-forget pattern as
   `record_persona_signal`) that upserts the new turn into the conversation
   document (`repository.append_turn`), creating the document on first turn.
3. Guest requests never include/require `conversation_id` — the field is
   simply absent, and the server does nothing conversation-related when
   `user_id is None`.
4. If the append fails, it fails silently from the client's perspective
   (best-effort persistence, consistent with how persona signal writes
   already behave) — it must never block or fail the streamed response.

### Read path

New Bearer-authenticated REST routes on `retrieval-api` (same auth
dependency used elsewhere for token validation):

- `GET /conversations` — list `{id, title, updated_at}` for the sidebar,
  newest first.
- `GET /conversations/{id}` — full conversation (messages) for reopening.
  404 if the id doesn't belong to the caller's `user_id`.
- `DELETE /conversations/{id}` — remove one. 404 under the same ownership
  check.

### Frontend changes (`packages/web`)

- Remove `CONVERSATIONS_KEY_PREFIX`, `conversationsKey`, `loadConversations`,
  `persistConversations`, and all `localStorage` call sites in `App.tsx`
  and any sidebar-state module.
- On login (or app load with a valid token), call `GET /conversations` to
  populate the sidebar list. On selecting a conversation, call
  `GET /conversations/{id}` to load its messages before rendering.
- Guests: conversation list starts empty each session, no fetch calls, plain
  React state for the active conversation.
- Sidebar/UI preference flags (collapsed/expanded etc.) become plain
  `useState`, resetting each page load — acceptable since they're cosmetic.
- Deleting a conversation calls `DELETE /conversations/{id}` and removes it
  from local state.

### Error handling

- REST routes: standard 401 for missing/invalid token, 404 for
  not-found-or-not-owned, 500 surfaced as a generic error toast on the
  frontend.
- Background persistence failures (Mongo write errors) are logged
  server-side only; they never surface to the client and never affect the
  streamed answer.

### Testing

- `packages/chat/tests` — unit tests for `repository.py` against a test
  Mongo (or mongomock/fixture, matching however `packages/persona/tests`
  does it) covering create/append/list/get/delete and ownership isolation
  between users.
- `packages/retrieval-api/tests` — tests for the new REST routes (auth
  required, 404 on cross-user access) and a test that a WS `/ws/search`
  request with a `user_id` triggers a conversation write (mirroring however
  the existing persona-write background task is tested).
- `packages/web` — update/replace existing `persistConversations`/
  `loadConversations` tests to cover the new fetch-based list/load, and add
  tests confirming no `localStorage` calls occur for guest or logged-in
  flows.

## Out of scope

- Semantic response caching (separate spec).
- Any change to WS transport shape (still one query per connection).
- Retention limits / conversation pruning.
- Migrating existing users' localStorage history into the DB — it is
  discarded; the app simply stops reading `localStorage` on this release.
