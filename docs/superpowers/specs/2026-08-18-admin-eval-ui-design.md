# Admin eval UI — local-only test-suite runner

Date: 2026-08-18
Status: proposed

## Problem

This repo has 4 eval scripts (`retrieval_eval.py`, `intent_eval.py`,
`collection_routing_eval.py`, `slm_intent_eval.py`) plus their `--limit`-style
CLI flags, but the only way to run one and read its output is a terminal: `uv
run python -m retrieval_api.<name>_eval`, scroll a wall of `PASS`/`FAIL` lines
mixed with noisy third-party log spam (Langfuse "Authentication error" /
"Context error" printed once per case when no Langfuse key is configured),
and hand-tally the summary. There's no live sense of progress on a run that
takes minutes, and no shareable view of a result without manually building an
artifact each time (as this session did for `slm_intent_eval`, by hand).

## Goals

- One local admin page: pick a suite, optionally cap it with `--limit`, run
  it, watch it complete with a live percent and a running pass/fail tally,
  see full per-case results in a table as they land.
- Zero non-signal output — no Langfuse log spam, no unrelated stdout, only
  structured per-case results.
- Gate the page behind a single hardcoded shared secret (`ADMIN_SECRET`) —
  not the user-facing JWT auth system, not a new user/role model.
- Support all 4 existing suites from one runner, without changing their
  underlying logic (`load_cases`, `check_*`, pipeline functions stay as-is).
- Survive a page refresh mid-development by caching the last completed run
  per suite in memory.

## Non-goals

- No run history beyond "the last one per suite" — no DB, no persisted run
  log, no multi-run comparison UI (that's what `compare_eval_runs.py`
  already does, separately, on disk).
- No queueing/concurrency for multiple simultaneous runs of the same suite —
  single admin user, single machine; a second run request for a suite
  already running is rejected, not queued.
- No production deployment story — this is a local dev tool. `ADMIN_SECRET`
  defaulting to unset (feature disabled) is fine for prod/staging.
- No change to the 4 eval scripts' pass/fail logic, datasets, or CLI
  behavior — the admin runner is a thin adapter that calls the same
  functions, not a reimplementation.
- Not fixing the eval scripts' own noisy stdout/Langfuse behavior when run
  from the CLI directly — this only silences it inside the admin runner's
  own process path (see "Log noise" below).

## Architecture

```
┌─────────────────┐   WS /ws/admin-eval, 1st msg {suite,token,limit}   ┌──────────────────────┐
│  packages/web    │ ──────────────────────────────────────────────────▶│  retrieval-api        │
│  /admin route     │◀──── {type: progress|case|done|error} events ────  │  admin_eval/ module    │
└─────────────────┘   GET /admin/api/eval-runs/{suite}  (cache read)   └──────────────────────┘
                                                                                        │
                                                                        calls existing  │
                                                                        load_cases /    ▼
                                                                        check_* / pipeline
                                                                        functions, unmodified
```

### Backend — auth gate

New `admin_secret: str | None = None` field on `common.config.Settings`
(env var `ADMIN_SECRET`). New dependency in a new
`retrieval_api/admin_eval/auth.py`:

```python
def require_admin(token: str) -> None:
    settings = get_settings()
    if not settings.admin_secret or token != settings.admin_secret:
        raise HTTPException(403)  # or close the WS with code 4403
```

Used by the cache-read REST endpoint (token as `X-Admin-Token` header, since
that's a normal fetch) and the WS route — but the WS route validates it from
the **first received JSON message**, not a URL query param: this repo's
existing `/ws/search` route (`ws.py`) already establishes that convention
(`await websocket.accept()`, then `receive_json()` for `query`/`mode`/
`access_token`/etc. as the first message) precisely so secrets never end up
in the URL — visible in server access logs, browser history, and `Referer`
headers otherwise. The admin WS route follows the same shape: accept, then
`{"suite": ..., "token": ..., "limit": ...}` as the first message, `4403`
close if the token check fails.
`admin_secret` unset (the default) disables the whole feature — both routes
403 unconditionally — so no prod/staging deployment needs to think about it
unless it opts in.

### Backend — suite registry (`retrieval_api/admin_eval/`)

One file per suite adapter, each exposing a single async generator with the
same shape:

```python
async def run(gateway_url: str, limit: int | None) -> AsyncIterator[dict]:
    cases = load_cases(...)  # the suite's own existing loader
    if limit:
        cases = cases[:limit]  # first-N slice, adapter-side - see retrieval_eval note below
    total = len(cases)
    for i, case in enumerate(cases, 1):
        result = await ...  # the suite's own existing per-case check logic
        yield {"type": "case", **result}
        yield {"type": "progress", "done": i, "total": total, "percent": round(i / total * 100)}
    yield {"type": "done", "summary": {...}}
```

A `SUITES` dict in `admin_eval/registry.py` maps a suite id (`"slm_intent"`,
`"intent"`, `"collection_routing"`, `"retrieval"`) to its adapter's `run`
function and a display name. The WS route looks up the suite by id from
this dict — an unknown id is a `{"type": "error", "reason": "unknown_suite"}`
close, not a 500.

Each adapter's `case` event carries whatever fields that suite's existing
`check_*` functions already produce (e.g. `slm_intent`'s
`{id, query, rewrite, status, catStatus, exp_cat, act_cat, exp_f, act_f,
filtersOk}` vs. `retrieval_eval`'s own different shape) — no attempt to force
a single common per-case schema across suites; the frontend renders generic
key/value detail for whatever comes back, keyed on `status` for the pass/fail
pill.

**`retrieval_eval` needs a materially different adapter than the other 3.**
Confirmed by reading `retrieval_eval.py` directly:

- Its own `--limit` CLI flag is *not* "first N cases" (that's `slm_intent_eval`
  /`intent_eval`/`collection_routing_eval`'s meaning) — it's the per-stage
  ES/Milvus top-K search depth (`default=50`), passed into `evaluate_case`.
  The admin UI's "cap to first N cases" control must not reuse that name/flag;
  the adapter does the `cases[:limit]` slice itself against the loaded list,
  independent of `evaluate_case`'s own `limit` kwarg (left at its default).
- `evaluate_case(case, gateway, es_client, milvus_client, ...)` needs real
  `es_client`/`milvus_client` resources acquired and closed around the run
  (`get_es_client`/`get_milvus_client`, `finally: await es_client.close();
  milvus_client.close()`) — the other 3 suites only need a `GatewayClient`.
  The `retrieval` adapter's `run()` sets these up itself, mirroring `_run()`'s
  own resource lifecycle in `retrieval_eval.py`.
- It's also far slower per case: real ES + dense/sparse Milvus + intent
  rewrite + reranker + synthesis + (optionally) the full agentic pipeline —
  seconds to tens of seconds per case, not the sub-second gateway-only calls
  the other 3 suites make. The `retrieval` adapter defaults `skip_agentic=True,
  skip_synthesis=True` (both already-existing `evaluate_case` kwargs) for a
  reasonably fast admin-UI run; an "include synthesis/agentic" checkbox is
  explicitly deferred (YAGNI) — full-depth runs stay a CLI job for now.

### Log noise

The Langfuse SDK's own logger emits "Authentication error" / "Context error"
warnings when `LANGFUSE_PUBLIC_KEY` isn't set, on every traced call — this is
third-party log output, not something the eval scripts print themselves.
`admin_eval/registry.py` sets `logging.getLogger("langfuse").setLevel(logging.CRITICAL)`
at import time, silencing it for any run started through the admin path,
without touching the eval scripts or their CLI behavior.

### Backend — WS route (`retrieval_api/admin_eval/router.py`)

`@router.websocket("/ws/admin-eval")`. Follows `/ws/search`'s own shape
exactly: `await websocket.accept()`, then `receive_json()` for the first
message `{"suite": ..., "token": ..., "limit": ...}`. On that message:

1. `require_admin(token)` — send `{"type": "error", "reason":
   "unauthorized"}` and close 4403 if it fails.
2. Look up `suite` in `SUITES` — `{"type": "error", "reason":
   "unknown_suite"}` and close if not found.
3. If this suite already has a run in progress (tracked in a small
   `dict[str, bool]` on app state), send `{"type": "error", "reason":
   "already_running"}` and close.
4. Otherwise, mark it running, iterate the adapter's `run(...)` generator
   in the same coroutine handling this connection (no detached background
   task — kept simple deliberately, see below), sending each yielded event
   over the WS as JSON, updating an in-memory `dict[str, dict]` cache
   (`app.state.admin_eval_cache[suite]`) with the running tally as `case`
   events arrive and the final result on `done`.
5. Mark the suite not-running in a `finally` block. A client disconnect
   mid-run cancels the run at the next `await` (same as any other FastAPI
   WS handler) — it does *not* keep running server-side to populate the
   cache. This is a deliberate YAGNI cut consistent with "no queueing" above:
   a local single-admin tool, re-running is one click, and decoupling the
   run from the connection (a detached `asyncio.Task`, mirroring `ws.py`'s
   `_background_tasks` pattern for the persona-signal write) adds real
   complexity — shared-state coordination between the task and a possible
   next connection for the same suite — for a case that's cheap to just
   retry.

### Backend — cache-read endpoint

`GET /admin/api/eval-runs/{suite}` (`X-Admin-Token` header via
`require_admin`) returns `app.state.admin_eval_cache.get(suite)` — `null`
if no run has completed yet this server lifetime. Used by the frontend on
page load/refresh so a completed run isn't lost, and on suite-switch to
show the last result before a new run starts.

### Frontend (`packages/web/src/admin/` — new route `/admin`)

- **Login gate**: single token input. No POST — the token is just held in
  `sessionStorage` (`admin_token`) and validated implicitly by the first
  real request (WS connect or cache-read); a 403 clears it and re-shows the
  login form with an error.
- **Suite picker**: 4 cards (one per `SUITES` entry, names hardcoded
  client-side to match the backend registry), a `limit` number input
  (empty = full dataset), a "Run" button.
- **Run view**: progress bar (from `progress` events' `percent`), a small
  live tally strip (pass count / fail count / total, updating per `case`
  event), and a table that appends a row per `case` event as it arrives —
  status pill (pass/fail/error) + an expandable row showing the full detail
  object as key/value pairs (reusing the general shape from the
  `slm_intent_eval` artifact built earlier this session, generalized to any
  suite's field set rather than that suite's specific columns).
- On mount for an already-known suite, fetch the cache-read endpoint first
  to show the last result immediately, before/without starting a new run —
  a run only starts on explicit "Run" click.

## Error handling

- Wrong/missing token → login form shows "invalid token," never a raw 403.
- WS drops mid-run (server restart, network blip) → frontend shows "run
  interrupted," offers "Run again"; the partially-filled table stays
  visible rather than clearing.
- `already_running` → frontend disables "Run" and shows "already running,"
  without erroring loudly — this is an expected state, not a failure.
- An individual case raising inside the adapter (e.g. gateway unreachable)
  → adapter yields `{"type": "case", "status": "error", "detail": {"error":
  str(exception)}}` and continues to the next case, matching the existing
  eval scripts' own `except Exception` → `ERROR {id}: {exception}` /
  continue pattern — one bad case never aborts the whole run.

## Testing

- **Backend**: one test per adapter mocking the suite's underlying
  `check_*`/pipeline call, asserting the yielded event sequence
  (`progress` count matches case count, `done` summary shape). One WS
  integration test (`TestClient`) covering: wrong token → 4403 close,
  unknown suite → error event, already-running → error event on a second
  concurrent connection, and a full happy-path run against a 2-case fake
  suite registered just for the test. One test for the cache-read endpoint
  (empty before any run, populated after).
- **Frontend**: component test for the login gate (bad token clears
  storage + shows error), and a test driving the run view off a scripted
  sequence of WS messages (mirrors the existing `useSearch.test.ts` /
  `useAgentSearch.test.ts` pattern already in this repo) asserting the
  progress bar and table update correctly, including an `already_running`
  and an interrupted-connection case.
