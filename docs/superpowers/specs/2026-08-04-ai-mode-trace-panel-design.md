# AI Mode trace panel — design

Date: 2026-08-04. Builds on `docs/retrieval-flow-current-state.md` (the
current-state audit of Instant vs AI Mode) and the existing web app from
`docs/superpowers/plans/2026-08-04-web-react-results-ui.md`.

## Problem

AI Mode's pipeline (SLM query rewrite → filter resolve → dense+sparse
hybrid Milvus retrieval → RRF merge → cross-encoder rerank → LLM synthesis)
is currently a black box to anyone using the web UI — only the final answer
and citations are visible. There's no way to see what queries were actually
sent to Milvus/ES, how the SLM rewrote the query, or which candidates the
reranker picked and why.

## Scope

**AI Mode only.** Instant is a single raw ES `multi_match` call + a single
Milvus dense-only call per query (see `docs/retrieval-flow-current-state.md`
§1) — there's no multi-step process worth visualizing there.

## Goal

A live-updating panel, visible only in Dev Mode, showing each AI Mode
pipeline stage's real inputs/outputs as they complete — not a summary added
after the fact, but streamed step-by-step while the query is still running.

## Architecture

```
ws.py "/ws/search"
  ├── instant_task (unchanged)
  └── ai_mode_task → run_ai_mode(..., on_step=emit)
                        emit() called after each pipeline stage completes
                        → asyncio.Lock-guarded websocket.send_json(
                              {"type": "ai_mode_trace", "step": ..., "data": ...})

Frontend: DevModeToggle on + mode != "instant" → App.tsx splits into 2-col grid
  left:  SearchBar + OverviewCard + DocumentsFeed (existing, unchanged)
  right: new TracePanel — accumulates ai_mode_trace messages via useSearch,
         renders one collapsible card per step, live-appending as they arrive
```

Final answer still arrives via the existing `ai_mode_done` / `ai_mode_error`
messages — trace steps are additive, not a replacement for that protocol.

## Backend: pipeline instrumentation

Thread an optional callback through the pipeline:

```python
OnStep = Callable[[str, dict], Awaitable[None]] | None

async def run_ai_mode(gateway, es_client, milvus_client, query: str, on_step: OnStep = None) -> dict: ...
```

Every stage function (`extract_intent`, `resolve_allowlist`, `retrieve`,
`rerank_and_prefetch`, `synthesize`) gains an optional `on_step` param and
calls `await on_step(step_name, payload)` immediately after producing its
result — using data already computed, no extra Milvus/ES calls added for
instrumentation's sake.

If `on_step` is `None` (e.g. any future non-websocket caller), stages skip
the call entirely — instrumentation must be zero-cost when unused.

If `on_step` raises (e.g. client disconnected mid-stream), the exception is
swallowed at the call site — a dead trace channel must never break the
pipeline or the final answer.

### Steps and payloads

Payloads are capped/truncated at the source to keep frames small and the UI
responsive — no unbounded lists, no full chunk text where a preview does:

| step | payload shape |
|---|---|
| `intent` | `{query, rewritten_query, intent, filters}` — full, tiny |
| `filters_resolved` | `{filters, doc_id_count, doc_id_sample}` — sample = first 10 |
| `milvus_dense` | `{collections: [{name, hit_count, top_hits}]}` — `top_hits` = top 5 of `{chunk_id, doc_id, score, text_preview}`, `text_preview` = first 200 chars |
| `milvus_sparse` | same shape as `milvus_dense` |
| `rrf_merge` | `{candidate_count, top_candidates}` — top 15 of `{chunk_id, doc_id, rrf_score, text_preview}` |
| `rerank` | `{considered_count, top_chunks}` — all 3 kept chunks, full text (small set, no truncation needed) |
| `synthesis_prompt` | `{prompt}` — full text; built from only 3 chunks so inherently small |

### Concurrency: serializing websocket sends

`instant_task` and `ai_mode_task` already run concurrently
(`asyncio.gather`-style task creation in `ws.py`). Today only one of them
sends at a time because the code `await`s `instant_task` fully before
touching `ai_mode_task`'s result. With live trace steps, `ai_mode_task` now
sends multiple times *while it's still running*, which can overlap with
`instant_task`'s completion send.

Fix: one `asyncio.Lock` in `ws.py`'s `search()` handler, wrapping every
`websocket.send_json(...)` call (instant result, each trace step, final
`ai_mode_done`/`ai_mode_error`) — `async with send_lock: await
websocket.send_json(...)`. No queue, no new infra — sends are already fast
and this only needs to prevent frame interleaving.

### Error handling

If the pipeline raises mid-stage, whatever trace steps already fired stay
sent — the panel shows real partial progress up to the point of failure.
The final message is still `ai_mode_error`, unchanged. No separate
"error" trace step type — the existing `ai_mode_error` message is enough
signal, and the panel simply stops updating.

## Frontend

- **`useSearch.ts`**: add `traceSteps: TraceStep[]` to hook state. Append
  on each `ai_mode_trace` message (in arrival order). Reset to `[]` when a
  new query is submitted.
- **`components/TracePanel.tsx`** (new): renders `traceSteps` as an ordered
  list of collapsible cards, one per step, in the order they arrived (so it
  visibly fills in top-to-bottom as the query runs). Each card:
  - Header: step name + a one-line computed summary (e.g. "Milvus dense · 7
    collections · 342 hits").
  - Body: any list beyond ~5 items is truncated by default with a "Show N
    more" button that expands to reveal the rest of what's **already in the
    payload** — pure local state, no refetch, so it can't introduce lag.
  - Visual style matches the existing `DocumentCard`/`OverviewCard` system
    already in the repo — no new design language introduced for this.
- **`App.tsx`**: when Dev Mode is on **and** the current query's `mode !==
  "instant"`, switch the main layout to a 2-column grid — left column is
  the existing content (`SearchBar` + `OverviewCard` + `DocumentsFeed`),
  right column is `<TracePanel />`. Otherwise (Dev Mode off, or an
  Instant-only query), layout stays single-column and `TracePanel` isn't
  rendered at all.

## Testing

- Backend: extend `test_ai_mode_pipeline.py` and the per-stage test files
  to assert `on_step` is called with the right step names and payload
  shapes; extend `test_ws_integration.py` for lock-serialized sends and
  correct `ai_mode_trace` message ordering relative to `instant_result` and
  `ai_mode_done`.
- Frontend: new `TracePanel.test.tsx` (renders steps in order, "Show more"
  expands without a refetch); extend `useSearch.test.ts` for
  `ai_mode_trace` accumulation and reset-on-new-query behavior.
