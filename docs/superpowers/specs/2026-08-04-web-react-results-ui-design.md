# Web UI redesign: React + Vite results feed (replaces Streamlit)

## Context

`packages/web` is currently a single-file Streamlit app (`src/web/app.py`) that talks to `retrieval-api`'s `/ws/search` websocket and renders a chat-style transcript: user query bubble, then an assistant turn with an "Instant results (ES + Milvus)" expander and an AI Mode answer/citations block. It was built as a debugging/validation tool during backend development and has served that purpose well, but it's a chat transcript, not a research-tool results page.

The target look-and-feel is `https://taxmann-aic.lovable.app/results` — confirmed by direct browser inspection (Claude in Chrome) to be a static Lovable-generated mockup with hardcoded demo data (fake user "R. Mehta, Sharma & Co. LLP", a canned "Overview" answer about ITC on employee transport that doesn't actually respond to the query in its own URL). It is a **visual/structural reference only**, not a live system we integrate with or whose behavior we need to match beyond layout.

This spec covers replacing the Streamlit app with a React + Vite frontend that adopts that page's structure (single-column: AI answer card on top, filterable document feed below) using our actual backend data. No new backend endpoints are introduced — the existing `/ws/search` contract (see `packages/retrieval-api/src/retrieval_api/ws.py`) is reused as-is.

## Scope

**In scope (v1):**
- Search bar (replaces `st.chat_input`)
- "Overview" card: AI Mode's synthesized answer, with inline numbered citation pills instead of raw `[doc_id]` brackets, a citation-chip row, and a collapsible "Show detailed reasoning" section
- "Documents" feed: card list built from Instant mode's ES + Milvus results, deduped by `doc_id`, no ES/Milvus source labels shown by default
- A dev-mode toggle that reveals per-card source (ES vs Milvus + collection) and raw scores, for our own debugging (replaces the current Streamlit "raw JSON" expander)
- Loading/error states for both the feed and the answer card

**Explicitly out of scope for v1** (per user decision during brainstorming):
- Left sidebar (New search / Recent / Library / Settings) — no auth, no query history backend exists yet
- Type filter tabs (Act / Rule / Circular / Editorial) — our corpus is case-law only right now (per `CLAUDE.md`); showing empty tabs for data we don't have would look broken
- Tag chips on cards (topic tags like "GST", "ITC") — no topic-tagging exists in our schema
- Sort options beyond "Relevance" (the only meaningful order we have)
- Any query-history, bookmarking, or collections backend

## Page layout

Single column, top to bottom:

1. **Search bar** — text input + Search button. On submit, opens a new `/ws/search` connection (mirrors the current Streamlit `run_query` flow) sending `{"query": ..., "mode": "both"}`. The `mode` field stays on the backend (added for debugging isolation) but the UI always uses `"both"` — a real user always wants both legs.

2. **Overview card** (AI Mode answer):
   - Renders `ai_mode_done.answer`, a markdown/plain-text string containing `[doc_id]` bracket citations.
   - The frontend maps each unique `doc_id` (in first-appearance order) to a running number 1, 2, 3… and replaces each `[doc_id]` occurrence with a small numbered pill matching the reference's inline citation markers.
   - Below the answer text, a row of citation chips — one per unique cited `doc_id`, in the same numbered order, each showing a short label (party names / heading from that doc's citation metadata, already available via `ai_mode_done.citations[doc_id]`) and a count of how many times it's cited in the text. Clicking a chip scrolls the matching Documents card into view (`scrollIntoView`) and briefly highlights it.
   - "Show detailed reasoning" is a collapsible section. Content: the rewritten query and extracted intent/filters. **Note:** the current `/ws/search` payload does not include `extract_intent`'s output in the response — only the final answer and citations. Making this collapsible section real (rather than a placeholder) requires a small backend addition: `ai_mode_done` needs an extra field carrying `{rewritten_query, intent, filters}` from `run_ai_mode`'s `intent_result`. This is a one-line addition to `ws.py`'s existing success-path dict and `run_ai_mode`'s return value, not a new endpoint. If deferred, this section can ship as a static "reasoning trace coming soon" placeholder instead — decide at implementation time.
   - If `ai_mode_error` is received instead: replace the card content with an inline error message (reuse the existing error string verbatim, e.g. "AI Mode is currently unavailable: <error>"). Do not hide the Documents feed — it renders independently.
   - While waiting for either message: a skeleton/loading placeholder in the card's place.

3. **Documents section**:
   - Header row: "Documents" label + result count + a "Relevance" sort label (non-interactive for v1, since it's the only order — kept as a visual anchor matching the reference, wired up later if a second sort order is ever meaningful).
   - No type filter tabs (see Scope).

4. **Result cards** — one per deduped document (see Data Mapping below). Each card:
   - Type badge: always "Case Law" (our only type today; the badge exists so the layout doesn't need restructuring when Act/Circular data eventually arrives)
   - "Cited N" badge, top-right — count of times this `doc_id` appears in the AI Mode answer's citations (0 if not cited; badge omitted when 0)
   - Title: derived from the ES document's heading/subheading/party-name fields (`otherinfo.partyname`/`otherinfo.judge` per the ES schema notes in `CLAUDE.md` — these are ~99% populated, unlike `court`/`act`/`section` which are ~0% populated per the earlier live-data audit)
   - Metadata line: date (`formatteddocumentdate`) + citation string, when available; omit court/act/section since they're essentially never populated in the real index right now
   - Snippet: the ES/Milvus snippet text as returned today, unchanged
   - Relevance bar + number: a 0–100 value normalized **within the current result set** (min-max scaled per query), purely a visual affordance — not a claim that ES fuzzy-match scores and Milvus vector-distance scores are on a comparable absolute scale. Two different queries' "82" are not comparable to each other; this mirrors what the reference page itself likely does (their scores are almost certainly a cosmetic/demo device, not a real fused signal, since building a genuinely comparable BM25-vs-vector score is nontrivial and out of scope here)
   - No tag chips (see Scope)
   - "Open" button: omitted for v1 — no document URL field exists anywhere in our schema (`common/schemas.py`) to link out to, so there is nothing for it to open
   - **Dev mode only**: a small badge showing `ES` or `Milvus:<collection_name>` and the raw underlying score

## Data mapping: backend → cards

`instant_result` currently looks like:
```json
{
  "type": "instant_result",
  "es": [{"doc_id": ..., "score": ..., "snippet": ...}, ...] | null,
  "es_error": string | null,
  "milvus": {"<collection>": [{"chunk_id", "doc_id", "text", "score"}, ...], ...} | null,
  "milvus_error": string | null
}
```

Merge algorithm (`lib/mergeResults.ts`), a **presentation merge, not a ranking fusion** (per the hard rule in `CLAUDE.md`: no blending ES/Milvus scores):

1. Start with ES hits, in ES's own score order. Each becomes a card keyed by `doc_id`.
2. Walk Milvus hits across all 7 collections, keeping only the best-scoring hit per `doc_id` within Milvus itself (`chunk_id`s from the same `doc_id` across collections collapse to one card). For any `doc_id` not already present from ES, append a card. For any `doc_id` already present from ES, do **not** re-rank or re-score it — just note (dev-mode only) that Milvus also matched it.
3. Final card order: ES cards first (their own order preserved), then Milvus-only cards (their own order preserved). No interleaving by combined score — that would be the ranking fusion the hard rule prohibits.
4. Card title/metadata for ES-sourced cards come from the ES snippet/fields already returned. Milvus-only cards (no ES hit for that `doc_id`) have less metadata available — title falls back to a truncated snippet of the chunk `text` since we don't have the ES document fields for it in the same response. (A future improvement could have the backend enrich Milvus-only `doc_id`s via `fetch_citations`, but that's a backend change out of scope here — flag it as a possible follow-up, not part of this spec.)

## Component architecture

```
packages/web/
  src/
    main.tsx                 - entry point
    App.tsx                  - top-level layout: SearchBar + OverviewCard + DocumentsFeed
    api/useSearch.ts          - websocket hook: takes a query, returns {instant, aiMode, loading, wsError}
    lib/mergeResults.ts        - ES/Milvus dedup merge (pure function, unit tested)
    lib/citations.ts           - [doc_id] bracket -> numbered pill mapping (pure function, unit tested)
    components/
      SearchBar.tsx
      OverviewCard.tsx        - AI answer + citation pills/chips + reasoning collapsible
      DocumentsFeed.tsx        - header + card list
      DocumentCard.tsx
      DevModeToggle.tsx
  index.html
  vite.config.ts
  package.json
  Dockerfile                  - vite build, served via a static file server (e.g. `vite preview` or nginx) instead of `streamlit run`
```

State: a single `useState` in `App.tsx` holding `{query, instant, aiMode, loading, devMode}`. No routing, no global state library — the page has one view.

Styling: plain CSS modules (one `.module.css` per component). No UI kit dependency, keeping the bundle small for an internal tool.

## Error handling

- `es_error` set, `es` null: Documents feed shows whatever Milvus returned; if both empty, feed shows "No results found."
- `milvus_error` set: same, feed just uses ES results; in dev mode the error string is visible per-section, in normal mode it's silent (matches user's stated preference: no ES/Milvus internals in the normal view)
- `ai_mode_error`: Overview card shows an inline error message instead of an answer; Documents feed is unaffected (it does not depend on AI Mode)
- WS connection failure/drop: a small dismissible banner at the top of the page; does not crash the rest of the UI
- Empty query submit: disable the Search button / no-op, same as today's Streamlit guard

## Testing

- Vitest + React Testing Library
- Unit tests for `mergeResults()` — covers: ES-only, Milvus-only, overlapping `doc_id`s across ES and multiple Milvus collections, both empty
- Unit tests for the citation pill/chip mapping in `lib/citations.ts` — covers: no citations, repeated citations to the same `doc_id`, citations appearing out of numeric order in the raw text
- No E2E test harness for v1 (matches project precedent — the Python backend also relies on manual `docker compose` + live verification for integration-level checks, not automated E2E)
- Manual verification: `docker compose up --build`, exercise the real `/ws/search` endpoint through the browser, confirm feed/answer/dev-mode toggle all behave with real data (same verification method used throughout this session's backend fixes)

## Follow-ups explicitly deferred (not part of this spec)

- Backend addition to include `intent_result` (rewritten query/intent/filters) in the `ai_mode_done` payload, needed to make "Show detailed reasoning" show real data instead of a placeholder
- Backend enrichment of Milvus-only cards with ES-sourced metadata (currently falls back to raw chunk text)
- Sidebar (Recent/Library/auth), type filter tabs, tag chips, additional sort orders — all blocked on data/backend features that don't exist yet
