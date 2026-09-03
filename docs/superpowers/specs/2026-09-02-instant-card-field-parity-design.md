# Instant-mode result-card field parity + doc_id visibility + hidden pagination — design

Branch: `feature/repotaxmannapi-exact-replica` (continues the same branch — this is scope
added to the same UI work, not a new feature branch).

## Context

`taxmann-research-ui/` was cloned to the repo root as a **read-only reference** (frontend
counterpart to `repotaxmannapi/`, same treatment: never edited, never built here, ground
truth only). Investigation (three passes, documented in session) found:

- The Angular frontend has **zero client-side ES logic** — it's a pure consumer of the
  `.NET` `repotaxmannapi` backend (`research/getSearchResult` etc). All query/boost/sum-vs-
  multiply logic was already ported in the prior `2026-09-01-repotaxmannapi-exact-replica`
  work. Nothing left to port on the query-building side.
- The reference product's global-search result card (`ISearchPageList`, the actual rendered
  struct — confirmed by reading `search-page-listing.component.html` in full) is thin:
  `Id, Heading1-4, CategoryName, GroupName, InfavourOf, CourtName, AuthorName`. No ES
  highlight/snippet field is shown on their card at all — our `subheading`-as-snippet is our
  own product decision, not a port gap.
- `CourtName` isn't a distinct ES field on their side either — it's a plain-text court
  abbreviation baked into `heading` (e.g. `"...(SC)"`), already covered by our `heading`.
  `Heading3`/`Heading4` have no analog in our schema (we only have `heading`/`subheading`) —
  a genuine, disclosed, unfixable gap.
- User explicitly widened scope beyond "just match their card": **"whatever we have in
  index we need to show it on ui"** — so this design also surfaces additional real,
  populated ES fields that neither product currently displays, confirmed via a live count
  query against the real index (`researchindex_aic_test`, 410,427 docs) run this session,
  not assumed from stale audit comments.
- User confirmed: skip all filters/facets/sort/spelling-correction UI (their product has
  these, we deliberately don't copy them — "just normal query based searching similar").
- User confirmed: doc_id should be visible always (testing), not just in dev mode.
- User confirmed: add real pagination (mirroring their `page`/`pageSize` contract) but keep
  it OFF by default behind a build-time env flag — current flat-20/client-slice-10 behavior
  must be bit-for-bit unchanged unless the flag is set.

## Live field-population audit (this session, real ES, `researchindex_aic_test`)

Confirmed dead (0/410427, do not surface):
`masterinfo.info.{court,bench,act,section}.name`, `masterinfo.citations.*`,
`searchcitation.formattedcitation.name`, `searchiltcitation.formattediltcitation.name`,
`url`, `displaydocumentdatestring`, `tariffinfo.*` (entire block — Tariff-group cards get no
type-specific fields, stays heading/subheading/badge-only, disclosed gap),
`searchboosttext`, `boostpopularity`, `incometaxactinfo`/`companyactinfo`/
`incometaxruleinfo`.

Confirmed real and populated (this session's live counts, plus prior-session-dated code
comments restated for the always-known fields):

| field | population | scope |
|---|---|---|
| `heading`, `subheading` | 100% | all — already shown |
| `categories.name` / `groups.group.name` | 100% | all — already shown (badge) |
| `otherinfo.judge.name` | 99.4% | CASELAWS |
| `otherinfo.partyname.name` | 100% | CASELAWS |
| `formatteddocumentdate` | 100% | all |
| `viewcount` | populated | all |
| `documenttypeboost` / `court_boost` | 100% / 99.9% | all — display only, **never** re-wired into ranking (CLAUDE.md rule 5 stays intact) |
| `otherinfo.fullcitation.name` | 60.34% (247,669/410,427) | CASELAWS (241,694), Experts Opinion (5,975) |
| `associates.act.name` | 60.98% (250,281/410,427) | CASELAWS, ACT, RULE, COMMENTARY, Experts Opinion |
| `associates.section.name` | 57.79% (237,183/410,427) | same spread as above |
| `associates.casereferred.name` | 24.09% (98,861/410,427) | mostly CASELAWS |

Dropped as too sparse to earn a card line: `associates.affirmreverse.name` (3.98%),
`associates.rule.name` (1.22%), `landmarkruling` (2.1%, also already excluded from ranking
per CLAUDE.md and not shown on the reference card either).

## Scope

### 1. Backend — extend Instant mode's doc-metadata fetch

`packages/common/src/common/es_client.py`: add a new function (or extend
`fetch_doc_categories`'s mget — same batched call, same doc_id keys, cheaper than a second
round-trip) that also pulls, per doc_id:
`otherinfo.judge.name`, `otherinfo.partyname.name`, `formatteddocumentdate`, `viewcount`,
`documenttypeboost`, `court_boost`, `otherinfo.fullcitation.name`, `associates.act.name`,
`associates.section.name`, `associates.casereferred.name`.

All fields optional/absent-safe (mirrors the reference product's own `*ngIf`-per-field
pattern — a field simply doesn't render when the doc has none, exactly like their card does
for non-caselaw content types).

`packages/retrieval-api/src/retrieval_api/instant/search.py::run_instant`: wire the new
fetch into the existing `doc_meta` step (same call site as today's category/group fetch),
extending `doc_meta`'s per-doc dict with the new fields. No change to `es`/`milvus`/
`reranked`/`grouped_es` shapes — this only enriches `doc_meta`, which every card already
reads for its badge.

### 2. Frontend — render the new fields + always-visible doc_id

`ChatMessageView.tsx` / `GroupedResultsPanel.tsx`:
- doc_id: move out of the `devMode &&` block, render unconditionally (small mono line under
  title) — score stays dev-mode-gated (internal debug, not a card field).
- New optional lines, each rendered only when present on that doc's `doc_meta` entry (same
  conditional pattern as the badge): judge, party (`InfavourOf`-analog), date
  (`formatteddocumentdate`), viewcount, citation (`fullcitation`), referenced act,
  referenced section, cases referred.
- `documenttypeboost`/`court_boost`: dev-mode-only debug numbers (same gate as score today)
  — display only, never feeds ranking.

### 3. Pagination, hidden by env flag

Backend: `raw_search` gains `page: int = 1, page_size: int = 20` (maps to ES `from`/`size`).
Default values reproduce today's exact behavior byte-for-byte when callers don't pass them.

Frontend: new build-time flag `VITE_ENABLE_PAGINATION` (unset/false by default).
- Off (default): unchanged — flat 20-result fetch, 10-per-page client-side slice, existing
  Prev/Next.
- On: real server pagination — `page` sent to the backend, re-fetch per page instead of
  slicing the same 20-item array.
No UI checkbox for this — purely env-gated, per explicit request.

### 4. Explicitly out of scope (confirmed)

Filters/facet UI, sort options, spelling-correction (`didyoumean`/`searchinsteadfor`),
Heading3/Heading4 (no schema analog), any of the confirmed-dead fields above,
`associates.affirmreverse`/`associates.rule` (too sparse), `landmarkruling` (too sparse,
also a settled ranking non-issue per CLAUDE.md).

### 5. Testing

- Backend: `packages/common/tests/test_es_client.py` — new/extended tests for the enriched
  doc-meta fetch (fields present/absent per content type) and for `raw_search`'s
  `page`/`page_size` params (default-preserving case is the critical regression guard).
- Frontend: `ChatMessageView.test.tsx` / `TracePanel.test.tsx` — always-visible doc_id;
  conditional rendering of each new optional field; `VITE_ENABLE_PAGINATION` off (byte-exact
  unchanged behavior) vs on (server-paged fetch) paths.

## Explicitly not touched

`raw_search`'s `boost_source` sum/multiply toggle and old/new query behavior — untouched,
this design only adds fields and pagination alongside it. No change to ranking, scoring,
`boost_mode`, or any of CLAUDE.md's hard rules.
