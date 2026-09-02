# Instant Card Field Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enrich Instant mode's result card with real, live-confirmed ES fields (judge/party/
date/viewcount/citation/associates), always show `doc_id`, and add a hidden-by-default,
env-flag-gated real pagination path — without touching `boost_source`/ranking logic at all.

**Architecture:** Backend: extend the existing `fetch_doc_categories` mget (one batched
per-doc_id call, already wired into `run_instant`) to also pull the new fields; add optional
`page`/`page_size` params to `raw_search` that map to ES `from`/`size`, defaulting to
byte-identical current behavior. Frontend: extend `DocMeta`'s TS shape to match, render the
new fields conditionally (mirrors the reference product's own per-field `*ngIf` pattern —
absent for content types that don't have them), move `doc_id` out of the `devMode` gate, and
thread a new `page` param through `useSearch.search()` gated by a `VITE_ENABLE_PAGINATION`
build-time flag that defaults off.

**Tech Stack:** Python (FastAPI/pytest), TypeScript/React (Vitest), Elasticsearch (real
index `researchindex_aic_test`, python `elasticsearch` client v8).

**Spec:** `docs/superpowers/specs/2026-09-02-instant-card-field-parity-design.md`

## Global Constraints

- `boost_source` (sum/repotaxmannapi) and old/new query toggle stay completely untouched —
  no ranking/scoring change anywhere in this plan.
- Every new backend field is optional/absent-safe — a doc without it must not error, must
  simply omit that key (mirrors `fetch_doc_categories`'s existing `category`/`group`
  fallback pattern).
- Confirmed-dead fields (0% populated, live-audited 2026-09-02) must never be added:
  `masterinfo.info.{court,bench,act,section}.name`, `masterinfo.citations.*`,
  `searchcitation.formattedcitation.name`, `searchiltcitation.formattediltcitation.name`,
  `url`, `displaydocumentdatestring`, `tariffinfo.*`, `searchboosttext`, `boostpopularity`,
  `incometaxactinfo`/`companyactinfo`/`incometaxruleinfo`.
- Default behavior (no `page`/`page_size` passed, `VITE_ENABLE_PAGINATION` unset) must stay
  byte-identical to today — this is an additive plan, not a redesign.
- No filters/facets/sort-option/spelling-correction UI — explicitly out of scope.

---

### Task 1: Backend — extend `fetch_doc_categories`'s mget with new fields

**Files:**
- Modify: `packages/common/src/common/es_client.py:963-995` (`fetch_doc_categories`)
- Test: `packages/common/tests/test_es_client.py`

**Interfaces:**
- Produces: `fetch_doc_categories(client, doc_ids: list[str]) -> dict[str, dict]` — same
  signature, return dict per doc_id now also carries (each key present only if the source
  doc had it): `judge: list[str]`, `party: list[str]`, `date: str | None`,
  `viewcount: int | None`, `documenttypeboost: int | None`, `court_boost: float | None`,
  `fullcitation: str | None` (first entry's `name`, joined if multiple), `referenced_act:
  list[str]`, `referenced_section: list[str]`, `cases_referred: list[str]`.

- [ ] **Step 1: Write the failing test**

Add to `packages/common/tests/test_es_client.py` (near the other `fetch_doc_categories`
tests, alongside the existing category/group ones):

```python
@pytest.mark.asyncio
async def test_fetch_doc_categories_includes_judge_party_date_viewcount_and_boost_debug_fields():
    client = FakeAsyncES(mget_docs={
        "d1": {
            "categories": [{"name": "Direct Tax Laws", "isprimarycat": 1}],
            "groups": {"group": {"name": "CASELAWS"}},
            "otherinfo": {
                "judge": [{"name": "V.K. KHANNA"}, {"name": "A.N. Varma"}],
                "partyname": [{"name": "Commissioner of Income-tax"}, {"name": "Munnalal Shrikishan"}],
                "fullcitation": [{"name": "[1987] 167 ITR 415 (Allahabad)"}],
            },
            "formatteddocumentdate": "1987-03-31T00:00:00",
            "viewcount": 70,
            "documenttypeboost": 4500,
            "court_boost": 233.2,
            "associates": {
                "act": [{"name": "Income-tax Act, 1961"}],
                "section": [{"name": "Section - 256"}],
                "casereferred": [{"name": "CIT vs. Laxmi Rattan Cotton Mills Co. Ltd."}],
            },
        },
    })

    results = await fetch_doc_categories(client, ["d1"])

    assert results["d1"]["judge"] == ["V.K. KHANNA", "A.N. Varma"]
    assert results["d1"]["party"] == ["Commissioner of Income-tax", "Munnalal Shrikishan"]
    assert results["d1"]["date"] == "1987-03-31T00:00:00"
    assert results["d1"]["viewcount"] == 70
    assert results["d1"]["documenttypeboost"] == 4500
    assert results["d1"]["court_boost"] == 233.2
    assert results["d1"]["fullcitation"] == "[1987] 167 ITR 415 (Allahabad)"
    assert results["d1"]["referenced_act"] == ["Income-tax Act, 1961"]
    assert results["d1"]["referenced_section"] == ["Section - 256"]
    assert results["d1"]["cases_referred"] == ["CIT vs. Laxmi Rattan Cotton Mills Co. Ltd."]


@pytest.mark.asyncio
async def test_fetch_doc_categories_omits_new_fields_when_doc_has_none_of_them():
    # An ACT-group doc, for example, has no judge/party/citation/associates at all -
    # this must not error or fabricate empty lists, just omit the keys, same fallback
    # philosophy as the existing category/group handling.
    client = FakeAsyncES(mget_docs={
        "d2": {"categories": [{"name": "Acts"}], "groups": {"group": {"name": "ACT"}}},
    })

    results = await fetch_doc_categories(client, ["d2"])

    assert results["d2"]["category"] == "Acts"
    assert results["d2"]["group"] == "ACT"
    for key in (
        "judge", "party", "date", "viewcount", "documenttypeboost", "court_boost",
        "fullcitation", "referenced_act", "referenced_section", "cases_referred",
    ):
        assert key not in results["d2"]


@pytest.mark.asyncio
async def test_fetch_doc_categories_requests_all_new_source_fields():
    client = FakeAsyncES(mget_docs={"d1": {}})

    await fetch_doc_categories(client, ["d1"])

    requested = client.mget_calls[0]["_source"]
    for field in (
        "otherinfo.judge.name", "otherinfo.partyname.name", "otherinfo.fullcitation.name",
        "formatteddocumentdate", "viewcount", "documenttypeboost", "court_boost",
        "associates.act.name", "associates.section.name", "associates.casereferred.name",
    ):
        assert field in requested
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/common/tests/test_es_client.py -k fetch_doc_categories -v`
Expected: FAIL (KeyError / new fields absent) — `fetch_doc_categories` doesn't fetch them yet.

- [ ] **Step 3: Implement**

Replace `fetch_doc_categories` (`packages/common/src/common/es_client.py:963-995`) with:

```python
async def fetch_doc_categories(client, doc_ids: list[str]) -> dict[str, dict]:
    """Instant mode's per-card metadata fetch: "category | group" badge, plus (this
    session, 2026-09-02) every other real, populated ES field confirmed usable on the
    result card - judge/party/citation/associates are CASELAWS-oriented and simply
    absent for other content types (same *ngIf-style fallback the reference product's
    own card uses); date/viewcount/boost-debug-numbers apply to every content type.
    Batched sibling of fetch_citations - one mget for every doc_id across ES, Milvus
    dense/sparse, and reranked cards, so metadata is consistent regardless of which
    engine surfaced a given doc (Milvus itself carries none of these fields).
    `categories` is a populated-on-every-doc but multi-valued list (verified live: a doc
    can carry 4+ subject-area tags at once) with no reliable primary flag -
    `isprimarycat` is only set on ~20% of the corpus (81k/410k docs). Picks the
    isprimarycat=1 entry when present, else the first entry. Raw category/group values
    are run through CATEGORY_DISPLAY_LABELS/GROUP_DISPLAY_LABELS; a value with no entry
    there is passed through unchanged rather than guessed at.

    Confirmed dead fields (0% populated, live-audited 2026-09-02) are deliberately never
    fetched here: masterinfo.info.{court,bench,act,section}.name, masterinfo.citations.*,
    searchcitation/searchiltcitation formattedcitation, url, displaydocumentdatestring,
    tariffinfo.* (Tariff-group cards get no type-specific fields at all - a real,
    disclosed gap), searchboosttext, boostpopularity, incometaxactinfo/companyactinfo/
    incometaxruleinfo."""
    if not doc_ids:
        return {}
    response = await client.mget(
        index=client.index, ids=doc_ids,
        _source=[
            "categories.name", "categories.isprimarycat", "groups.group.name",
            "otherinfo.judge.name", "otherinfo.partyname.name", "otherinfo.fullcitation.name",
            "formatteddocumentdate", "viewcount", "documenttypeboost", "court_boost",
            "associates.act.name", "associates.section.name", "associates.casereferred.name",
        ],
    )
    results: dict[str, dict] = {}
    for doc in response["docs"]:
        if not doc.get("found"):
            continue
        source = doc["_source"]
        categories = source.get("categories") or []
        primary = next((c for c in categories if c.get("isprimarycat") == 1), None)
        category_name = (primary or categories[0])["name"] if categories else None
        category = CATEGORY_DISPLAY_LABELS.get(category_name, category_name)
        group_name = source.get("groups", {}).get("group", {}).get("name")
        group = GROUP_DISPLAY_LABELS.get(group_name, group_name)
        entry: dict = {"category": category, "group": group}

        otherinfo = source.get("otherinfo") or {}
        judges = [j["name"] for j in otherinfo.get("judge") or [] if j.get("name")]
        if judges:
            entry["judge"] = judges
        parties = [p["name"] for p in otherinfo.get("partyname") or [] if p.get("name")]
        if parties:
            entry["party"] = parties
        citations = [c["name"] for c in otherinfo.get("fullcitation") or [] if c.get("name")]
        if citations:
            entry["fullcitation"] = citations[0]

        if source.get("formatteddocumentdate") is not None:
            entry["date"] = source["formatteddocumentdate"]
        if source.get("viewcount") is not None:
            entry["viewcount"] = source["viewcount"]
        if source.get("documenttypeboost") is not None:
            entry["documenttypeboost"] = source["documenttypeboost"]
        if source.get("court_boost") is not None:
            entry["court_boost"] = source["court_boost"]

        associates = source.get("associates") or {}
        acts = [a["name"] for a in associates.get("act") or [] if a.get("name")]
        if acts:
            entry["referenced_act"] = acts
        sections = [s["name"] for s in associates.get("section") or [] if s.get("name")]
        if sections:
            entry["referenced_section"] = sections
        cases = [c["name"] for c in associates.get("casereferred") or [] if c.get("name")]
        if cases:
            entry["cases_referred"] = cases

        results[doc["_id"]] = entry
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/common/tests/test_es_client.py -k fetch_doc_categories -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/common/src/common/es_client.py packages/common/tests/test_es_client.py
git commit -m "feat(instant-card): extend fetch_doc_categories with judge/party/date/citation/associates fields

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: Backend — `raw_search` gains `page`/`page_size` (default-preserving)

**Files:**
- Modify: `packages/common/src/common/es_client.py:682-736` (`raw_search`)
- Test: `packages/common/tests/test_es_client.py`

**Interfaces:**
- Consumes: nothing new from Task 1.
- Produces: `raw_search(client, query, limit=20, boost=False, boost_source="sum", page=1,
  page_size=None) -> list[dict]` — same return shape. When `page_size` is `None` (every
  existing caller), behavior is byte-identical to today (`size=limit`, no `from_`). When
  `page_size` is set, ES `from_=(page-1)*page_size`, `size=page_size`, ignoring `limit`.

- [ ] **Step 1: Write the failing test**

Add to `packages/common/tests/test_es_client.py`:

```python
@pytest.mark.asyncio
async def test_raw_search_defaults_preserve_no_from_and_limit_as_size():
    client = FakeAsyncES(search_hits=[{"_source": {"id": "d1"}, "_score": 1.0}])

    await raw_search(client, "query", limit=20)

    assert client.size_calls[-1] == 20
    assert client.from_calls[-1] is None  # no `from_` sent at all when page_size is unset


@pytest.mark.asyncio
async def test_raw_search_page_size_maps_to_es_from_and_size():
    client = FakeAsyncES(search_hits=[{"_source": {"id": "d1"}, "_score": 1.0}])

    await raw_search(client, "query", page=3, page_size=10)

    assert client.size_calls[-1] == 10
    assert client.from_calls[-1] == 20  # (page 3 - 1) * page_size 10


@pytest.mark.asyncio
async def test_raw_search_page_size_defaults_page_to_1():
    client = FakeAsyncES(search_hits=[{"_source": {"id": "d1"}, "_score": 1.0}])

    await raw_search(client, "query", page_size=10)

    assert client.size_calls[-1] == 10
    assert client.from_calls[-1] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/common/tests/test_es_client.py -k "raw_search_page or raw_search_defaults_preserve" -v`
Expected: FAIL — `FakeAsyncES` has no `from_calls` attribute / `search()` doesn't accept `from_` yet.

- [ ] **Step 3: Implement**

In `packages/common/tests/test_es_client.py`, extend `FakeAsyncES` (around line 38-58) to
track `from_`:

```python
class FakeAsyncES:
    def __init__(self, search_hits=None, mget_docs=None, index="test_index", aggs_response=None):
        self.search_hits = search_hits or []
        self.mget_docs = mget_docs or {}
        self.search_calls = []
        self.aggs_calls = []
        self.size_calls = []
        self.from_calls = []
        self.highlight_calls = []
        self.source_calls = []
        self.mget_calls = []
        self.index = index
        self.aggs_response = aggs_response or {}

    async def search(self, index, query, size, highlight=None, _source=None, aggs=None, from_=None):
        self.search_calls.append(query)
        self.aggs_calls.append(aggs)
        self.size_calls.append(size)
        self.from_calls.append(from_)
        self.highlight_calls.append(highlight)
        self.source_calls.append(_source)
        self.searched_index = index
        return {"hits": {"hits": self.search_hits}, "aggregations": self.aggs_response}
```

In `packages/common/src/common/es_client.py`, change `raw_search`'s signature (line 682-684)
and the ES call (line 726):

```python
async def raw_search(
    client, query: str, limit: int = 20, boost: bool = False, boost_source: str = "sum",
    page: int = 1, page_size: int | None = None,
) -> list[dict]:
```

Keep the full existing docstring, and append a new paragraph:

```python
    """[... existing docstring unchanged ...]

    page/page_size (added 2026-09-02, hidden-by-default pagination for the Instant-mode
    UI): page_size=None (every existing caller) reproduces prior behavior exactly -
    `size=limit`, no `from_` sent at all. Passing page_size switches to real ES paging -
    `from_=(page-1)*page_size`, `size=page_size` - and `limit` is ignored in that case."""
```

Then change the search call (was: `response = await client.search(index=client.index, query=field_query, size=limit)`):

```python
    if page_size is not None:
        response = await client.search(
            index=client.index, query=field_query, size=page_size, from_=(page - 1) * page_size,
        )
    else:
        response = await client.search(index=client.index, query=field_query, size=limit)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/common/tests/test_es_client.py -v`
Expected: PASS (full file, to confirm the `FakeAsyncES` signature change didn't break any
other test that calls `client.search` positionally with extra args).

- [ ] **Step 5: Commit**

```bash
git add packages/common/src/common/es_client.py packages/common/tests/test_es_client.py
git commit -m "feat(instant-card): add page/page_size to raw_search, default-preserving

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: Backend — thread `page`/`page_size` through `run_instant` → `ws.py`

**Files:**
- Modify: `packages/retrieval-api/src/retrieval_api/instant/search.py:54-86,185-260`
  (`_run_es`, `run_instant`)
- Modify: `packages/retrieval-api/src/retrieval_api/ws.py:135-145,274-280`
- Test: `packages/retrieval-api/tests/test_instant_search.py`

**Interfaces:**
- Consumes: `raw_search(..., page=1, page_size=None)` from Task 2.
- Produces: `run_instant(..., page: int = 1, page_size: int | None = None)`; `_run_es(...,
  page: int = 1, page_size: int | None = None)`. WS message accepts optional `page`/
  `page_size` keys, defaulting to `1`/`None` (today's behavior).

- [ ] **Step 1: Write the failing test**

Find the existing `_run_es`/`run_instant` test setup in
`packages/retrieval-api/tests/test_instant_search.py` (search for `raw_search` mock/patch
pattern already used there) and add:

```python
@pytest.mark.asyncio
async def test_run_instant_passes_page_and_page_size_through_to_raw_search(monkeypatch):
    import retrieval_api.instant.search as search_module

    captured = {}

    async def fake_raw_search(client, query, limit=20, boost=False, boost_source="sum", page=1, page_size=None):
        captured["page"] = page
        captured["page_size"] = page_size
        return [{"doc_id": "d1", "score": 4.2}]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"ruling": []}

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    await run_instant(
        gateway=gateway, es_client=object(), milvus_client=object(), query="section 80HH",
        page=2, page_size=10,
    )

    assert captured == {"page": 2, "page_size": 10}


@pytest.mark.asyncio
async def test_run_instant_defaults_page_to_1_and_page_size_to_none(monkeypatch):
    import retrieval_api.instant.search as search_module

    captured = {}

    async def fake_raw_search(client, query, limit=20, boost=False, boost_source="sum", page=1, page_size=None):
        captured["page"] = page
        captured["page_size"] = page_size
        return [{"doc_id": "d1", "score": 4.2}]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"ruling": []}

    monkeypatch.setattr(search_module, "raw_search", fake_raw_search)
    monkeypatch.setattr(search_module, "hybrid_search", fake_hybrid_search)

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    await run_instant(gateway=gateway, es_client=object(), milvus_client=object(), query="section 80HH")

    assert captured == {"page": 1, "page_size": None}
```

(`AsyncMock` is already imported at the top of this test file — same import every other
test above uses for `gateway`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/retrieval-api/tests/test_instant_search.py -k page -v`
Expected: FAIL — `run_instant()` doesn't accept `page`/`page_size` yet.

- [ ] **Step 3: Implement**

In `packages/retrieval-api/src/retrieval_api/instant/search.py`:

`_run_es` signature (line 54-56) becomes:

```python
async def _run_es(
    es_client, query: str, on_step: OnStep | None, boost: bool = False, skip_cutoff: bool = False,
    boost_source: str = "sum", page: int = 1, page_size: int | None = None,
) -> tuple[list[dict] | None, str | None]:
```

Its `raw_search` call (line 64) becomes:

```python
            raw_results = await raw_search(
                es_client, query, limit=_ES_LIMIT, boost=boost, boost_source=boost_source,
                page=page, page_size=page_size,
            )
```

`run_instant`'s signature (line 185-189) gains `page: int = 1, page_size: int | None = None`,
and its `es_task` construction (line 240-243) becomes:

```python
        es_task = (
            _run_es(
                es_client, query, on_step, boost=boost, skip_cutoff=label == "KEYWORD",
                boost_source=boost_source, page=page, page_size=page_size,
            )
            if plan["es"] else None
        )
```

In `packages/retrieval-api/src/retrieval_api/ws.py`, after the existing `boost_source` line
(144):

```python
    # Hidden-by-default real pagination (feature/repotaxmannapi-exact-replica,
    # 2026-09-02) - page/page_size are only ever sent by the frontend when
    # VITE_ENABLE_PAGINATION is set; every other caller omits them and gets today's
    # exact flat-20 behavior (raw_search's page_size=None default).
    page = message.get("page", 1)
    page_size = message.get("page_size")
```

And in the `run_instant(...)` call (line 274-277), add `page=page, page_size=page_size,`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/retrieval-api/tests/test_instant_search.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/instant/search.py packages/retrieval-api/src/retrieval_api/ws.py packages/retrieval-api/tests/test_instant_search.py
git commit -m "feat(instant-card): thread page/page_size from ws message through to raw_search

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: Frontend — `DocMeta` type + always-visible `doc_id`

**Files:**
- Modify: `packages/web/src/api/useSearch.ts:6-9` (`DocMeta`)
- Modify: `packages/web/src/components/ChatMessageView.tsx:296-317`
- Modify: `packages/web/src/components/GroupedResultsPanel.tsx:67-86`
- Test: `packages/web/src/components/ChatMessageView.test.tsx`
- Test: `packages/web/src/components/TracePanel.test.tsx` (only if it asserts doc_id is
  dev-mode-gated today — check first; update only if so)

**Interfaces:**
- Produces: `DocMeta` now has all-optional fields: `category: string | null`, `group:
  string | null`, `judge?: string[]`, `party?: string[]`, `date?: string`, `viewcount?:
  number`, `documenttypeboost?: number`, `court_boost?: number`, `fullcitation?: string`,
  `referenced_act?: string[]`, `referenced_section?: string[]`, `cases_referred?: string[]`.

- [ ] **Step 1: Write the failing test**

Add to `packages/web/src/components/ChatMessageView.test.tsx` (near the existing `describe`
blocks, reusing the file's own `assistantMessage` helper):

```ts
describe('ChatMessageView result card — doc_id and enriched metadata', () => {
  const instant: ResultState['instant'] = {
    es: [{ doc_id: 'd1', score: 5, heading: 'h1', subheading: 's1' }],
    es_error: null, milvus: null, milvus_sparse: null, milvus_error: null,
    doc_meta: {
      d1: {
        category: 'Direct Tax Laws', group: 'Case Laws',
        judge: ['V.K. KHANNA'], party: ['Commissioner of Income-tax'],
        date: '1987-03-31T00:00:00', viewcount: 70,
      },
    },
  }

  it('shows doc_id even outside dev mode', () => {
    render(<ChatMessageView message={assistantMessage(instant)} devMode={false} onOpenDocument={() => {}} />)
    expect(screen.getByText('d1')).toBeInTheDocument()
  })

  it('shows judge, party, date, and viewcount when present on doc_meta', () => {
    render(<ChatMessageView message={assistantMessage(instant)} devMode={false} onOpenDocument={() => {}} />)
    expect(screen.getByText(/V.K. KHANNA/)).toBeInTheDocument()
    expect(screen.getByText(/Commissioner of Income-tax/)).toBeInTheDocument()
    expect(screen.getByText(/1987-03-31/)).toBeInTheDocument()
    expect(screen.getByText(/70/)).toBeInTheDocument()
  })

  it('omits judge/party lines entirely for a doc with no such doc_meta fields', () => {
    const noExtras: ResultState['instant'] = {
      ...instant,
      doc_meta: { d1: { category: 'Acts', group: 'Acts' } },
    }
    render(<ChatMessageView message={assistantMessage(noExtras)} devMode={false} onOpenDocument={() => {}} />)
    expect(screen.queryByText(/V.K. KHANNA/)).not.toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/web && npx vitest run src/components/ChatMessageView.test.tsx -t "result card"`
Expected: FAIL — `doc_id` still hidden outside dev mode, no judge/party/date/viewcount
rendering exists yet.

- [ ] **Step 3: Implement**

`packages/web/src/api/useSearch.ts:6-9`, replace `DocMeta`:

```ts
export interface DocMeta {
  category: string | null
  group: string | null
  judge?: string[]
  party?: string[]
  date?: string
  viewcount?: number
  documenttypeboost?: number
  court_boost?: number
  fullcitation?: string
  referenced_act?: string[]
  referenced_section?: string[]
  cases_referred?: string[]
}
```

`packages/web/src/components/ChatMessageView.tsx`, replace the card block (lines 296-317):

```tsx
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-sm font-medium truncate" style={{ color: 'var(--text)' }}>
                  {card.heading ? highlightMatches(card.heading, query) : card.doc_id}
                </span>
                {devMode && (
                  <span className="text-xs shrink-0 font-mono" style={{ color: 'var(--text-faint)' }}>
                    {card.score.toFixed(3)}
                  </span>
                )}
              </div>
              <span className="text-xs font-mono mt-1 block truncate" style={{ color: 'var(--text-faint)' }}>
                {card.doc_id}
              </span>
              {devMode && (
                <div className="flex items-center gap-2 mt-1 text-xs" style={{ color: 'var(--text-faint)' }}>
                  <span className="uppercase tracking-wide px-1.5 py-0.5 rounded" style={{ background: 'var(--surface-raised)' }}>
                    {card.source === 'es'
                      ? 'ES'
                      : card.source === 'reranked'
                        ? 'Reranked'
                        : `Milvus ${card.source === 'milvus_dense' ? 'dense' : 'sparse'}:${card.collection}`}
                  </span>
                  {devMode && meta?.documenttypeboost !== undefined && (
                    <span className="font-mono">dtb:{meta.documenttypeboost}</span>
                  )}
                  {devMode && meta?.court_boost !== undefined && (
                    <span className="font-mono">cb:{meta.court_boost}</span>
                  )}
                </div>
              )}
              <CardMetaLines meta={meta} />
```

Add a small shared render helper right above `InstantPane` (used by both this pane and
`GroupedResultsPanel`, so define it in a place both can import — put it in
`packages/web/src/lib/cardMeta.tsx`, a new file):

```tsx
// packages/web/src/lib/cardMeta.tsx
import type { DocMeta } from '../api/useSearch'

/**
 * Renders the optional per-doc metadata lines ported from the reference product's
 * result card (judge/party/date/viewcount/citation/associates) - each line is
 * conditional on that field being present, mirroring the reference product's own
 * per-field *ngIf pattern (a Rule/Act doc simply has no judge/party, same as
 * their card). Not shown: score (dev-mode only, internal) and boost-debug numbers
 * (rendered separately, dev-mode only, by the caller).
 */
export function CardMetaLines({ meta }: { meta: DocMeta | undefined }) {
  if (!meta) return null
  return (
    <>
      {(meta.judge?.length || meta.party?.length) && (
        <p className="text-xs mt-1" style={{ color: 'var(--text-faint)' }}>
          {meta.judge?.length ? `Judge: ${meta.judge.join(', ')}` : null}
          {meta.judge?.length && meta.party?.length ? ' · ' : null}
          {meta.party?.length ? `Party: ${meta.party.join(', ')}` : null}
        </p>
      )}
      {(meta.date || meta.viewcount !== undefined) && (
        <p className="text-xs mt-0.5" style={{ color: 'var(--text-faint)' }}>
          {meta.date ? meta.date.slice(0, 10) : null}
          {meta.date && meta.viewcount !== undefined ? ' · ' : null}
          {meta.viewcount !== undefined ? `${meta.viewcount} views` : null}
        </p>
      )}
      {meta.fullcitation && (
        <p className="text-xs mt-0.5 truncate" style={{ color: 'var(--text-faint)' }}>{meta.fullcitation}</p>
      )}
      {(meta.referenced_act?.length || meta.referenced_section?.length) && (
        <p className="text-xs mt-0.5" style={{ color: 'var(--text-faint)' }}>
          {[...(meta.referenced_act ?? []), ...(meta.referenced_section ?? [])].join(', ')}
        </p>
      )}
      {meta.cases_referred?.length ? (
        <p className="text-xs mt-0.5 truncate" style={{ color: 'var(--text-faint)' }}>
          Cases referred: {meta.cases_referred.join('; ')}
        </p>
      ) : null}
    </>
  )
}
```

Import it in `ChatMessageView.tsx`: `import { CardMetaLines } from '../lib/cardMeta'`.

`packages/web/src/components/GroupedResultsPanel.tsx`, replace lines 67-86:

```tsx
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-sm font-medium truncate" style={{ color: 'var(--text)' }}>
                  {doc.heading ? highlightMatches(doc.heading, query) : doc.doc_id}
                </span>
                {devMode && (
                  <span className="text-xs shrink-0 font-mono" style={{ color: 'var(--text-faint)' }}>
                    {doc.score.toFixed(3)}
                  </span>
                )}
              </div>
              {doc.subheading && (
                <p className="text-sm mt-1 line-clamp-2" style={{ color: 'var(--text-muted)' }}>
                  {highlightMatches(doc.subheading, query)}
                </p>
              )}
              <span className="text-xs font-mono mt-1 block truncate" style={{ color: 'var(--text-faint)' }}>
                {doc.doc_id}
              </span>
              <CardMetaLines meta={docMeta?.[doc.doc_id]} />
```

Add the same import at the top of `GroupedResultsPanel.tsx`:
`import { CardMetaLines } from '../lib/cardMeta'`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/web && npx vitest run src/components/ChatMessageView.test.tsx src/components/TracePanel.test.tsx`
Expected: PASS. If `TracePanel.test.tsx` had an assertion that doc_id is dev-mode-gated
in the grouped panel, update that one assertion to match the new always-visible behavior —
don't touch anything else in that file.

- [ ] **Step 5: Commit**

```bash
git add packages/web/src/api/useSearch.ts packages/web/src/components/ChatMessageView.tsx packages/web/src/components/GroupedResultsPanel.tsx packages/web/src/lib/cardMeta.tsx packages/web/src/components/ChatMessageView.test.tsx packages/web/src/components/TracePanel.test.tsx
git commit -m "feat(instant-card): always show doc_id, render judge/party/date/citation/associates

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 5: Frontend — `VITE_ENABLE_PAGINATION` flag threads `page` through `useSearch`

**Files:**
- Modify: `packages/web/src/api/useSearch.ts:53-94`
- Modify: `packages/web/src/App.tsx` (near `runQuery`, lines ~147-151, and the Prev/Next
  handlers passed into `ChatMessageView`/`InstantPane`)
- Test: `packages/web/src/api/useSearch.test.ts`
- Create: `packages/web/.env.example` entry (see Step 3)

**Interfaces:**
- Consumes: backend `page`/`page_size` support from Task 3.
- Produces: `useSearch(...).search(query, trace, mode, rrf, autoRoute, conversationId,
  boost, boostSource, page?, pageSize?)` — new trailing optional params, defaulting to
  `undefined` (omitted from the WS payload entirely, so the backend's own `1`/`None`
  defaults apply — today's behavior, unchanged).

- [ ] **Step 1: Write the failing test**

Add to `packages/web/src/api/useSearch.test.ts` (find this file's existing WebSocket-mock
pattern for asserting the sent payload, and follow it):

```ts
it('omits page/page_size from the payload when not passed', () => {
  const { result } = renderHook(() => useSearch('ws://test'))
  act(() => {
    result.current.search('cgst', true)
  })
  const socket = MockWebSocket.instances[0]
  act(() => {
    socket.emit('open')
  })
  const sent = JSON.parse(socket.sent[0])
  expect(sent).not.toHaveProperty('page')
  expect(sent).not.toHaveProperty('page_size')
})

it('includes page/page_size in the payload when passed', () => {
  const { result } = renderHook(() => useSearch('ws://test'))
  act(() => {
    result.current.search('cgst', true, 'instant', false, false, undefined, false, 'sum', 2, 10)
  })
  const socket = MockWebSocket.instances[0]
  act(() => {
    socket.emit('open')
  })
  expect(JSON.parse(socket.sent[0])).toMatchObject({ page: 2, page_size: 10 })
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/web && npx vitest run src/api/useSearch.test.ts -t "page"`
Expected: FAIL — `search()` doesn't accept `page`/`pageSize` yet.

- [ ] **Step 3: Implement**

`packages/web/src/api/useSearch.ts`, extend the returned `search` type (line 58-61):

```ts
  search: (
    query: string, trace: boolean, mode?: SearchMode, rrf?: boolean, autoRoute?: boolean,
    conversationId?: string, boost?: boolean, boostSource?: 'sum' | 'repotaxmannapi',
    page?: number, pageSize?: number,
  ) => void
```

The `search` callback's parameter list (line 67-71):

```ts
    (
      query: string, trace: boolean, mode: SearchMode = 'both', rrf: boolean = false,
      autoRoute: boolean = false, conversationId?: string, boost: boolean = false,
      boostSource: 'sum' | 'repotaxmannapi' = 'sum', page?: number, pageSize?: number,
    ) => {
```

The payload construction (line 88-90) — only include the keys when actually passed, so
omitting them reproduces today's exact WS message:

```ts
        const payload: Record<string, unknown> = {
          query, mode, trace, rrf, auto_route: autoRoute, boost, boost_source: boostSource,
        }
        if (page !== undefined) payload.page = page
        if (pageSize !== undefined) payload.page_size = pageSize
        if (accessToken) payload.access_token = accessToken
        if (conversationId) payload.conversation_id = conversationId
```

`packages/web/src/App.tsx`: add the env-flag read near the other top-level consts (same
file, near where `readDevModeFromUrl` is defined):

```ts
// Hidden-by-default real pagination (2026-09-02) - off unless explicitly built with this
// flag set; when off, InstantPane's existing client-side 10-per-page slice over a flat
// 20-result fetch is completely unchanged. No UI checkbox for this, deliberately -
// purely a build-time flag.
const PAGINATION_ENABLED = import.meta.env.VITE_ENABLE_PAGINATION === 'true'
```

In `runQuery` (around line 147-151), only reaching for page 1 initially — real per-page
re-fetching is wired via a `fetchPage` callback passed down to `InstantPane` instead of
inline here, so `runQuery` itself is otherwise unchanged. Add, near `runQuery`:

```ts
  const [instantPage, setInstantPage] = useState(1)

  function fetchInstantPage(conversationId: string, question: string, page: number) {
    if (!PAGINATION_ENABLED) return
    setInstantPage(page)
    classicSearch.search(question, true, 'instant', rrf, autoRoute, undefined, boost, boostSource, page, 20)
  }
```

Reset `instantPage` to `1` wherever `runQuery` already resets other per-query state (same
`useEffect`/call site pattern the file uses for `page`/`setPage` inside `ChatMessageView` —
this is a *separate* piece of state from `InstantPane`'s own internal `page`, deliberately:
`PAGINATION_ENABLED` off means `InstantPane`'s existing internal pagination fully applies
and this new state is simply unused).

This task does NOT wire `fetchInstantPage` into `ChatMessageView`'s Prev/Next buttons yet —
that requires passing a callback prop through `ChatMessageView` → `InstantPane`, which only
matters when `PAGINATION_ENABLED` is true. Add it now as Task 6 (kept separate so this
task's diff stays reviewable on its own: "the flag and the WS plumbing exist" vs "the UI
actually uses it end to end").

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/web && npx vitest run src/api/useSearch.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/web/src/api/useSearch.ts packages/web/src/App.tsx packages/web/src/api/useSearch.test.ts
git commit -m "feat(instant-card): thread optional page/page_size through useSearch, add VITE_ENABLE_PAGINATION flag

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 6: Frontend — wire real pagination into `InstantPane`'s Prev/Next when the flag is on

**Files:**
- Modify: `packages/web/src/components/ChatMessageView.tsx:126-188,325-347`
- Modify: `packages/web/src/App.tsx` (pass `fetchInstantPage` down)
- Test: `packages/web/src/components/ChatMessageView.test.tsx`

**Interfaces:**
- Consumes: `fetchInstantPage(conversationId, question, page)` from Task 5.
- Produces: `InstantPane` accepts an optional `onFetchPage?: (page: number) => void` and
  `paginationEnabled?: boolean` prop; when `paginationEnabled` is false/omitted, behavior is
  100% today's (client-side slice) — this prop threading is the only change to
  `InstantPane`'s public shape.

- [ ] **Step 1: Write the failing test**

Add to `packages/web/src/components/ChatMessageView.test.tsx`:

```ts
describe('InstantPane pagination — hidden by default', () => {
  it('Next button calls onFetchPage with page 2 when paginationEnabled is true', () => {
    const manyEsHits = Array.from({ length: 25 }, (_, i) => ({ doc_id: `d${i}`, score: 1, heading: `h${i}`, subheading: '' }))
    const instant: ResultState['instant'] = { es: manyEsHits, es_error: null, milvus: null, milvus_sparse: null, milvus_error: null }
    const onFetchPage = vi.fn()
    render(
      <ChatMessageView
        message={assistantMessage(instant)} devMode={false} onOpenDocument={() => {}}
        paginationEnabled={true} onFetchPage={onFetchPage}
      />,
    )
    fireEvent.click(screen.getByText('Next'))
    expect(onFetchPage).toHaveBeenCalledWith(2)
  })

  it('does not require onFetchPage when paginationEnabled is false (default, unchanged behavior)', () => {
    const manyEsHits = Array.from({ length: 25 }, (_, i) => ({ doc_id: `d${i}`, score: 1, heading: `h${i}`, subheading: '' }))
    const instant: ResultState['instant'] = { es: manyEsHits, es_error: null, milvus: null, milvus_sparse: null, milvus_error: null }
    render(<ChatMessageView message={assistantMessage(instant)} devMode={false} onOpenDocument={() => {}} />)
    fireEvent.click(screen.getByText('Next'))
    expect(screen.getByText(/Page 2 of/)).toBeInTheDocument() // client-side slice still works exactly as before
  })
})
```

(`ChatMessageView`'s own props need `paginationEnabled`/`onFetchPage` threaded down to
`InstantPane` — check the exact prop-drilling path in the file first; both new props are
optional so every existing call site/test in this file keeps compiling untouched.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/web && npx vitest run src/components/ChatMessageView.test.tsx -t pagination`
Expected: FAIL — `onFetchPage`/`paginationEnabled` props don't exist yet.

- [ ] **Step 3: Implement**

`packages/web/src/components/ChatMessageView.tsx`: thread `paginationEnabled?: boolean` and
`onFetchPage?: (page: number) => void` from `ChatMessageView`'s own props down into
`InstantPane`'s props (line 126), and change the Next/Prev handlers (line 328, 339):

```tsx
function InstantPane({
  result, devMode, onOpenDocument, query, paginationEnabled = false, onFetchPage,
}: {
  result: ResultState | undefined; devMode: boolean; onOpenDocument: (docId: string) => void; query: string
  paginationEnabled?: boolean; onFetchPage?: (page: number) => void
}) {
```

```tsx
          <button
            onClick={() => {
              const next = Math.max(0, page - 1)
              setPage(next)
              if (paginationEnabled) onFetchPage?.(next + 1)
            }}
            disabled={clampedPage === 0}
            ...
          >
            Prev
          </button>
          ...
          <button
            onClick={() => {
              const next = Math.min(pageCount - 1, page + 1)
              setPage(next)
              if (paginationEnabled) onFetchPage?.(next + 1)
            }}
            disabled={clampedPage >= pageCount - 1}
            ...
          >
            Next
          </button>
```

`ChatMessageView`'s own top-level props/signature gains the same two optional props and
passes them straight through to `InstantPane`'s JSX call site.

`packages/web/src/App.tsx`: pass `paginationEnabled={PAGINATION_ENABLED}` and
`onFetchPage={(page) => fetchInstantPage(conversationId, question, page)}` at the
`ChatMessageView` call site (using the `conversationId`/`question` already in scope there).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/web && npx vitest run src/components/ChatMessageView.test.tsx`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/web/src/components/ChatMessageView.tsx packages/web/src/App.tsx packages/web/src/components/ChatMessageView.test.tsx
git commit -m "feat(instant-card): wire real server pagination into InstantPane when VITE_ENABLE_PAGINATION is on

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 7: `.env.example` documentation + full-suite verification

**Files:**
- Modify: `packages/web/.env.example` (create the entry if the file doesn't already list
  build-time flags; check first)

**Interfaces:** none (docs + verification only).

- [ ] **Step 1: Check for `packages/web/.env.example`**

Run: `ls packages/web/.env.example 2>/dev/null || echo "no file"` — if it doesn't exist,
check whether `packages/web/vite.config.ts` or `README.md` documents env vars elsewhere and
follow that pattern instead of inventing a new file.

- [ ] **Step 2: Document the flag**

If `.env.example` exists (or following whatever pattern Step 1 found), add:

```
# Hidden-by-default real server-side pagination for Instant mode's result list.
# Off (unset/false, default): unchanged flat-20-fetch + 10-per-page client-side slice.
# true: Prev/Next re-fetch from the server via page/page_size instead of slicing.
VITE_ENABLE_PAGINATION=false
```

- [ ] **Step 3: Run the full affected test scope**

Run: `uv run pytest packages/common/tests/test_es_client.py packages/retrieval-api/tests/test_instant_search.py -v`
Run: `cd packages/web && npx vitest run src/api/useSearch.test.ts src/components/ChatMessageView.test.tsx src/components/TracePanel.test.tsx`
Expected: all PASS. (Per this repo's own guidance — do not run the full monorepo `pytest`/
`vitest` suite, scope to these files.)

- [ ] **Step 4: Commit**

```bash
git add packages/web/.env.example
git commit -m "docs(instant-card): document VITE_ENABLE_PAGINATION flag

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Explicitly not in this plan

- No changes to `boost_source`/ranking/scoring anywhere.
- No filters/facets/sort-option/spelling-correction UI.
- No Heading3/Heading4 (no schema analog — disclosed, unfixable gap).
- No `associates.affirmreverse`/`associates.rule` (too sparse — 3.98%/1.22%), no
  `landmarkruling` (2.1%, also a settled ranking non-issue per CLAUDE.md).
- No re-audit of the "unaudited, needs live check" bucket beyond what Task 1's live queries
  already confirmed this session (`masterinfo.citations.*` etc. came back confirmed dead;
  nothing was left unresolved).
