# ES Sparse-Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover lexical/keyword search for the 5 Milvus collections whose `sparse_vector` field was dropped (`ruling`, `act_section`, `rule_section`, `article_section`, `commentary_section`) by falling back to Elasticsearch, without touching the 6 collections that still have native Milvus sparse search.

**Architecture:** A new `sparse_fallback_search()` in `common/es_client.py` queries ES filtered by `groups.group.name` (mapped 1:1 from the 5 gap collections), extracts one ~1024-token snippet per doc via ES highlighting, and returns rows in the same `dict[collection, list[row]]` shape `hybrid_search` already produces. `retrieve.py` runs it in parallel with the existing Milvus dense/sparse calls and merges its output into `sparse_by_collection`. `_flatten()` is reworked to rank Milvus-native and ES-origin rows **locally within their own source** and interleave by rank — never comparing raw scores across sources — before handing the result to the existing, unchanged `rrf_merge()`.

**Tech Stack:** Python 3.11, pytest + pytest-asyncio, `elasticsearch` async client, `pymilvus`, new dependency `tiktoken` (`cl100k_base` tokenizer).

**Spec:** `docs/superpowers/specs/2026-08-17-milvus-sparse-es-fallback-design.md`

## Global Constraints

- Only `ruling`, `act_section`, `rule_section`, `article_section`, `commentary_section` get ES fallback. The 6 collections with real `sparse_vector` (`case_summary`, `digest`, `headnotes`, `facts`, `held`, `metadata`) are untouched.
- Category filter mapping is fixed: `act_section`→`ACT`, `rule_section`→`RULE`, `commentary_section`→`COMMENTARY`, `article_section`→`Experts Opinion`, `ruling`→`CASELAWS`.
- No raw-score comparison between ES's BM25 score and Milvus's native BM25-Function score, anywhere (CLAUDE.md hard rule 3). All cross-source fusion is rank-position-based only.
- Snippet target: ~1024 tokens, `tiktoken` `cl100k_base` — not a character count.
- Fetch cap: 20 docs total per ES fallback call; no single `groups.group.name` value may claim more than 15 of those 20 when multiple gap-groups are routed together in one call.
- One snippet per doc, one ES call per query (never one ES call per gap-collection).
- `chunk_id` for ES-origin rows: `f"es:{doc_id}:0"`, tagged `source: "es_fallback"`.
- Dense search is entirely unaffected — Voyage embeddings are comparable across all 11 collections regardless of source; `_flatten()`'s existing global sort-by-score stays correct there.

---

### Task 1: `ES_GROUP_FOR_COLLECTION` mapping

**Files:**
- Modify: `packages/common/src/common/schemas.py`
- Test: `packages/common/tests/test_schemas.py`

**Interfaces:**
- Produces: `ES_GROUP_FOR_COLLECTION: dict[str, str]` — Milvus collection name → ES `groups.group.name` value. Consumed by Task 4 (`sparse_fallback_search`) and Task 6 (`retrieve()`'s gap-collection detection).

- [ ] **Step 1: Write the failing test**

Add to `packages/common/tests/test_schemas.py`:

```python
def test_es_group_for_collection_covers_every_sparse_missing_collection():
    from common.schemas import ES_GROUP_FOR_COLLECTION, MILVUS_COLLECTIONS, SPARSE_VECTOR_COLLECTIONS

    gap_collections = set(MILVUS_COLLECTIONS) - SPARSE_VECTOR_COLLECTIONS
    assert set(ES_GROUP_FOR_COLLECTION.keys()) == gap_collections


def test_es_group_for_collection_values():
    from common.schemas import ES_GROUP_FOR_COLLECTION

    assert ES_GROUP_FOR_COLLECTION == {
        "ruling": "CASELAWS",
        "act_section": "ACT",
        "rule_section": "RULE",
        "article_section": "Experts Opinion",
        "commentary_section": "COMMENTARY",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/common/tests/test_schemas.py -k es_group_for_collection -v`
Expected: FAIL with `ImportError: cannot import name 'ES_GROUP_FOR_COLLECTION'`

- [ ] **Step 3: Write minimal implementation**

In `packages/common/src/common/schemas.py`, add directly below the existing `SPARSE_VECTOR_COLLECTIONS` definition (after line 20):

```python
# ES's groups.group.name field maps 1:1 onto 4 of these 5 by name (ACT/RULE/COMMENTARY/
# CASELAWS). article_section is the exception - a naming mismatch, not a data problem:
# verified live against 20 doc_ids spanning the full id range (20/20 consistent), and
# independently confirmed by a teammate familiar with the ingestion side. See
# docs/superpowers/specs/2026-08-17-milvus-sparse-es-fallback-design.md.
ES_GROUP_FOR_COLLECTION: dict[str, str] = {
    "ruling": "CASELAWS",
    "act_section": "ACT",
    "rule_section": "RULE",
    "article_section": "Experts Opinion",
    "commentary_section": "COMMENTARY",
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/common/tests/test_schemas.py -k es_group_for_collection -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/common/src/common/schemas.py packages/common/tests/test_schemas.py
git commit -m "feat(common): add ES_GROUP_FOR_COLLECTION mapping for sparse-fallback routing"
```

---

### Task 2: Token-budget snippet trimmer

**Files:**
- Modify: `packages/common/src/common/es_client.py`
- Modify: `packages/common/pyproject.toml`
- Test: `packages/common/tests/test_es_client.py`

**Interfaces:**
- Produces: `_trim_to_token_budget(text: str, target_tokens: int = 1024) -> str` in `common/es_client.py`. Consumed by Task 4.

- [ ] **Step 1: Add the dependency**

In `packages/common/pyproject.toml`, add `"tiktoken>=0.7"` to the `dependencies` list (alongside `elasticsearch`).

Run: `uv sync --all-packages`

- [ ] **Step 2: Write the failing test**

Add to `packages/common/tests/test_es_client.py`:

```python
def test_trim_to_token_budget_returns_short_text_unchanged():
    from common.es_client import _trim_to_token_budget

    text = "short text well under budget"
    assert _trim_to_token_budget(text, target_tokens=1024) == text


def test_trim_to_token_budget_centers_and_trims_oversized_text():
    from common.es_client import _trim_to_token_budget
    import tiktoken

    tokenizer = tiktoken.get_encoding("cl100k_base")
    # Build text whose token count is well over budget, with a distinct marker word
    # positioned near the middle - a naive "trim from the end" implementation would
    # keep the marker; a naive "trim from the start" implementation would drop it.
    # Centered trimming keeps it either way, which is what we're asserting.
    before = " ".join(f"word{i}" for i in range(2000))
    after = " ".join(f"word{i}" for i in range(2000, 4000))
    text = f"{before} MARKER {after}"

    trimmed = _trim_to_token_budget(text, target_tokens=100)

    trimmed_tokens = tokenizer.encode(trimmed)
    assert len(trimmed_tokens) == 100
    assert "MARKER" in trimmed
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest packages/common/tests/test_es_client.py -k trim_to_token_budget -v`
Expected: FAIL with `ImportError: cannot import name '_trim_to_token_budget'`

- [ ] **Step 4: Write minimal implementation**

In `packages/common/src/common/es_client.py`, add near the top (after the existing imports, before `_BOOST_PROFILES`):

```python
import tiktoken

# Same tokenizer tm-dp/packages/data-pipeline/src/data_pipeline/chunking.py uses for its
# CHUNK_SIZE_TOKENS=1024 splitter cap - matching it here keeps ES-fallback snippets from
# being systematically under-scored by the reranker for carrying less context than the
# real Milvus chunks they compete against. See
# docs/superpowers/specs/2026-08-17-milvus-sparse-es-fallback-design.md.
_SNIPPET_TOKENIZER = tiktoken.get_encoding("cl100k_base")
_SNIPPET_TARGET_TOKENS = 1024


def _trim_to_token_budget(text: str, target_tokens: int = _SNIPPET_TARGET_TOKENS) -> str:
    """Trims text to at most target_tokens tokens, centered - never expands short text.
    ES's own highlighter already centers a fragment on the best-scoring match; this only
    caps an oversized fragment down to budget, trimming evenly from both ends so a match
    positioned anywhere near the middle of the requested (oversized) fragment survives."""
    ids = _SNIPPET_TOKENIZER.encode(text)
    if len(ids) <= target_tokens:
        return text
    excess = len(ids) - target_tokens
    start = excess // 2
    return _SNIPPET_TOKENIZER.decode(ids[start : start + target_tokens])
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest packages/common/tests/test_es_client.py -k trim_to_token_budget -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add packages/common/src/common/es_client.py packages/common/tests/test_es_client.py packages/common/pyproject.toml uv.lock
git commit -m "feat(common): add tiktoken-based snippet token-budget trimmer"
```

---

### Task 3: Per-group starvation cap

**Files:**
- Modify: `packages/common/src/common/es_client.py`
- Test: `packages/common/tests/test_es_client.py`

**Interfaces:**
- Consumes: nothing new (pure function over plain dicts).
- Produces: `_cap_group_shares(hits: list[dict], limit: int, group_cap: int) -> list[dict]` — each `hit` dict must have a `"_group"` key. Consumed by Task 4.

- [ ] **Step 1: Write the failing test**

Add to `packages/common/tests/test_es_client.py`:

```python
def test_cap_group_shares_single_group_is_unaffected():
    from common.es_client import _cap_group_shares

    hits = [{"_group": "CASELAWS", "id": i} for i in range(20)]
    result = _cap_group_shares(hits, limit=20, group_cap=15)

    assert result == hits


def test_cap_group_shares_trims_dominant_group_and_backfills_from_minority():
    from common.es_client import _cap_group_shares

    # 18 CASELAWS hits (relevance rank 1-18) + 2 Experts Opinion hits (rank 19-20) -
    # naive top-20 would return 18 CASELAWS + 2 Experts Opinion. The cap should trim
    # CASELAWS to 15 and backfill the 3 freed slots from Experts Opinion's next-best
    # hits (which don't exist here beyond the 2 already present, so this asserts what
    # DOES exist survives and CASELAWS is capped, not that phantom hits appear).
    caselaws_hits = [{"_group": "CASELAWS", "id": f"cl{i}"} for i in range(18)]
    eo_hits = [{"_group": "Experts Opinion", "id": f"eo{i}"} for i in range(2)]
    hits = caselaws_hits + eo_hits  # already in relevance order

    result = _cap_group_shares(hits, limit=20, group_cap=15)

    result_caselaws = [h for h in result if h["_group"] == "CASELAWS"]
    result_eo = [h for h in result if h["_group"] == "Experts Opinion"]
    assert len(result_caselaws) == 15
    assert result_caselaws == caselaws_hits[:15]
    assert result_eo == eo_hits


def test_cap_group_shares_backfills_to_full_limit_when_minority_group_has_more():
    from common.es_client import _cap_group_shares

    # 18 CASELAWS + 10 Experts Opinion, all in relevance order (interleaved isn't
    # required - the function must not assume a particular pre-existing interleave).
    caselaws_hits = [{"_group": "CASELAWS", "id": f"cl{i}"} for i in range(18)]
    eo_hits = [{"_group": "Experts Opinion", "id": f"eo{i}"} for i in range(10)]
    hits = caselaws_hits + eo_hits

    result = _cap_group_shares(hits, limit=20, group_cap=15)

    assert len(result) == 20
    result_caselaws = [h for h in result if h["_group"] == "CASELAWS"]
    result_eo = [h for h in result if h["_group"] == "Experts Opinion"]
    assert len(result_caselaws) == 15
    assert len(result_eo) == 5
    assert result_caselaws == caselaws_hits[:15]
    assert result_eo == eo_hits[:5]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/common/tests/test_es_client.py -k cap_group_shares -v`
Expected: FAIL with `ImportError: cannot import name '_cap_group_shares'`

- [ ] **Step 3: Write minimal implementation**

In `packages/common/src/common/es_client.py`, add after `_trim_to_token_budget`:

```python
def _cap_group_shares(hits: list[dict], limit: int, group_cap: int) -> list[dict]:
    """hits must already be in ES relevance order (ES's own default sort). Caps any single
    group's share of the top `limit` hits at `group_cap`, backfilling freed slots from the
    other routed group(s)' next-best hits (in their own relevance order) rather than
    returning fewer than `limit` total. With only one group present, this is a no-op past
    the limit slice - the cap only ever engages with 2+ groups in the same call. See
    "Per-group starvation cap" in
    docs/superpowers/specs/2026-08-17-milvus-sparse-es-fallback-design.md."""
    if len({hit["_group"] for hit in hits}) <= 1:
        return hits[:limit]

    taken_counts: dict[str, int] = {}
    kept: list[dict] = []
    skipped: list[dict] = []
    for hit in hits:
        group = hit["_group"]
        if taken_counts.get(group, 0) < group_cap:
            kept.append(hit)
            taken_counts[group] = taken_counts.get(group, 0) + 1
        else:
            skipped.append(hit)
        if len(kept) == limit:
            break

    if len(kept) < limit:
        kept.extend(skipped[: limit - len(kept)])
    return kept[:limit]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/common/tests/test_es_client.py -k cap_group_shares -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/common/src/common/es_client.py packages/common/tests/test_es_client.py
git commit -m "feat(common): add per-group starvation cap for shared ES fallback calls"
```

---

### Task 4: `sparse_fallback_search()`

**Files:**
- Modify: `packages/common/src/common/es_client.py`
- Test: `packages/common/tests/test_es_client.py`

**Interfaces:**
- Consumes: `_trim_to_token_budget` (Task 2), `_cap_group_shares` (Task 3), `ES_GROUP_FOR_COLLECTION` (Task 1, from `common.schemas`), existing `build_query_preview` (this file).
- Produces: `async def sparse_fallback_search(client, query: str, groups: list[str], doc_id_allowlist: list[str] | None = None, limit: int = 20, group_cap: int = 15) -> dict[str, list[dict]]` — same `dict[collection, list[row]]` shape as `common.milvus_client.hybrid_search`. Each row: `{"chunk_id": str, "doc_id": str, "text": str, "score": float, "source": "es_fallback"}`. Consumed by Task 6.

- [ ] **Step 1: Write the failing test**

Add to `packages/common/tests/test_es_client.py` (reuse the existing `FakeAsyncES` class already in this file):

```python
@pytest.mark.asyncio
async def test_sparse_fallback_search_filters_by_group_and_partitions_by_collection():
    client = FakeAsyncES(search_hits=[
        {
            "_source": {"id": "d1", "groups": {"group": {"name": "CASELAWS"}}},
            "_score": 9.0,
            "highlight": {"fullcontent": ["snippet about the ruling"]},
        },
        {
            "_source": {"id": "d2", "groups": {"group": {"name": "Experts Opinion"}}},
            "_score": 7.0,
            "highlight": {"fullcontent": ["snippet about the article"]},
        },
    ], index="researchindex_aic_test")

    from common.es_client import sparse_fallback_search

    result = await sparse_fallback_search(client, "query text", groups=["CASELAWS", "Experts Opinion"])

    assert result == {
        "ruling": [{
            "chunk_id": "es:d1:0", "doc_id": "d1", "text": "snippet about the ruling",
            "score": 9.0, "source": "es_fallback",
        }],
        "article_section": [{
            "chunk_id": "es:d2:0", "doc_id": "d2", "text": "snippet about the article",
            "score": 7.0, "source": "es_fallback",
        }],
    }


@pytest.mark.asyncio
async def test_sparse_fallback_search_applies_doc_id_allowlist_and_highlight_config():
    client = FakeAsyncES(search_hits=[], index="researchindex_aic_test")

    from common.es_client import sparse_fallback_search

    await sparse_fallback_search(client, "query text", groups=["ACT"], doc_id_allowlist=["d1", "d2"])

    query = client.search_calls[0]
    must_clauses = query["bool"]["must"]
    assert {"terms": {"groups.group.name.keyword": ["ACT"]}} in must_clauses
    assert {"terms": {"id": ["d1", "d2"]}} in must_clauses


@pytest.mark.asyncio
async def test_sparse_fallback_search_skips_hits_missing_highlight_or_unknown_group():
    client = FakeAsyncES(search_hits=[
        {"_source": {"id": "d1", "groups": {"group": {"name": "CASELAWS"}}}, "_score": 9.0, "highlight": {}},
        {"_source": {"id": "d2", "groups": {"group": {"name": "Nonsense Group"}}}, "_score": 8.0,
         "highlight": {"fullcontent": ["x"]}},
    ], index="researchindex_aic_test")

    from common.es_client import sparse_fallback_search

    result = await sparse_fallback_search(client, "query text", groups=["CASELAWS"])

    assert result == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/common/tests/test_es_client.py -k sparse_fallback_search -v`
Expected: FAIL with `ImportError: cannot import name 'sparse_fallback_search'`

- [ ] **Step 3: Write minimal implementation**

In `packages/common/src/common/es_client.py`, add after `_cap_group_shares` (and add `from common.schemas import ES_GROUP_FOR_COLLECTION, MASTERINFO_CITATION_FIELDS` to the existing import line near the top):

```python
_ES_FALLBACK_LIMIT = 20
_ES_FALLBACK_GROUP_CAP = 15
_ES_HIGHLIGHT_FRAGMENT_CHARS = 6000  # oversized on purpose - _trim_to_token_budget cuts to ~1024 tokens after

_COLLECTION_FOR_ES_GROUP = {group: collection for collection, group in ES_GROUP_FOR_COLLECTION.items()}


async def sparse_fallback_search(
    client, query: str, groups: list[str], doc_id_allowlist: list[str] | None = None,
    limit: int = _ES_FALLBACK_LIMIT, group_cap: int = _ES_FALLBACK_GROUP_CAP,
) -> dict[str, list[dict]]:
    """ES fallback for lexical search on the Milvus collections whose sparse_vector was
    dropped. One ES call per query regardless of how many gap-collections are routed
    together - `groups` is the list of ES groups.group.name values to search (mapped from
    the routed gap-collections via ES_GROUP_FOR_COLLECTION), OR'd into one filter. Returns
    rows partitioned back into the same dict[collection, list[row]] shape
    common.milvus_client.hybrid_search returns, via the inverse of that same mapping. See
    docs/superpowers/specs/2026-08-17-milvus-sparse-es-fallback-design.md."""
    field_query = build_query_preview(query)["es_query"]
    must: list[dict] = [{"terms": {"groups.group.name.keyword": groups}}]
    if doc_id_allowlist:
        must.append({"terms": {"id": doc_id_allowlist}})
    must.append(field_query)

    # Requesting more than `limit` when multiple groups are routed gives _cap_group_shares
    # a real pool to draw from - without this, ES's own top-`limit` (sorted globally by
    # score) could already be dominated by one group before the cap ever sees the rest.
    fetch_size = limit if len(groups) <= 1 else limit * len(groups)

    response = await client.search(
        index=client.index,
        query={"bool": {"must": must}},
        size=fetch_size,
        highlight={"fields": {"fullcontent": {
            "fragment_size": _ES_HIGHLIGHT_FRAGMENT_CHARS, "number_of_fragments": 1,
        }}},
    )

    hits = []
    for hit in response["hits"]["hits"]:
        source = hit["_source"]
        group = source.get("groups", {}).get("group", {}).get("name")
        fragments = hit.get("highlight", {}).get("fullcontent")
        if group not in _COLLECTION_FOR_ES_GROUP or not fragments:
            continue
        hits.append({
            "_group": group, "_doc_id": source["id"], "_snippet": fragments[0], "_score": hit["_score"],
        })

    capped = _cap_group_shares(hits, limit, group_cap)

    by_collection: dict[str, list[dict]] = {}
    for hit in capped:
        collection = _COLLECTION_FOR_ES_GROUP[hit["_group"]]
        row = {
            "chunk_id": f"es:{hit['_doc_id']}:0",
            "doc_id": hit["_doc_id"],
            "text": _trim_to_token_budget(hit["_snippet"]),
            "score": hit["_score"],
            "source": "es_fallback",
        }
        by_collection.setdefault(collection, []).append(row)
    return by_collection
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/common/tests/test_es_client.py -k sparse_fallback_search -v`
Expected: PASS

- [ ] **Step 5: Run the full common package test suite**

Run: `uv run pytest packages/common/tests -v`
Expected: PASS (no regressions in existing `es_client`/`schemas`/`milvus_client` tests)

- [ ] **Step 6: Commit**

```bash
git add packages/common/src/common/es_client.py packages/common/tests/test_es_client.py
git commit -m "feat(common): add sparse_fallback_search for ES-backed lexical search on gap collections"
```

---

### Task 5: `_flatten()` — source-aware local-rank interleave

**Files:**
- Modify: `packages/retrieval-api/src/retrieval_api/ai_mode/retrieve.py`
- Test: `packages/retrieval-api/tests/test_ai_mode_retrieve.py`

**Interfaces:**
- Consumes: nothing new (still takes `dict[str, list[dict]]`, same as today).
- Produces: `_flatten(by_collection: dict[str, list[dict]]) -> list[dict]` — **behavior change**: when any row carries `"source": "es_fallback"`, ranks native and ES-origin rows locally by their own `score`, then interleaves by rank (round-robin, longer list's remainder appended in its own order) instead of one global sort-by-score. When no row carries that tag, behavior is byte-for-byte identical to today (plain global sort by `score` — this is what keeps `retrieval_eval.py`'s direct `_flatten()` calls and the dense pass unaffected).

- [ ] **Step 1: Write the failing test**

Add to `packages/retrieval-api/tests/test_ai_mode_retrieve.py`:

```python
def test_flatten_plain_sort_unchanged_when_no_es_fallback_rows():
    from retrieval_api.ai_mode.retrieve import _flatten

    by_collection = {
        "held": [{"chunk_id": "h1", "score": 3.0}],
        "facts": [{"chunk_id": "f1", "score": 9.0}, {"chunk_id": "f2", "score": 1.0}],
    }
    result = _flatten(by_collection)

    assert [row["chunk_id"] for row in result] == ["f1", "h1", "f2"]


def test_flatten_interleaves_native_and_es_fallback_rows_by_local_rank():
    from retrieval_api.ai_mode.retrieve import _flatten

    # Native rows carry high raw scores (Milvus BM25 Function scale), ES rows carry low
    # raw scores (ES BM25 scale) - a naive global sort-by-score would put every native
    # row ahead of every ES row regardless of true relevance. Interleaving must not do
    # that: each source's own #1 gets equal footing.
    by_collection = {
        "held": [
            {"chunk_id": "n1", "score": 100.0},
            {"chunk_id": "n2", "score": 90.0},
        ],
        "ruling": [
            {"chunk_id": "e1", "score": 2.0, "source": "es_fallback"},
            {"chunk_id": "e2", "score": 1.0, "source": "es_fallback"},
        ],
    }
    result = _flatten(by_collection)

    assert [row["chunk_id"] for row in result] == ["n1", "e1", "n2", "e2"]


def test_flatten_interleave_appends_longer_lists_remainder_in_rank_order():
    from retrieval_api.ai_mode.retrieve import _flatten

    by_collection = {
        "held": [
            {"chunk_id": "n1", "score": 100.0},
            {"chunk_id": "n2", "score": 90.0},
            {"chunk_id": "n3", "score": 80.0},
        ],
        "ruling": [{"chunk_id": "e1", "score": 2.0, "source": "es_fallback"}],
    }
    result = _flatten(by_collection)

    assert [row["chunk_id"] for row in result] == ["n1", "e1", "n2", "n3"]


def test_flatten_interleave_handles_es_only_input():
    from retrieval_api.ai_mode.retrieve import _flatten

    by_collection = {
        "ruling": [
            {"chunk_id": "e1", "score": 2.0, "source": "es_fallback"},
            {"chunk_id": "e2", "score": 5.0, "source": "es_fallback"},
        ],
    }
    result = _flatten(by_collection)

    assert [row["chunk_id"] for row in result] == ["e2", "e1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_retrieve.py -k flatten -v`
Expected: `test_flatten_plain_sort_unchanged_when_no_es_fallback_rows` PASSes already (current behavior); the other three FAIL (current `_flatten` does one global sort, doesn't interleave).

- [ ] **Step 3: Write minimal implementation**

In `packages/retrieval-api/src/retrieval_api/ai_mode/retrieve.py`, add `from itertools import zip_longest` to the imports, then replace the existing `_flatten` function (lines 23-25):

```python
def _flatten(by_collection: dict[str, list[dict]]) -> list[dict]:
    all_rows = [row for rows in by_collection.values() for row in rows]
    es_rows = [row for row in all_rows if row.get("source") == "es_fallback"]
    if not es_rows:
        return sorted(all_rows, key=lambda row: row["score"], reverse=True)

    native_rows = [row for row in all_rows if row.get("source") != "es_fallback"]
    ranked_native = sorted(native_rows, key=lambda row: row["score"], reverse=True)
    ranked_es = sorted(es_rows, key=lambda row: row["score"], reverse=True)

    interleaved: list[dict] = []
    for native_row, es_row in zip_longest(ranked_native, ranked_es):
        if native_row is not None:
            interleaved.append(native_row)
        if es_row is not None:
            interleaved.append(es_row)
    return interleaved
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_retrieve.py -k flatten -v`
Expected: PASS (all four)

- [ ] **Step 5: Run the full retrieval-api test suite to check for regressions**

Run: `uv run pytest packages/retrieval-api/tests -v`
Expected: PASS. `retrieval_eval.py`'s direct `_flatten()` calls pass plain Milvus dicts with no `source` key, so they hit the unchanged code path.

- [ ] **Step 6: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/ai_mode/retrieve.py packages/retrieval-api/tests/test_ai_mode_retrieve.py
git commit -m "feat(retrieval-api): rank ES-fallback rows locally, interleave with Milvus-native rows by rank

Prevents raw-score comparison between ES's BM25 score and Milvus's
native BM25-Function score (CLAUDE.md hard rule 3) when a sparse pass
mixes both sources. No behavior change when no ES-fallback rows are
present - retrieval_eval.py's direct _flatten() calls and the dense
pass are both unaffected."
```

---

### Task 6: Wire ES fallback into `retrieve()`

**Files:**
- Modify: `packages/retrieval-api/src/retrieval_api/ai_mode/retrieve.py`
- Modify: `packages/retrieval-api/src/retrieval_api/ai_mode/pipeline.py`
- Test: `packages/retrieval-api/tests/test_ai_mode_retrieve.py`
- Test: `packages/retrieval-api/tests/test_ai_mode_pipeline.py`

**Interfaces:**
- Consumes: `sparse_fallback_search` (Task 4), `ES_GROUP_FOR_COLLECTION` (Task 1), `_flatten` (Task 5, unchanged call site).
- Produces: `retrieve()`'s signature changes from `retrieve(gateway, milvus_client, search_query, doc_id_allowlist, intent=None, on_step=None)` to `retrieve(gateway, milvus_client, es_client, search_query, doc_id_allowlist, intent=None, on_step=None)` — **breaking signature change**, every call site must be updated in this task.

- [ ] **Step 1: Update the existing `retrieve()` test for the new signature**

In `packages/retrieval-api/tests/test_ai_mode_retrieve.py`, the existing `test_retrieve_embeds_search_query_and_merges_dense_sparse` (around line 62) calls `retrieve()` with no `intent` argument. `collections_for_intent(None or [])` returns **all 11 collections**, which includes the 5 gap collections — so this test must also stub `sparse_fallback_search`, or it will try a real ES call through `es_client=object()`. Replace the test with:

```python
@pytest.mark.asyncio
async def test_retrieve_embeds_search_query_and_merges_dense_sparse(monkeypatch):
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        if dense_vector is not None:
            return {"ruling": [{"chunk_id": "a", "doc_id": "d1", "text": "t", "score": 0.9}]}
        return {}  # ruling has no native sparse_vector - matches real SPARSE_VECTOR_COLLECTIONS behavior

    async def fake_sparse_fallback_search(client, query, groups, doc_id_allowlist=None):
        return {"ruling": [{
            "chunk_id": "es:d1:0", "doc_id": "d1", "text": "t", "score": 5.0, "source": "es_fallback",
        }]}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(module, "sparse_fallback_search", fake_sparse_fallback_search)

    result = await retrieve(
        gateway, milvus_client=object(), es_client=object(), search_query="q", doc_id_allowlist=["d1"],
    )

    gateway.embed.assert_awaited_once_with(role="query_embed", text="q")
    assert result[0]["chunk_id"] == "a"
```

- [ ] **Step 2: Add a test asserting the ES fallback call is made for gap collections and skipped otherwise**

Add to `packages/retrieval-api/tests/test_ai_mode_retrieve.py`:

```python
@pytest.mark.asyncio
async def test_retrieve_calls_es_fallback_only_for_routed_gap_collections(monkeypatch):
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]
    es_calls = []

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"case_summary": [{"chunk_id": "a", "doc_id": "d1", "text": "t", "score": 1.0}]}

    async def fake_sparse_fallback_search(client, query, groups, doc_id_allowlist=None):
        es_calls.append(groups)
        return {}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(module, "sparse_fallback_search", fake_sparse_fallback_search)

    # intent "caselaws" routes to case_summary/digest/headnotes/facts/held/ruling/metadata -
    # "ruling" is the one gap collection in that set, mapped to ES group CASELAWS.
    await retrieve(gateway, milvus_client=object(), es_client=object(), search_query="q",
                    doc_id_allowlist=None, intent=["caselaws"])

    assert es_calls == [["CASELAWS"]]


@pytest.mark.asyncio
async def test_retrieve_skips_es_fallback_when_no_gap_collection_routed(monkeypatch):
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]
    es_calls = []

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"held": [{"chunk_id": "a", "doc_id": "d1", "text": "t", "score": 1.0}]}

    async def fake_sparse_fallback_search(client, query, groups, doc_id_allowlist=None):
        es_calls.append(groups)
        return {}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(module, "sparse_fallback_search", fake_sparse_fallback_search)

    # No such single-collection intent tag exists today that avoids every gap collection
    # except by routing to a strict subset - this test constructs that condition directly
    # by monkeypatching collections_for_intent so the test doesn't depend on future intent
    # taxonomy changes.
    monkeypatch.setattr(module, "collections_for_intent", lambda intent: ["held"])

    await retrieve(gateway, milvus_client=object(), es_client=object(), search_query="q",
                    doc_id_allowlist=None, intent=["caselaws"])

    assert es_calls == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_retrieve.py -v`
Expected: FAIL — `retrieve()` doesn't accept `es_client` yet (`TypeError: retrieve() got an unexpected keyword argument 'es_client'`), and `sparse_fallback_search` isn't imported into the `retrieve` module yet.

- [ ] **Step 4: Write the implementation**

In `packages/retrieval-api/src/retrieval_api/ai_mode/retrieve.py`:

Add imports (alongside the existing ones at the top of the file):

```python
import asyncio

from common.es_client import sparse_fallback_search
from common.schemas import ES_GROUP_FOR_COLLECTION, SPARSE_VECTOR_COLLECTIONS, collections_for_intent
```

Replace the `retrieve()` function signature and body (currently lines 28-96) with:

```python
async def retrieve(
    gateway: GatewayClient,
    milvus_client,
    es_client,
    search_query: str,
    doc_id_allowlist: list[str] | None,
    intent: list[str] | None = None,
    on_step: OnStep | None = None,
) -> list[dict]:
    collections = collections_for_intent(intent or [])
    gap_collections = [
        c for c in collections if c not in SPARSE_VECTOR_COLLECTIONS and c in ES_GROUP_FOR_COLLECTION
    ]

    dense_vector = await gateway.embed(role="query_embed", text=search_query)

    async def _run_es_fallback(allowlist):
        if not gap_collections:
            return {}
        groups = [ES_GROUP_FOR_COLLECTION[c] for c in gap_collections]
        return await sparse_fallback_search(es_client, search_query, groups, doc_id_allowlist=allowlist)

    dense_by_collection, sparse_by_collection, es_sparse_by_collection = await asyncio.gather(
        hybrid_search(
            milvus_client, collections=collections, dense_vector=dense_vector,
            sparse_query_text=search_query, doc_id_allowlist=doc_id_allowlist, limit=50,
        ),
        hybrid_search(
            milvus_client, collections=collections, dense_vector=None,
            sparse_query_text=search_query, doc_id_allowlist=doc_id_allowlist, limit=50,
        ),
        _run_es_fallback(doc_id_allowlist),
    )
    sparse_by_collection.update(es_sparse_by_collection)

    # Circuit breaker: a resolved doc_id_allowlist that's non-empty but the wrong kind of
    # document for these collections silently zeroes every collection even though an
    # unfiltered search would find real matches. If the allowlist was non-empty but
    # produced zero hits everywhere, retry once unfiltered rather than returning nothing -
    # the embedding is already computed, so this only costs the extra round-trips.
    # Retries against the SAME routed collection set - a routed-but-genuinely-wrong-
    # category query should surface as zero results, not silently widen to every
    # collection (that would defeat the point of routing).
    if doc_id_allowlist and not any(dense_by_collection.values()) and not any(sparse_by_collection.values()):
        if on_step is not None:
            await on_step("filter_fallback", {
                "reason": "doc_id_allowlist matched zero Milvus results across every routed collection; retrying unfiltered",
                "doc_id_allowlist_count": len(doc_id_allowlist),
            })
        dense_by_collection, sparse_by_collection, es_sparse_by_collection = await asyncio.gather(
            hybrid_search(
                milvus_client, collections=collections, dense_vector=dense_vector,
                sparse_query_text=search_query, doc_id_allowlist=None, limit=50,
            ),
            hybrid_search(
                milvus_client, collections=collections, dense_vector=None,
                sparse_query_text=search_query, doc_id_allowlist=None, limit=50,
            ),
            _run_es_fallback(None),
        )
        sparse_by_collection.update(es_sparse_by_collection)

    if on_step is not None:
        await on_step("milvus_dense", collection_trace(dense_by_collection))
        await on_step("milvus_sparse", collection_trace(sparse_by_collection))

    # RRF fusion weight is always neutral - category does not drive dense/sparse
    # weighting (considered during brainstorming, explicitly rejected; see
    # docs/superpowers/specs/2026-08-14-category-collection-routing-design.md).
    merged = rrf_merge(_flatten(dense_by_collection), _flatten(sparse_by_collection))

    if on_step is not None:
        top_candidates = [
            {
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "rrf_score": row["rrf_score"],
                "text_preview": row["text"][:200],
            }
            for row in merged[:15]
        ]
        await on_step("rrf_merge", {
            "candidate_count": len(merged), "top_candidates": top_candidates,
            "dense_weight": 1.0, "sparse_weight": 1.0,
        })

    return merged
```

Note: `sparse_by_collection.update(es_sparse_by_collection)` is safe without a key-collision check — `hybrid_search`'s own sparse pass (`common/milvus_client.py:63-64`) already excludes every collection not in `SPARSE_VECTOR_COLLECTIONS`, so `sparse_by_collection` can never already have a key for a `gap_collections` entry.

- [ ] **Step 5: Update `pipeline.py`'s call site**

In `packages/retrieval-api/src/retrieval_api/ai_mode/pipeline.py`, the `retrieve()` call (currently lines 32-35):

```python
                candidates = await retrieve(
                    gateway, milvus_client, intent_result["search_query"], doc_id_allowlist,
                    intent_result["intent"], on_step=on_step,
                )
```

becomes:

```python
                candidates = await retrieve(
                    gateway, milvus_client, es_client, intent_result["search_query"], doc_id_allowlist,
                    intent_result["intent"], on_step=on_step,
                )
```

(`es_client` is already an in-scope parameter of `run_ai_mode`.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_retrieve.py packages/retrieval-api/tests/test_ai_mode_pipeline.py -v`
Expected: PASS

- [ ] **Step 7: Run the full retrieval-api and common test suites**

Run: `uv run pytest packages/retrieval-api/tests packages/common/tests -v`
Expected: PASS. Check specifically for any other direct caller of `retrieve()` the grep in this plan's prep didn't catch (`retrieval_eval.py` imports `_flatten`/`rrf_merge` directly, not `retrieve()`, so it's unaffected — confirm this is still true by checking its imports).

- [ ] **Step 8: Commit**

```bash
git add packages/retrieval-api/src/retrieval_api/ai_mode/retrieve.py packages/retrieval-api/src/retrieval_api/ai_mode/pipeline.py packages/retrieval-api/tests/test_ai_mode_retrieve.py
git commit -m "feat(retrieval-api): wire ES sparse-fallback into retrieve()'s dense/sparse gather

retrieve() gains a required es_client parameter. ES fallback runs in
the same asyncio.gather as the Milvus dense/sparse calls whenever the
routed collection set includes a gap collection (ruling/act_section/
rule_section/article_section/commentary_section), decided upfront by
static SPARSE_VECTOR_COLLECTIONS membership - no sequencing cost."
```

---

### Task 7: CLAUDE.md hard rule 3 rewrite

**Files:**
- Modify: `CLAUDE.md`

**Interfaces:** none (documentation only).

- [ ] **Step 1: Rewrite rule 3**

In `CLAUDE.md`, replace the existing rule 3:

```markdown
3. **No raw-score ranking fusion between ES and Milvus.** `doc_id` is join-only (citation lookup, filter allowlist) by default. Don't blend ES's lexical score and Milvus's cosine/BM25-distance score directly — they're on incomparable scales. The one sanctioned exception: Instant mode's opt-in `rerank` toggle (`instant/rerank.py::rrf_merge_by_doc_id`) fuses ES + Milvus dense + Milvus sparse by *rank position* via RRF, not raw score — rank-based fusion sidesteps the incomparable-scale problem this rule exists to prevent. Don't extend raw-score blending elsewhere off the back of this exception.
```

With:

```markdown
3. **No raw-score ranking fusion between ES and Milvus.** `doc_id` is join-only (citation lookup, filter allowlist) by default. Don't blend ES's lexical score and Milvus's cosine/BM25-distance score directly — they're on incomparable scales. Two sanctioned exceptions, both rank-based, never raw-score:
   - Instant mode's opt-in `rerank` toggle (`instant/rerank.py::rrf_merge_by_doc_id`) fuses ES + Milvus dense + Milvus sparse by *rank position* via RRF.
   - AI Mode's ES sparse-fallback (`ai_mode/retrieve.py::_flatten()`, for `ruling`/`act_section`/`rule_section`/`article_section`/`commentary_section` — the collections whose Milvus `sparse_vector` was dropped) ranks Milvus-native and ES-origin sparse hits *locally within their own source*, then interleaves by rank position to build the list `rrf_merge()` fuses — raw scores from the two sources are never compared. See `docs/superpowers/specs/2026-08-17-milvus-sparse-es-fallback-design.md`.

   Don't extend raw-score blending elsewhere off the back of either exception.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update hard rule 3 for the ES sparse-fallback rank-based fusion exception"
```

---

### Task 8: End-to-end retrieval tests for gap-only and mixed-intent routing

**Files:**
- Test: `packages/retrieval-api/tests/test_ai_mode_retrieve.py`

**Interfaces:** none new — this task only adds coverage over Tasks 5+6's combined behavior.

- [ ] **Step 1: Write the tests**

Add to `packages/retrieval-api/tests/test_ai_mode_retrieve.py`:

```python
@pytest.mark.asyncio
async def test_retrieve_gap_only_intent_still_returns_ranked_results(monkeypatch):
    """intent=["acts"] routes only to act_section, a gap collection with no native sparse -
    the sparse pass must not silently degrade to empty just because Milvus sparse skips it."""
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        if dense_vector is not None:
            return {"act_section": [{"chunk_id": "d1", "doc_id": "doc1", "text": "dense hit", "score": 0.8}]}
        return {}  # act_section excluded from native sparse, same as production SPARSE_VECTOR_COLLECTIONS

    async def fake_sparse_fallback_search(client, query, groups, doc_id_allowlist=None):
        assert groups == ["ACT"]
        return {"act_section": [{
            "chunk_id": "es:doc1:0", "doc_id": "doc1", "text": "es hit",
            "score": 4.0, "source": "es_fallback",
        }]}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(module, "sparse_fallback_search", fake_sparse_fallback_search)

    result = await retrieve(gateway, milvus_client=object(), es_client=object(), search_query="q",
                             doc_id_allowlist=None, intent=["acts"])

    chunk_ids = {row["chunk_id"] for row in result}
    assert chunk_ids == {"d1", "es:doc1:0"}


@pytest.mark.asyncio
async def test_retrieve_mixed_intent_produces_both_native_and_es_origin_rows(monkeypatch):
    """intent=["caselaws", "articles"] routes case_summary/digest/headnotes/facts/held/
    metadata (native sparse) + ruling + article_section (both gap collections, one shared
    ES call)."""
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        if dense_vector is not None:
            return {c: [{"chunk_id": f"dense-{c}", "doc_id": f"doc-{c}", "text": "t", "score": 0.5}] for c in collections}
        return {"held": [{"chunk_id": "native-held", "doc_id": "doc-held", "text": "t", "score": 9.0}]}

    async def fake_sparse_fallback_search(client, query, groups, doc_id_allowlist=None):
        assert sorted(groups) == sorted(["CASELAWS", "Experts Opinion"])
        return {
            "ruling": [{"chunk_id": "es-ruling", "doc_id": "doc-ruling", "text": "t", "score": 3.0, "source": "es_fallback"}],
            "article_section": [{"chunk_id": "es-article", "doc_id": "doc-article", "text": "t", "score": 2.0, "source": "es_fallback"}],
        }

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(module, "sparse_fallback_search", fake_sparse_fallback_search)

    result = await retrieve(gateway, milvus_client=object(), es_client=object(), search_query="q",
                             doc_id_allowlist=None, intent=["caselaws", "articles"])

    chunk_ids = {row["chunk_id"] for row in result}
    assert "native-held" in chunk_ids
    assert "es-ruling" in chunk_ids
    assert "es-article" in chunk_ids
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest packages/retrieval-api/tests/test_ai_mode_retrieve.py -v`
Expected: PASS (these exercise Tasks 5+6's already-implemented behavior — no new production code expected in this task; if either fails, it means Task 5 or 6's implementation has a gap, fix there before proceeding).

- [ ] **Step 3: Run the entire test suite**

Run: `uv run pytest` (from repo root)
Expected: PASS, all 4 packages.

- [ ] **Step 4: Commit**

```bash
git add packages/retrieval-api/tests/test_ai_mode_retrieve.py
git commit -m "test(retrieval-api): cover gap-only and mixed-intent ES-fallback routing end-to-end"
```

---

## Self-Review Notes

- **Spec coverage:** category filter mapping → Task 1; snippet extraction (highlight + token trim) → Tasks 2+4; per-group starvation cap → Task 3; row shape/synthetic chunk_id → Task 4; ranking/interleave/no-raw-score-mixing → Task 5; one-ES-call-per-query/mixed-intent partitioning → Task 4+6/8; `doc_id_allowlist` respected in ES fallback → Task 4 (query) + Task 6 (circuit breaker retry); timing/parallelism → Task 6; dense pass untouched → Task 5 (no-op path) + Task 8's assertions only exercise sparse-side rows; CLAUDE.md rule 3 rewrite → Task 7. No spec section without a task.
- **Placeholder scan:** no TBD/TODO, every step has real code, every test has real assertions.
- **Type consistency:** `retrieve()`'s new `es_client` positional parameter matches across Task 6's implementation and every test call site added in Tasks 6 and 8. `sparse_fallback_search`'s signature (`client, query, groups, doc_id_allowlist=None, limit=..., group_cap=...`) is identical between its Task 4 definition and every call site in Task 6. Row shape (`chunk_id`/`doc_id`/`text`/`score`/`source`) is identical across Tasks 4, 5, 6, 8's fixtures.
