from unittest.mock import AsyncMock
import pytest

from retrieval_api.ai_mode.retrieve import rrf_merge, retrieve, _flatten


def test_rrf_merge_combines_and_ranks_by_reciprocal_rank():
    dense = [{"chunk_id": "a", "text": "A"}, {"chunk_id": "b", "text": "B"}]
    sparse = [{"chunk_id": "b", "text": "B"}, {"chunk_id": "c", "text": "C"}]

    merged = rrf_merge(dense, sparse, k=60)

    ids = [row["chunk_id"] for row in merged]
    assert ids[0] == "b"  # appears rank-1 sparse + rank-2 dense: highest combined score
    assert set(ids) == {"a", "b", "c"}
    assert merged[0]["rrf_score"] > merged[-1]["rrf_score"]


def test_rrf_merge_dedupes_by_chunk_id():
    dense = [{"chunk_id": "a", "text": "A"}]
    sparse = [{"chunk_id": "a", "text": "A"}]

    merged = rrf_merge(dense, sparse)

    assert len(merged) == 1


def test_rrf_merge_default_weights_are_neutral():
    dense = [{"chunk_id": "a", "text": "A"}, {"chunk_id": "b", "text": "B"}]
    sparse = [{"chunk_id": "b", "text": "B"}, {"chunk_id": "c", "text": "C"}]

    merged = rrf_merge(dense, sparse, k=60)

    ids = [row["chunk_id"] for row in merged]
    assert ids[0] == "b"
    assert set(ids) == {"a", "b", "c"}


def test_rrf_merge_upweights_dense_list_over_sparse_when_explicitly_passed():
    # rrf_merge itself still accepts explicit weights (used by callers other than
    # retrieve(), and by these direct unit tests) - only retrieve() no longer
    # resolves non-neutral weights from intent.
    dense = [{"chunk_id": "a", "text": "A"}]
    sparse = [{"chunk_id": "c", "text": "C"}]

    merged = rrf_merge(dense, sparse, k=60, dense_weight=1.5, sparse_weight=0.5)

    assert merged[0]["chunk_id"] == "a"
    assert merged[0]["rrf_score"] > merged[1]["rrf_score"]


def test_rrf_merge_upweights_sparse_list_over_dense_when_explicitly_passed():
    dense = [{"chunk_id": "a", "text": "A"}]
    sparse = [{"chunk_id": "c", "text": "C"}]

    merged = rrf_merge(dense, sparse, k=60, dense_weight=0.5, sparse_weight=1.5)

    assert merged[0]["chunk_id"] == "c"
    assert merged[0]["rrf_score"] > merged[1]["rrf_score"]


@pytest.mark.asyncio
async def test_retrieve_embeds_search_query_and_merges_dense_sparse(monkeypatch):
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        if dense_vector is not None:
            return {"ruling": [{"chunk_id": "a", "doc_id": "d1", "text": "t", "score": 0.9}]}
        return {}  # ruling has no native sparse_vector - matches real SPARSE_VECTOR_COLLECTIONS behavior

    async def fake_sparse_fallback_search(client, query, groups, doc_id_allowlist=None, boost=False):
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


@pytest.mark.asyncio
async def test_retrieve_uses_raw_query_for_es_fallback_not_rewritten_search_query(monkeypatch):
    """ES fallback must search the user's own words - the LLM-rewritten search_query is what
    Milvus embeds/searches on, but ES already has its own query-shape/phrase/synonym handling
    tuned against real user phrasing (_build_field_query), and a rewrite on top of that risks
    drifting the exact match the boost toggle's phrase boosts depend on."""
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]
    seen_queries = []

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        seen_queries.append(("hybrid_search", sparse_query_text))
        return {}

    async def fake_sparse_fallback_search(client, query, groups, doc_id_allowlist=None, boost=False):
        seen_queries.append(("sparse_fallback_search", query))
        return {}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(module, "sparse_fallback_search", fake_sparse_fallback_search)

    await retrieve(
        gateway, milvus_client=object(), es_client=object(), search_query="rewritten by LLM",
        doc_id_allowlist=None, intent=["acts"], raw_query="section 55",
    )

    gateway.embed.assert_awaited_once_with(role="query_embed", text="rewritten by LLM")
    assert ("sparse_fallback_search", "section 55") in seen_queries
    assert ("hybrid_search", "rewritten by LLM") in seen_queries


@pytest.mark.asyncio
async def test_retrieve_falls_back_to_search_query_for_es_fallback_when_raw_query_omitted(monkeypatch):
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]
    seen_queries = []

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {}

    async def fake_sparse_fallback_search(client, query, groups, doc_id_allowlist=None, boost=False):
        seen_queries.append(query)
        return {}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(module, "sparse_fallback_search", fake_sparse_fallback_search)

    await retrieve(
        gateway, milvus_client=object(), es_client=object(), search_query="rewritten by LLM",
        doc_id_allowlist=None, intent=["acts"],
    )

    assert seen_queries == ["rewritten by LLM"]


@pytest.mark.asyncio
async def test_retrieve_forwards_boost_flag_to_sparse_fallback_search(monkeypatch):
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]
    seen_boost = []

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {}

    async def fake_sparse_fallback_search(client, query, groups, doc_id_allowlist=None, boost=False):
        seen_boost.append(boost)
        return {}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(module, "sparse_fallback_search", fake_sparse_fallback_search)

    await retrieve(
        gateway, milvus_client=object(), es_client=object(), search_query="q", doc_id_allowlist=None,
        intent=["acts"], boost=True,
    )

    assert seen_boost == [True]


@pytest.mark.asyncio
async def test_retrieve_defaults_boost_to_false(monkeypatch):
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]
    seen_boost = []

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {}

    async def fake_sparse_fallback_search(client, query, groups, doc_id_allowlist=None, boost=False):
        seen_boost.append(boost)
        return {}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(module, "sparse_fallback_search", fake_sparse_fallback_search)

    await retrieve(
        gateway, milvus_client=object(), es_client=object(), search_query="q", doc_id_allowlist=None,
        intent=["acts"],
    )

    assert seen_boost == [False]


@pytest.mark.asyncio
async def test_retrieve_calls_es_fallback_only_for_routed_gap_collections(monkeypatch):
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]
    es_calls = []

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"case_summary": [{"chunk_id": "a", "doc_id": "d1", "text": "t", "score": 1.0}]}

    async def fake_sparse_fallback_search(client, query, groups, doc_id_allowlist=None, boost=False):
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

    async def fake_sparse_fallback_search(client, query, groups, doc_id_allowlist=None, boost=False):
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


def test_collection_trace_caps_top_hits_at_five_and_builds_preview():
    from retrieval_api.trace_utils import collection_trace

    rows = [{"chunk_id": f"c{i}", "doc_id": "d1", "text": "x" * 250, "score": float(i)} for i in range(7)]
    trace = collection_trace({"ruling": rows})

    assert trace == {
        "collections": [{
            "name": "ruling",
            "hit_count": 7,
            "top_hits": [
                {"chunk_id": f"c{i}", "doc_id": "d1", "score": float(i), "text_preview": "x" * 200, "origin": "milvus"}
                for i in range(5)
            ],
        }]
    }


def test_collection_trace_tags_es_fallback_rows_with_es_origin():
    from retrieval_api.trace_utils import collection_trace

    rows = [{"chunk_id": "es:d1:0", "doc_id": "d1", "text": "x", "score": 1.0, "source": "es_fallback"}]
    trace = collection_trace({"act_section": rows})

    assert trace["collections"][0]["top_hits"][0]["origin"] == "es"


@pytest.mark.asyncio
async def test_retrieve_emits_dense_sparse_and_rrf_merge_steps(monkeypatch):
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        if dense_vector is not None:
            return {"ruling": [{"chunk_id": "a", "doc_id": "d1", "text": "dense text", "score": 0.9}]}
        return {"ruling": [{"chunk_id": "b", "doc_id": "d1", "text": "sparse text", "score": 5.0}]}

    async def fake_sparse_fallback_search(client, query, groups, doc_id_allowlist=None, boost=False):
        return {}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(module, "sparse_fallback_search", fake_sparse_fallback_search)
    steps = []

    async def on_step(step, data):
        steps.append(step)

    result = await module.retrieve(
        gateway, milvus_client=object(), es_client=object(), search_query="q", doc_id_allowlist=None, on_step=on_step,
        milvus_sparse_enabled=True,
    )

    assert steps == ["ai_milvus_dense", "ai_milvus_sparse", "ai_rrf_merge"]
    assert {row["chunk_id"] for row in result} == {"a", "b"}


@pytest.mark.asyncio
async def test_retrieve_ai_milvus_dense_trace_includes_the_query_it_searched_with(monkeypatch):
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {}

    async def fake_sparse_fallback_search(client, query, groups, doc_id_allowlist=None, boost=False):
        return {}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(module, "sparse_fallback_search", fake_sparse_fallback_search)
    traces = {}

    async def on_step(step, data):
        traces[step] = data

    await module.retrieve(
        gateway, milvus_client=object(), es_client=object(), search_query="rewritten by LLM",
        doc_id_allowlist=None, intent=["acts"], on_step=on_step, raw_query="section 55",
    )

    assert traces["ai_milvus_dense"]["query"] == "rewritten by LLM"
    assert traces["ai_milvus_sparse"]["milvus_query"] == "rewritten by LLM"
    assert traces["ai_milvus_sparse"]["es_query"] == "section 55"
    es_query_body = traces["ai_milvus_sparse"]["es_query_body"]
    assert {"terms": {"groups.group.name.keyword": ["ACT"]}} in es_query_body["bool"]["must"]


@pytest.mark.asyncio
async def test_retrieve_ai_milvus_sparse_trace_omits_es_query_when_no_gap_collection_routed(monkeypatch):
    """A category set that happens to route zero gap collections must not show an es_query -
    no ES call was made, so there's nothing to preview."""
    import retrieval_api.ai_mode.retrieve as module
    from common.schemas import CATEGORY_COLLECTIONS

    monkeypatch.setitem(CATEGORY_COLLECTIONS, "acts", ["case_summary"])

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {}

    es_fallback_called = False

    async def fake_sparse_fallback_search(client, query, groups, doc_id_allowlist=None, boost=False):
        nonlocal es_fallback_called
        es_fallback_called = True
        return {}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(module, "sparse_fallback_search", fake_sparse_fallback_search)
    traces = {}

    async def on_step(step, data):
        traces[step] = data

    await module.retrieve(
        gateway, milvus_client=object(), es_client=object(), search_query="q",
        doc_id_allowlist=None, intent=["acts"], on_step=on_step, milvus_sparse_enabled=True,
    )

    assert not es_fallback_called
    assert "es_query" not in traces["ai_milvus_sparse"]
    assert "es_query_body" not in traces["ai_milvus_sparse"]


@pytest.mark.asyncio
async def test_retrieve_always_uses_neutral_rrf_weighting(monkeypatch):
    """Category no longer drives RRF weighting at all (rejected during
    brainstorming - see docs/superpowers/specs/2026-08-14-category-collection-
    routing-design.md). Every intent value, including ones that would have
    skewed weighting under the old 4-value enum, must resolve to (1.0, 1.0)."""
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"ruling": [{"chunk_id": "a", "doc_id": "d1", "text": "t", "score": 0.9}]}

    async def fake_sparse_fallback_search(client, query, groups, doc_id_allowlist=None, boost=False):
        return {}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(module, "sparse_fallback_search", fake_sparse_fallback_search)
    steps = []

    async def on_step(step, data):
        steps.append((step, data))

    await module.retrieve(
        gateway, milvus_client=object(), es_client=object(), search_query="q", doc_id_allowlist=None,
        intent=["caselaws"], on_step=on_step,
    )

    rrf_step = next(data for step, data in steps if step == "ai_rrf_merge")
    assert rrf_step["dense_weight"] == 1.0
    assert rrf_step["sparse_weight"] == 1.0


@pytest.mark.asyncio
async def test_retrieve_routes_collections_by_intent(monkeypatch):
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]
    seen_collections = []

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        seen_collections.append(collections)
        return {}

    async def fake_sparse_fallback_search(client, query, groups, doc_id_allowlist=None, boost=False):
        return {}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(module, "sparse_fallback_search", fake_sparse_fallback_search)

    await module.retrieve(
        gateway, milvus_client=object(), es_client=object(), search_query="q", doc_id_allowlist=None, intent=["acts"],
        milvus_sparse_enabled=True,
    )

    # This test verifies retrieve.py passes the routed collection set through to hybrid_search
    # unchanged for both the dense and sparse passes - it does NOT reflect what actually gets
    # searched in production. act_section has no sparse_vector field (excluded from
    # common.schemas.SPARSE_VECTOR_COLLECTIONS), so the real hybrid_search further drops it from
    # the sparse pass, meaning an intent=["acts"]-only query's sparse pass searches zero
    # collections. That filtering is hybrid_search's own responsibility and is covered by
    # packages/common/tests/test_milvus_client.py::test_hybrid_search_skips_sparse_search_for_ruling_collection
    # (same mechanism, "ruling" collection), not retrieve.py's.
    assert seen_collections == [["act_section"], ["act_section"]]  # dense pass, sparse pass


@pytest.mark.asyncio
async def test_retrieve_defaults_to_all_collections_when_intent_omitted(monkeypatch):
    import retrieval_api.ai_mode.retrieve as module
    from common.schemas import MILVUS_COLLECTIONS

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]
    seen_collections = []

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        seen_collections.append(collections)
        return {}

    async def fake_sparse_fallback_search(client, query, groups, doc_id_allowlist=None, boost=False):
        return {}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(module, "sparse_fallback_search", fake_sparse_fallback_search)

    await module.retrieve(
        gateway, milvus_client=object(), es_client=object(), search_query="q", doc_id_allowlist=None,
        milvus_sparse_enabled=True,
    )

    assert seen_collections[0] == MILVUS_COLLECTIONS


@pytest.mark.asyncio
async def test_retrieve_falls_back_to_unfiltered_when_allowlist_zeroes_everything(monkeypatch):
    """A resolved doc_id_allowlist that's non-empty but wrong-typed/disjoint from the
    target Milvus collections must not silently return zero candidates when an
    unfiltered search would find real matches - retry once unfiltered instead,
    within the same routed collection set."""
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]
    seen_collections = []

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        seen_collections.append(collections)
        if doc_id_allowlist is not None:
            return {"ruling": []}
        return {"ruling": [{"chunk_id": "a", "doc_id": "d1", "text": "t", "score": 0.9}]}

    async def fake_sparse_fallback_search(client, query, groups, doc_id_allowlist=None, boost=False):
        return {}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(module, "sparse_fallback_search", fake_sparse_fallback_search)
    steps = []

    async def on_step(step, data):
        steps.append((step, data))

    result = await module.retrieve(
        gateway, milvus_client=object(), es_client=object(), search_query="q",
        doc_id_allowlist=["wrong-doc-id"], intent=["caselaws"], on_step=on_step,
    )

    assert result[0]["chunk_id"] == "a"
    assert [step for step, _ in steps] == ["filter_fallback", "ai_milvus_dense", "ai_milvus_sparse", "ai_rrf_merge"]
    fallback_data = next(data for step, data in steps if step == "filter_fallback")
    assert fallback_data["doc_id_allowlist_count"] == 1
    gateway.embed.assert_awaited_once()  # retry reuses the already-computed embedding
    # every hybrid_search call (both the initial pair and the retry pair) used the
    # same routed collection set - the retry drops the allowlist, not the routing.
    assert all(collections == ["case_summary", "digest", "headnotes", "facts", "held", "ruling", "metadata"] for collections in seen_collections)


@pytest.mark.asyncio
async def test_retrieve_does_not_fall_back_when_no_allowlist_was_applied(monkeypatch):
    """Zero hits with no allowlist at all is just a genuinely empty result, not the
    disjoint-allowlist failure mode - must not trigger a pointless retry."""
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]
    calls = []

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        calls.append(doc_id_allowlist)
        return {"ruling": []}

    async def fake_sparse_fallback_search(client, query, groups, doc_id_allowlist=None, boost=False):
        return {}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(module, "sparse_fallback_search", fake_sparse_fallback_search)

    result = await module.retrieve(
        gateway, milvus_client=object(), es_client=object(), search_query="q", doc_id_allowlist=None,
        milvus_sparse_enabled=True,
    )

    assert result == []
    assert len(calls) == 2  # dense + sparse, no retry


@pytest.mark.asyncio
async def test_retrieve_survives_es_fallback_raising(monkeypatch):
    """ES was never in the retrieve() path before this branch - a fallback path must
    degrade gracefully, not escalate an ES hiccup (timeout, 5xx,
    index.highlight.max_analyzed_offset on a long judgment) into a total query failure
    for dense/native-sparse results that would otherwise have been fine."""
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        if dense_vector is not None:
            return {"ruling": [{"chunk_id": "a", "doc_id": "d1", "text": "dense hit", "score": 0.9}]}
        return {}

    async def fake_sparse_fallback_search(client, query, groups, doc_id_allowlist=None, boost=False):
        raise RuntimeError("index.highlight.max_analyzed_offset")

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(module, "sparse_fallback_search", fake_sparse_fallback_search)
    steps = []

    async def on_step(step, data):
        steps.append((step, data))

    result = await module.retrieve(
        gateway, milvus_client=object(), es_client=object(), search_query="q",
        doc_id_allowlist=None, intent=["caselaws"], on_step=on_step,
    )

    assert [row["chunk_id"] for row in result] == ["a"]
    assert "es_fallback_degraded" in [step for step, _ in steps]


def test_flatten_plain_sort_unchanged_when_no_es_fallback_rows():
    by_collection = {
        "held": [{"chunk_id": "h1", "score": 3.0}],
        "facts": [{"chunk_id": "f1", "score": 9.0}, {"chunk_id": "f2", "score": 1.0}],
    }
    result = _flatten(by_collection)

    assert [row["chunk_id"] for row in result] == ["f1", "h1", "f2"]


def test_flatten_interleaves_native_and_es_fallback_rows_by_local_rank():
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
    by_collection = {
        "ruling": [
            {"chunk_id": "e1", "score": 2.0, "source": "es_fallback"},
            {"chunk_id": "e2", "score": 5.0, "source": "es_fallback"},
        ],
    }
    result = _flatten(by_collection)

    assert [row["chunk_id"] for row in result] == ["e2", "e1"]


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

    async def fake_sparse_fallback_search(client, query, groups, doc_id_allowlist=None, boost=False):
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

    async def fake_sparse_fallback_search(client, query, groups, doc_id_allowlist=None, boost=False):
        assert sorted(groups) == sorted(["CASELAWS", "Experts Opinion"])
        return {
            "ruling": [{"chunk_id": "es-ruling", "doc_id": "doc-ruling", "text": "t", "score": 3.0, "source": "es_fallback"}],
            "article_section": [{"chunk_id": "es-article", "doc_id": "doc-article", "text": "t", "score": 2.0, "source": "es_fallback"}],
        }

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(module, "sparse_fallback_search", fake_sparse_fallback_search)

    result = await retrieve(gateway, milvus_client=object(), es_client=object(), search_query="q",
                             doc_id_allowlist=None, intent=["caselaws", "articles"], milvus_sparse_enabled=True)

    chunk_ids = {row["chunk_id"] for row in result}
    assert "native-held" in chunk_ids
    assert "es-ruling" in chunk_ids
    assert "es-article" in chunk_ids


@pytest.mark.asyncio
async def test_retrieve_skips_native_milvus_sparse_pass_by_default(monkeypatch):
    """milvus_sparse_enabled defaults to False - retrieve() must not call
    hybrid_search for the native sparse (dense_vector=None) pass at all unless the
    caller explicitly opts in. ES sparse-fallback for gap collections is unaffected."""
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]
    sparse_pass_called = False

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        nonlocal sparse_pass_called
        if dense_vector is None:
            sparse_pass_called = True
            return {}
        return {"held": [{"chunk_id": "a", "doc_id": "d1", "text": "t", "score": 0.9}]}

    async def fake_sparse_fallback_search(client, query, groups, doc_id_allowlist=None, boost=False):
        return {}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(module, "sparse_fallback_search", fake_sparse_fallback_search)

    result = await module.retrieve(
        gateway, milvus_client=object(), es_client=object(), search_query="q", doc_id_allowlist=None,
    )

    assert not sparse_pass_called
    assert [row["chunk_id"] for row in result] == ["a"]


@pytest.mark.asyncio
async def test_retrieve_omits_ai_milvus_sparse_step_when_nothing_ran(monkeypatch):
    """No gap collections routed and native Milvus sparse disabled means nothing
    contributed to the sparse pass at all - the step must not be emitted (an empty
    'Milvus sparse search' card with zero collections is useless noise)."""
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        return {"held": [{"chunk_id": "a", "doc_id": "d1", "text": "t", "score": 0.9}]}

    async def fake_sparse_fallback_search(client, query, groups, doc_id_allowlist=None, boost=False):
        return {}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(module, "collections_for_intent", lambda intent: ["held"])
    monkeypatch.setattr(module, "sparse_fallback_search", fake_sparse_fallback_search)
    steps = []

    async def on_step(step, data):
        steps.append(step)

    await module.retrieve(
        gateway, milvus_client=object(), es_client=object(), search_query="q",
        doc_id_allowlist=None, intent=["caselaws"], on_step=on_step,
    )

    assert steps == ["ai_milvus_dense", "ai_rrf_merge"]
