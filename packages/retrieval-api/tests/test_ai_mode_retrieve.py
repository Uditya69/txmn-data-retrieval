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


def test_collection_trace_caps_top_hits_at_five_and_builds_preview():
    from retrieval_api.trace_utils import collection_trace

    rows = [{"chunk_id": f"c{i}", "doc_id": "d1", "text": "x" * 250, "score": float(i)} for i in range(7)]
    trace = collection_trace({"ruling": rows})

    assert trace == {
        "collections": [{
            "name": "ruling",
            "hit_count": 7,
            "top_hits": [
                {"chunk_id": f"c{i}", "doc_id": "d1", "score": float(i), "text_preview": "x" * 200}
                for i in range(5)
            ],
        }]
    }


@pytest.mark.asyncio
async def test_retrieve_emits_dense_sparse_and_rrf_merge_steps(monkeypatch):
    import retrieval_api.ai_mode.retrieve as module

    gateway = AsyncMock()
    gateway.embed.return_value = [0.1, 0.2]

    async def fake_hybrid_search(client, collections, dense_vector, sparse_query_text, doc_id_allowlist=None, limit=50):
        if dense_vector is not None:
            return {"ruling": [{"chunk_id": "a", "doc_id": "d1", "text": "dense text", "score": 0.9}]}
        return {"ruling": [{"chunk_id": "b", "doc_id": "d1", "text": "sparse text", "score": 5.0}]}

    async def fake_sparse_fallback_search(client, query, groups, doc_id_allowlist=None):
        return {}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(module, "sparse_fallback_search", fake_sparse_fallback_search)
    steps = []

    async def on_step(step, data):
        steps.append(step)

    result = await module.retrieve(
        gateway, milvus_client=object(), es_client=object(), search_query="q", doc_id_allowlist=None, on_step=on_step,
    )

    assert steps == ["milvus_dense", "milvus_sparse", "rrf_merge"]
    assert {row["chunk_id"] for row in result} == {"a", "b"}


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

    async def fake_sparse_fallback_search(client, query, groups, doc_id_allowlist=None):
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

    rrf_step = next(data for step, data in steps if step == "rrf_merge")
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

    async def fake_sparse_fallback_search(client, query, groups, doc_id_allowlist=None):
        return {}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(module, "sparse_fallback_search", fake_sparse_fallback_search)

    await module.retrieve(
        gateway, milvus_client=object(), es_client=object(), search_query="q", doc_id_allowlist=None, intent=["acts"],
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

    async def fake_sparse_fallback_search(client, query, groups, doc_id_allowlist=None):
        return {}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(module, "sparse_fallback_search", fake_sparse_fallback_search)

    await module.retrieve(gateway, milvus_client=object(), es_client=object(), search_query="q", doc_id_allowlist=None)

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

    async def fake_sparse_fallback_search(client, query, groups, doc_id_allowlist=None):
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
    assert [step for step, _ in steps] == ["filter_fallback", "milvus_dense", "milvus_sparse", "rrf_merge"]
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

    async def fake_sparse_fallback_search(client, query, groups, doc_id_allowlist=None):
        return {}

    monkeypatch.setattr(module, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(module, "sparse_fallback_search", fake_sparse_fallback_search)

    result = await module.retrieve(gateway, milvus_client=object(), es_client=object(), search_query="q", doc_id_allowlist=None)

    assert result == []
    assert len(calls) == 2  # dense + sparse, no retry


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
